import torch
import role_low_rank as RLR
import struct_role as SRB
MECHANISM = 'target_col_mask'
VERSION = 1
QUERY_ROLES = (RLR.ROLE_CTA_TARGET, RLR.ROLE_CPA_HEAD, RLR.ROLE_CPA_TAIL)
EXTRA_COLS = {RLR.ROLE_CTA_TARGET: (), RLR.ROLE_CPA_HEAD: (), RLR.ROLE_CPA_TAIL: ('head',)}
RULE = {'cta_target': 'its own column', 'cpa_head': 'its own column', 'cpa_tail': 'its own column + the head column'}
ALLOWED_BASE = ('sdpa', 'eager')
_MASK_DTYPE = {'sdpa': None, 'eager': torch.float32}
NO_COL = -1

def columns_from_struct(struct_ids):
    st = torch.as_tensor(struct_ids)
    (_row, col, _hdr) = SRB.unpack_struct(st)
    return torch.where(st != SRB.STRUCT_NONE, col.long(), torch.full_like(col.long(), NO_COL))

def _role_is(role_ids, wanted):
    r = torch.as_tensor(role_ids)
    out = torch.zeros_like(r, dtype=torch.bool)
    for w in wanted:
        out |= r == w
    return out

def head_column_flags(cols, role_ids):
    cols = torch.as_tensor(cols)
    is_head = _role_is(role_ids, (RLR.ROLE_CPA_HEAD,)) & (cols >= 0)
    n_col = int(cols.max().item()) + 1 if bool((cols >= 0).any()) else 1
    flags = torch.zeros(cols.shape[0], n_col, dtype=torch.bool, device=cols.device)
    if bool(is_head.any()):
        (b_idx, t_idx) = is_head.nonzero(as_tuple=True)
        flags[b_idx, cols[b_idx, t_idx]] = True
    return flags

def build_mask(cols_full, role_full, cache_position, pad_mask, dtype=None):
    cols = torch.as_tensor(cols_full).long()
    role = torch.as_tensor(role_full).long()
    if cols.dim() != 2 or role.shape != cols.shape:
        raise ValueError(f'cols/roles must both be [B,T]; got {tuple(cols.shape)} and {tuple(role.shape)}')
    dev = cols.device
    pad = torch.as_tensor(pad_mask).to(dev).bool()
    if pad.dim() != 2 or pad.shape[0] != cols.shape[0]:
        raise ValueError(f'pad mask must be [B,T_k] with B={cols.shape[0]}; got {tuple(pad.shape)}')
    cp = torch.as_tensor(cache_position).to(dev).long().reshape(-1)
    role = role.to(dev)
    (B, Tp) = cols.shape
    Tk = pad.shape[1]
    if Tk <= Tp:
        (cols_k, role_k) = (cols[:, :Tk], role[:, :Tk])
    else:
        pad_n = Tk - Tp
        cols_k = torch.cat([cols, torch.full((B, pad_n), NO_COL, dtype=torch.long, device=dev)], 1)
        role_k = torch.cat([role, torch.zeros(B, pad_n, dtype=torch.long, device=dev)], 1)
    cell_k = cols_k >= 0
    in_prompt = cp < Tp
    idx_q = cp.clamp(max=Tp - 1)
    cols_q = torch.where(in_prompt[None, :], cols[:, idx_q], torch.full_like(cols[:, idx_q], NO_COL))
    role_q = torch.where(in_prompt[None, :], role[:, idx_q], torch.zeros_like(role[:, idx_q]))
    queried_q = _role_is(role_q, QUERY_ROLES) & (cols_q >= 0)
    same_col = cols_q[:, :, None] == cols_k[:, None, :]
    permitted = same_col
    tail_q = role_q == RLR.ROLE_CPA_TAIL
    if bool(tail_q.any()):
        flags = head_column_flags(cols, role)
        k_in_head = torch.zeros(B, Tk, dtype=torch.bool, device=dev)
        safe_k = cols_k.clamp(min=0, max=flags.shape[1] - 1)
        k_in_head = torch.gather(flags, 1, safe_k) & cell_k
        permitted = permitted | tail_q[:, :, None] & k_in_head[:, None, :]
    idx_k = torch.arange(Tk, device=dev)
    allowed = cp[None, :, None] >= idx_k[None, None, :]
    allowed = allowed.expand(B, -1, -1).clone()
    allowed &= pad[:, None, :]
    allowed &= ~(queried_q[:, :, None] & cell_k[:, None, :] & ~permitted)
    dead = ~allowed.any(-1)
    if bool(dead.any()):
        self_k = cp.clamp(max=Tk - 1)[None, :].expand(B, -1)
        allowed[dead] = allowed[dead].scatter(-1, self_k[dead][:, None], True)
    allowed = allowed.unsqueeze(1)
    if dtype is None:
        return allowed
    out = torch.zeros(allowed.shape, dtype=dtype, device=dev)
    return out.masked_fill_(~allowed, torch.finfo(dtype).min)

class _Ctx:
    active = False
    cols_full = None
    role_full = None
    mask_dtype = None
    n_applied = 0
    n_skipped = 0
CTX = _Ctx()

def set_context(cols_full, role_full):
    CTX.cols_full = torch.as_tensor(cols_full)
    CTX.role_full = torch.as_tensor(role_full)
    CTX.active = True

def set_context_from_ids(struct_ids, role_ids):
    set_context(columns_from_struct(struct_ids), role_ids)

def clear_context():
    CTX.cols_full = None
    CTX.role_full = None
    CTX.active = False

def reset_stats():
    CTX.n_applied = 0
    CTX.n_skipped = 0

def stats():
    return {'forwards_masked': CTX.n_applied, 'forwards_unmasked': CTX.n_skipped}

def _mask_prehook(module, args, kwargs):
    if not CTX.active or CTX.cols_full is None:
        CTX.n_skipped += 1
        return None
    am = kwargs.get('attention_mask', None)
    if am is not None and getattr(am, 'ndim', 0) == 4:
        return None
    ids = kwargs.get('input_ids', None)
    if ids is None and args:
        ids = args[0]
    if ids is None and kwargs.get('inputs_embeds', None) is not None:
        ids = kwargs['inputs_embeds']
    if ids is None:
        CTX.n_skipped += 1
        return None
    n_q = int(ids.shape[1])
    cp = kwargs.get('cache_position', None)
    dev = ids.device
    if am is None:
        n_k = n_q if cp is None else int(torch.as_tensor(cp).reshape(-1)[-1]) + 1
        am = torch.ones(ids.shape[0], n_k, dtype=torch.bool, device=dev)
    if cp is None:
        cp = torch.arange(am.shape[1] - n_q, am.shape[1], device=dev)
    kwargs = dict(kwargs)
    kwargs['attention_mask'] = build_mask(CTX.cols_full.to(dev), CTX.role_full.to(dev), cp, am, dtype=CTX.mask_dtype)
    CTX.n_applied += 1
    return (args, kwargs)

def attach(model, base_impl='sdpa'):
    if base_impl not in ALLOWED_BASE:
        raise ValueError(f'--{MECHANISM} needs --attn_impl in {list(ALLOWED_BASE)} (got {base_impl!r}): flash_attention_2 cannot take a custom 4D attention mask.')
    if getattr(model, '_tgt_col_mask_handles', None):
        raise RuntimeError(f'{MECHANISM} is already attached to this model')
    dt = _MASK_DTYPE[base_impl]
    if base_impl == 'eager':
        dt = next((p.dtype for p in model.parameters() if p.is_floating_point), torch.float32)
    CTX.mask_dtype = dt
    reset_stats()
    inner = model.get_base_model() if hasattr(model, 'get_base_model') else model
    hosts = [model] if inner is model else [model, inner]
    handles = [h.register_forward_pre_hook(_mask_prehook, with_kwargs=True) for h in hosts]
    model._tgt_col_mask_handles = handles
    return {'mechanism': MECHANISM, 'version': VERSION, 'inference_only': True, 'rule': dict(RULE), 'base_attn_impl': base_impl, 'mask_kind': 'bool' if dt is None else f'additive {dt}', 'prehook_on': [type(h).__name__ for h in hosts], 'query_roles': [RLR.ROLE_NAMES[r] for r in QUERY_ROLES], 'cut_keys': 'table CELL content outside the permitted column(s); header cells included', 'untouched_keys': 'instruction, candidate list, query lines, chat/control tokens, Markdown delimiters and separator row, answer, padding', 'n_new_parameters': 0}

def detach(model):
    for h in getattr(model, '_tgt_col_mask_handles', []) or []:
        h.remove()
    model._tgt_col_mask_handles = []
    clear_context()

def audit(struct_ids, role_ids):
    cols = columns_from_struct(struct_ids)
    role = torch.as_tensor(role_ids)
    cell = cols >= 0
    q = _role_is(role, QUERY_ROLES) & cell
    return {'n_examples': int(cols.shape[0]), 'n_cell_tokens': int(cell.sum()), 'n_queried_cell_tokens': int(q.sum()), 'n_cta_target': int((_role_is(role, (RLR.ROLE_CTA_TARGET,)) & cell).sum()), 'n_cpa_head': int((_role_is(role, (RLR.ROLE_CPA_HEAD,)) & cell).sum()), 'n_cpa_tail': int((_role_is(role, (RLR.ROLE_CPA_TAIL,)) & cell).sum()), 'n_examples_without_a_queried_cell': int((~q.any(1)).sum()), 'n_roles_without_a_column': int((_role_is(role, QUERY_ROLES) & ~cell).sum())}
