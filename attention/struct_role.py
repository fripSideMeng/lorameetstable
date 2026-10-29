import json
import os
from pathlib import Path
import torch
import torch.nn as nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask, eager_mask
import attn_lora as AL
import recv_gate as RCV
import role_low_rank as RLR
MECHANISM = 'struct_role_bias'
VERSION = 1
(REL_SAME_ROW, REL_SAME_COL, REL_TARGET_KEY, REL_CONTEXT_KEY) = (0, 1, 2, 3)
REL_NAMES = ('same_row', 'same_column', 'target_key', 'context_key')
N_REL = len(REL_NAMES)
STRUCTURE_RELATIONS = (REL_SAME_ROW, REL_SAME_COL)
ROLE_RELATIONS = (REL_TARGET_KEY, REL_CONTEXT_KEY)
RELATION_SETS = {'structure': STRUCTURE_RELATIONS, 'all': (0, 1, 2, 3)}
DEFAULT_RELATION_SET = 'structure'
STATE_FILE = 'struct_role_bias.pt'
CONFIG_FILE = 'struct_role_bias_config.json'
_REG_NAME = 'struct_role_bias'
ALLOWED_BASE = ('sdpa', 'eager')
_BASE_MASK_FN = {'sdpa': sdpa_mask, 'eager': eager_mask}
PARAM_DTYPE = torch.float32
MAX_GEN_BATCH = 2
STRUCT_NONE = 0
_COL_BITS = 20
_COL_MASK = (1 << _COL_BITS) - 1
(KEY_NONE, KEY_TARGET, KEY_CONTEXT) = (0, 1, 2)
TARGET_ROLES = (RLR.ROLE_CTA_TARGET, RLR.ROLE_CPA_HEAD, RLR.ROLE_CPA_TAIL)
CONTEXT_ROLES = (RLR.ROLE_CTA_CONTEXT, RLR.ROLE_CPA_CONTEXT)
(_REL_NONE, _REL_ROW, _REL_COL) = (0, 1, 2)

def pack_struct(row, col, is_header):
    row = int(row)
    col = int(col)
    if row < 0 or col < 0:
        raise ValueError(f'row/col ids must be non-negative (got {row}, {col})')
    if col > _COL_MASK:
        raise ValueError(f'column id {col} does not fit in {_COL_BITS} bits')
    return (row << _COL_BITS | col) << 1 | int(bool(is_header))

def unpack_struct(code):
    c = torch.as_tensor(code).long()
    hdr = c & 1
    rest = c >> 1
    return (rest >> _COL_BITS, rest & _COL_MASK, hdr)

def build_struct(tok, rendered, expect_ids=None):
    if not getattr(tok, 'is_fast', False):
        raise RuntimeError(f'{MECHANISM} requires a fast tokenizer (needs return_offsets_mapping)')
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    (ids, offs) = (enc['input_ids'], enc['offset_mapping'])
    if expect_ids is not None and list(ids) != list(expect_ids):
        raise RuntimeError(f"structural tokenization ({len(ids)} tokens) does not reproduce the model's own prompt tokenization ({len(expect_ids)} tokens). The row/column coordinates must be aligned to the ids the model is fed; refusing to train on a misaligned sequence.")
    (rows, cols, hdr, cnt, _) = AL.structure_from_offsets(rendered, offs)
    return [pack_struct(rows[i], cols[i], hdr[i]) if cnt[i] else STRUCT_NONE for i in range(len(ids))]

def pad_struct(struct_lists, width, side='right'):
    out = []
    for s in struct_lists:
        n = width - len(s)
        if n < 0:
            raise ValueError('struct sequence longer than the padded width')
        out.append([STRUCT_NONE] * n + list(s) if side == 'left' else list(s) + [STRUCT_NONE] * n)
    return out

def struct_from_attention_mask(struct_lists, attn_mask):
    T = attn_mask.shape[1]
    rows = []
    for (s, m) in zip(struct_lists, attn_mask.tolist()):
        n_pad = len(m) - int(sum(m))
        if any((m[k] == 0 for k in range(n_pad, T))) or any((m[k] == 1 for k in range(n_pad))):
            raise RuntimeError("struct ids can only be aligned to a LEFT-padded batch; set tok.padding_side='left'")
        if n_pad + len(s) != T:
            raise RuntimeError(f'left-pad width {n_pad} + {len(s)} struct ids != padded length {T}')
        rows.append([STRUCT_NONE] * n_pad + list(s))
    return torch.tensor(rows, dtype=torch.long)

def _role_lut(members, device):
    lut = torch.zeros(RLR.N_ROLES, dtype=torch.uint8, device=device)
    for r in members:
        lut[int(r)] = 1
    return lut

def receiver_mask_from_regions(region_ids):
    return RCV.receiver_mask_from_regions(region_ids)

def key_class_from_roles(role_ids):
    r = torch.as_tensor(role_ids).long()
    dev = r.device
    tgt = _role_lut(TARGET_ROLES, dev)[r]
    ctx = _role_lut(CONTEXT_ROLES, dev)[r]
    return (tgt * KEY_TARGET + ctx * KEY_CONTEXT).to(torch.uint8)

def resolve_relations(relations):
    if relations is None:
        relations = DEFAULT_RELATION_SET
    if isinstance(relations, str):
        if relations not in RELATION_SETS:
            raise ValueError(f'relation set must be one of {sorted(RELATION_SETS)} (got {relations!r})')
        return tuple(RELATION_SETS[relations])
    out = []
    for r in relations:
        i = REL_NAMES.index(r) if isinstance(r, str) else int(r)
        if not 0 <= i < N_REL:
            raise ValueError(f'unknown relation {r!r}')
        out.append(i)
    if not out:
        raise ValueError('at least one relation must be active; an empty set is the baseline')
    return tuple(sorted(set(out)))

def set_name(active):
    for (name, members) in RELATION_SETS.items():
        if tuple(active) == tuple(members):
            return name
    return '+'.join((REL_NAMES[i] for i in active))

class StructRoleBias(nn.Module):

    def __init__(self, n_layers, n_heads, relations=DEFAULT_RELATION_SET):
        super().__init__()
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.active = resolve_relations(relations)
        self.beta = nn.Parameter(torch.zeros(self.n_layers, self.n_heads, N_REL, dtype=PARAM_DTYPE))

    @property
    def roles_active(self):
        return any((i in self.active for i in ROLE_RELATIONS))

    def n_params(self):
        return int(self.n_layers * self.n_heads * len(self.active))

    def summary(self):
        with torch.no_grad():
            b = self.beta.detach().float()
            return {'n_layers': self.n_layers, 'n_heads': self.n_heads, 'relations': list(REL_NAMES), 'relations_active': [REL_NAMES[i] for i in self.active], 'relation_set': set_name(self.active), 'roles_active': self.roles_active, 'n_beta_params': self.n_params(), 'n_beta_entries': int(self.beta.numel()), 'per_relation_absmax': {n: float(b[:, :, i].abs().max()) for (i, n) in enumerate(REL_NAMES)}, 'per_relation_absmean': {n: float(b[:, :, i].abs().mean()) for (i, n) in enumerate(REL_NAMES)}, 'per_relation_mean': {n: float(b[:, :, i].mean()) for (i, n) in enumerate(REL_NAMES)}}

    def per_layer_table(self):
        with torch.no_grad():
            b = self.beta.detach().float().cpu()
            return {n: [float(x) for x in b[:, :, i].mean(dim=1).tolist()] for (i, n) in enumerate(REL_NAMES)}

def snapshot(model):
    return get_bias(model).beta.detach().clone().cpu()

def stats(model, init_snapshot=None):
    bias = get_bias(model)
    out = bias.summary()
    out['per_layer_head_mean'] = bias.per_layer_table()
    if init_snapshot is not None:
        with torch.no_grad():
            d = bias.beta.detach().float().cpu() - init_snapshot.float()
            out['move_from_init'] = {n: float(d[:, :, i].norm()) for (i, n) in enumerate(REL_NAMES)}
            out['move_from_init_total'] = float(d.norm())
    return out

class _Ctx:
    active = False
    bias_module = None
    base_attn_forward = None
    struct_ids = None
    key_cls = None
    recv = None
    rel_full = None
    n_biased = 0
    n_passthrough = 0
CTX = _Ctx()

def set_context(struct_ids, role_ids=None, region_ids=None):
    s = torch.as_tensor(struct_ids).long()
    if s.dim() != 2:
        raise ValueError(f'struct_ids must be [B,T]; got {tuple(s.shape)}')
    need_roles = CTX.bias_module is None or CTX.bias_module.roles_active
    if need_roles:
        missing = [k for (k, v) in (('role_ids', role_ids), ('region_ids', region_ids)) if v is None]
        if missing:
            raise ValueError(f"{MECHANISM}: {', '.join(missing)} missing, but {[REL_NAMES[i] for i in ROLE_RELATIONS]} are active -- the target/context KEYS come from the role ids and the RECEIVER queries from the region ids.")
    CTX.struct_ids = s
    CTX.key_cls = CTX.recv = None
    if role_ids is not None:
        r = torch.as_tensor(role_ids)
        if r.shape != s.shape:
            raise ValueError(f'role_ids {tuple(r.shape)} != struct_ids {tuple(s.shape)}')
        CTX.key_cls = key_class_from_roles(r).to(s.device)
    if region_ids is not None:
        g = torch.as_tensor(region_ids)
        if g.shape != s.shape:
            raise ValueError(f'region_ids {tuple(g.shape)} != struct_ids {tuple(s.shape)}')
        CTX.recv = receiver_mask_from_regions(g).to(s.device)
    CTX.rel_full = None
    CTX.active = True

def clear_context():
    CTX.active = False
    CTX.struct_ids = CTX.key_cls = CTX.recv = CTX.rel_full = None

def reset_stats():
    CTX.n_biased = CTX.n_passthrough = 0

def forward_stats():
    return {'layer_forwards_biased': CTX.n_biased, 'layer_forwards_passthrough': CTX.n_passthrough}

def _relation_from_codes(code_q, code_k):
    (r_q, c_q, h_q) = unpack_struct(code_q)
    (r_k, c_k, h_k) = unpack_struct(code_k)
    cnt_q = (code_q != STRUCT_NONE).unsqueeze(2)
    cnt_k = (code_k != STRUCT_NONE).unsqueeze(1)
    (ri, rj) = (r_q.unsqueeze(2), r_k.unsqueeze(1))
    (ci, cj) = (c_q.unsqueeze(2), c_k.unsqueeze(1))
    (hi, hj) = (h_q.unsqueeze(2).bool(), h_k.unsqueeze(1).bool())
    both = cnt_q & cnt_k
    same_row = both & ~hi & ~hj & (ri == rj) & (ci != cj)
    same_col = both & (ci == cj) & (ri != rj)
    rel = same_row.to(torch.uint8)
    rel += same_col.to(torch.uint8) * _REL_COL
    return rel

def _current_relation(q_len, k_len, dev, cache_position):
    code = CTX.struct_ids.to(dev)
    kcls = None if CTX.key_cls is None else CTX.key_cls.to(dev)
    recv = None if CTX.recv is None else CTX.recv.to(dev)
    T = code.shape[1]
    if q_len == T and k_len == T:
        if CTX.rel_full is None or CTX.rel_full.device != dev:
            CTX.rel_full = _relation_from_codes(code, code)
        return (CTX.rel_full, kcls, recv)
    q_pos = torch.arange(q_len, device=dev) if cache_position is None else torch.as_tensor(cache_position).to(dev).long().reshape(-1)
    k_pos = torch.arange(k_len, device=dev)
    qi = q_pos.clamp(max=T - 1)
    ki = k_pos.clamp(max=T - 1)
    in_prompt_q = (q_pos < T)[None, :]
    code_q = torch.where(in_prompt_q, code[:, qi], torch.zeros_like(code[:, qi]))
    code_k = torch.where((k_pos < T)[None, :], code[:, ki], torch.zeros_like(code[:, ki]))
    kcls_k = None if kcls is None else torch.where((k_pos < T)[None, :], kcls[:, ki], torch.zeros_like(kcls[:, ki]))
    recv_q = None if recv is None else recv[:, qi] & in_prompt_q | ~in_prompt_q
    return (_relation_from_codes(code_q, code_k), kcls_k, recv_q)
_BWD_TILE_ELEMS = 1 << 21

class _StructRoleBiasFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, beta_l, relation, key_cls, recv_q, out_dtype, active):
        roles = any((i in active for i in ROLE_RELATIONS))
        if roles:
            ctx.save_for_backward(relation, key_cls, recv_q)
        else:
            ctx.save_for_backward(relation)
        ctx.active = tuple(active)
        b = beta_l.to(out_dtype)
        H = b.shape[0]
        bias = None
        if REL_SAME_ROW in active:
            bias = b[:, REL_SAME_ROW].view(1, H, 1, 1) * (relation == _REL_ROW).unsqueeze(1)
        if REL_SAME_COL in active:
            col = (relation == _REL_COL).unsqueeze(1)
            if bias is None:
                bias = b[:, REL_SAME_COL].view(1, H, 1, 1) * col
            else:
                bias.addcmul_(b[:, REL_SAME_COL].view(1, H, 1, 1), col.to(out_dtype))
        if roles:
            rq = recv_q[:, None, :, None].to(out_dtype)
            for (rel_i, cls) in ((REL_TARGET_KEY, KEY_TARGET), (REL_CONTEXT_KEY, KEY_CONTEXT)):
                if rel_i not in active:
                    continue
                kk = (key_cls == cls)[:, None, None, :].to(out_dtype)
                if bias is None:
                    bias = b[:, rel_i].view(1, H, 1, 1) * rq * kk
                else:
                    bias.addcmul_(b[:, rel_i].view(1, H, 1, 1) * rq, kk)
        return bias

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        relation = saved[0]
        key_cls = saved[1] if len(saved) > 1 else None
        recv_q = saved[2] if len(saved) > 2 else None
        active = ctx.active
        (B, H, Tq, Tk) = grad_out.shape
        g = grad_out.new_zeros((H, N_REL), dtype=torch.float32)
        tgt = None if key_cls is None else (key_cls == KEY_TARGET)[:, None, None, :]
        ctxk = None if key_cls is None else (key_cls == KEY_CONTEXT)[:, None, None, :]
        step = max(1, _BWD_TILE_ELEMS // max(1, H * Tk))
        for i in range(0, Tq, step):
            go = grad_out[:, :, i:i + step]
            rel = relation[:, None, i:i + step]
            if REL_SAME_ROW in active:
                g[:, REL_SAME_ROW] += (go * (rel == _REL_ROW)).sum(dim=(0, 2, 3), dtype=torch.float32)
            if REL_SAME_COL in active:
                g[:, REL_SAME_COL] += (go * (rel == _REL_COL)).sum(dim=(0, 2, 3), dtype=torch.float32)
            if recv_q is None:
                continue
            rq = recv_q[:, None, i:i + step, None]
            if REL_TARGET_KEY in active:
                g[:, REL_TARGET_KEY] += (go * rq * tgt).sum(dim=(0, 2, 3), dtype=torch.float32)
            if REL_CONTEXT_KEY in active:
                g[:, REL_CONTEXT_KEY] += (go * rq * ctxk).sum(dim=(0, 2, 3), dtype=torch.float32)
        return (g, None, None, None, None, None)

def struct_role_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    base = CTX.base_attn_forward
    if not CTX.active or CTX.bias_module is None or CTX.struct_ids is None:
        CTX.n_passthrough += 1
        return base(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    beta_l = CTX.bias_module.beta[int(module.layer_idx)]
    active = CTX.bias_module.active
    if not torch.is_grad_enabled() and (not bool(torch.any(beta_l[:, list(active)] != 0))):
        CTX.n_passthrough += 1
        return base(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    (q_len, k_len) = (query.shape[2], key.shape[2])
    n_q_heads = query.shape[1]
    if int(beta_l.shape[0]) != int(n_q_heads):
        raise RuntimeError(f'{MECHANISM}: beta has {int(beta_l.shape[0])} heads but this layer runs with {int(n_q_heads)} query heads. beta is indexed by QUERY head.')
    (dev, dtype) = (query.device, query.dtype)
    cache_position = kwargs.get('cache_position', None)
    (relation, key_cls, recv_q) = _current_relation(q_len, k_len, dev, cache_position)
    add_mask = AL._to_additive_mask(attention_mask, q_len, k_len, dtype, dev, cache_position)
    bias = _StructRoleBiasFn.apply(beta_l, relation, key_cls, recv_q, dtype, active)
    bias.add_(add_mask)
    CTX.n_biased += 1
    return base(module, query, key, value, bias, scaling=scaling, dropout=dropout, **kwargs)
ALL_ATTENTION_FUNCTIONS.register(_REG_NAME, struct_role_attention_forward)
AttentionMaskInterface.register(_REG_NAME, _BASE_MASK_FN['sdpa'])

def count_query_heads(model):
    for cfg in AL._iter_configs(model):
        n = getattr(cfg, 'num_attention_heads', None)
        if n:
            return int(n)
    raise RuntimeError('count_query_heads: no config exposes num_attention_heads')

def attach(model, base_impl='sdpa', n_layers=None, n_heads=None, relations=DEFAULT_RELATION_SET):
    if base_impl not in ALLOWED_BASE:
        raise ValueError(f'--struct_role_bias needs --attn_impl in {list(ALLOWED_BASE)} (got {base_impl!r}): flash_attention_2 cannot take the additive structural mask.')
    if getattr(model, 'struct_role_bias', None) is not None:
        raise RuntimeError(f'{MECHANISM} is already attached to this model')
    n_layers = int(n_layers) if n_layers else AL.count_layers(model)
    n_heads = int(n_heads) if n_heads else count_query_heads(model)
    bias = StructRoleBias(n_layers, n_heads, relations=relations)
    dev = next((p.device for p in model.parameters()), torch.device('cpu'))
    bias.to(device=dev, dtype=PARAM_DTYPE)
    model.struct_role_bias = bias
    if base_impl == 'eager':
        import transformers.models.llama.modeling_llama as _llama
        CTX.base_attn_forward = _llama.eager_attention_forward
    else:
        CTX.base_attn_forward = ALL_ATTENTION_FUNCTIONS[base_impl]
    AttentionMaskInterface.register(_REG_NAME, _BASE_MASK_FN[base_impl])
    CTX.bias_module = bias
    reset_stats()
    n_set = 0
    for m in model.modules():
        cfg = getattr(m, 'config', None)
        if cfg is not None and hasattr(cfg, '_attn_implementation') and m.__class__.__name__.endswith('Attention'):
            cfg._attn_implementation = _REG_NAME
            n_set += 1
    for cfg in AL._iter_configs(model):
        if hasattr(cfg, '_attn_implementation'):
            cfg._attn_implementation = _REG_NAME
    if n_set == 0:
        raise RuntimeError(f'{MECHANISM}: no *Attention submodules found to patch')
    _act = [REL_NAMES[i] for i in bias.active]
    _off = [n for n in REL_NAMES if n not in _act]
    return {'mechanism': MECHANISM, 'version': VERSION, 'relations': list(REL_NAMES), 'relations_active': _act, 'relations_pinned_at_zero': _off, 'relation_set': set_name(bias.active), 'roles_active': bias.roles_active, 'n_layers': n_layers, 'n_heads': n_heads, 'n_beta_params': bias.n_params(), 'param_count_formula': f'{n_layers} layers x {n_heads} query heads x {len(bias.active)} active relation(s) = {bias.n_params():,} trainable' + (f" ({', '.join(_off)} pinned at 0, never in the graph)" if _off else ''), 'beta_shape': [n_layers, n_heads, N_REL], 'beta_dtype': str(PARAM_DTYPE), 'init': 'zeros (bit-for-bit vanilla attention at init)', 'base_attn_impl': base_impl, 'attention_modules_patched': n_set, 'indexing': "beta[layer, QUERY head, relation]; this module never touches K/V -- it hands the base kernel one additive [B, Hq, q, k] mask. NOTE: with any 4D mask transformers' sdpa path calls repeat_kv (enable_gqa is used only when attention_mask is None), so K/V are expanded to the query-head count inside the kernel, exactly as they are for a vanilla PADDED batch.", 'scopes': {n: s for (n, s) in (('same_row', 'table-cell query -> table-cell key'), ('same_column', 'table-cell query -> table-cell key'), ('target_key', 'post-table receiver query -> target table-cell key'), ('context_key', 'post-table receiver query -> context table-cell key')) if n in _act}, 'metadata_needed': ['struct_ids', 'role_ids', 'region_ids'] if bias.roles_active else ['struct_ids'], 'receiver_scope': 'post-table prompt positions (task declaration / query + the teacher-forced answer) + every generated position', 'target_roles': [RLR.ROLE_NAMES[r] for r in TARGET_ROLES], 'context_roles': [RLR.ROLE_NAMES[r] for r in CONTEXT_ROLES], 'bias_tensor': '[B, n_query_heads, q, k] in the compute dtype, current layer only'}

def clamp_gen_batch(bs, where='generation'):
    bs = int(bs)
    if bs <= MAX_GEN_BATCH:
        return bs
    print(f'[{MECHANISM}] {where} batch size {bs} -> {MAX_GEN_BATCH}: the per-layer [B, n_query_heads, S, S] bias makes a wide PREFILL the memory peak, and an evaluation prompt is not capped by max_seq_len.')
    return MAX_GEN_BATCH

def needs_roles_for(relations):
    return any((i in resolve_relations(relations) for i in ROLE_RELATIONS))

def needs_roles(model=None):
    bank = getattr(model, 'struct_role_bias', None) if model is not None else CTX.bias_module
    return True if bank is None else bool(bank.roles_active)

def get_bias(model):
    b = getattr(model, 'struct_role_bias', None)
    if b is None:
        raise RuntimeError(f'{MECHANISM} is not attached to this model')
    return b

def detach(model, base_impl='sdpa'):
    for m in model.modules():
        cfg = getattr(m, 'config', None)
        if cfg is not None and getattr(cfg, '_attn_implementation', None) == _REG_NAME:
            cfg._attn_implementation = base_impl
    for cfg in AL._iter_configs(model):
        if getattr(cfg, '_attn_implementation', None) == _REG_NAME:
            cfg._attn_implementation = base_impl
    CTX.bias_module = None
    clear_context()
    if getattr(model, 'struct_role_bias', None) is not None:
        del model.struct_role_bias

def save(model, out_dir, extra=None):
    bias = get_bias(model)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'beta': bias.beta.detach().to('cpu', PARAM_DTYPE), 'relations': list(REL_NAMES), 'relations_active': [REL_NAMES[i] for i in bias.active]}, out / STATE_FILE)
    cfg = {'struct_role_bias': True, 'mechanism': MECHANISM, 'version': VERSION, 'relations': list(REL_NAMES), 'relations_active': [REL_NAMES[i] for i in bias.active], 'relation_set': set_name(bias.active), 'n_layers': bias.n_layers, 'n_heads': bias.n_heads, 'n_beta_params': bias.n_params(), 'state_file': STATE_FILE}
    if extra:
        cfg.update(extra)
    with open(out / CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)
    return str(out / STATE_FILE)

def load(model, adapter_dir):
    bias = get_bias(model)
    p = Path(adapter_dir) / STATE_FILE
    if not p.exists():
        raise FileNotFoundError(f'{p} not found: this adapter carries no {MECHANISM} state')
    state = torch.load(p, map_location='cpu')
    beta = state['beta'] if isinstance(state, dict) and 'beta' in state else state
    names = list(state.get('relations', REL_NAMES)) if isinstance(state, dict) else list(REL_NAMES)
    if names != list(REL_NAMES):
        raise RuntimeError(f"{MECHANISM}: the checkpoint's relation order {names} is not this build's {list(REL_NAMES)}; beta's last axis would be reinterpreted.")
    if tuple(beta.shape) != tuple(bias.beta.shape):
        raise RuntimeError(f'{MECHANISM}: checkpoint beta {tuple(beta.shape)} does not match the attached {tuple(bias.beta.shape)} (layers x query heads x relations).')
    saved_active = state.get('relations_active') if isinstance(state, dict) else None
    if saved_active is not None and list(saved_active) != [REL_NAMES[i] for i in bias.active]:
        raise RuntimeError(f'{MECHANISM}: the checkpoint was trained with {list(saved_active)} active but the attached bank has {[REL_NAMES[i] for i in bias.active]}. Attach with relations=relations_from_adapter(dir) so the run is served the way it was trained.')
    with torch.no_grad():
        bias.beta.copy_(beta.to(bias.beta.device, PARAM_DTYPE))
    return {'loaded_from': str(p), 'beta_shape': list(beta.shape), 'relations_active': [REL_NAMES[i] for i in bias.active]}

def relations_from_adapter(adapter_dir):
    for src in (STATE_FILE, CONFIG_FILE):
        p = Path(adapter_dir) / src
        if not p.exists():
            continue
        d = torch.load(p, map_location='cpu') if src == STATE_FILE else json.load(open(p))
        act = d.get('relations_active') if isinstance(d, dict) else None
        if act:
            return resolve_relations(list(act))
    return resolve_relations('all')

def read_config(adapter_dir):
    p = Path(adapter_dir) / CONFIG_FILE
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def is_struct_role_adapter(adapter_dir):
    return bool(adapter_dir) and os.path.exists(os.path.join(adapter_dir, STATE_FILE))

def training_audit(enc_struct, enc_roles=None, enc_regions=None, enc_tasks=None):
    per_task = {}
    roles_on = enc_roles is not None and enc_regions is not None
    for (i, codes) in enumerate(enc_struct):
        c = torch.tensor(codes, dtype=torch.long)
        kc = key_class_from_roles(torch.tensor(enc_roles[i], dtype=torch.long)) if roles_on else torch.zeros(1, dtype=torch.uint8)
        rv = receiver_mask_from_regions(torch.tensor(enc_regions[i], dtype=torch.long)) if roles_on else torch.zeros(1, dtype=torch.bool)
        t = enc_tasks[i] if enc_tasks is not None else 'all'
        d = per_task.setdefault(t, {'examples': 0, 'cell_tokens': 0, 'target_tokens': 0, 'context_tokens': 0, 'receiver_tokens': 0, 'no_cells': 0, 'no_roles': 0, 'no_receiver': 0})
        d['examples'] += 1
        n_cell = int((c != STRUCT_NONE).sum())
        d['cell_tokens'] += n_cell
        n_t = int((kc == KEY_TARGET).sum())
        n_c = int((kc == KEY_CONTEXT).sum())
        n_r = int(rv.sum())
        d['target_tokens'] += n_t
        d['context_tokens'] += n_c
        d['receiver_tokens'] += n_r
        if n_cell == 0:
            d['no_cells'] += 1
        if n_t == 0 and n_c == 0:
            d['no_roles'] += 1
        if n_r == 0:
            d['no_receiver'] += 1
    return {'per_task': per_task, 'n_examples': len(enc_struct), 'roles_counted': roles_on}

def format_audit(audit, active=None):
    act = list(REL_NAMES) if active is None else [REL_NAMES[i] for i in active]
    L = [f'[{MECHANISM}] active relations: ' + ', '.join(act) + ('   (the role pair acts on post-table receiver queries only)' if any((REL_NAMES[i] in act for i in ROLE_RELATIONS)) else '')]
    roles_on = audit.get('roles_counted', True)
    for (t, d) in sorted(audit['per_task'].items()):
        line = f"[{MECHANISM}]   {t}: {d['examples']:,} examples, cell tokens={d['cell_tokens']:,}, examples without a table={d['no_cells']}"
        if roles_on:
            line += f", target keys={d['target_tokens']:,}, context keys={d['context_tokens']:,}, receivers={d['receiver_tokens']:,}, without a target/context role={d['no_roles']}, without a receiver={d['no_receiver']}"
        L.append(line)
    return '\n'.join(L)
