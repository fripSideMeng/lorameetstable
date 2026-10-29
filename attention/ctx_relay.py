import json
import os
import torch
import recv_gate as RCV
import role_low_rank as RLR
MECHANISM = 'ctx_relay_mask'
VERSION = 1
MANIFEST_KEY = 'ctx_relay_mask'
MARKER = 'ctx_relay.json'
RELAY = {'cta': 'context -> target -> answer', 'cpa': 'context -> {head, tail} -> answer'}
CONTEXT_ROLES = (RLR.ROLE_CTA_CONTEXT, RLR.ROLE_CPA_CONTEXT)
RELAY_ROLES = (RLR.ROLE_CTA_TARGET, RLR.ROLE_CPA_HEAD, RLR.ROLE_CPA_TAIL)
ALLOWED_BASE = ('sdpa', 'eager')
REQUIRES = ('target_end',)
_MASK_DTYPE = {'sdpa': None, 'eager': torch.float32}

def _role_lut(keep, device):
    lut = torch.zeros(RLR.N_ROLES, dtype=torch.bool, device=device)
    if keep:
        lut[list(keep)] = True
    return lut

def context_mask_from_roles(role_ids):
    r = torch.as_tensor(role_ids)
    return _role_lut(CONTEXT_ROLES, r.device)[r.long()]

def relay_mask_from_roles(role_ids):
    r = torch.as_tensor(role_ids)
    return _role_lut(RELAY_ROLES, r.device)[r.long()]

def receiver_mask_from_regions(region_ids):
    return RCV.receiver_mask_from_regions(region_ids)

def masks_for_batch(role_ids, region_ids):
    r = torch.as_tensor(role_ids)
    g = torch.as_tensor(region_ids)
    if r.shape != g.shape:
        raise ValueError(f'role_ids {tuple(r.shape)} and region_ids {tuple(g.shape)} must agree')
    return (context_mask_from_roles(r), receiver_mask_from_regions(g))

def build_mask(context_full, recv_full, cache_position, pad_mask, dtype=None):
    ctx = torch.as_tensor(context_full).bool()
    rcv = torch.as_tensor(recv_full).bool()
    if ctx.dim() != 2 or rcv.shape != ctx.shape:
        raise ValueError(f'context/receiver masks must both be [B,T]; got {tuple(ctx.shape)} and {tuple(rcv.shape)}')
    dev = ctx.device
    pad = torch.as_tensor(pad_mask).to(dev).bool()
    if pad.dim() != 2 or pad.shape[0] != ctx.shape[0]:
        raise ValueError(f'pad mask must be [B,T_k] with B={ctx.shape[0]}; got {tuple(pad.shape)}')
    cp = torch.as_tensor(cache_position).to(dev).long().reshape(-1)
    (B, Tp) = ctx.shape
    Tk = pad.shape[1]
    rcv = rcv.to(dev)
    if Tk <= Tp:
        ctx_k = ctx[:, :Tk]
    else:
        ctx_k = torch.cat([ctx, torch.zeros(B, Tk - Tp, dtype=torch.bool, device=dev)], dim=1)
    in_prompt = cp < Tp
    rcv_q = rcv[:, cp.clamp(max=Tp - 1)] & in_prompt[None, :] | (~in_prompt)[None, :]
    idx_k = torch.arange(Tk, device=dev)
    allowed = cp[None, :, None] >= idx_k[None, None, :]
    allowed = allowed.expand(B, -1, -1).clone()
    allowed &= pad[:, None, :]
    allowed &= ~(rcv_q[:, :, None] & ctx_k[:, None, :])
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
    context_full = None
    recv_full = None
    mask_dtype = None
    n_applied = 0
    n_skipped = 0
CTX = _Ctx()

def set_context(context_mask, receiver_mask):
    c = torch.as_tensor(context_mask).bool()
    r = torch.as_tensor(receiver_mask).bool()
    if c.dim() != 2 or r.shape != c.shape:
        raise ValueError(f'context/receiver masks must both be [B,T]; got {tuple(c.shape)} and {tuple(r.shape)}')
    dev = c.device
    (CTX.context_full, CTX.recv_full) = (c, r.to(dev))
    CTX.active = True

def set_context_from_ids(role_ids, region_ids):
    (c, r) = masks_for_batch(role_ids, region_ids)
    set_context(c, r)
    return (c, r)

def clear_context():
    CTX.active = False
    CTX.context_full = CTX.recv_full = None

def reset_stats():
    CTX.n_applied = CTX.n_skipped = 0

def stats():
    return {'forwards_masked': CTX.n_applied, 'forwards_unmasked': CTX.n_skipped}

def _mask_prehook(module, args, kwargs):
    if not CTX.active or CTX.context_full is None:
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
    kwargs['attention_mask'] = build_mask(CTX.context_full.to(dev), CTX.recv_full.to(dev), cp, am, dtype=CTX.mask_dtype)
    CTX.n_applied += 1
    return (args, kwargs)

def attach(model, base_impl='sdpa'):
    if base_impl not in ALLOWED_BASE:
        raise ValueError(f'--ctx_relay_mask needs --attn_impl in {list(ALLOWED_BASE)} (got {base_impl!r}): flash_attention_2 cannot take a custom 4D attention mask.')
    if getattr(model, '_ctx_relay_handles', None):
        raise RuntimeError('ctx_relay is already attached to this model')
    dt = _MASK_DTYPE[base_impl]
    if base_impl == 'eager':
        dt = next((p.dtype for p in model.parameters() if p.is_floating_point), torch.float32)
    CTX.mask_dtype = dt
    reset_stats()
    inner = model.get_base_model() if hasattr(model, 'get_base_model') else model
    hosts = [model] if inner is model else [model, inner]
    handles = [h.register_forward_pre_hook(_mask_prehook, with_kwargs=True) for h in hosts]
    model._ctx_relay_handles = handles
    return {'mechanism': MECHANISM, 'version': VERSION, 'relay': dict(RELAY), 'base_attn_impl': base_impl, 'mask_kind': 'bool' if dt is None else f'additive {dt}', 'prehook_on': [type(h).__name__ for h in hosts], 'context_roles': [RLR.ROLE_NAMES[r] for r in CONTEXT_ROLES], 'relay_roles': [RLR.ROLE_NAMES[r] for r in RELAY_ROLES], 'receiver_scope': 'post-table prompt positions + every generated position', 'requires': list(REQUIRES), 'n_new_parameters': 0}

def detach(model):
    for h in getattr(model, '_ctx_relay_handles', []) or []:
        h.remove()
    model._ctx_relay_handles = []
    clear_context()

def write_marker(out_dir, info=None):
    os.makedirs(out_dir, exist_ok=True)
    rec = {MANIFEST_KEY: True, 'mechanism': MECHANISM, 'version': VERSION, 'relay': dict(RELAY)}
    if info:
        rec.update({k: v for (k, v) in info.items() if k not in rec})
    p = os.path.join(out_dir, MARKER)
    with open(p, 'w') as f:
        json.dump(rec, f, indent=2)
    return p

def read_marker(adapter_dir):
    if not adapter_dir:
        return None
    for fn in (MARKER, 'train_manifest.json', 'prompt_format.json'):
        p = os.path.join(adapter_dir, fn)
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        if MANIFEST_KEY in d:
            return bool(d[MANIFEST_KEY])
        if isinstance(d.get('args'), dict) and MANIFEST_KEY in d['args']:
            return bool(d['args'][MANIFEST_KEY])
    return None

def training_audit(enc_roles, enc_regions, enc_tasks=None):
    per_task = {}
    n_no_recv = 0
    n_no_relay = 0
    n_no_context = 0
    n_orphan_ex = 0
    n_orphan_tok = 0
    for (i, (roles, regions)) in enumerate(zip(enc_roles, enc_regions)):
        r = torch.tensor(roles, dtype=torch.long).unsqueeze(0)
        g = torch.tensor(regions, dtype=torch.long).unsqueeze(0)
        cm = context_mask_from_roles(r)[0]
        rm = relay_mask_from_roles(r)[0]
        ctx = int(cm.sum())
        rel = int(rm.sum())
        rcv = int(receiver_mask_from_regions(g).sum())
        orph = 0
        if ctx and rel:
            last_relay = int(torch.nonzero(rm).flatten().max())
            orph = int((torch.nonzero(cm).flatten() > last_relay).sum())
        t = enc_tasks[i] if enc_tasks is not None else 'all'
        d = per_task.setdefault(t, {'examples': 0, 'context_tokens': 0, 'relay_tokens': 0, 'receiver_tokens': 0, 'no_receiver': 0, 'no_relay': 0, 'no_context': 0, 'inert': 0, 'orphan_examples': 0, 'orphan_tokens': 0})
        d['examples'] += 1
        d['context_tokens'] += ctx
        d['relay_tokens'] += rel
        d['receiver_tokens'] += rcv
        if rcv == 0:
            d['no_receiver'] += 1
            n_no_recv += 1
        if rel == 0:
            d['no_relay'] += 1
            n_no_relay += 1
        if ctx == 0:
            d['no_context'] += 1
            n_no_context += 1
        if ctx == 0 or rcv == 0:
            d['inert'] += 1
        if orph:
            d['orphan_examples'] += 1
            d['orphan_tokens'] += orph
            n_orphan_ex += 1
            n_orphan_tok += orph
    return {'per_task': per_task, 'n_no_receiver': n_no_recv, 'n_no_relay': n_no_relay, 'n_no_context': n_no_context, 'n_examples': len(enc_roles), 'n_orphan_examples': n_orphan_ex, 'n_orphan_context_tokens': n_orphan_tok}

def format_audit(audit):
    L = [f'[{MECHANISM}] relay: ' + '; '.join((f'{k.upper()} {v}' for (k, v) in RELAY.items()))]
    for (t, d) in sorted(audit['per_task'].items()):
        L.append(f"[{MECHANISM}]   {t}: {d['examples']:,} examples, context={d['context_tokens']:,} tok (cut from the answer), relay={d['relay_tokens']:,} tok, receivers={d['receiver_tokens']:,} tok, inert={d['inert']}, orphan={d['orphan_examples']} ex / {d['orphan_tokens']:,} tok")
    if audit.get('n_orphan_examples'):
        L.append(f"[{MECHANISM}] NOTE: {audit['n_orphan_examples']} example(s) carry {audit['n_orphan_context_tokens']:,} context token(s) positioned AFTER their last designated-column token, so for those tokens the mask is a deletion rather than a relay. These are the prompts --target_end could not canonicalize (a ragged table row leaves the corpus column order in place).")
    return '\n'.join(L)
