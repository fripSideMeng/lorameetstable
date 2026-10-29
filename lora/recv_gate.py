import inspect
import json
import math
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import role_low_rank as RLR
from lora_internal_per_token import find_markdown_table_span
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask, eager_mask
MECHANISM = 'recv_pc_vlora_gate'
VERSION = 1
FORMULA = 'g = 1 + tanh(W[task,layer,stream] @ zQ_recv); correction = scaling_V * B_V((gP-1)*rP + (gC-1)*rC); m_tilde = m + correction'
GATE_FUNCTION = '1 + tanh'
CONTROLLER_INPUT = 'q_proj LoRA rank code (zQ) at the receiver position'
CONTROLLED_QUANTITY = 'attention-retrieved v_proj LoRA rank residual (zV), per stream'
RECEIVER_SCOPE = 'query + answer/decoding positions'
MODES = ('off', 'v')
DEFAULT_MODE = 'off'
ALLOWED_BASE = ('sdpa', 'eager')
_REG_NAME = 'recv_pc_gate'
_BASE_MASK_FN = {'sdpa': sdpa_mask, 'eager': eager_mask}
CONTROLLER_SOURCE_MODULE = 'q_proj'
CONTROLLED_MODULE = 'v_proj'
STREAMS = ('primary', 'context')
GATE_TASKS = ('cta', 'cpa', 'tabfact')
TASK_NONE = 0
TASK_INDEX = {t: i + 1 for (i, t) in enumerate(GATE_TASKS)}
N_TASK_ROWS = len(GATE_TASKS) + 1
PRIMARY_TASKS = ('cta', 'cpa')
TASK_SEMANTICS = {'cta': {'primary': 'the target entity column', 'context': 'the other table columns'}, 'cpa': {'primary': 'the head and tail columns as ONE stream', 'context': 'the other table columns'}, 'tabfact': {'primary': None, 'context': 'the entire serialized table region'}}
TASK_STREAM_ROLES = {'cta': {'primary': (RLR.ROLE_CTA_TARGET,), 'context': (RLR.ROLE_CTA_CONTEXT,)}, 'cpa': {'primary': (RLR.ROLE_CPA_HEAD, RLR.ROLE_CPA_TAIL), 'context': (RLR.ROLE_CPA_CONTEXT,)}}
RLR_NAME = RLR.ROLE_NAMES
PRIMARY_ROLES = tuple((r for t in TASK_STREAM_ROLES.values() for r in t['primary']))
CONTEXT_ROLES = tuple((r for t in TASK_STREAM_ROLES.values() for r in t['context']))
ROLE_NONE = RLR.ROLE_NONE
(R_PAD, R_PRE, R_TABLE, R_POST) = (-1, 0, 1, 2)
REGION_NAME = {R_PAD: 'pad', R_PRE: 'pre_table', R_TABLE: 'table', R_POST: 'post_table'}
RECEIVER_REGIONS = (R_POST,)
PARAM_DTYPE = torch.float32
STATE_FILE = 'recv_gate.pt'
CONFIG_FILE = 'recv_gate_config.json'
GATE_STOP_GRADIENT = True
STOP_GRADIENT_TENSORS = ('zQ (q_proj LoRA rank code; the controller input)', 'rP / rC (the retrieved primary and context streams, hence the second attention call, its query/key and the cached zV behind it)', 'B_V (v_proj LoRA B, used to expand the rank-space correction)')

def canonical_task(slug):
    base = str(slug).lower() if slug else ''
    for t in GATE_TASKS:
        if base == t or base.startswith(t + '_'):
            return t
    return None

def task_row(slug):
    t = canonical_task(slug)
    return TASK_NONE if t is None else TASK_INDEX[t]

def task_rows(slugs, device=None):
    return torch.tensor([task_row(s) for s in slugs], dtype=torch.long, device=device)

def normalize_exclude_target(tasks):
    if tasks is None:
        return ()
    if isinstance(tasks, str):
        tasks = [tasks]
    seen = set()
    for t in tasks:
        c = canonical_task(t)
        if c is None:
            raise ValueError(f'target-available toggle names {t!r}, which is not one of {GATE_TASKS}')
        if c not in PRIMARY_TASKS:
            raise ValueError(f"target-available toggle names {t!r}, but {c} has no primary stream ({TASK_SEMANTICS[c]['context']} is its only stream), so the toggle would have no effect. Tasks with a primary span: {list(PRIMARY_TASKS)}")
        seen.add(c)
    return tuple((t for t in GATE_TASKS if t in seen))

def target_available_by_task(exclude_target_tasks=()):
    excl = normalize_exclude_target(exclude_target_tasks)
    return {t: t not in excl for t in PRIMARY_TASKS}

def has_primary_stream(task):
    return task in PRIMARY_TASKS

def primary_gated(task, exclude_target_tasks=()):
    return has_primary_stream(task) and target_available_by_task(exclude_target_tasks)[task]

def primary_on_rows(exclude_target_tasks=(), dtype=PARAM_DTYPE, device=None):
    vals = [0.0] + [1.0 if primary_gated(t, exclude_target_tasks) else 0.0 for t in GATE_TASKS]
    return torch.tensor(vals, dtype=dtype, device=device)

def _role_lut(keep, device):
    lut = torch.zeros(RLR.N_ROLES, dtype=torch.bool, device=device)
    if keep:
        lut[list(keep)] = True
    return lut

def stream_masks_from_roles(role_ids):
    r = torch.as_tensor(role_ids)
    rl = r.long()
    p = _role_lut(PRIMARY_ROLES, r.device)[rl]
    c = _role_lut(CONTEXT_ROLES, r.device)[rl]
    return (p, c)

def stream_masks(task_ids, role_ids, region_ids):
    tid = torch.as_tensor(task_ids).reshape(-1).long()
    r = torch.as_tensor(role_ids)
    g = torch.as_tensor(region_ids)
    if r.shape != g.shape:
        raise ValueError(f'role_ids {tuple(r.shape)} and region_ids {tuple(g.shape)} must agree')
    if tid.shape[0] != r.shape[0]:
        raise ValueError(f'task rows {tid.shape[0]} != mask rows {r.shape[0]}')
    dev = r.device
    tid = tid.to(dev)
    (p_role, c_role) = stream_masks_from_roles(r)
    table = g.to(dev) == R_TABLE
    whole_table = torch.zeros_like(tid, dtype=torch.bool)
    for t in GATE_TASKS:
        if not has_primary_stream(t):
            whole_table |= tid == TASK_INDEX[t]
    whole_table = whole_table.view(-1, 1)
    live = (tid != TASK_NONE).view(-1, 1)
    primary = p_role & ~whole_table & live
    context = torch.where(whole_table, table, c_role) & live
    return (primary, context)

def receiver_mask_from_regions(region_ids):
    r = torch.as_tensor(region_ids)
    out = torch.zeros(r.shape, dtype=torch.bool, device=r.device)
    for reg in RECEIVER_REGIONS:
        out |= r == reg
    return out

def masks_for_slug(slug, role_ids, region_ids):
    r = torch.as_tensor(role_ids)
    tid = torch.full((r.shape[0],), task_row(slug), dtype=torch.long, device=r.device)
    (p, c) = stream_masks(tid, r, region_ids)
    return (p, c, receiver_mask_from_regions(region_ids))

def build_regions(tok, raw_prompt, rendered, expect_ids=None):
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    (ids, offs) = (enc['input_ids'], enc['offset_mapping'])
    if expect_ids is not None and list(ids) != list(expect_ids):
        raise RuntimeError(f"region tokenization ({len(ids)} tokens) does not reproduce the model's own prompt tokenization ({len(expect_ids)} tokens). Receiver positions must be aligned to the ids the model is fed; refusing to train on a misaligned region sequence.")
    span = _table_char_span(raw_prompt, rendered)
    if span is None:
        return [R_PRE] * len(ids)
    (t0, t1) = span
    out = []
    for (a, b) in offs:
        if b <= a:
            b = a + 1
        if b <= t0:
            out.append(R_PRE)
        elif a >= t1:
            out.append(R_POST)
        else:
            out.append(R_TABLE)
    return out

def _table_char_span(content, templated):
    for cand in (content, str(content).strip()):
        base = templated.rfind(cand)
        if base < 0:
            continue
        span = find_markdown_table_span(cand)
        if span is None:
            return None
        return (base + span[0], base + span[1])
    return None

def pad_regions(region_lists, width, side='right'):
    out = []
    for r in region_lists:
        n = width - len(r)
        if n < 0:
            raise ValueError('region sequence longer than the padded width')
        out.append([R_PAD] * n + list(r) if side == 'left' else list(r) + [R_PAD] * n)
    return out

def regions_from_attention_mask(region_lists, attn_mask):
    T = attn_mask.shape[1]
    rows = []
    for (r, m) in zip(region_lists, attn_mask.tolist()):
        n_pad = len(m) - int(sum(m))
        if any((m[k] == 0 for k in range(n_pad, T))) or any((m[k] == 1 for k in range(n_pad))):
            raise RuntimeError("region ids can only be aligned to a LEFT-padded batch; set tok.padding_side='left'")
        if n_pad + len(r) != T:
            raise RuntimeError(f'left-pad width {n_pad} + {len(r)} region ids != padded length {T}')
        rows.append([R_PAD] * n_pad + list(r))
    return torch.tensor(rows, dtype=torch.long)

class _Ctx:
    active = False
    bank = None
    base_attn_forward = None
    apply_gate = True
    task_id = None
    rows = ()
    primary_full = None
    context_full = None
    recv_full = None
    valid_full = None
    answer_full = None
    cur_primary = None
    cur_context = None
    cur_recv = None
    cur_valid = None
    cur_answer = None
    fresh = True
CTX = _Ctx()

def set_gate_context(task_id, primary_mask, context_mask, recv_mask, valid_mask=None, answer_mask=None):
    tid = torch.as_tensor(task_id).reshape(-1).long()
    p = torch.as_tensor(primary_mask).bool()
    c = torch.as_tensor(context_mask).bool()
    rv = torch.as_tensor(recv_mask).bool()
    if p.dim() != 2 or c.shape != p.shape or rv.shape != p.shape:
        raise ValueError(f'masks must all be [B,T]; got {tuple(p.shape)}, {tuple(c.shape)}, {tuple(rv.shape)}')
    if tid.shape[0] != p.shape[0]:
        raise ValueError(f'task_id rows {tid.shape[0]} != mask rows {p.shape[0]}')
    if int(tid.min()) < 0 or int(tid.max()) >= N_TASK_ROWS:
        raise ValueError(f'task rows must be in [0,{N_TASK_ROWS}); got {sorted(set(tid.tolist()))}')
    if bool((p & c).any()):
        raise ValueError('primary and context masks overlap; a sender belongs to one stream at most')
    dev = p.device
    (c, rv, tid) = (c.to(dev), rv.to(dev), tid.to(dev))
    CTX.task_id = tid
    (CTX.primary_full, CTX.context_full, CTX.recv_full) = (p, c, rv)
    CTX.valid_full = None if valid_mask is None else torch.as_tensor(valid_mask).bool().to(dev)
    CTX.answer_full = None if answer_mask is None else torch.as_tensor(answer_mask).bool().to(dev)
    CTX.rows = tuple(sorted({int(v) for v in tid.tolist()}))
    CTX.active = True
    _clear_slices()

def set_gate_context_for_slug(slug, primary_mask, context_mask, recv_mask, valid_mask=None, answer_mask=None, device=None):
    b = torch.as_tensor(primary_mask).shape[0]
    tid = torch.full((b,), task_row(slug), dtype=torch.long, device=device if device is not None else None)
    set_gate_context(tid, primary_mask, context_mask, recv_mask, valid_mask, answer_mask)

def clear_gate_context():
    CTX.active = False
    CTX.task_id = None
    CTX.rows = ()
    CTX.primary_full = CTX.context_full = CTX.recv_full = None
    CTX.valid_full = CTX.answer_full = None
    _clear_slices()
    if CTX.bank is not None:
        CTX.bank.clear_caches()

def _clear_slices():
    CTX.cur_primary = CTX.cur_context = CTX.cur_recv = None
    CTX.cur_valid = CTX.cur_answer = None
    CTX.fresh = True

def set_apply_gate(flag):
    CTX.apply_gate = bool(flag)
    return CTX.apply_gate

def _mask_prehook(module, args, kwargs):
    if not CTX.active or CTX.primary_full is None:
        return None
    cp = kwargs.get('cache_position', None)
    ids = kwargs.get('input_ids', None)
    if ids is None and args:
        ids = args[0]
    T = CTX.primary_full.shape[1]
    if cp is None:
        n = int(ids.shape[1]) if ids is not None else T
        cp = torch.arange(n, device=CTX.primary_full.device)
    cp = cp.to(CTX.primary_full.device).long()
    CTX.fresh = bool(int(cp[0]) == 0)
    k_len = int(cp[-1]) + 1
    if k_len <= T:
        CTX.cur_primary = CTX.primary_full[:, :k_len]
        CTX.cur_context = CTX.context_full[:, :k_len]
    else:
        pad = torch.zeros(CTX.primary_full.shape[0], k_len - T, dtype=torch.bool, device=CTX.primary_full.device)
        CTX.cur_primary = torch.cat([CTX.primary_full, pad], dim=1)
        CTX.cur_context = torch.cat([CTX.context_full, pad], dim=1)
    idx = cp.clamp(max=T - 1)
    in_prompt = cp < T
    CTX.cur_recv = CTX.recv_full[:, idx] & in_prompt | ~in_prompt
    CTX.cur_valid = None if CTX.valid_full is None else CTX.valid_full[:, idx] | ~in_prompt
    CTX.cur_answer = None if CTX.answer_full is None else CTX.answer_full[:, idx] & in_prompt | ~in_prompt
    return None
ACC_FIELDS = ('n', 'sum_g', 'sum_abs_dev', 'sum_sq_dev', 'n_lt1', 'n_gt1', 'sum_sq_rank_corr', 'sum_sq_rank_stream', 'n_recv')

def blank_acc():
    d = {k: 0.0 for k in ACC_FIELDS}
    d['max_abs_dev'] = 0.0
    return d

def merge_acc(dst, rec):
    for k in ACC_FIELDS:
        dst[k] += rec[k]
    dst['max_abs_dev'] = max(dst['max_abs_dev'], rec['max_abs_dev'])
    return dst

def _finish_acc(rec):
    n = rec['n']
    out = {'n_values': int(n), 'n_receivers': int(rec['n_recv']), 'mean_g': rec['sum_g'] / n if n else 1.0, 'mean_abs_dev': rec['sum_abs_dev'] / n if n else 0.0, 'rms_dev': math.sqrt(rec['sum_sq_dev'] / n) if n else 0.0, 'max_abs_dev': rec['max_abs_dev'], 'frac_lt_1': rec['n_lt1'] / n if n else 0.0, 'frac_gt_1': rec['n_gt1'] / n if n else 0.0}
    den = rec['sum_sq_rank_stream']
    out['rank_space_rel_change'] = math.sqrt(rec['sum_sq_rank_corr'] / den) if den > 0 else 0.0
    return out

class RecvGateBank(nn.Module):

    def __init__(self, n_layers, r, tasks=GATE_TASKS, exclude_target_tasks=()):
        super().__init__()
        self.n_layers = int(n_layers)
        self.r = int(r)
        self.tasks = tuple(tasks)
        self.exclude_target_tasks = normalize_exclude_target(exclude_target_tasks)
        self.W_primary = nn.ParameterList([nn.Parameter(torch.zeros(len(self.tasks), self.r, self.r, dtype=PARAM_DTYPE)) for _ in range(self.n_layers)])
        self.W_context = nn.ParameterList([nn.Parameter(torch.zeros(len(self.tasks), self.r, self.r, dtype=PARAM_DTYPE)) for _ in range(self.n_layers)])
        self.register_buffer('primary_on', primary_on_rows(self.exclude_target_tasks, dtype=PARAM_DTYPE), persistent=False)
        self.enabled = True
        self.collect = False
        self.acc = {}
        self.layers = {}
        self.zq = {}
        self.zv = {}
        self.zv_cache = {}

    def streams(self):
        return {'primary': self.W_primary, 'context': self.W_context}

    def n_params(self):
        return sum((p.numel() for p in self.parameters() if p.requires_grad))

    def expected_n_params(self):
        return self.n_layers * len(STREAMS) * len(self.tasks) * self.r * self.r

    def target_available(self):
        return target_available_by_task(self.exclude_target_tasks)

    def extra_repr(self):
        return f'layers={self.n_layers}, r={self.r}, tasks={self.tasks}, streams={STREAMS}, target_available={self.target_available()}'

    def select(self, stream, layer, tid, dtype, device):
        w = self.streams()[stream][layer].to(device=device, dtype=dtype)
        sel = F.one_hot(tid, N_TASK_ROWS).to(dtype)[:, 1:]
        return (sel @ w.reshape(w.shape[0], -1)).view(-1, self.r, self.r)

    def clear_caches(self):
        self.zq.clear()
        self.zv.clear()
        self.zv_cache.clear()

    def note_zq(self, layer, z):
        self.zq[layer] = z

    def note_zv(self, layer, z, reset):
        self.zv[layer] = z
        if reset or layer not in self.zv_cache:
            self.zv_cache[layer] = z
        else:
            self.zv_cache[layer] = torch.cat([self.zv_cache[layer], z], dim=1)

    def reset_stats(self):
        self.acc = {}

    def _accumulate(self, layer, stream, tid_dev, dev_g, r_stream, recv, answer):
        if not bool(recv.any()):
            return
        d = dev_g.detach().float()
        rs = r_stream.detach().float()
        corr = d.unsqueeze(2) * rs
        halves = [('all', recv)] if answer is None else [('query', recv & ~answer), ('answer', recv & answer)]
        for row in CTX.rows:
            if stream == 'primary' and float(self.primary_on[row]) == 0.0:
                continue
            in_row = (tid_dev == row).unsqueeze(1)
            for (half, hmask) in halves:
                sel = hmask & in_row
                if not bool(sel.any()):
                    continue
                m = sel.unsqueeze(-1)
                dv = d * m
                n_recv = float(sel.sum().item())
                stacked = torch.stack([dv.sum(), dv.abs().sum(), (dv * dv).sum(), dv.abs().max(), (dv < 0).sum().to(dv.dtype), (dv > 0).sum().to(dv.dtype), (corr * sel[:, :, None, None]).pow(2).sum(), (rs * sel[:, :, None, None]).pow(2).sum()]).double().tolist()
                rec = blank_acc()
                rec['n'] = n_recv * self.r
                rec['sum_g'] = stacked[0] + rec['n']
                rec['sum_abs_dev'] = stacked[1]
                rec['sum_sq_dev'] = stacked[2]
                rec['max_abs_dev'] = stacked[3]
                rec['n_lt1'] = stacked[4]
                rec['n_gt1'] = stacked[5]
                rec['sum_sq_rank_corr'] = stacked[6]
                rec['sum_sq_rank_stream'] = stacked[7]
                rec['n_recv'] = n_recv
                merge_acc(self.acc.setdefault((row, int(layer), stream, half), blank_acc()), rec)

def recv_gate_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    base = CTX.base_attn_forward
    (out, weights) = base(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    bank = CTX.bank
    if not CTX.active or bank is None or (not bank.enabled) or (not CTX.apply_gate):
        return (out, weights)
    layer = getattr(module, 'layer_idx', None)
    if layer is None or layer not in bank.layers:
        return (out, weights)
    if CTX.cur_recv is None or not bool(CTX.cur_recv.any()):
        return (out, weights)
    refs = bank.layers[layer]
    zq = bank.zq.get(layer)
    zv = bank.zv_cache.get(layer)
    if zq is None or zv is None:
        raise RuntimeError(f'layer {layer}: the receiver gate has no rank codes for this forward. q_proj/v_proj LoRA hooks did not fire -- the adapter may be disabled or swapped.')
    (B, H, Tq, hd) = query.shape
    (KV, Tk) = (key.shape[1], key.shape[2])
    if zv.shape[1] != Tk:
        raise RuntimeError(f'layer {layer}: cached v_proj rank codes cover {zv.shape[1]} positions but attention has {Tk} keys. The zV cache and the KV cache have drifted apart.')
    if CTX.cur_primary.shape[1] != Tk:
        raise RuntimeError(f'layer {layer}: sender masks cover {CTX.cur_primary.shape[1]} positions but attention has {Tk} keys.')
    dt = zv.dtype
    adt = query.dtype
    dev = zv.device
    tid = CTX.task_id.to(dev)
    pm = CTX.cur_primary.to(dev).unsqueeze(-1).to(dt)
    cm = CTX.cur_context.to(dev).unsqueeze(-1).to(dt)
    v_cat = torch.cat([zv * pm, zv * cm], dim=-1).to(adt)
    v_cat = v_cat.unsqueeze(1).expand(B, KV, Tk, v_cat.shape[-1])
    (r_cat, _) = base(module, query, key, v_cat, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    r = bank.r
    sg = GATE_STOP_GRADIENT
    (rP, rC) = (r_cat[..., :r].to(dt), r_cat[..., r:].to(dt))
    if sg:
        (rP, rC) = (rP.detach(), rC.detach())
    zq_d = zq.to(dev)
    if sg:
        zq_d = zq_d.detach()
    wP = bank.select('primary', layer, tid, dt, dev)
    wC = bank.select('context', layer, tid, dt, dev)
    devP = torch.tanh(torch.bmm(zq_d, wP.transpose(1, 2)))
    devC = torch.tanh(torch.bmm(zq_d, wC.transpose(1, 2)))
    devP = devP * bank.primary_on.to(device=dev, dtype=dt)[tid].view(-1, 1, 1)
    rmask = CTX.cur_recv.to(dev).unsqueeze(-1).to(dt)
    devP = devP * rmask
    devC = devC * rmask
    if bank.collect:
        recv_b = CTX.cur_recv.to(dev)
        ans_b = None if CTX.cur_answer is None else CTX.cur_answer.to(dev)
        bank._accumulate(layer, 'primary', tid, devP, rP, recv_b, ans_b)
        bank._accumulate(layer, 'context', tid, devC, rC, recv_b, ans_b)
    rank_corr = devP.unsqueeze(2) * rP + devC.unsqueeze(2) * rC
    Bv = refs.lora_B.weight
    Bv = (Bv.detach() if sg else Bv).to(device=dev, dtype=dt)
    if Bv.shape[0] != KV * hd:
        raise RuntimeError(f'layer {layer}: v_proj lora_B is {tuple(Bv.shape)} but attention reports {KV} kv heads x {hd} head_dim = {KV * hd}.')
    G = H // KV
    if G * KV != H:
        raise RuntimeError(f'layer {layer}: {H} query heads is not a multiple of {KV} kv heads')
    Bv = Bv.view(KV, hd, r)
    corr = torch.einsum('btkgr,khr->btkgh', rank_corr.view(B, Tq, KV, G, r), Bv)
    corr = corr.reshape(B, Tq, H, hd) * float(refs.scaling)
    return (out + corr.to(out.dtype), weights)
ALL_ATTENTION_FUNCTIONS.register(_REG_NAME, recv_gate_attention_forward)

class _LayerRefs:
    __slots__ = ('layer', 'q_proj', 'v_proj', 'lora_A_q', 'lora_A_v', 'lora_B', 'scaling')

    def __init__(self, layer, q_proj, v_proj, lora_A_q, lora_A_v, lora_B, scaling):
        self.layer = layer
        (self.q_proj, self.v_proj) = (q_proj, v_proj)
        (self.lora_A_q, self.lora_A_v, self.lora_B) = (lora_A_q, lora_A_v, lora_B)
        self.scaling = scaling

def _active_adapter(mod):
    act = getattr(mod, 'active_adapters', None)
    if act:
        return act[0]
    act = getattr(mod, 'active_adapter', None)
    if isinstance(act, str):
        return act
    if isinstance(act, (list, tuple)) and act:
        return act[0]
    keys = list(getattr(mod, 'lora_A', {}).keys())
    if len(keys) != 1:
        raise RuntimeError(f'cannot resolve a single active LoRA adapter (have {keys})')
    return keys[0]

def _parse_layer(name):
    m = re.search('\\.layers\\.(\\d+)\\.', name)
    return int(m.group(1)) if m else None

def find_gated_layers(model):
    (q_by_layer, v_by_layer) = ({}, {})
    for (name, mod) in model.named_modules():
        if not hasattr(mod, 'lora_A'):
            continue
        layer = _parse_layer(name)
        if layer is None:
            continue
        if name.endswith('.' + CONTROLLER_SOURCE_MODULE):
            q_by_layer[layer] = (name, mod)
        elif name.endswith('.' + CONTROLLED_MODULE):
            v_by_layer[layer] = (name, mod)
    out = {}
    for layer in sorted(set(q_by_layer) & set(v_by_layer)):
        (_, q) = q_by_layer[layer]
        (_, v) = v_by_layer[layer]
        (aq, av) = (_active_adapter(q), _active_adapter(v))
        out[layer] = _LayerRefs(layer, q, v, q.lora_A[aq], v.lora_A[av], v.lora_B[av], float(v.scaling[av]))
    missing_q = sorted(set(v_by_layer) - set(q_by_layer))
    missing_v = sorted(set(q_by_layer) - set(v_by_layer))
    if missing_q or missing_v:
        raise RuntimeError(f'the receiver gate needs LoRA on BOTH q_proj (the controller input) and v_proj (the controlled residual); layers missing q_proj: {missing_q}, missing v_proj: {missing_v}')
    return out

def _lora_rank(refs):
    ranks = {int(r.lora_A_q.out_features) for r in refs.values()}
    ranks |= {int(r.lora_A_v.out_features) for r in refs.values()}
    if len(ranks) != 1:
        raise RuntimeError(f'expected one LoRA rank across q/v, found {sorted(ranks)}')
    return ranks.pop()

def _iter_configs(model):
    seen = []
    c = getattr(model, 'config', None)
    if c is not None:
        seen.append(c)
    base = getattr(model, 'base_model', None)
    if base is not None:
        for obj in (base, getattr(base, 'model', None)):
            c = getattr(obj, 'config', None)
            if c is not None:
                seen.append(c)
    return seen

def _zq_hook(bank, layer):

    def hook(_module, _args, z):
        bank.note_zq(layer, z)
        return None
    return hook

def _zv_hook(bank, layer):

    def hook(_module, _args, z):
        bank.note_zv(layer, z, reset=CTX.fresh)
        return None
    return hook

def attach(model, mode='v', tasks=GATE_TASKS, exclude_target_tasks=(), base_impl='sdpa'):
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES} (got {mode!r})')
    if mode == 'off':
        return {'lora_recv_gate': 'off'}
    if base_impl not in ALLOWED_BASE:
        raise ValueError(f'base_impl must be one of {ALLOWED_BASE} (got {base_impl!r}); flash_attention_2 cannot be guaranteed to reproduce the same alpha.')
    if getattr(model, 'recv_gate_bank', None) is not None:
        raise RuntimeError('a receiver gate is already attached to this model')
    for cfg in _iter_configs(model):
        drop = float(getattr(cfg, 'attention_dropout', 0.0) or 0.0)
        if drop != 0.0:
            raise RuntimeError(f'attention_dropout={drop} would give the second attention call a DIFFERENT dropout mask from the first, so the retrieved streams would not ride on the ordinary alpha. Refusing to attach.')
    refs = find_gated_layers(model)
    if not refs:
        raise RuntimeError('no decoder layer carries LoRA on both q_proj and v_proj')
    r = _lora_rank(refs)
    n_layers = max(refs) + 1
    if sorted(refs) != list(range(n_layers)):
        raise RuntimeError(f'expected a contiguous layer range, found {sorted(refs)}')
    bank = RecvGateBank(n_layers, r, tasks=tasks, exclude_target_tasks=exclude_target_tasks)
    bank.layers = refs
    dev = next(model.parameters()).device
    bank.to(device=dev, dtype=PARAM_DTYPE)
    model.recv_gate_bank = bank
    CTX.bank = bank
    CTX.apply_gate = True
    handles = []
    for (layer, ref) in refs.items():
        handles.append(ref.lora_A_q.register_forward_hook(_zq_hook(bank, layer)))
        handles.append(ref.lora_A_v.register_forward_hook(_zv_hook(bank, layer)))
    inner = model.get_base_model() if hasattr(model, 'get_base_model') else model
    hosts = [model] if inner is model else [model, inner]
    for h in hosts:
        handles.append(h.register_forward_pre_hook(_mask_prehook, with_kwargs=True))
    model._recv_gate_handles = handles
    model._recv_gate_prev_attn = {id(c): getattr(c, '_attn_implementation', None) for c in _iter_configs(model)}
    if base_impl == 'eager':
        import transformers.models.llama.modeling_llama as _llama
        CTX.base_attn_forward = _llama.eager_attention_forward
    else:
        CTX.base_attn_forward = ALL_ATTENTION_FUNCTIONS[base_impl]
    AttentionMaskInterface.register(_REG_NAME, _BASE_MASK_FN[base_impl])
    n_set = 0
    for m in model.modules():
        cfg = getattr(m, 'config', None)
        if cfg is not None and hasattr(cfg, '_attn_implementation') and m.__class__.__name__.endswith('Attention'):
            cfg._attn_implementation = _REG_NAME
            n_set += 1
    for cfg in _iter_configs(model):
        if hasattr(cfg, '_attn_implementation'):
            cfg._attn_implementation = _REG_NAME
    if n_set == 0:
        raise RuntimeError('attach: no *Attention submodules found to point at the gate kernel')
    n_params = bank.n_params()
    if n_params != bank.expected_n_params():
        raise RuntimeError(f'controller parameter count {n_params} != expected {bank.expected_n_params()} = {n_layers} layers x {len(STREAMS)} streams x {len(bank.tasks)} tasks x {r}x{r}')
    return {'lora_recv_gate': mode, 'mechanism': MECHANISM, 'version': VERSION, 'formula': FORMULA, 'gate_function': GATE_FUNCTION, 'gate_stop_gradient': GATE_STOP_GRADIENT, 'stop_gradient_tensors': list(STOP_GRADIENT_TENSORS), 'receiver_scope': RECEIVER_SCOPE, 'controller_input': CONTROLLER_INPUT, 'controlled_quantity': CONTROLLED_QUANTITY, 'streams': list(STREAMS), 'task_specific': True, 'stream_specific': True, 'head_specific_controller': False, 'zero_init': True, 'primary_tasks': list(PRIMARY_TASKS), 'task_semantics': {t: TASK_SEMANTICS[t] for t in bank.tasks}, 'base_attn_impl': base_impl, 'attn_impl_registered': _REG_NAME, 'n_layers': n_layers, 'n_attention_modules_patched': n_set, 'lora_r': r, 'n_gate_params': n_params, 'gate_tasks': list(bank.tasks), 'task_rows': {'none': TASK_NONE, **{t: TASK_INDEX[t] for t in GATE_TASKS}}, 'exclude_target_tasks': list(bank.exclude_target_tasks), 'target_available_by_task': bank.target_available(), 'prehook_on': [type(h).__name__ for h in hosts]}

def detach(model, restore_attn_impl='sdpa'):
    for h in getattr(model, '_recv_gate_handles', []) or []:
        h.remove()
    model._recv_gate_handles = []
    for m in model.modules():
        cfg = getattr(m, 'config', None)
        if cfg is not None and getattr(cfg, '_attn_implementation', None) == _REG_NAME and m.__class__.__name__.endswith('Attention'):
            cfg._attn_implementation = restore_attn_impl
    for cfg in _iter_configs(model):
        if getattr(cfg, '_attn_implementation', None) == _REG_NAME:
            cfg._attn_implementation = restore_attn_impl
    if getattr(model, 'recv_gate_bank', None) is not None:
        model.recv_gate_bank = None
    CTX.bank = None
    clear_gate_context()

def get_bank(model):
    bank = getattr(model, 'recv_gate_bank', None)
    if bank is None:
        raise RuntimeError('no receiver gate is attached to this model')
    return bank

def is_attached(model):
    return getattr(model, 'recv_gate_bank', None) is not None

def minimal_logits_kwargs(model):
    inner = model.get_base_model() if hasattr(model, 'get_base_model') else model
    try:
        params = inspect.signature(inner.forward).parameters
    except (TypeError, ValueError):
        return {}
    for name in ('logits_to_keep', 'num_logits_to_keep'):
        if name in params:
            return {name: 1}
    return {}

def probe_masks(task_id, T, device=None):
    tid = torch.as_tensor(task_id).reshape(-1).long()
    B = int(tid.shape[0])
    q = max(1, T // 4)
    regions = torch.full((B, T), R_PRE, dtype=torch.long)
    regions[:, q:3 * q] = R_TABLE
    regions[:, 3 * q:] = R_POST
    roles = torch.full((B, T), ROLE_NONE, dtype=torch.long)
    for b in range(B):
        t = {v: k for (k, v) in TASK_INDEX.items()}.get(int(tid[b]))
        if t is None or not has_primary_stream(t):
            continue
        ctx = TASK_STREAM_ROLES[t]['context']
        prim = TASK_STREAM_ROLES[t]['primary']
        roles[b, q:2 * q] = ctx[0]
        for (i, rid) in enumerate(prim):
            lo = 2 * q + i * q // len(prim)
            hi = 2 * q + (i + 1) * q // len(prim)
            roles[b, lo:hi] = rid
    (p, c) = stream_masks(tid, roles, regions)
    rv = receiver_mask_from_regions(regions)
    if device is not None:
        (p, c, rv) = (p.to(device), c.to(device), rv.to(device))
    return (p, c, rv, roles, regions)

def verify_zero_gate(model, input_ids, attention_mask=None, task_id=None, primary_mask=None, context_mask=None, recv_mask=None, atol=0.0):
    bank = get_bank(model)
    (B, T) = (int(input_ids.shape[0]), int(input_ids.shape[1]))
    if B < N_TASK_ROWS:
        raise ValueError(f'the probe must exercise all {N_TASK_ROWS} gate rows; got B={B}')
    dev = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if task_id is None:
        task_id = torch.arange(B, dtype=torch.long) % N_TASK_ROWS
    task_id = torch.as_tensor(task_id).reshape(-1).long()
    if primary_mask is None or context_mask is None or recv_mask is None:
        (_p, _c, _rv, _, _) = probe_masks(task_id, T)
        primary_mask = _p if primary_mask is None else primary_mask
        context_mask = _c if context_mask is None else context_mask
        recv_mask = _rv if recv_mask is None else recv_mask
    was_training = model.training
    model.eval()
    (prev_collect, prev_apply) = (bank.collect, CTX.apply_gate)
    bank.reset_stats()
    try:
        with torch.no_grad():
            set_gate_context(task_id, primary_mask, context_mask, recv_mask, valid_mask=attention_mask.bool())
            set_apply_gate(False)
            ref = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
            set_apply_gate(True)
            bank.collect = True
            set_gate_context(task_id, primary_mask, context_mask, recv_mask, valid_mask=attention_mask.bool())
            got = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        diff = float((got - ref).abs().max().item())
        stats = gate_activation_stats(model)
    finally:
        bank.collect = prev_collect
        set_apply_gate(prev_apply)
        clear_gate_context()
        if was_training:
            model.train()
    rows_seen = sorted({int(v) for v in task_id.tolist()})
    if rows_seen != list(range(N_TASK_ROWS)):
        raise RuntimeError(f'the probe exercised gate rows {rows_seen}, expected {list(range(N_TASK_ROWS))}')
    total = stats.get('total') or {}
    if not total.get('n_values'):
        raise RuntimeError('the gate collected no statistics: the kernel never ran, so this check proves nothing about the mechanism.')
    if total['mean_g'] != 1.0 or total['max_abs_dev'] != 0.0:
        raise RuntimeError(f"at zero init the gate must be exactly 1: mean_g={total['mean_g']!r} max|g-1|={total['max_abs_dev']!r}")
    if diff > atol:
        raise RuntimeError(f'zero-initialized receiver gate is NOT identity: max|logit diff| = {diff:.3e} (tolerance {atol:g}). Investigate the implementation rather than loosening this.')
    return {'max_abs_logit_diff': diff, 'rows_exercised': rows_seen, 'mean_g': total['mean_g'], 'max_abs_g_minus_1': total['max_abs_dev'], 'n_values': total['n_values'], 'atol': atol}

def verify_zero_gate_generation(model, input_ids, attention_mask=None, task_id=None, primary_mask=None, context_mask=None, recv_mask=None, max_new_tokens=3):
    bank = get_bank(model)
    (B, T) = (int(input_ids.shape[0]), int(input_ids.shape[1]))
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if task_id is None:
        task_id = torch.arange(B, dtype=torch.long) % N_TASK_ROWS
    task_id = torch.as_tensor(task_id).reshape(-1).long()
    if primary_mask is None or context_mask is None or recv_mask is None:
        (_p, _c, _rv, _, _) = probe_masks(task_id, T)
        primary_mask = _p if primary_mask is None else primary_mask
        context_mask = _c if context_mask is None else context_mask
        recv_mask = _rv if recv_mask is None else recv_mask
    was_training = model.training
    model.eval()
    prev_apply = CTX.apply_gate
    try:
        outs = {}
        for (name, flag) in (('off', False), ('on', True)):
            set_gate_context(torch.as_tensor(task_id), primary_mask, context_mask, recv_mask, valid_mask=attention_mask.bool())
            set_apply_gate(flag)
            with torch.no_grad():
                outs[name] = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
            clear_gate_context()
    finally:
        set_apply_gate(prev_apply)
        clear_gate_context()
        if was_training:
            model.train()
    same = bool(torch.equal(outs['off'], outs['on']))
    if not same:
        raise RuntimeError('zero-initialized receiver gate changed CACHED generation. The mechanism is not identity through decoding; investigate rather than relaxing the check.')
    return {'max_new_tokens': int(max_new_tokens), 'tokens_identical': same, 'n_generated': int(outs['on'].shape[1] - T)}

def gate_activation_stats(model, per_layer=True):
    bank = get_bank(model)
    row_name = {TASK_NONE: 'none'}
    row_name.update({TASK_INDEX[t]: t for t in GATE_TASKS})
    total = blank_acc()
    (by_task, by_stream, by_task_stream, by_half, by_layer) = ({}, {}, {}, {}, {})
    for ((row, layer, stream, half), rec) in bank.acc.items():
        task = row_name.get(row, str(row))
        merge_acc(total, rec)
        merge_acc(by_task.setdefault(task, blank_acc()), rec)
        merge_acc(by_stream.setdefault(stream, blank_acc()), rec)
        merge_acc(by_task_stream.setdefault(f'{task}/{stream}', blank_acc()), rec)
        merge_acc(by_half.setdefault(half, blank_acc()), rec)
        if per_layer:
            merge_acc(by_layer.setdefault(f'{task}/{stream}/L{layer}', blank_acc()), rec)
    out = {'collected': bool(bank.acc), 'total': _finish_acc(total), 'by_task': {k: _finish_acc(v) for (k, v) in sorted(by_task.items())}, 'by_stream': {k: _finish_acc(v) for (k, v) in sorted(by_stream.items())}, 'by_task_stream': {k: _finish_acc(v) for (k, v) in sorted(by_task_stream.items())}, 'by_receiver_half': {k: _finish_acc(v) for (k, v) in sorted(by_half.items())}}
    if per_layer:
        out['by_task_stream_layer'] = {k: _finish_acc(v) for (k, v) in sorted(by_layer.items())}
    out['target_available_by_task'] = bank.target_available()
    out['task_semantics'] = {t: TASK_SEMANTICS[t] for t in bank.tasks}
    out['primary_applicable_by_task'] = {}
    for task in bank.tasks:
        live = primary_gated(task, bank.exclude_target_tasks)
        out['primary_applicable_by_task'][task] = live
        key = f'{task}/primary'
        if live:
            continue
        reason = 'no primary span: ' + str(TASK_SEMANTICS[task]['context']) + ' is its only stream' if not has_primary_stream(task) else 'target held out by the target-available toggle'
        if key in out['by_task_stream'] and out['by_task_stream'][key]['n_values']:
            raise RuntimeError(f"{task}: its primary stream is inert ({reason}) so it must contribute no gate observations, but {out['by_task_stream'][key]['n_values']} were recorded.")
        out['by_task_stream'][key] = {'applicable': False, 'reason': reason, 'n_values': 0, 'n_receivers': 0}
    return out

def gate_weight_stats(model, init_snapshot=None):
    bank = get_bank(model)
    out = {}
    for (stream, plist) in bank.streams().items():
        for (ti, task) in enumerate(bank.tasks):
            (fros, moves, n_zero) = ([], [], 0)
            for layer in range(bank.n_layers):
                w = plist[layer].detach()[ti]
                f = float(torch.linalg.matrix_norm(w.float()).item())
                fros.append(f)
                if f == 0.0:
                    n_zero += 1
                if init_snapshot is not None:
                    w0 = init_snapshot[stream][layer][ti].to(w.device)
                    moves.append(float(torch.linalg.matrix_norm((w - w0).float()).item()))
            rec = {'mean_fro': sum(fros) / len(fros), 'max_fro': max(fros), 'n_exactly_zero_layers': n_zero, 'n_layers': len(fros)}
            if moves:
                rec['mean_move_from_init'] = sum(moves) / len(moves)
                rec['max_move_from_init'] = max(moves)
            out[f'{task}/{stream}'] = rec
    return out

def snapshot(model):
    bank = get_bank(model)
    return {s: [p.detach().clone() for p in plist] for (s, plist) in bank.streams().items()}

def save(model, out_dir, extra=None):
    bank = get_bank(model)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({k: v.cpu() for (k, v) in bank.state_dict().items()}, os.path.join(out_dir, STATE_FILE))
    cfg = {'lora_recv_gate': 'v', 'mechanism': MECHANISM, 'version': VERSION, 'formula': FORMULA, 'gate_function': GATE_FUNCTION, 'gate_stop_gradient': GATE_STOP_GRADIENT, 'stop_gradient_tensors': list(STOP_GRADIENT_TENSORS), 'receiver_scope': RECEIVER_SCOPE, 'controller_input': CONTROLLER_INPUT, 'controlled_quantity': CONTROLLED_QUANTITY, 'streams': list(STREAMS), 'task_specific': True, 'stream_specific': True, 'head_specific_controller': False, 'zero_init': True, 'exclude_target_tasks': list(bank.exclude_target_tasks), 'target_available_by_task': bank.target_available(), 'gate_tasks': list(bank.tasks), 'primary_tasks': list(PRIMARY_TASKS), 'task_semantics': {t: TASK_SEMANTICS[t] for t in bank.tasks}, 'task_rows': {'none': TASK_NONE, **{t: TASK_INDEX[t] for t in GATE_TASKS}}, 'n_layers': bank.n_layers, 'lora_r': bank.r, 'n_gate_params': bank.n_params(), 'state_file': STATE_FILE}
    if extra:
        cfg.update(extra)
    with open(os.path.join(out_dir, CONFIG_FILE), 'w') as f:
        json.dump(cfg, f, indent=2)
    return cfg

def load(model, adapter_dir, strict=True):
    bank = get_bank(model)
    path = os.path.join(adapter_dir, STATE_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f'no {STATE_FILE} in {adapter_dir}')
    cfg = read_config(adapter_dir)
    if cfg:
        _seq_keys = ('gate_tasks', 'exclude_target_tasks')
        for (key, mine) in (('gate_tasks', tuple(bank.tasks)), ('lora_r', bank.r), ('n_layers', bank.n_layers), ('exclude_target_tasks', tuple(bank.exclude_target_tasks))):
            theirs = cfg.get(key)
            if theirs is None:
                continue
            if (key in _seq_keys) != isinstance(theirs, (list, tuple)):
                raise RuntimeError(f'{adapter_dir}: {key} is {theirs!r} ({type(theirs).__name__}), but this field must be ' + ('a list' if key in _seq_keys else 'a scalar') + '; this side-car is malformed.')
            mismatch = tuple(theirs) != tuple(mine) if key in _seq_keys else theirs != mine
            if mismatch:
                raise RuntimeError(f'{adapter_dir} was trained with {key}={theirs!r} but this model is attached with {mine!r}; refusing to serve it under a different configuration.')
        if cfg.get('mechanism', MECHANISM) != MECHANISM:
            raise RuntimeError(f"{adapter_dir} holds mechanism {cfg.get('mechanism')!r}, not {MECHANISM!r}")
    state = torch.load(path, map_location='cpu')
    (missing, unexpected) = bank.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        missing = [m for m in missing if not m.endswith('primary_on')]
        unexpected = [u for u in unexpected if not u.endswith('primary_on')]
        if missing or unexpected:
            raise RuntimeError(f'receiver-gate state mismatch: missing={missing} unexpected={unexpected}')
    bank.to(dtype=PARAM_DTYPE)
    cfg = dict(cfg)
    cfg['trained_with_gate_stop_gradient'] = bool(cfg.get('gate_stop_gradient', False))
    return cfg

def is_recv_gate_adapter(adapter_dir):
    if not adapter_dir:
        return False
    return os.path.exists(os.path.join(adapter_dir, STATE_FILE))

def read_config(adapter_dir):
    if not adapter_dir:
        return {}
    p = os.path.join(adapter_dir, CONFIG_FILE)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)
