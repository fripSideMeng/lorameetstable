from __future__ import annotations
import json
import os
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
import recv_gate as RCV
MECHANISM = 'table_attn_aux'
VERSION = 1
MANIFEST_KEY = 'attn_table_aux'
DEFAULT_LAMBDA = 0.05
DEFAULT_BETA = 10.0
_REG_NAME = 'table_attn_aux'
ALLOWED_BASE = ('sdpa',)
USE_SDPA_REDUCTION = True
VDIM = 8

def tokenize_target(tok, text):
    ids = list(tok(str(text), add_special_tokens=False)['input_ids'])
    if tok.eos_token_id is not None:
        ids = ids + [tok.eos_token_id]
    return tuple(ids)

def build_trie(seqs):
    root = {}
    for s in seqs:
        node = root
        for t in s:
            node = node.setdefault(int(t), {})
    return root

def decision_indices(root, gold):
    out = []
    node = root
    for (t, tokid) in enumerate(gold):
        if not node:
            return (out, False)
        if len(node) > 1:
            out.append(t)
        node = node.get(int(tokid))
        if node is None:
            return (out, False)
    return (out, True)

class TaskCandidates:

    def __init__(self, slug, targets, tok):
        self.slug = slug
        self.labels = sorted({str(x) for x in targets})
        self.seqs = [tokenize_target(tok, s) for s in self.labels]
        self.root = build_trie(self.seqs)

    def __len__(self):
        return len(self.labels)

def build_task_candidates(tok, task_slugs, targets):
    from tqdm import tqdm
    by_task = {}
    for (slug, tgt) in tqdm(list(zip(task_slugs, targets)), desc='attn_aux: label space', unit='ex'):
        by_task.setdefault(slug, set()).add(str(tgt))
    return {s: TaskCandidates(s, v, tok) for (s, v) in sorted(by_task.items())}

def receivers_for_example(labels_row, trie_root):
    sup = [i for (i, v) in enumerate(labels_row) if int(v) != -100]
    if not sup:
        return ([], 'no_supervised_tokens')
    (first, last) = (sup[0], sup[-1])
    if len(sup) != last - first + 1:
        return ([], 'non_contiguous_answer')
    if first == 0:
        return ([], 'answer_at_position_zero')
    gold = [int(labels_row[i]) for i in sup]
    (idxs, matched) = decision_indices(trie_root, gold)
    if not matched:
        return ([], 'gold_not_in_candidate_set')
    if not idxs:
        return ([], 'no_decision_position')
    return ([first + t - 1 for t in idxs], None)

class CTX:
    active = False
    base_attn_forward = None
    recv_idx = None
    recv_valid = None
    allow = None
    table_f = None
    table_v = None
    ex_valid = None
    m_layers = None
    n_recv = None
    skips = None
    n_forwards = 0
    n_skipped_forwards = 0

def clear_context():
    CTX.active = False
    CTX.recv_idx = CTX.recv_valid = CTX.allow = CTX.table_f = CTX.table_v = None
    CTX.ex_valid = CTX.m_layers = CTX.n_recv = None
    CTX.skips = None

def reset_stats():
    CTX.n_forwards = 0
    CTX.n_skipped_forwards = 0

def stats():
    return {'forwards_measured': int(CTX.n_forwards), 'forwards_skipped': int(CTX.n_skipped_forwards), 'reduction': 'sdpa' if USE_SDPA_REDUCTION else 'explicit'}

def pad_receivers(per_ex, device=None):
    B = len(per_ex)
    P = max(1, max((len(p) for p in per_ex), default=1))
    recv_idx = torch.zeros(B, P, dtype=torch.long, device=device)
    recv_valid = torch.zeros(B, P, dtype=torch.bool, device=device)
    for (b, pos) in enumerate(per_ex):
        if len(pos):
            recv_idx[b, :len(pos)] = torch.as_tensor(list(pos), dtype=torch.long, device=device)
            recv_valid[b, :len(pos)] = True
    return (recv_idx, recv_valid)

def receivers_from_labels(labels, task_slugs, candidates):
    lab = torch.as_tensor(labels).detach().cpu().tolist()
    (per_ex, skips) = ([], {})
    for (b, row) in enumerate(lab):
        cand = candidates.get(task_slugs[b])
        if cand is None:
            per_ex.append([])
            skips['no_candidates_for_task'] = skips.get('no_candidates_for_task', 0) + 1
            continue
        (pos, reason) = receivers_for_example(row, cand.root)
        if reason is not None:
            skips[reason] = skips.get(reason, 0) + 1
        per_ex.append(pos)
    return (per_ex, skips)

def set_context(recv_idx, recv_valid, region_ids, attention_mask, skips=None):
    region_ids = torch.as_tensor(region_ids)
    recv_idx = torch.as_tensor(recv_idx)
    recv_valid = torch.as_tensor(recv_valid).bool()
    (B, S) = region_ids.shape
    dev = region_ids.device
    if recv_idx.shape != recv_valid.shape or recv_idx.shape[0] != B:
        raise ValueError(f'receiver shapes {tuple(recv_idx.shape)}/{tuple(recv_valid.shape)} do not match a batch of {B} examples')
    recv_idx = recv_idx.to(dev)
    recv_valid = recv_valid.to(dev)
    if bool(recv_valid.any()) and int(recv_idx[recv_valid].max().item()) >= S:
        raise ValueError('a receiver position falls outside the padded sequence; the positions were built against a different tokenization')
    if attention_mask is None:
        valid = torch.ones(B, S, dtype=torch.bool, device=dev)
    else:
        valid = torch.as_tensor(attention_mask).to(dev).bool()
    table = (region_ids == RCV.R_TABLE) & valid
    skips = dict(skips or {})
    no_table = recv_valid.any(dim=1) & ~table.any(dim=1)
    if bool(no_table.any()):
        skips['no_table_tokens'] = skips.get('no_table_tokens', 0) + int(no_table.sum().item())
        recv_valid = recv_valid & ~no_table[:, None]
    keys = torch.arange(S, device=dev)
    allow = (keys[None, None, :] <= recv_idx[:, :, None]) & valid[:, None, :]
    fallback = torch.zeros(S, dtype=torch.bool, device=dev)
    fallback[0] = True
    allow = torch.where(recv_valid[:, :, None], allow, fallback[None, None, :])
    CTX.recv_idx = recv_idx
    CTX.recv_valid = recv_valid
    CTX.allow = allow.unsqueeze(1)
    CTX.table_f = table.to(torch.float32)
    CTX.table_v = {}
    CTX.n_recv = recv_valid.sum(dim=1)
    CTX.ex_valid = CTX.n_recv > 0
    CTX.m_layers = {}
    CTX.skips = skips
    CTX.active = True
    return {'n_examples': int(B), 'n_examples_used': int(CTX.ex_valid.sum().item()), 'n_receivers': int(CTX.n_recv.sum().item()), 'skips': dict(skips)}

def set_context_from_labels(labels, region_ids, attention_mask, task_slugs, candidates):
    (per_ex, skips) = receivers_from_labels(labels, task_slugs, candidates)
    (idx, valid) = pad_receivers(per_ex, device=torch.as_tensor(region_ids).device)
    return set_context(idx, valid, region_ids, attention_mask, skips=skips)

def _select_q(query, P, d):
    (B, H) = (query.shape[0], query.shape[1])
    gather_idx = CTX.recv_idx[:, None, :, None].expand(B, H, P, d)
    return torch.gather(query, 2, gather_idx)

def _table_values(B, Hkv, S, dtype, device):
    key = (str(dtype), str(device), int(Hkv))
    v = CTX.table_v.get(key)
    if v is None:
        v = CTX.table_f.to(dtype=dtype, device=device)[:, None, :, None]
        v = v.expand(B, Hkv, S, VDIM).contiguous()
        CTX.table_v[key] = v
    return v

def _mass_sdpa(query, key, scaling, P):
    (B, H, S, d) = query.shape
    Hkv = key.shape[1]
    q_sel = _select_q(query, P, d)
    v = _table_values(B, Hkv, S, query.dtype, query.device)
    out = torch.nn.functional.scaled_dot_product_attention(q_sel, key, v, attn_mask=CTX.allow, dropout_p=0.0, scale=float(scaling), enable_gqa=H != Hkv)
    return out[..., 0].float()

def _mass_explicit(query, key, scaling, P):
    (B, H, S, d) = query.shape
    Hkv = key.shape[1]
    n_rep = H // Hkv
    q_sel = _select_q(query, P, d).view(B, Hkv, n_rep, P, d)
    logits = torch.matmul(q_sel, key.unsqueeze(2).transpose(-1, -2))
    logits = logits.reshape(B, H, P, S).float() * float(scaling)
    logits = logits.masked_fill(~CTX.allow, float('-inf'))
    probs = torch.softmax(logits, dim=-1)
    return (probs * CTX.table_f[:, None, None, :]).sum(-1)

def _accumulate(module, query, key, scaling):
    (B, H, S, d) = query.shape
    if key.shape[2] != S:
        return
    if CTX.recv_idx.shape[0] != B or CTX.allow.shape[-1] != S:
        raise RuntimeError(f'{MECHANISM}: context is [{CTX.recv_idx.shape[0]}, {CTX.allow.shape[-1]}] but the layer sees [{B}, {S}]; the per-batch context was not refreshed.')
    P = CTX.recv_idx.shape[1]
    if scaling is None:
        scaling = float(d) ** (-0.5)
    m = (_mass_sdpa if USE_SDPA_REDUCTION else _mass_explicit)(query, key, scaling, P)
    w = CTX.recv_valid[:, None, :].to(m.dtype)
    denom = (CTX.n_recv.to(m.dtype) * H).clamp(min=1.0)
    CTX.m_layers[int(getattr(module, 'layer_idx', len(CTX.m_layers)))] = (m * w).sum((1, 2)) / denom

def aux_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    base = CTX.base_attn_forward or ALL_ATTENTION_FUNCTIONS['sdpa']
    if CTX.active and torch.is_grad_enabled():
        _accumulate(module, query, key, scaling)
        CTX.n_forwards += 1
    else:
        CTX.n_skipped_forwards += 1
    return base(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
ALL_ATTENTION_FUNCTIONS.register(_REG_NAME, aux_attention_forward)
AttentionMaskInterface.register(_REG_NAME, sdpa_mask)

def attach(model, base_impl='sdpa'):
    if base_impl not in ALLOWED_BASE:
        raise ValueError(f'base_impl must be one of {ALLOWED_BASE} (got {base_impl!r})')
    CTX.base_attn_forward = ALL_ATTENTION_FUNCTIONS[base_impl]
    AttentionMaskInterface.register(_REG_NAME, sdpa_mask)
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
        raise RuntimeError(f'{MECHANISM}: no *Attention submodules found to patch')
    return n_set

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

def aux_loss(beta=None):
    if not CTX.active or not CTX.m_layers:
        return (None, {'n_layers': 0, 'n_examples_used': 0})
    order = sorted(CTX.m_layers)
    stack = torch.stack([CTX.m_layers[l] for l in order], dim=0)
    valid = CTX.ex_valid
    if not bool(valid.any()):
        return (None, {'n_layers': len(order), 'n_examples_used': 0})
    stack = stack[:, valid]
    mean_m = stack.mean(dim=0, keepdim=True)
    var_per_example = ((stack - mean_m) ** 2).sum(dim=0)
    loss = var_per_example.mean()
    info = {'n_layers': len(order), 'n_examples_used': int(valid.sum().item()), 'n_receivers': int(CTX.n_recv.sum().item()), 'var_mean': float(loss.detach().item()), 'mean_table_mass': float(stack.mean(dim=0).mean().detach().item()), 'total_table_mass_mean': float(stack.sum(dim=0).mean().detach().item()), 'depth_std_mean': float((var_per_example / max(1, len(order))).sqrt().mean().detach().item()), 'm_per_layer': [float(x) for x in stack.mean(dim=1).detach().tolist()], 'layer_ids': [int(x) for x in order]}
    return (loss, info)

class AuxLogger:

    def __init__(self, n_layers=0):
        self.reset()
        self.n_layers = n_layers

    def reset(self):
        self.n = 0
        self.ce = 0.0
        self.aux = 0.0
        self.total = 0.0
        self.var = 0.0
        self.mean_mass = 0.0
        self.total_mass = 0.0
        self.depth_std = 0.0
        self.recv = 0.0
        self.ex_used = 0
        self.ex_seen = 0
        self.m_layer = None
        self.skips = {}

    def update(self, ce, aux, total, info, batch_info):
        self.n += 1
        self.ce += float(ce)
        self.aux += float(aux)
        self.total += float(total)
        self.var += float(info.get('var_mean', 0.0))
        self.mean_mass += float(info.get('mean_table_mass', 0.0))
        self.total_mass += float(info.get('total_table_mass_mean', 0.0))
        self.depth_std += float(info.get('depth_std_mean', 0.0))
        self.recv += float(info.get('n_receivers', 0))
        self.ex_used += int(info.get('n_examples_used', 0))
        self.ex_seen += int(batch_info.get('n_examples', 0))
        ml = info.get('m_per_layer')
        if ml:
            if self.m_layer is None or len(self.m_layer) != len(ml):
                self.m_layer = [0.0] * len(ml)
            for (i, v) in enumerate(ml):
                self.m_layer[i] += v
        for (k, v) in batch_info.get('skips', {}).items():
            self.skips[k] = self.skips.get(k, 0) + int(v)

    def summary(self):
        n = max(1, self.n)
        return {'steps': self.n, 'ce': self.ce / n, 'aux': self.aux / n, 'total': self.total / n, 'var_mean': self.var / n, 'mean_table_mass': self.mean_mass / n, 'total_table_mass_mean': self.total_mass / n, 'depth_std_mean': self.depth_std / n, 'receivers_per_batch': self.recv / n, 'examples_used': self.ex_used, 'examples_seen': self.ex_seen, 'm_per_layer': None if self.m_layer is None else [v / n for v in self.m_layer], 'skips': dict(self.skips)}

    def line(self, peak_mem=None):
        s = self.summary()
        ml = s['m_per_layer'] or []
        head = f"[{MECHANISM}] ce={s['ce']:.4f} aux={s['aux']:.6f} total={s['total']:.4f} var={s['var_mean']:.6f} depth_std={s['depth_std_mean']:.4f} mean_mass={s['mean_table_mass']:.4f} total_mass={s['total_table_mass_mean']:.3f} recv/batch={s['receivers_per_batch']:.2f} ex_used={s['examples_used']}/{s['examples_seen']}"
        if peak_mem is not None:
            head += f' peak_mem={peak_mem:.2f}GiB'
        if s['skips']:
            head += f" skips={s['skips']}"
        if ml:
            head += '\n  m_l = ' + ' '.join((f'{i}:{v:.3f}' for (i, v) in enumerate(ml)))
        return head
MARKER = 'attn_table_aux.json'

def write_marker(out_dir, info=None):
    payload = {'mechanism': MECHANISM, 'version': VERSION}
    if info:
        payload.update(info)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, MARKER), 'w') as f:
        json.dump(payload, f, indent=2)
    return payload
