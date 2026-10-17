import json
import os
import re
import torch
import recv_gate as RCV
import role_low_rank as RLR
MECHANISM = 'target_sandwich'
VERSION = 3
MANIFEST_KEY = 'target_sandwich'
MASK_KEY = 'sandwich_mask'
MARKER = 'target_sandwich.json'
(COPY_NONE, COPY_FRONT, COPY_CONTEXT, COPY_BACK) = (0, 1, 2, 3)
N_COPY = 4
COPY_NAMES = {COPY_NONE: 'none', COPY_FRONT: 't_front', COPY_CONTEXT: 'context', COPY_BACK: 't_back'}
COPY_RANK = {COPY_NONE: 0, COPY_CONTEXT: 1, COPY_BACK: 2, COPY_FRONT: 3}
SANDWICH = {'cta': 'T_front | context | T_back  (query -> T_back)', 'cpa': 'H,T_front | context | H,T_back  (query -> the last pair)'}
ALLOWED_BASE = ('sdpa', 'eager')
_COL_NAME_RE = re.compile('col\\d+')
_MASK_DTYPE = {'sdpa': None, 'eager': torch.float32}

def target_columns(slug, text):
    import evaluate as E
    if slug not in E._SHUFFLE_TASKS:
        return (None, 'task_not_supported')
    if slug.startswith('cpa'):
        (mh, mt) = (E._CPA_HEAD_RE.search(text), E._CPA_TAIL_RE.search(text))
        if not (mh and mt):
            return (None, 'no_markers')
        names = [mh.group(1), mt.group(1)]
    else:
        m = E._CTA_TARGET_RE.search(text)
        if not m:
            return (None, 'no_marker')
        names = [m.group(1)]
    tm = E._MD_TABLE_RE.search(text)
    if not tm:
        return (None, 'no_table')
    header = [c.strip() for c in E._md_cells(tm.group(0).split('\n')[0])]
    idx = []
    for nm in names:
        if nm not in header:
            m = _COL_NAME_RE.search(nm)
            nm = m.group(0) if m else nm
        if nm not in header:
            return (None, 'target_not_in_header')
        idx.append(header.index(nm))
    if len(set(idx)) != len(idx):
        return (None, 'duplicate_target_columns')
    return (idx, None)

def sandwich_prompt(slug, text):
    import evaluate as E
    (targets, why) = target_columns(slug, text)
    if targets is None:
        return (text, {'skipped': True, 'reason': why})
    tm = E._MD_TABLE_RE.search(text)
    rows = tm.group(0).split('\n')
    header = [c.strip() for c in E._md_cells(rows[0])]
    n = len(header)
    k = len(targets)
    ctx = [c for c in range(n) if c not in targets]
    order = list(targets) + ctx + list(targets)
    n_new = len(order)
    new_rows = []
    for (ri, row) in enumerate(rows):
        cs = E._md_cells(row)
        if len(cs) != n:
            return (text, {'skipped': True, 'reason': 'ragged_row'})
        if ri == 0:
            new_rows.append('|' + '|'.join((E._relabel_header_cell(cs[o], p) for (p, o) in enumerate(order))) + '|')
        else:
            new_rows.append('|' + '|'.join((cs[o] for o in order)) + '|')
    new_text = text[:tm.start()] + '\n'.join(new_rows) + text[tm.end():]
    back = list(range(n_new - k, n_new))
    if slug.startswith('cpa'):
        new_text = E._sub_marker(new_text, E._CPA_HEAD_RE, f'col{back[0]}')
        new_text = E._sub_marker(new_text, E._CPA_TAIL_RE, f'col{back[1]}')
    else:
        new_text = E._sub_marker(new_text, E._CTA_TARGET_RE, f'col{back[0]}')
    return (new_text, {'skipped': False, 'n': n, 'n_new': n_new, 'targets': targets, 'front': list(range(k)), 'back': back, 'context': list(range(k, n_new - k)), 'already_front': targets == list(range(k)), 'already_back': targets == list(range(n - k, n))})

class _Transform:
    considered = 0
    sandwiched = 0
    not_supported_task = 0
    skipped = None
TRANSFORM = _Transform()

def reset_transform_stats():
    import collections
    TRANSFORM.considered = TRANSFORM.sandwiched = TRANSFORM.not_supported_task = 0
    TRANSFORM.skipped = collections.Counter()

def transform_stats():
    return {'considered': TRANSFORM.considered, 'sandwiched': TRANSFORM.sandwiched, 'not_supported_task': TRANSFORM.not_supported_task, 'skipped': dict(TRANSFORM.skipped or {})}

def sandwich_row_text(slug, text):
    if TRANSFORM.skipped is None:
        reset_transform_stats()
    base = str(slug or '').lower()
    if not (base.startswith('cta') or base.startswith('cpa')):
        TRANSFORM.not_supported_task += 1
        return (text, False)
    TRANSFORM.considered += 1
    (new_text, meta) = sandwich_prompt(slug, text)
    if meta.get('skipped'):
        TRANSFORM.skipped[f"{base.split('_')[0]}:{meta.get('reason')}"] += 1
        return (text, False)
    TRANSFORM.sandwiched += 1
    return (new_text, True)

def format_sandwich(text, meta, max_show=12):
    if meta.get('skipped'):
        return f"[{MECHANISM}] sandwich SKIPPED ({meta.get('reason')})"
    return f"[{MECHANISM}] sandwich N={meta['n']} -> {meta['n_new']} visible columns; target(s) {meta['targets']} -> front {meta['front']} + back {meta['back']}; context {meta['context'][:max_show]}; query -> col{meta['back'][0]}" + (f"..col{meta['back'][-1]}" if len(meta['back']) > 1 else '')

def content_equality_audit(slug, text):
    import evaluate as E
    (targets, why) = target_columns(slug, text)
    if targets is None:
        return {'ok': False, 'reason': why}
    k = len(targets)
    tm = E._MD_TABLE_RE.search(text)
    if not tm:
        return {'ok': False, 'reason': 'no_table'}
    rows = tm.group(0).split('\n')
    w = len(E._md_cells(rows[0]))
    if w < 2 * k:
        return {'ok': False, 'reason': 'too_few_columns', 'n_visible': w}
    (bad, checked) = ([], 0)
    for (ri, row) in enumerate(rows):
        if ri == 0:
            continue
        cs = E._md_cells(row)
        if len(cs) != w:
            return {'ok': False, 'reason': 'ragged_row', 'row': ri}
        for j in range(k):
            checked += 1
            if cs[j].strip() != cs[w - k + j].strip():
                bad.append({'row': ri, 'front': j, 'back': w - k + j, 'front_text': cs[j], 'back_text': cs[w - k + j]})
    return {'ok': not bad, 'n_checked': checked, 'n_bad': len(bad), 'first_bad': bad[:3]}

def is_sandwiched(slug, text, targets=None):
    if targets is None:
        (targets, _) = target_columns(slug, text)
        if targets is None:
            return False
    import evaluate as E
    tm = E._MD_TABLE_RE.search(text)
    if not tm:
        return False
    k = len(targets)
    n = len(E._md_cells(tm.group(0).split('\n')[0]))
    if n < 2 * k or targets != list(range(n - k, n)):
        return False
    a = content_equality_audit(slug, text)
    return bool(a.get('ok')) and a.get('n_checked', 0) > 0

def char_copy_ids(slug, text, targets=None):
    cls = [COPY_NONE] * len(text)
    if targets is None:
        (targets, _) = target_columns(slug, text)
    if targets is None:
        return cls
    k = len(targets)
    if not is_sandwiched(slug, text, targets):
        return cls
    (col, content, _headers, content_cols) = RLR.scan_table_columns(text)
    if not content_cols:
        return cls
    cols_sorted = sorted(content_cols)
    back = {t + 1 for t in targets}
    front = set(cols_sorted[:k])
    if front & back:
        return cls
    for g in range(len(text)):
        if not content[g]:
            continue
        c = col[g]
        if c in front:
            cls[g] = COPY_FRONT
        elif c in back:
            cls[g] = COPY_BACK
        elif c in content_cols:
            cls[g] = COPY_CONTEXT
    return cls

def _span_copy(span):
    (best, best_rank) = (COPY_NONE, 0)
    for c in span:
        r = COPY_RANK[c]
        if r > best_rank:
            (best, best_rank) = (c, r)
    return best

def tokens_copy_ids(tok, slug, text, targets=None):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    (ids, offs) = (enc['input_ids'], enc['offset_mapping'])
    cmap = char_copy_ids(slug, text, targets=targets)
    cls = [_span_copy(cmap[a:b]) if b > a else COPY_NONE for (a, b) in offs]
    return (ids, cls)

def build_copy_ids(tok, raw_prompt, rendered, task, expect_ids=None):
    slug = str(task or '')
    (targets, _) = target_columns(slug, raw_prompt)
    (ids, cls) = tokens_copy_ids(tok, slug, rendered, targets=targets)
    if expect_ids is not None and list(ids) != list(expect_ids):
        raise RuntimeError(f"copy-id tokenization ({len(ids)} tokens) does not reproduce the model's own prompt tokenization ({len(expect_ids)} tokens). The mask must be aligned to the ids the model is fed; refusing to train on a misaligned copy-id sequence.")
    return cls

def pad_copy_ids(copy_lists, width, side='right'):
    out = []
    for c in copy_lists:
        n = width - len(c)
        if n < 0:
            raise ValueError('copy-id sequence longer than the padded width')
        out.append([COPY_NONE] * n + list(c) if side == 'left' else list(c) + [COPY_NONE] * n)
    return out

def copy_ids_from_attention_mask(copy_lists, attn_mask):
    T = attn_mask.shape[1]
    rows = []
    for (c, m) in zip(copy_lists, attn_mask.tolist()):
        n_pad = len(m) - int(sum(m))
        if any((m[k] == 0 for k in range(n_pad, T))) or any((m[k] == 1 for k in range(n_pad))):
            raise ValueError('copy ids can only be aligned to a LEFT-padded batch')
        if len(c) != T - n_pad:
            raise ValueError(f'copy-id sequence of {len(c)} does not fill the {T - n_pad} unpadded slots')
        rows.append([COPY_NONE] * n_pad + list(c))
    return torch.tensor(rows, dtype=torch.long)

def _class_lut(keep, device):
    lut = torch.zeros(N_COPY, dtype=torch.bool, device=device)
    if keep:
        lut[list(keep)] = True
    return lut

def front_mask_from_ids(copy_ids):
    c = torch.as_tensor(copy_ids)
    return _class_lut((COPY_FRONT,), c.device)[c.long()]

def back_mask_from_ids(copy_ids):
    c = torch.as_tensor(copy_ids)
    return _class_lut((COPY_BACK,), c.device)[c.long()]

def context_mask_from_ids(copy_ids):
    c = torch.as_tensor(copy_ids)
    return _class_lut((COPY_CONTEXT,), c.device)[c.long()]

def receiver_mask_from_regions(region_ids):
    return RCV.receiver_mask_from_regions(region_ids)

def masks_for_batch(copy_ids, region_ids=None):
    c = torch.as_tensor(copy_ids)
    if region_ids is None:
        return (front_mask_from_ids(c), back_mask_from_ids(c), torch.zeros_like(c, dtype=torch.bool))
    g = torch.as_tensor(region_ids)
    if c.shape != g.shape:
        raise ValueError(f'copy_ids {tuple(c.shape)} and region_ids {tuple(g.shape)} must agree')
    return (front_mask_from_ids(c), back_mask_from_ids(c), receiver_mask_from_regions(g))

def build_mask(front_full, cut_q_full, cache_position, pad_mask, dtype=None):
    frt = torch.as_tensor(front_full).bool()
    cut = torch.as_tensor(cut_q_full).bool()
    if frt.dim() != 2 or cut.shape != frt.shape:
        raise ValueError(f'front/cut masks must both be [B,T]; got {tuple(frt.shape)} and {tuple(cut.shape)}')
    dev = frt.device
    pad = torch.as_tensor(pad_mask).to(dev).bool()
    if pad.dim() != 2 or pad.shape[0] != frt.shape[0]:
        raise ValueError(f'pad mask must be [B,T_k] with B={frt.shape[0]}; got {tuple(pad.shape)}')
    cp = torch.as_tensor(cache_position).to(dev).long().reshape(-1)
    (B, Tp) = frt.shape
    Tk = pad.shape[1]
    cut = cut.to(dev)
    if Tk <= Tp:
        frt_k = frt[:, :Tk]
    else:
        frt_k = torch.cat([frt, torch.zeros(B, Tk - Tp, dtype=torch.bool, device=dev)], dim=1)
    in_prompt = cp < Tp
    cut_q = cut[:, cp.clamp(max=Tp - 1)] & in_prompt[None, :]
    idx_k = torch.arange(Tk, device=dev)
    allowed = cp[None, :, None] >= idx_k[None, None, :]
    allowed = allowed.expand(B, -1, -1).clone()
    allowed &= pad[:, None, :]
    allowed &= ~(cut_q[:, :, None] & frt_k[:, None, :])
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
    front_full = None
    cut_full = None
    mask_dtype = None
    n_applied = 0
    n_skipped = 0
CTX = _Ctx()

def set_context(front_mask, cut_mask):
    f = torch.as_tensor(front_mask).bool()
    c = torch.as_tensor(cut_mask).bool()
    if f.dim() != 2 or c.shape != f.shape:
        raise ValueError(f'front/cut masks must both be [B,T]; got {tuple(f.shape)} and {tuple(c.shape)}')
    (CTX.front_full, CTX.cut_full) = (f, c.to(f.device))
    CTX.active = True

def set_context_from_ids(copy_ids, region_ids=None):
    (front, back, recv) = masks_for_batch(copy_ids, region_ids)
    set_context(front, back)
    return (front, back, recv)

def clear_context():
    CTX.active = False
    CTX.front_full = CTX.cut_full = None

def reset_stats():
    CTX.n_applied = CTX.n_skipped = 0

def stats():
    return {'forwards_masked': CTX.n_applied, 'forwards_unmasked': CTX.n_skipped}

def _mask_prehook(module, args, kwargs):
    if not CTX.active or CTX.front_full is None:
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
    kwargs['attention_mask'] = build_mask(CTX.front_full.to(dev), CTX.cut_full.to(dev), cp, am, dtype=CTX.mask_dtype)
    CTX.n_applied += 1
    return (args, kwargs)

def attach(model, base_impl='sdpa'):
    if base_impl not in ALLOWED_BASE:
        raise ValueError(f'--target_sandwich needs --attn_impl in {list(ALLOWED_BASE)} (got {base_impl!r}): flash_attention_2 cannot take a custom 4D attention mask.')
    if getattr(model, '_tgt_sandwich_handles', None):
        raise RuntimeError('tgt_sandwich is already attached to this model')
    dt = _MASK_DTYPE[base_impl]
    if base_impl == 'eager':
        dt = next((p.dtype for p in model.parameters() if p.is_floating_point), torch.float32)
    CTX.mask_dtype = dt
    reset_stats()
    inner = model.get_base_model() if hasattr(model, 'get_base_model') else model
    hosts = [model] if inner is model else [model, inner]
    handles = [h.register_forward_pre_hook(_mask_prehook, with_kwargs=True) for h in hosts]
    model._tgt_sandwich_handles = handles
    return {'mechanism': MECHANISM, 'version': VERSION, 'layout': dict(SANDWICH), 'base_attn_impl': base_impl, 'mask_kind': 'bool' if dt is None else f'additive {dt}', 'prehook_on': [type(h).__name__ for h in hosts], 'cut_keys': 'T_front cell content', 'cut_queries': 'T_back cell content only (no query/answer/generated position is cut)', 'n_new_parameters': 0}

def detach(model):
    for h in getattr(model, '_tgt_sandwich_handles', []) or []:
        h.remove()
    model._tgt_sandwich_handles = []
    clear_context()

def read_mask_marker(adapter_dir):
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
        if MASK_KEY in d:
            return bool(d[MASK_KEY])
        if isinstance(d.get('args'), dict) and MASK_KEY in d['args']:
            return bool(d['args'][MASK_KEY])
    return None

def write_marker(out_dir, info=None):
    os.makedirs(out_dir, exist_ok=True)
    rec = {MANIFEST_KEY: True, 'mechanism': MECHANISM, 'version': VERSION, 'layout': dict(SANDWICH)}
    if info and MASK_KEY in info:
        rec[MASK_KEY] = bool(info[MASK_KEY])
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

def training_audit(enc_copy, enc_regions, enc_tasks=None):
    per_task = {}
    n_no_front = n_no_back = n_no_context = n_no_recv = n_unequal = 0
    for (i, (copies, regions)) in enumerate(zip(enc_copy, enc_regions)):
        c = torch.tensor(copies, dtype=torch.long).unsqueeze(0)
        g = torch.tensor(regions, dtype=torch.long).unsqueeze(0)
        n_f = int(front_mask_from_ids(c).sum())
        n_b = int(back_mask_from_ids(c).sum())
        n_c = int(context_mask_from_ids(c).sum())
        n_r = int(receiver_mask_from_regions(g).sum())
        t = enc_tasks[i] if enc_tasks is not None else 'all'
        d = per_task.setdefault(t, {'examples': 0, 'front_tokens': 0, 'back_tokens': 0, 'context_tokens': 0, 'receiver_tokens': 0, 'no_front': 0, 'no_back': 0, 'no_context': 0, 'no_receiver': 0, 'unequal_copies': 0, 'inert': 0})
        d['examples'] += 1
        d['front_tokens'] += n_f
        d['back_tokens'] += n_b
        d['context_tokens'] += n_c
        d['receiver_tokens'] += n_r
        if n_f == 0:
            d['no_front'] += 1
            n_no_front += 1
        if n_b == 0:
            d['no_back'] += 1
            n_no_back += 1
        if n_c == 0:
            d['no_context'] += 1
            n_no_context += 1
        if n_r == 0:
            d['no_receiver'] += 1
            n_no_recv += 1
        if n_f and n_b and (n_f != n_b):
            d['unequal_copies'] += 1
            n_unequal += 1
        if n_f == 0:
            d['inert'] += 1
    return {'per_task': per_task, 'n_examples': len(enc_copy), 'n_no_front': n_no_front, 'n_no_back': n_no_back, 'n_no_context': n_no_context, 'n_no_receiver': n_no_recv, 'n_unequal_copies': n_unequal}

def format_audit(audit):
    L = [f'[{MECHANISM}] layout: ' + '; '.join((f'{k.upper()} {v}' for (k, v) in SANDWICH.items()))]
    for (t, d) in sorted(audit['per_task'].items()):
        L.append(f"[{MECHANISM}]   {t}: {d['examples']:,} examples, front={d['front_tokens']:,} tok (cut from T_back), back={d['back_tokens']:,} tok, context={d['context_tokens']:,} tok, receivers={d['receiver_tokens']:,} tok, inert={d['inert']}, unequal_copies={d['unequal_copies']}")
    if audit.get('n_no_context'):
        L.append(f"[{MECHANISM}] NOTE: {audit['n_no_context']} example(s) have NO context column between the two copies, so there is no relay for the front copy to run through and the mask is a deletion rather than a sandwich for them.")
    if audit.get('n_unequal_copies'):
        L.append(f"[{MECHANISM}] NOTE: {audit['n_unequal_copies']} example(s) tokenize their two copies to DIFFERENT token counts -- the cells are byte-identical, but the tokenizer splits them differently at their two positions in the row.")
    return '\n'.join(L)
