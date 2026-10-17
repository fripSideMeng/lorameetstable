from __future__ import annotations
import random
import re
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import evaluate as E
_COL_NUM_RE = re.compile('col(\\d+)')
_SEP_CELL_RE = re.compile('^:?-+:?$')

def _canonical(task: str) -> str:
    s = str(task).lower()
    for base in ('cta', 'cpa'):
        if s == base or s.startswith(base + '_'):
            return base
    return s

def parse_columns(content: str):
    from lora_internal_per_token import find_markdown_table_span
    span = find_markdown_table_span(content)
    if span is None:
        return 'no_table'
    (tstart, tend) = span
    block = content[tstart:tend]
    char_to_col = np.full(len(content), -1, dtype=np.int32)
    line_start = tstart
    header_n = None
    for (li, line) in enumerate(block.split('\n')):
        pipes = [i for (i, ch) in enumerate(line) if ch == '|']
        n_cells = len(pipes) - 1
        if len(pipes) < 2:
            line_start += len(line) + 1
            continue
        if header_n is None:
            if n_cells < 2:
                return 'single_column'
            header_n = n_cells
            line_start += len(line) + 1
            continue
        if n_cells != header_n:
            return 'ragged_row'
        cells = E._md_cells(line)
        is_sep = bool(cells) and all((_SEP_CELL_RE.match(c.strip()) for c in cells))
        if not is_sep:
            for k in range(header_n):
                cs = line_start + pipes[k] + 1
                ce = line_start + pipes[k + 1]
                char_to_col[cs:ce] = k
        line_start += len(line) + 1
    if header_n is None:
        return 'no_table'
    return (header_n, char_to_col)

def token_columns(content: str, coff: int, offsets, n_prompt: int):
    parsed = parse_columns(content)
    if isinstance(parsed, str):
        return (None, parsed)
    (n_cols, char_to_col) = parsed
    col = np.full(n_prompt, -1, dtype=np.int64)
    clen = len(content)
    for t in range(min(n_prompt, len(offsets))):
        (a, b) = offsets[t]
        if a == b:
            continue
        (ca, cb) = (a - coff, b - coff)
        if cb <= 0 or ca >= clen:
            continue
        seg = char_to_col[max(ca, 0):min(cb, clen)]
        seg = seg[seg >= 0]
        if seg.size == 0:
            continue
        col[t] = int(np.bincount(seg).argmax())
    return (col, n_cols)

def mandatory_columns(task: str, content: str):
    base = _canonical(task)

    def _colnum(m):
        if not m:
            return None
        g = _COL_NUM_RE.search(m.group(1))
        return int(g.group(1)) if g else None
    if base == 'cta':
        c = _colnum(E._CTA_TARGET_RE.search(content))
        return None if c is None else {c}
    if base == 'cpa':
        h = _colnum(E._CPA_HEAD_RE.search(content))
        t = _colnum(E._CPA_TAIL_RE.search(content))
        return None if h is None or t is None else {h, t}
    return None

def mask_columns_prompt(task: str, text: str, keep):
    base = _canonical(task)
    keep = sorted(set((int(k) for k in keep)))
    if base == 'cta':
        m = E._CTA_TARGET_RE.search(text)
        if not m:
            return (text, {'skipped': True, 'reason': 'no_marker'})
        tgt_names = [m.group(1)]
    elif base == 'cpa':
        (mh, mt) = (E._CPA_HEAD_RE.search(text), E._CPA_TAIL_RE.search(text))
        if not (mh and mt):
            return (text, {'skipped': True, 'reason': 'no_markers'})
        tgt_names = [mh.group(1), mt.group(1)]
    else:
        return (text, {'skipped': True, 'reason': 'task_not_supported'})
    tm = E._MD_TABLE_RE.search(text)
    if not tm:
        return (text, {'skipped': True, 'reason': 'no_table'})
    rows = tm.group(0).split('\n')
    header = [c.strip() for c in E._md_cells(rows[0])]
    n = len(header)
    if n < 2:
        return (text, {'skipped': True, 'reason': 'too_few_columns', 'n_cols': n})
    if not keep or keep[0] < 0 or keep[-1] >= n:
        return (text, {'skipped': True, 'reason': 'keep_out_of_range'})
    try:
        targets = [header.index(nm) for nm in tgt_names]
    except ValueError:
        return (text, {'skipped': True, 'reason': 'target_not_in_header'})
    if any((t not in keep for t in targets)):
        return (text, {'skipped': True, 'reason': 'mandatory_dropped'})
    new_pos = {orig: p for (p, orig) in enumerate(keep)}
    new_rows = []
    for (ri, row) in enumerate(rows):
        cs = E._md_cells(row)
        if len(cs) != n:
            return (text, {'skipped': True, 'reason': 'ragged_row'})
        if ri == 0:
            new_rows.append('|' + '|'.join((E._relabel_header_cell(cs[orig], new_pos[orig]) for orig in keep)) + '|')
        else:
            new_rows.append('|' + '|'.join((cs[orig] for orig in keep)) + '|')
    new_text = text[:tm.start()] + '\n'.join(new_rows) + text[tm.end():]
    if base == 'cpa':
        new_text = E._sub_marker(new_text, E._CPA_HEAD_RE, f'col{new_pos[targets[0]]}')
        new_text = E._sub_marker(new_text, E._CPA_TAIL_RE, f'col{new_pos[targets[1]]}')
    else:
        new_text = E._sub_marker(new_text, E._CTA_TARGET_RE, f'col{new_pos[targets[0]]}')
    return (new_text, {'skipped': False, 'n_cols': n, 'kept': keep, 'targets': targets, 'new_targets': [new_pos[t] for t in targets]})

def build_attrib_item(tok, row, max_seq_len, model_name, task_prefix):
    import train_lora as T
    iid = row['id']
    task = row.get('task')
    content = T.with_task_prefix(row['input'], task, task_prefix)
    target_ids = tok(row['target'], add_special_tokens=False)['input_ids']
    if tok.eos_token_id is not None:
        target_ids = target_ids + [tok.eos_token_id]
    prompt_ids = T._encode_prompt(tok, content, model_name)
    if len(prompt_ids) + len(target_ids) > max_seq_len:
        (content, prompt_ids) = T._trim_table_rows(content, tok, len(target_ids), max_seq_len, model_name)
    full_ids = prompt_ids + target_ids
    if len(full_ids) > max_seq_len:
        return {'skip': 'over_len', 'instance_id': iid}
    labels = [-100] * len(prompt_ids) + target_ids
    if T.has_chat_template(tok):
        tstr = T.apply_model_chat_template(tok, [{'role': 'user', 'content': content}], model_name=model_name, tokenize=False, add_generation_prompt=True)
        enc = tok(tstr, add_special_tokens=False, return_offsets_mapping=True)
        cstrip = content.strip()
        lead = len(content) - len(content.lstrip())
        pos = tstr.find(cstrip)
        coff = pos - lead if pos >= 0 else -1
    else:
        tstr = content
        enc = tok(tstr, add_special_tokens=True, return_offsets_mapping=True)
        coff = 0
    if coff < 0 or list(enc['input_ids']) != list(prompt_ids):
        return {'skip': 'offset_mismatch', 'instance_id': iid}
    (col_of_tok, n_cols) = token_columns(content, coff, enc['offset_mapping'], len(prompt_ids))
    if col_of_tok is None:
        return {'skip': n_cols, 'instance_id': iid}
    mand = mandatory_columns(task, content)
    if mand is None:
        return {'skip': 'no_marker', 'instance_id': iid}
    col_full = np.full(len(full_ids), -1, dtype=np.int64)
    col_full[:len(col_of_tok)] = col_of_tok
    return {'instance_id': iid, 'task': task, 'input_ids': full_ids, 'labels': labels, 'col_of_tok': col_full, 'n_cols': int(n_cols), 'mandatory': set((int(c) for c in mand))}

def _find_v_proj_modules(model):
    mods = [(n, m) for (n, m) in model.named_modules() if n.endswith('self_attn.v_proj')]
    if not mods:
        mods = [(n, m) for (n, m) in model.named_modules() if n.endswith('.v_proj')]
    return mods
_ALPHAS = (0.0, 0.5, 1.0)

@torch.enable_grad()
def attribution_sweep(model, tok, items, batch_size, device):
    was_training = model.training
    prev_cache = model.config.use_cache
    model.eval()
    model.config.use_cache = False
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    gate_holder = {'gate': None}

    def _hook(module, inputs, output):
        g = gate_holder['gate']
        if g is None:
            return output
        return output * g.to(output.dtype)
    v_mods = _find_v_proj_modules(model)
    if not v_mods:
        raise RuntimeError('attribution_sweep: no self_attn.v_proj modules found to gate')
    handles = [m.register_forward_hook(_hook) for (_, m) in v_mods]
    results = {}
    model.zero_grad(set_to_none=True)
    try:
        for start in tqdm(range(0, len(items), batch_size), desc='grad_col_mask attribution', unit='batch'):
            batch = items[start:start + batch_size]
            Lmax = max((len(it['input_ids']) for it in batch))
            B = len(batch)
            C = max((it['n_cols'] for it in batch))
            input_ids = torch.full((B, Lmax), pad_id, dtype=torch.long)
            attn = torch.zeros((B, Lmax), dtype=torch.long)
            labels = torch.full((B, Lmax), -100, dtype=torch.long)
            col_index = torch.zeros((B, Lmax), dtype=torch.long)
            is_opt = torch.zeros((B, Lmax), dtype=torch.bool)
            for (i, it) in enumerate(batch):
                L = len(it['input_ids'])
                input_ids[i, :L] = torch.tensor(it['input_ids'], dtype=torch.long)
                attn[i, :L] = 1
                labels[i, :L] = torch.tensor(it['labels'], dtype=torch.long)
                col = torch.from_numpy(it['col_of_tok'][:L].astype(np.int64))
                optmask = col >= 0
                mand = it['mandatory']
                if mand:
                    in_mand = torch.isin(col.clamp(min=0), torch.tensor(sorted(mand), dtype=torch.long))
                    optmask = optmask & ~in_mand
                col_index[i, :L] = col.clamp(min=0)
                is_opt[i, :L] = optmask
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            labels = labels.to(device)
            col_index = col_index.to(device)
            is_opt = is_opt.to(device)
            zgrads = {}
            inst_loss = {}
            for alpha in _ALPHAS:
                Z = torch.full((B, C), float(alpha), dtype=torch.float32, device=device, requires_grad=True)
                Zg = torch.gather(Z, 1, col_index)
                gate = torch.where(is_opt, Zg, torch.ones_like(Zg)).unsqueeze(-1)
                gate_holder['gate'] = gate
                logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
                shift_logits = logits[:, :-1, :].float()
                shift_labels = labels[:, 1:]
                tok_loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=-100, reduction='none').view(B, -1)
                per_inst = tok_loss.sum(dim=1)
                loss = per_inst.sum()
                loss.backward()
                zgrads[alpha] = Z.grad.detach().float().cpu().numpy()
                inst_loss[alpha] = per_inst.detach().float().cpu().numpy()
                gate_holder['gate'] = None
                model.zero_grad(set_to_none=True)
                del Z, Zg, gate, logits, shift_logits, tok_loss, per_inst, loss
            for (i, it) in enumerate(batch):
                nc = it['n_cols']
                mand = it['mandatory']
                col = it['col_of_tok']
                valid = col >= 0
                counts = np.bincount(col[valid], minlength=nc).astype(np.int64) if valid.any() else np.zeros(nc, dtype=np.int64)
                scores = np.zeros(nc, dtype=np.float64)
                gnorms = np.zeros(nc, dtype=np.float64)
                g0 = np.zeros(nc, dtype=np.float64)
                g05 = np.zeros(nc, dtype=np.float64)
                g1 = np.zeros(nc, dtype=np.float64)
                for c in range(nc):
                    if c in mand:
                        continue
                    a0 = float(zgrads[0.0][i, c])
                    a05 = float(zgrads[0.5][i, c])
                    a1 = float(zgrads[1.0][i, c])
                    integral = (a0 + 4.0 * a05 + a1) / 6.0
                    scores[c] = -integral
                    gnorms[c] = abs(scores[c])
                    (g0[c], g05[c], g1[c]) = (a0, a05, a1)
                results[it['instance_id']] = {'scores': scores, 'grad_norms': gnorms, 'tok_counts': counts, 'n_cols': nc, 'grad_alpha_0': g0, 'grad_alpha_0_5': g05, 'grad_alpha_1': g1, 'loss_alpha_0': float(inst_loss[0.0][i]), 'loss_alpha_1': float(inst_loss[1.0][i])}
            model.zero_grad(set_to_none=True)
    finally:
        for hd in handles:
            hd.remove()
        gate_holder['gate'] = None
    model.config.use_cache = prev_cache
    if was_training:
        model.train()
    model.zero_grad(set_to_none=True)
    return results

def build_masks(items, attrib, method, seed):
    masks = {}
    diag = {}
    for it in items:
        if 'skip' in it:
            diag[it['instance_id']] = {'skipped': it['skip']}
            continue
        iid = it['instance_id']
        res = attrib.get(iid)
        if res is None:
            diag[iid] = {'skipped': 'no_attribution'}
            continue
        nc = res['n_cols']
        mand = it['mandatory']
        scores = res['scores']
        optional = [c for c in range(nc) if c not in mand]
        grad_keep = [c for c in optional if scores[c] > 0.0]
        k = len(grad_keep)
        if method == 'gradient':
            sel_optional = grad_keep
        elif method == 'random':
            rng = random.Random((seed, str(iid)))
            sel_optional = sorted(rng.sample(optional, k)) if k <= len(optional) else list(optional)
        else:
            raise ValueError(f'unknown col_mask method: {method}')
        retained = sorted(set(mand) | set(sel_optional))
        retained_set = set(retained)
        n_opt = len(optional)
        n_pos = int(sum((1 for c in optional if scores[c] > 0.0)))
        n_neg = int(sum((1 for c in optional if scores[c] < 0.0)))
        g0 = res.get('grad_alpha_0')
        g05 = res.get('grad_alpha_0_5')
        g1 = res.get('grad_alpha_1')
        path = {}
        n_signflip = 0
        sum_integrated = 0.0
        for c in optional:
            local_score = -float(g1[c]) if g1 is not None else 0.0
            integ = float(scores[c])
            sum_integrated += integ
            flip = (integ > 0.0) != (local_score > 0.0)
            if flip:
                n_signflip += 1
            path[int(c)] = {'local_grad_alpha_0': round(float(g0[c]), 6) if g0 is not None else None, 'local_grad_alpha_0_5': round(float(g05[c]), 6) if g05 is not None else None, 'local_grad_alpha_1': round(float(g1[c]), 6) if g1 is not None else None, 'integrated_score': round(integ, 6), 'local_score_alpha_1': round(local_score, 6), 'signflip_vs_local': bool(flip), 'retained': bool(c in retained_set)}
        L0 = res.get('loss_alpha_0')
        L1 = res.get('loss_alpha_1')
        completeness = None
        if L0 is not None and L1 is not None:
            completeness = {'loss_alpha_0': round(float(L0), 6), 'loss_alpha_1': round(float(L1), 6), 'sum_integrated_score': round(sum_integrated, 6), 'completeness_error': round(sum_integrated - (float(L0) - float(L1)), 6)}
        diag[iid] = {'n_cols': nc, 'mandatory': sorted(mand), 'retained': retained, 'n_optional': n_opt, 'n_optional_retained': len(sel_optional), 'n_total_retained': len(retained), 'n_optional_positive': n_pos, 'n_optional_negative': n_neg, 'n_signflip_vs_local': n_signflip, 'frac_optional_retained': len(sel_optional) / n_opt if n_opt else 1.0, 'scores': {int(c): round(float(scores[c]), 6) for c in range(nc)}, 'grad_norms': {int(c): round(float(res['grad_norms'][c]), 6) for c in range(nc)}, 'tok_counts': {int(c): int(res['tok_counts'][c]) for c in range(nc)}, 'path': path, 'completeness': completeness}
        if len(retained) < nc:
            masks[iid] = retained
    return (masks, diag)

def summarize(diag):
    attributed = [d for d in diag.values() if 'skipped' not in d]
    skipped = Counter((d['skipped'] for d in diag.values() if 'skipped' in d))
    n = len(attributed)
    if n == 0:
        return {'n_attributed': 0, 'n_skipped': int(sum(skipped.values())), 'skip_reasons': dict(skipped)}
    fr = np.array([d['frac_optional_retained'] for d in attributed], dtype=np.float64)
    retained_counts = np.array([d['n_total_retained'] for d in attributed], dtype=np.int64)
    opt = np.array([d['n_optional'] for d in attributed], dtype=np.int64)
    opt_ret = np.array([d['n_optional_retained'] for d in attributed], dtype=np.int64)
    pos = int(sum((d['n_optional_positive'] for d in attributed)))
    neg = int(sum((d['n_optional_negative'] for d in attributed)))
    tot_opt = int(opt.sum())
    hist = Counter((int(c) for c in retained_counts))
    signflips = int(sum((d.get('n_signflip_vs_local', 0) for d in attributed)))
    cerr = np.array([d['completeness']['completeness_error'] for d in attributed if d.get('completeness') is not None], dtype=np.float64)
    out = {'n_attributed': n, 'n_skipped': int(sum(skipped.values())), 'skip_reasons': dict(skipped), 'mean_frac_optional_retained': float(fr.mean()), 'median_frac_optional_retained': float(np.median(fr)), 'pct_instances_zero_optional_retained': float(100.0 * (opt_ret == 0).mean()), 'pct_instances_all_optional_retained': float(100.0 * np.mean([o == 0 or r == o for (o, r) in zip(opt, opt_ret)])), 'mean_total_retained_columns': float(retained_counts.mean()), 'retained_column_count_histogram': {str(k): v for (k, v) in sorted(hist.items())}, 'total_optional_columns': tot_opt, 'frac_optional_positive': pos / tot_opt if tot_opt else 0.0, 'frac_optional_negative': neg / tot_opt if tot_opt else 0.0, 'total_signflips_vs_local': signflips, 'frac_optional_signflip_vs_local': signflips / tot_opt if tot_opt else 0.0}
    if cerr.size:
        out['mean_completeness_error'] = float(cerr.mean())
        out['mean_abs_completeness_error'] = float(np.abs(cerr).mean())
        out['median_abs_completeness_error'] = float(np.median(np.abs(cerr)))
    return out
