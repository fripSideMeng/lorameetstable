import collections
import hashlib
import json
import random
import samerow as SR
import table_reorder as TRO
SLUG = 'structrel'
LABEL_ROW = 'same_row'
LABEL_COL = 'same_column'
LABEL_NEITHER = 'neither'
LABELS = (LABEL_ROW, LABEL_COL, LABEL_NEITHER)
INSTRUCTION = 'You are given a table. Below the table are two cell values taken from it; each of the two values occurs exactly once in the table.\n\nYour task is to decide how the two values are positioned in the table: whether they come from the SAME row, from the SAME column, or from neither.\n\nNo explanation and return the final answer as a JSON object, {"label": "<LABEL>"}, where "<LABEL>" is "' + LABEL_ROW + '" if the two values are in the same row, "' + LABEL_COL + '" if the two values are in the same column, or "' + LABEL_NEITHER + '" if they share neither a row nor a column.'
_TABLE_FIRST = False

def set_table_first(on):
    global _TABLE_FIRST
    _TABLE_FIRST = bool(on)
    return _TABLE_FIRST

def table_first_enabled():
    return _TABLE_FIRST

def build_prompt(block_text, value_a, value_b):
    text = f'{INSTRUCTION}\n\nTable:\n{block_text}\n\nValue A: {value_a}\nValue B: {value_b}\n'
    if _TABLE_FIRST:
        text = TRO.move_tables_first(text)[0]
    return text

def target_for(label):
    return json.dumps({'label': label})

def max_target_tokens(tok):
    n = max((len(tok(target_for(lab), add_special_tokens=False)['input_ids']) for lab in LABELS))
    return n + (1 if tok.eos_token_id is not None else 0)

def pick_triple(block_text, header, rows):
    ncols = len(header)
    if ncols < 2 or len(rows) < 2:
        return (None, 'too_small')
    counts = collections.Counter()
    for c in header:
        counts[SR.norm_value(c)] += 1
    for row in rows:
        for c in row:
            counts[SR.norm_value(c)] += 1

    def eligible(r, c):
        n = SR.norm_value(rows[r][c])
        return bool(n) and counts[n] == 1
    good = {c: [r for r in range(len(rows)) if eligible(r, c)] for c in range(ncols)}
    cands = []
    for ca in range(ncols):
        if len(good[ca]) < 2:
            continue
        gset_a = set(good[ca])
        for cb in range(ncols):
            if cb == ca or len(good[cb]) < 2:
                continue
            inter = [r for r in good[cb] if r in gset_a]
            if inter:
                cands.append((ca, cb, tuple(inter)))
    if not cands:
        return (None, 'no_unambiguous_triple')
    seed = int.from_bytes(hashlib.blake2b(block_text.encode('utf-8', 'replace'), digest_size=8).digest(), 'big')
    rng = random.Random(seed)
    (ca, cb, pos_rows) = cands[rng.randrange(len(cands))]
    r_pos = pos_rows[rng.randrange(len(pos_rows))]
    a_rest = [r for r in good[ca] if r != r_pos]
    b_rest = [r for r in good[cb] if r != r_pos]
    r2 = a_rest[rng.randrange(len(a_rest))]
    r_neg = b_rest[rng.randrange(len(b_rest))]
    return ((ca, cb, r_pos, r2, r_neg), None)

def triple_values(rows, got):
    (ca, cb, r_pos, r2, r_neg) = got
    return (rows[r_pos][ca], rows[r_pos][cb], rows[r2][ca], rows[r_neg][cb])

def _fits(header_line, sep_line, kept_lines, rows, k, encode, max_len, reserve):
    (va, vb) = SR._worst_case_values(rows, k)
    block = '\n'.join([header_line, sep_line] + kept_lines[:k])
    return len(encode(build_prompt(block, va, vb))) + reserve <= max_len

def fit_rows(header_line, sep_line, kept_lines, rows, encode, max_len, reserve):
    n = len(rows)
    if n == 0 or _fits(header_line, sep_line, kept_lines, rows, n, encode, max_len, reserve):
        return n
    (lo, hi) = (0, n)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _fits(header_line, sep_line, kept_lines, rows, mid, encode, max_len, reserve):
            lo = mid
        else:
            hi = mid
    return lo

def synthesize(src_row, max_rows, encode=None, max_len=None, reserve=0):
    tables = SR.find_tables(src_row['input'])
    if not tables:
        return (None, 'no_table')
    parsed = [SR.parse_table(h, s, d, max_rows) for (h, s, d) in tables]
    parsed.sort(key=lambda t: -len(t[2]))
    (block_text, header, rows, kept_lines) = parsed[0]
    if len(header) < 2:
        return (None, 'single_column')
    if len(rows) < 2:
        return (None, 'too_few_rows')
    budgeted = encode is not None and max_len is not None
    (header_line, sep_line) = block_text.split('\n')[:2]
    k = len(rows)
    if budgeted:
        k = fit_rows(header_line, sep_line, kept_lines, rows, encode, max_len, reserve)
        if k < 2:
            return (None, 'too_long')
    while True:
        if k < len(rows):
            block_text = '\n'.join([header_line, sep_line] + kept_lines[:k])
            (rows, kept_lines) = (rows[:k], kept_lines[:k])
        (got, reason) = pick_triple(block_text, header, rows)
        if got is None:
            return (None, reason)
        (va, b_row, b_col, b_neg) = triple_values(rows, got)
        if not budgeted or all((len(encode(build_prompt(block_text, va, vb))) + reserve <= max_len for vb in (b_row, b_col, b_neg))):
            break
        k -= 1
        if k < 2:
            return (None, 'too_long')
    (ca, cb, r_pos, r2, r_neg) = got
    th = SR.table_hash(block_text)
    base_meta = {'pair_group': th, 'table_sha1': th, 'src_task': src_row.get('task'), 'src_id': src_row.get('id'), 'src_dataset': src_row.get('dataset'), 'col_a': ca, 'col_b': cb, 'row_anchor': r_pos, 'row_same_col': r2, 'row_neither': r_neg, 'header_a': header[ca], 'header_b': header[cb], 'value_a': va, 'n_rows': len(rows), 'n_cols': len(header)}
    out = []
    for (label, vb, rb, cb_) in ((LABEL_ROW, b_row, r_pos, cb), (LABEL_COL, b_col, r2, ca), (LABEL_NEITHER, b_neg, r_neg, cb)):
        out.append({'id': f'{SLUG}-{th[:12]}-{label}', 'task': SLUG, 'dataset': f"{SLUG}/{src_row.get('task', '?')}", 'input': build_prompt(block_text, va, vb), 'target': target_for(label), 'accept': None, 'meta': dict(base_meta, label=label, value_b=vb, row_b=rb, col_b=cb_)})
    return (out, None)

def synthesize_pool(src_rows, max_rows=20, encode=None, max_len=None, reserve=0, seen=None, limit=None, tag='', progress=None):
    seen = set() if seen is None else seen
    stats = collections.Counter()
    out = []
    it = src_rows if progress is None else progress(src_rows)
    for r in it:
        if limit is not None and stats['tables_used'] >= limit:
            break
        stats['src_rows'] += 1
        (triple, reason) = synthesize(r, max_rows, encode=encode, max_len=max_len, reserve=reserve)
        if triple is None:
            stats[f'skip_{reason}'] += 1
            continue
        th = triple[0]['meta']['table_sha1']
        if th in seen:
            stats['skip_duplicate_table'] += 1
            continue
        seen.add(th)
        out.extend(triple)
        stats['tables_used'] += 1
    stats['examples'] = len(out)
    print(f"[{SLUG}{tag}] {stats['src_rows']} source rows -> {stats['tables_used']} usable tables -> {len(out)} examples (skipped: " + ', '.join((f'{k[5:]}={v}' for (k, v) in sorted(stats.items()) if k.startswith('skip_'))) + ')')
    return (out, dict(stats))

def verify_rows(rows, strict=True):
    (bad, checked) = ([], 0)
    counts = collections.Counter()
    for r in rows:
        text = r['input']
        tables = SR.find_tables(text)
        if len(tables) != 1:
            bad.append((r['id'], f'{len(tables)} tables in the prompt'))
            continue
        (h, sep, d) = tables[0]
        (_, header, grid, _) = SR.parse_table(h, sep, d, 10 ** 9)
        va = text.split('\nValue A: ', 1)[1].split('\n', 1)[0]
        vb = text.split('\nValue B: ', 1)[1].split('\n', 1)[0]
        (pos, ok) = ({}, True)
        for (which, v) in (('a', va), ('b', vb)):
            hits = [(ri, ci) for (ri, row) in enumerate(grid) for (ci, c) in enumerate(row) if SR.norm_value(c) == SR.norm_value(v)]
            if len(hits) != 1:
                bad.append((r['id'], f'value {which} occurs {len(hits)} times in the rendered table'))
                ok = False
                break
            pos[which] = hits[0]
        if not ok:
            continue
        same_row = pos['a'][0] == pos['b'][0]
        same_col = pos['a'][1] == pos['b'][1]
        if same_row and same_col:
            bad.append((r['id'], 'the two values are the same cell'))
            continue
        want = LABEL_ROW if same_row else LABEL_COL if same_col else LABEL_NEITHER
        gold = json.loads(r['target'])['label']
        if want != gold:
            bad.append((r['id'], f'gold={gold} recomputed={want}'))
            continue
        counts[gold] += 1
        checked += 1
    if bad and strict:
        for (i, (rid, why)) in enumerate(bad[:10]):
            print(f'[{SLUG}:VERIFY] {rid}: {why}')
        raise SystemExit(f'[{SLUG}] {len(bad)}/{len(rows)} examples failed gold re-derivation from the rendered table -- refusing to train on them')
    return {'checked': checked, 'failed': len(bad), 'label_counts': dict(counts)}
