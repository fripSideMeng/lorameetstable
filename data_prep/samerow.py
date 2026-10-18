import collections
import hashlib
import json
import random
import re
_SEP_CELL_RE = re.compile('^\\s*:?-{2,}:?\\s*$')
SLUG = 'samerow'
LABEL_SAME = 'same_row'
LABEL_DIFF = 'different_row'
SOURCE_TASKS = ('cta', 'cpa', 'hvm', 'tabfact')
INSTRUCTION = 'You are given a table. Below the table are two cell values taken from it; each of the two values occurs exactly once in the table.\n\nYour task is to decide whether the two values come from the SAME row of the table.\n\nNo explanation and return the final answer as a JSON object, {"label": "<LABEL>"}, where "<LABEL>" is either "' + LABEL_SAME + '" if the two values are in the same row, or "' + LABEL_DIFF + '" if they are in different rows.'

def norm_value(v):
    return ' '.join(str(v).split()).lower()

def md_cells(line):
    s = line.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    return s.split('|')

def _is_separator(line, ncols):
    cells = md_cells(line)
    return len(cells) == ncols and all((_SEP_CELL_RE.match(c) for c in cells))

def find_tables(text):
    lines = text.split('\n')
    tables = []
    i = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith('|'):
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].lstrip().startswith('|'):
            j += 1
        block = lines[i:j]
        i = j
        if len(block) >= 3 and _is_separator(block[1], len(md_cells(block[0]))):
            tables.append((block[0], block[1], block[2:]))
    return tables

def parse_table(header_line, sep_line, data_lines, max_rows):
    header = [c.strip() for c in md_cells(header_line)]
    (rows, kept_lines) = ([], [])
    for ln in data_lines:
        cells = md_cells(ln)
        if len(cells) != len(header):
            continue
        rows.append([c.strip() for c in cells])
        kept_lines.append(ln)
        if len(rows) >= max_rows:
            break
    block_text = '\n'.join([header_line, sep_line] + kept_lines)
    return (block_text, header, rows, kept_lines)

def build_prompt(block_text, value_a, value_b):
    return f'{INSTRUCTION}\n\nTable:\n{block_text}\n\nValue A: {value_a}\nValue B: {value_b}\n'

def _worst_case_values(rows, k):
    cells = sorted((c for r in rows[:k] for c in r), key=len, reverse=True)
    if not cells:
        return ('', '')
    if len(cells) == 1:
        return (cells[0], cells[0])
    return (cells[0], cells[1])

def _fits(header_line, sep_line, kept_lines, rows, k, encode, max_len, reserve):
    (va, vb) = _worst_case_values(rows, k)
    block = '\n'.join([header_line, sep_line] + kept_lines[:k])
    return len(encode(build_prompt(block, va, vb))) + reserve <= max_len

def _pair_fits(block_text, rows, got, encode, max_len, reserve):
    (ca, cb, r_pos, r_neg) = got
    va = rows[r_pos][ca]
    return all((len(encode(build_prompt(block_text, va, rows[r_b][cb]))) + reserve <= max_len for r_b in (r_pos, r_neg)))

def fit_rows(header_line, sep_line, kept_lines, rows, encode, max_len, reserve=0):
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

def _pick(block_text, header, rows):
    ncols = len(header)
    if ncols < 2 or len(rows) < 2:
        return None
    counts = collections.Counter()
    for c in header:
        counts[norm_value(c)] += 1
    for row in rows:
        for c in row:
            counts[norm_value(c)] += 1

    def eligible(r, c):
        n = norm_value(rows[r][c])
        return bool(n) and counts[n] == 1
    good_in_col = {c: [r for r in range(len(rows)) if eligible(r, c)] for c in range(ncols)}
    cands = []
    for ca in range(ncols):
        for cb in range(ncols):
            if ca == cb or len(good_in_col[cb]) < 2:
                continue
            pos_rows = [r for r in good_in_col[ca] if eligible(r, cb)]
            if pos_rows:
                cands.append((ca, cb, tuple(pos_rows)))
    if not cands:
        return None
    seed = int.from_bytes(hashlib.blake2b(block_text.encode('utf-8', 'replace'), digest_size=8).digest(), 'big')
    rng = random.Random(seed)
    (ca, cb, pos_rows) = cands[rng.randrange(len(cands))]
    r_pos = pos_rows[rng.randrange(len(pos_rows))]
    neg_rows = [r for r in good_in_col[cb] if r != r_pos]
    r_neg = neg_rows[rng.randrange(len(neg_rows))]
    return (ca, cb, r_pos, r_neg)

def table_hash(block_text):
    return hashlib.sha1(block_text.encode('utf-8', 'replace')).hexdigest()

def synthesize(src_row, max_rows, encode=None, max_len=None, reserve=0):
    tables = find_tables(src_row['input'])
    if not tables:
        return (None, 'no_table')
    parsed = [parse_table(h, s, d, max_rows) for (h, s, d) in tables]
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
            (block_text, rows, kept_lines) = ('\n'.join([header_line, sep_line] + kept_lines[:k]), rows[:k], kept_lines[:k])
        got = _pick(block_text, header, rows)
        if got is None:
            return (None, 'no_unambiguous_pair')
        if not budgeted or _pair_fits(block_text, rows, got, encode, max_len, reserve):
            break
        k -= 1
        if k < 2:
            return (None, 'too_long')
    (ca, cb, r_pos, r_neg) = got
    th = table_hash(block_text)
    va = rows[r_pos][ca]
    out = []
    for (label, r_b) in ((LABEL_SAME, r_pos), (LABEL_DIFF, r_neg)):
        vb = rows[r_b][cb]
        out.append({'id': f'samerow-{th[:12]}-{label}', 'task': SLUG, 'dataset': f"samerow/{src_row.get('task', '?')}", 'input': build_prompt(block_text, va, vb), 'target': json.dumps({'label': label}), 'accept': None, 'meta': {'pair_group': th, 'table_sha1': th, 'label': label, 'src_task': src_row.get('task'), 'src_id': src_row.get('id'), 'src_dataset': src_row.get('dataset'), 'col_a': ca, 'col_b': cb, 'row_a': r_pos, 'row_b': r_b, 'header_a': header[ca], 'header_b': header[cb], 'value_a': va, 'value_b': vb, 'n_rows': len(rows), 'n_cols': len(header)}})
    return (out, None)

def synthesize_pool(src_rows, max_rows=20, exclude_tables=None, tag='', encode=None, max_len=None, reserve=0):
    exclude = set(exclude_tables or ())
    stats = collections.Counter()
    seen = set()
    rows = []
    for r in src_rows:
        stats['src_rows'] += 1
        (pair, reason) = synthesize(r, max_rows, encode=encode, max_len=max_len, reserve=reserve)
        if pair is None:
            stats[f'skip_{reason}'] += 1
            continue
        th = pair[0]['meta']['table_sha1']
        if th in seen:
            stats['skip_duplicate_table'] += 1
            continue
        seen.add(th)
        if th in exclude:
            stats['skip_excluded_table'] += 1
            continue
        rows.extend(pair)
        stats['tables_used'] += 1
    stats['examples'] = len(rows)
    print(f"[samerow{tag}] {stats['src_rows']} source rows -> {stats['tables_used']} usable tables -> {len(rows)} examples (skipped: no_table={stats['skip_no_table']} single_column={stats['skip_single_column']} too_few_rows={stats['skip_too_few_rows']} ambiguous={stats['skip_no_unambiguous_pair']} too_long={stats['skip_too_long']} duplicate_table={stats['skip_duplicate_table']} excluded={stats['skip_excluded_table']})")
    return (rows, dict(stats))

def sample_balanced(pool, n, rng):
    groups = collections.defaultdict(list)
    for r in pool:
        groups[r['meta']['pair_group']].append(r)
    for g in groups.values():
        g.sort(key=lambda r: r['meta']['label'] != LABEL_SAME)
    keys = sorted(groups)
    rng.shuffle(keys)
    out = []
    for k in keys:
        if len(out) >= n:
            break
        out.extend(groups[k])
    if len(out) < n:
        raise ValueError(f'[samerow] pool has {len(pool)} examples from {len(keys)} tables < needed {n}; lower the (per-task) budget or widen the source pool.')
    out = out[:n]
    rng.shuffle(out)
    return out
