_EMPTY_CELL_TOKENS = {'undefined', 'null', 'n/a', 'none', 'nan', 'na', '-', '--', '?'}

def cell_is_blank(v):
    s = str(v).strip()
    return not s or s.lower() in _EMPTY_CELL_TOKENS

def empty_context_positions(rows, ctx_positions, row_idx=None):
    idx = range(len(rows)) if row_idx is None else list(row_idx)
    if not idx:
        return set()
    return {c for c in ctx_positions if all((cell_is_blank(rows[i][c]) for i in idx))}

def select_context_columns(task, target_positions, ctx_positions, ncols, max_context_cols):
    ctx_set = set(ctx_positions)
    K = max_context_cols
    if task == 'cpa' and len(target_positions) >= 2:
        (lo, hi) = (min(target_positions), max(target_positions))
        order = [j for j in range(lo + 1, hi) if j in ctx_set]
        (left_start, right_start) = (lo, hi)
    else:
        t = target_positions[0]
        order = []
        left_start = right_start = t
    for d in range(1, ncols + 1):
        for p in (left_start - d, right_start + d):
            if p in ctx_set:
                order.append(p)
    return sorted(order[:K])
_COL_DOC_ROW_CAP = 400
_col_docs_cache = {}

def _col_docs(rows, ncols, seed, table_id, cache_key=None):
    if cache_key is not None and cache_key in _col_docs_cache:
        return _col_docs_cache[cache_key]
    import random
    n = len(rows)
    idx = range(n)
    if n > _COL_DOC_ROW_CAP:
        idx = sorted(random.Random(_table_row_seed(seed, f'{table_id}#coldocs')).sample(range(n), _COL_DOC_ROW_CAP))
    docs = [' '.join((_clean_cell(rows[i][c]) for i in idx)) for c in range(ncols)]
    if cache_key is not None:
        _col_docs_cache[cache_key] = docs
    return docs

def select_context_columns_random(ctx_positions, max_context_cols, seed, table_id):
    import random
    ctx = list(ctx_positions)
    K = max_context_cols
    if K >= len(ctx):
        return sorted(ctx)
    rng = random.Random(_table_row_seed(seed, f'{table_id}#cols'))
    return sorted(rng.sample(ctx, K))

def select_context_columns_coverage(rows, ctx_positions, max_context_cols, seed, table_id, ncols=None, cache_key=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    ctx = sorted(ctx_positions)
    K = max_context_cols
    if K >= len(ctx):
        return ctx
    docs_all = _col_docs(rows, len(rows[0]) if ncols is None else ncols, seed, table_id, cache_key)
    _TL._coverage_stats['col_calls'] += 1
    try:
        X = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4)).fit_transform([docs_all[c] for c in ctx])
    except ValueError:
        _TL._coverage_stats['col_empty_vocab'] += 1
        return select_context_columns_random(ctx, K, seed, table_id)
    S = (X @ X.T).toarray()
    m = S.shape[0]
    if m > 1:
        off = S[~np.eye(m, dtype=bool)]
        if float(off.std()) < _COVERAGE_FLAT_STD:
            _TL._coverage_stats['col_flat_sim'] += 1
    cov = np.zeros(m)
    picked = []
    for step in range(min(K, m)):
        gains = S.sum(axis=0) if step == 0 else np.maximum(S - cov[:, None], 0.0).sum(axis=0)
        gains[picked] = -np.inf
        p = int(np.argmax(gains))
        picked.append(p)
        cov = np.maximum(cov, S[:, p])
    return sorted((ctx[p] for p in picked))

def column_keep_positions(task, target_positions, ncols, max_context_cols, rows=None, drop_empty=False, row_idx=None, method='nearest', seed=0, table_id='', cache_key=None):
    tset = set(target_positions)
    ctx_positions = [j for j in range(ncols) if j not in tset]
    empties = set()
    if drop_empty and rows:
        empties = empty_context_positions(rows, ctx_positions, row_idx)
    keepable = [c for c in ctx_positions if c not in empties]
    if not max_context_cols or max_context_cols <= 0:
        return sorted(set(keepable) | tset) if empties else None
    if len(keepable) <= max_context_cols:
        return sorted(set(keepable) | tset) if empties else None
    if method == 'random':
        kept_ctx = select_context_columns_random(keepable, max_context_cols, seed, table_id)
    elif method == 'coverage_col':
        kept_ctx = select_context_columns_coverage(rows, keepable, max_context_cols, seed, table_id, ncols, cache_key)
    else:
        kept_ctx = select_context_columns(task, target_positions, keepable, ncols, max_context_cols)
    return sorted(set(kept_ctx) | tset)
import numpy as np

def _table_row_seed(seed, table_id):
    import hashlib
    return int(hashlib.md5(f'{seed}|{table_id}'.encode()).hexdigest()[:8], 16)

def _tfidf_row_scores(docs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        X = TfidfVectorizer().fit_transform(docs)
    except ValueError:
        return None
    return [float(v) for v in X.sum(axis=1).A.ravel()]
_NULL_CELLS = {'', 'nan', 'none', 'null', 'n/a', 'na', 'undefined', '-'}
_COVERAGE_CAND_CAP = 1200
_COVERAGE_FLAT_STD = 0.02

def _clean_cell(v):
    if v is None:
        return ''
    s = str(v)
    return '' if s.strip().lower() in _NULL_CELLS else s

def _similarity_select_order(docs, max_rows, seed, table_id, objective, max_keep=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import random
    n = len(docs)
    cand = list(range(n))
    if n > _COVERAGE_CAND_CAP:
        crng = random.Random(_table_row_seed(seed, table_id))
        cand = sorted(crng.sample(range(n), _COVERAGE_CAND_CAP))
    _TL._coverage_stats['calls'] += 1
    try:
        X = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4)).fit_transform([docs[i] for i in cand])
    except ValueError:
        _TL._coverage_stats['empty_vocab'] += 1
        return None
    S = (X @ X.T).toarray()
    m = S.shape[0]
    if m > 1:
        off = S[~np.eye(m, dtype=bool)]
        if float(off.std()) < _COVERAGE_FLAT_STD:
            _TL._coverage_stats['flat_sim'] += 1
    k = min(max_rows, m)
    if max_keep is not None:
        k = min(k, max(0, int(max_keep)))
    if objective == 'central':
        colsum = S.sum(axis=0)
        picked = list(np.argsort(-colsum, kind='stable')[:k])
    else:
        cov = np.zeros(m)
        picked = []
        for step in range(k):
            gains = S.sum(axis=0) if step == 0 else np.maximum(S - cov[:, None], 0.0).sum(axis=0)
            gains[picked] = -np.inf
            p = int(np.argmax(gains))
            picked.append(p)
            cov = np.maximum(cov, S[:, p])
    return [cand[int(p)] for p in picked]

def _target_docs(rows, target_pos):
    return [' '.join((_clean_cell(rows[i][p]) for p in target_pos)) for i in range(len(rows))]

def _row_docs(rows):
    return [' '.join((_clean_cell(c) for c in rows[i])) for i in range(len(rows))]

def _coverage_select_order(rows, max_rows, target_pos, seed, table_id):
    return _similarity_select_order(_target_docs(rows, target_pos), max_rows, seed, table_id, 'facility')

def _coverage_row_select_order(rows, max_rows, seed, table_id):
    return _similarity_select_order(_row_docs(rows), max_rows, seed, table_id, 'facility')

def _central_row_select_order(rows, max_rows, seed, table_id):
    return _similarity_select_order(_row_docs(rows), max_rows, seed, table_id, 'central')

def _dedup_select_order(rows, max_rows, target_pos, seed, table_id):
    import random
    seen = {}
    for i in range(len(rows)):
        key = ' '.join((_clean_cell(rows[i][p]) for p in target_pos))
        if not key.strip():
            continue
        if key not in seen:
            seen[key] = i
    if not seen:
        return None
    reps = list(seen.values())
    if len(reps) > max_rows:
        reps = random.Random(_table_row_seed(seed, table_id)).sample(reps, max_rows)
    return reps

def _coverage_col_orders(rows, seed, table_id, col_positions, max_keep=None):
    n = len(rows)
    orders = {}
    for c in col_positions:
        docs = [_clean_cell(rows[i][c]) for i in range(n)]
        order = _similarity_select_order(docs, n, seed, f'{table_id}#col{c}', 'facility', max_keep=max_keep)
        if order is None:
            order = list(range(n))
        if len(order) < n:
            seen = set(order)
            order = order + [i for i in range(n) if i not in seen]
        orders[c] = order
    return orders

def build_coverage_col_block(kind, col_names, rows, df, orders, k, render_cols, reindex=False):
    import prepare_data as _PD
    n = len(rows)
    k = max(0, min(k, n))
    sub_names = [f'col{i}' for i in range(len(render_cols))] if reindex else [col_names[c] for c in render_cols]
    synth = [[rows[orders[c][r]][c] if r < len(orders[c]) else '' for c in render_cols] for r in range(k)]
    if kind == 'sotab':
        return _PD.render_md(synth, sub_names, len(synth), 60)
    import pandas as pd
    return pd.DataFrame(synth, columns=sub_names).to_markdown(index=False)

def _select_raw_rows(rows, method, max_rows, seed, table_id, target_pos=None, full_rank=False):
    import random
    n = len(rows)
    k = n if full_rank else max_rows
    if method == 'dedup':
        order = _dedup_select_order(rows, k, target_pos, seed, table_id)
        if order is None:
            order = list(range(min(k, n)))
        prio_rank = {i: r for (r, i) in enumerate(order)}
        return (sorted(order), prio_rank)
    if not full_rank and n <= max_rows:
        idx = list(range(n))
        return (idx, {i: r for (r, i) in enumerate(idx)})
    if method == 'tfidf':

        def cell(v):
            return '' if v is None else str(v)
        tset = set(target_pos or [])
        prim_docs = [' '.join((cell(r[p]) for p in target_pos)) for r in rows]
        ctx_docs = [' '.join((cell(c) for (j, c) in enumerate(r) if j not in tset)) for r in rows]
        prim = _tfidf_row_scores(prim_docs)
        ctx = _tfidf_row_scores(ctx_docs)
        if prim is None and ctx is None:
            order = list(range(k))
        else:
            tot = [(prim[i] if prim else 0.0) + (ctx[i] if ctx else 0.0) for i in range(n)]
            order = sorted(range(n), key=lambda i: (-tot[i], i))[:k]
    elif method == 'coverage':
        order = _coverage_select_order(rows, k, target_pos, seed, table_id)
        if order is None:
            order = list(range(k))
    elif method == 'coverage_row':
        order = _coverage_row_select_order(rows, k, seed, table_id)
        if order is None:
            order = list(range(k))
    elif method == 'central_row':
        order = _central_row_select_order(rows, k, seed, table_id)
        if order is None:
            order = list(range(k))
    elif method == 'head':
        order = list(range(k))
    else:
        rng = random.Random(_table_row_seed(seed, table_id))
        order = rng.sample(range(n), k)
    if full_rank and len(order) < n:
        seen = set(order)
        order = order + [i for i in range(n) if i not in seen]
    prio_rank = {i: r for (r, i) in enumerate(order)}
    return (sorted(order), prio_rank)
import train_lora as _TL
