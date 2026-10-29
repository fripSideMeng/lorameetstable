import argparse
import collections
import os
import json
import random
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
ROOT = Path(os.environ.get('LORA_GEN_ROOT') or '.')
T2D = ROOT / 't2d/data'
MMTU_PARQUET = ROOT / 'MMTU/data/train-00000-of-00001.parquet'
OUT = ROOT / 'data_prepared'
TASKS = {'cta': dict(mmtu='Column-type-annotation', slug='cta_t2d', gt='CTA', key='DBpedia ontology class', marker='Input Table:'), 'cpa': dict(mmtu='Columns-property-anotation', slug='cpa_t2d', gt='CPA', key='DBpedia property', marker='Input table:')}
ROUNDS = [1, 2, 3, 4]

def mmtu_csv_set(df, mmtu_task):
    return {json.loads(m)['csv'] for m in df[df.task == mmtu_task]['metadata']}

def mmtu_label_hist(df, mmtu_task):
    return collections.Counter((json.loads(m)['label'] for m in df[df.task == mmtu_task]['metadata']))

def mmtu_preamble(df, mmtu_task, marker):
    pres = {p[:p.find(marker)] for p in df[df.task == mmtu_task]['prompt']}
    if len(pres) != 1:
        raise RuntimeError(f'expected a single {mmtu_task} preamble, got {len(pres)}')
    return pres.pop()

def read_gt(gt_prefix, i):
    rows = []
    p = T2D / f'r{i}/gt/{gt_prefix}_Round{i}_gt.csv'
    with open(p) as f:
        import csv as _csv
        for r in _csv.reader(f):
            if r:
                rows.append(r)
    return rows
_TABLE_CACHE = {}

def load_table(round_i, name):
    key = (round_i, name)
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    fp = T2D / f'r{round_i}/tables/{name}.csv'
    df = None
    if fp.exists():
        try:
            df = pd.read_csv(fp)
            df.columns = [f'col{j}' for j in range(df.shape[1])]
        except Exception:
            df = None
    _TABLE_CACHE[key] = df
    return df

class Fitter:

    def __init__(self, tok, model_name, cap):
        (self.tok, self.model_name, self.cap) = (tok, model_name, cap)
        self.is_qwen = 'qwen3' in str(model_name).lower()

    def ntok(self, text):
        kw = {}
        if self.is_qwen:
            kw['enable_thinking'] = False
        return len(self.tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True, **kw))

    def fit(self, build, n_rows, reserve):
        budget = self.cap - reserve
        full = build(n_rows)
        if self.ntok(full) <= budget:
            return (full, n_rows, True)
        (lo, hi, best) = (1, n_rows, None)
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.ntok(build(mid)) <= budget:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            return (build(1), 1, False)
        return (build(best), best, True)

def make_builder(task, preamble, df, spec_cols):
    if task == 'cta':
        col = spec_cols

        def build(k):
            md = df.head(k).to_markdown(index=False)
            return preamble + f'Input Table:\n{md}\n\nTarget entity column: col{col}\n\n'
    else:
        (a, b) = spec_cols

        def build(k):
            md = df.head(k).to_markdown(index=False)
            return preamble + f'Input table:\n{md}\n\nHead column: col{a}\n\nTail column: col{b}\n\n'
    return build

def build_pool(task, conf, exclude, label_set):
    seen = set()
    pool = []
    n_overlap = n_offlabel = n_badtable = n_badcol = 0
    for i in ROUNDS:
        for r in read_gt(conf['gt'], i):
            name = r[0]
            if task == 'cta':
                col = int(r[1])
                label = r[2]
                cols = (col,)
            else:
                col_a = int(r[1])
                col_b = int(r[2])
                label = r[3]
                cols = (col_a, col_b)
            if name in exclude:
                n_overlap += 1
                continue
            if label not in label_set:
                n_offlabel += 1
                continue
            df = load_table(i, name)
            if df is None:
                n_badtable += 1
                continue
            if any((c >= df.shape[1] for c in cols)):
                n_badcol += 1
                continue
            key = (i, name, cols, label)
            if key in seen:
                continue
            seen.add(key)
            inst = {'round': i, 'csv': name, 'label': label}
            if task == 'cta':
                inst['col'] = cols[0]
            else:
                (inst['col_a'], inst['col_b']) = cols
            pool.append(inst)
    stats = {'overlap_skipped': n_overlap, 'offlabel_skipped': n_offlabel, 'unreadable_skipped': n_badtable, 'badcol_skipped': n_badcol, 'pool_size': len(pool)}
    return (pool, stats)

def plan_counts(label_hist, pool_by_label, scale, target_n):
    mtot = sum(label_hist.values())
    targets = {}
    if scale == 'replica':
        for (l, c) in label_hist.items():
            targets[l] = min(c, len(pool_by_label.get(l, [])))
    else:
        for (l, c) in label_hist.items():
            t = round(c / mtot * target_n)
            targets[l] = min(t, len(pool_by_label.get(l, [])))
    return targets

def split_sample(task, pool, label_hist, val_ratio, scale, target_n, seed):
    rng = random.Random(seed)
    pool_by_label = collections.defaultdict(list)
    for inst in pool:
        pool_by_label[inst['label']].append(inst)
    tables = sorted({inst['csv'] for inst in pool})
    rng.shuffle(tables)
    n_val_tbl = int(round(val_ratio * len(tables)))
    val_tables = set(tables[:n_val_tbl])
    train_tables = set(tables[n_val_tbl:])
    targets = plan_counts(label_hist, pool_by_label, scale, target_n)
    (train, val) = ([], [])
    shortfalls = {}
    for (l, T) in sorted(targets.items()):
        insts = list(pool_by_label.get(l, []))
        rng.shuffle(insts)
        tr_av = [x for x in insts if x['csv'] in train_tables]
        va_av = [x for x in insts if x['csv'] in val_tables]
        v_t = int(round(val_ratio * T))
        v_use = min(v_t, len(va_av))
        t_use = min(T - v_use, len(tr_av))
        if t_use < T - v_use:
            extra = min(T - v_use - t_use, len(va_av) - v_use)
            v_use += max(0, extra)
        val.extend(va_av[:v_use])
        train.extend(tr_av[:t_use])
        got = t_use + v_use
        if got < T:
            shortfalls[l] = {'target': T, 'got': got}
    report = {'n_tables_total': len(tables), 'n_train_tables': len(train_tables), 'n_val_tables': len(val_tables), 'n_train': len(train), 'n_val': len(val), 'n_combined': len(train) + len(val), 'n_labels': len(targets), 'label_shortfalls': shortfalls, 'combined_target': sum(targets.values())}
    return (train, val, report)

def materialize(task, conf, preamble, instances, fitter, split, log_examples=0):
    (rows, trimmed, over) = ([], 0, 0)
    for (idx, inst) in enumerate(tqdm(instances, desc=f"{conf['slug']}:{split}", unit='ex')):
        df = load_table(inst['round'], inst['csv'])
        n_rows = df.shape[0]
        target = json.dumps({conf['key']: inst['label']}, ensure_ascii=False)
        reserve = len(fitter.tok(target, add_special_tokens=False)['input_ids']) + 1 + 4
        if task == 'cta':
            build = make_builder('cta', preamble, df, inst['col'])
        else:
            build = make_builder('cpa', preamble, df, (inst['col_a'], inst['col_b']))
        (prompt, used, fit_ok) = fitter.fit(build, n_rows, reserve)
        if not fit_ok:
            over += 1
        if used < n_rows:
            trimmed += 1
        meta = {'csv': inst['csv'], 'round': inst['round'], 'label': inst['label'], 'rows_total': int(n_rows), 'rows_used': int(used), 'src_dataset': 'SemTab2019'}
        if task == 'cta':
            meta['col'] = inst['col']
        else:
            meta['col_a'] = inst['col_a']
            meta['col_b'] = inst['col_b']
        rows.append({'task': task, 'dataset': 'T2D', 'input': prompt, 'target': target, 'accept': [inst['label']], 'meta': meta})
        if log_examples and idx < log_examples:
            print('=' * 90)
            print(f"[{conf['slug']}/{split} #{idx}] csv={inst['csv']} round={inst['round']} rows_used={used}/{n_rows}")
            print(prompt)
            print('--- target ---', target)
    return (rows, {'trimmed': trimmed, 'over_len': over})

def write_split(slug, split, rows):
    d = OUT / slug
    d.mkdir(parents=True, exist_ok=True)
    p = d / f'{split}.jsonl'
    with open(p, 'w') as f:
        for (i, r) in enumerate(rows):
            r = dict(r, id=f'{slug}-{split}-{i}')
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=['cta', 'cpa'], choices=['cta', 'cpa'])
    ap.add_argument('--model', default=str(ROOT / 'Qwen3-4B-Instruct-2507'))
    ap.add_argument('--input_cap', type=int, default=2048)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--scale', default='replica', choices=['replica', 'capped'])
    ap.add_argument('--target_n', type=int, default=3000)
    ap.add_argument('--leakage', default='per_task', choices=['per_task', 'union'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--preview', type=int, default=0)
    args = ap.parse_args()
    df = pd.read_parquet(MMTU_PARQUET)
    cea = 'Cell-entity-annotation'
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    fitter = Fitter(tok, args.model, args.input_cap)
    for task in args.tasks:
        conf = TASKS[task]
        label_hist = mmtu_label_hist(df, conf['mmtu'])
        label_set = set(label_hist)
        preamble = mmtu_preamble(df, conf['mmtu'], conf['marker'])
        if args.leakage == 'per_task':
            exclude = mmtu_csv_set(df, conf['mmtu'])
        else:
            exclude = mmtu_csv_set(df, TASKS['cta']['mmtu']) | mmtu_csv_set(df, TASKS['cpa']['mmtu']) | mmtu_csv_set(df, cea)
        (pool, pstats) = build_pool(task, conf, exclude, label_set)
        print(f"\n### {conf['slug']}: MMTU labels={len(label_set)} MMTU instances={sum(label_hist.values())} | {pstats}")
        (train, val, report) = split_sample(task, pool, label_hist, args.val_ratio, args.scale, args.target_n, args.seed)
        print(f"   split: combined={report['n_combined']}/{report['combined_target']} train={report['n_train']} val={report['n_val']} tables(train/val)={report['n_train_tables']}/{report['n_val_tables']} shortfall_labels={len(report['label_shortfalls'])}")
        if args.preview:
            materialize(task, conf, preamble, train[:args.preview], fitter, 'train', log_examples=args.preview)
            continue
        (train_rows, ts) = materialize(task, conf, preamble, train, fitter, 'train')
        (val_rows, vs) = materialize(task, conf, preamble, val, fitter, 'val')
        ptr = write_split(conf['slug'], 'train', train_rows)
        pva = write_split(conf['slug'], 'val', val_rows)
        train_tbls = {r['meta']['csv'] for r in train_rows}
        val_tbls = {r['meta']['csv'] for r in val_rows}
        assert not train_tbls & val_tbls, 'train/val share a table!'
        assert not train_tbls & exclude, 'train contains an excluded MMTU table!'
        assert not val_tbls & exclude, 'val contains an excluded MMTU table!'
        comb = collections.Counter((r['meta']['label'] for r in train_rows + val_rows))
        manifest = {'slug': conf['slug'], 'task': task, 'src_dataset': 'SemTab2019/T2D', 'mmtu_task': conf['mmtu'], 'template': 'MMTU-verbatim', 'json_key': conf['key'], 'leakage_mode': args.leakage, 'n_excluded_tables': len(exclude), 'scale': args.scale, 'target_n': args.target_n if args.scale == 'capped' else None, 'val_ratio': args.val_ratio, 'seed': args.seed, 'input_cap': args.input_cap, 'tokenizer': args.model, 'pool_stats': pstats, 'split_report': report, 'materialize_train': ts, 'materialize_val': vs, 'mmtu_label_counts': dict(label_hist), 'combined_label_counts': dict(comb), 'train_label_counts': dict(collections.Counter((r['meta']['label'] for r in train_rows))), 'val_label_counts': dict(collections.Counter((r['meta']['label'] for r in val_rows))), 'train_tables': sorted(train_tbls), 'val_tables': sorted(val_tbls)}
        with open(OUT / conf['slug'] / 'split_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"   wrote {ptr} ({len(train_rows)}) and {pva} ({len(val_rows)}); train_trim={ts['trimmed']}/over={ts['over_len']} val_trim={vs['trimmed']}/over={vs['over_len']}")
        print(f"   manifest -> {OUT / conf['slug'] / 'split_manifest.json'}")
if __name__ == '__main__':
    main()
