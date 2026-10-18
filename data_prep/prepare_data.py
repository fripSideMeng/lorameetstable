import argparse, json, os, random, sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
ROOT = Path(os.environ.get('LORA_GEN_ROOT') or '.')
SOTAB = ROOT / 'SOTAB'
TGPT = ROOT / 'Table-GPT/train_large/train_large_HeaderValueMatching.jsonl'
MMTU_PARQUET = ROOT / 'MMTU/data/train-00000-of-00001.parquet'
TPL_DIR = ROOT / 'templates'
OUT = ROOT / 'data_prepared'

def load_tpl(slug):
    return (TPL_DIR / f'{slug}.txt').read_text()

def cell_str(v, max_cell):
    if v is None:
        return ''
    if isinstance(v, list):
        v = ', '.join((str(x) for x in v[:3]))
    s = str(v).replace('\n', ' ').replace('|', '/').strip()
    return s[:max_cell]

def render_md(rows, col_names, max_rows, max_cell):
    out = ['| ' + ' | '.join(col_names) + ' |', '| ' + ' | '.join(['---'] * len(col_names)) + ' |']
    for r in rows[:max_rows]:
        out.append('| ' + ' | '.join((cell_str(c, max_cell) for c in r)) + ' |')
    return '\n'.join(out)

class Renderer:

    def __init__(self, tok, input_cap, max_cell, start_rows, min_rows):
        (self.tok, self.cap) = (tok, input_cap)
        (self.max_cell, self.start_rows, self.min_rows) = (max_cell, start_rows, min_rows)

    def ntok(self, text):
        return len(self.tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True))

    def fit(self, build_with_table, reserve=0):
        budget = self.cap - reserve
        for r in range(self.start_rows, 0, -1):
            prompt = build_with_table(r)
            if self.ntok(prompt) <= budget:
                return (prompt, r, r < self.min_rows)
        return (build_with_table(1), 1, True)

def load_sotab_table(path, max_cell):
    df = pd.read_json(path, lines=True, compression='gzip')
    df.columns = [int(c) for c in df.columns]
    cols = sorted(df.columns)
    return (cols, df[cols].values.tolist())

def cand_block(labels, header):
    return header + '\n'.join((f'- {l}' for l in labels))

def build_sotab(task, gt_path, table_dir, rend, max_cell, target_n, seed, labels=None, cand_header='', tables=None):
    slug = task
    tpl = load_tpl(slug)
    gt = pd.read_csv(gt_path)
    label_col = 'label'
    if labels is None:
        labels = sorted(gt[label_col].astype(str).unique())
    cb = cand_block(labels, cand_header)
    groups = list(gt.groupby('table_name'))
    if tables is not None:
        keep = set(tables)
        groups = [(t, g) for (t, g) in groups if t in keep]
    rng = random.Random(seed)
    rng.shuffle(groups)
    (rows, dropped, trunc) = ([], 0, 0)
    pbar = tqdm(groups, desc=f'{slug}:{gt_path.name}', unit='tbl')
    for (tname, grp) in pbar:
        if target_n is not None and len(rows) >= target_n:
            break
        path = table_dir / tname
        if not path.exists():
            dropped += 1
            continue
        try:
            (cols, table_rows) = load_sotab_table(path, max_cell)
        except Exception as e:
            dropped += 1
            if dropped <= 3:
                print(f'  WARN {tname}: {e}', file=sys.stderr)
            continue
        col_names = [f'col{c}' for c in cols]
        for (_, r) in grp.iterrows():
            gold = str(r[label_col])
            if task == 'cta':
                col = int(r['column_index'])
                target = json.dumps({'column type': gold}, ensure_ascii=False)

                def build(mr, col=col):
                    return tpl.replace('<<CANDIDATES>>', cb).replace('<<TABLE>>', render_md(table_rows, col_names, mr, max_cell)).replace('<<COL>>', str(col))
            else:
                (head, tail) = (int(r['main_column_index']), int(r['column_index']))
                target = json.dumps({'property': gold}, ensure_ascii=False)

                def build(mr, head=head, tail=tail):
                    return tpl.replace('<<CANDIDATES>>', cb).replace('<<TABLE>>', render_md(table_rows, col_names, mr, max_cell)).replace('<<HEAD>>', str(head)).replace('<<TAIL>>', str(tail))
            reserve = len(rend.tok(target, add_special_tokens=False)['input_ids']) + 1 + 4
            (prompt, used, over) = rend.fit(build, reserve=reserve)
            if over:
                trunc += 1
            rows.append({'task': slug, 'dataset': 'SOTAB', 'input': prompt, 'target': target, 'accept': [gold], 'meta': {'table': tname, 'rows_used': used, 'col': int(r['column_index'])}})
        pbar.set_postfix(n=len(rows))
    return (rows, dropped, trunc, labels)

def extract_tgpt_hvm(prompt):
    td = prompt.rfind('Table Data:')
    if td < 0:
        return None
    ch = prompt.find('Candidate column headers:', td)
    if ch < 0:
        return None
    table_block = prompt[td + len('Table Data:'):ch]
    rest = prompt[ch + len('Candidate column headers:'):]
    for stop in ['Return the final result', '## Output', 'Output:', '\nA:']:
        k = rest.find(stop)
        if k >= 0:
            rest = rest[:k]
    candidates = []
    for ln in rest.strip().splitlines():
        c = ln.strip().lstrip('-').strip().strip('*').strip()
        if c and set(c) - set('*-: '):
            candidates.append(c)
    rows = []
    for ln in table_block.strip().splitlines():
        ln = ln.strip()
        if not ln.startswith('|'):
            continue
        cells = [c.strip() for c in ln.strip('|').split('|')]
        if set(''.join(cells)) <= set('-: '):
            continue
        if all((c == '' for c in cells)):
            continue
        rows.append(cells)
    if not rows or not candidates:
        return None
    return (rows, candidates)

def build_hvm_train(rend, max_cell):
    tpl = load_tpl('hvm')
    (rows, fail) = ([], 0)
    with open(TGPT) as f:
        lines = f.readlines()
    for line in tqdm(lines, desc='hvm:Table-GPT', unit='ex'):
        d = json.loads(line)
        ext = extract_tgpt_hvm(d['prompt'])
        try:
            gold = json.loads(d['completion'])['column_headers']
        except Exception:
            gold = None
        if ext is None or not isinstance(gold, list):
            fail += 1
            continue
        (table_rows, candidates) = ext
        ncol = max((len(r) for r in table_rows))
        if not len(candidates) == ncol == len(gold):
            fail += 1
            continue
        norm_rows = [r + [''] * (ncol - len(r)) for r in table_rows]
        col_names = [f'Col_{i}' for i in range(ncol)]
        cand_text = '\n'.join(candidates)
        target = json.dumps({'column_headers': gold}, ensure_ascii=False)

        def build(mr):
            return tpl.replace('<<TABLE>>', render_md(norm_rows, col_names, mr, max_cell)).replace('<<CANDIDATES>>', cand_text)
        reserve = len(rend.tok(target, add_special_tokens=False)['input_ids']) + 1 + 4
        (prompt, used, over) = rend.fit(build, reserve=reserve)
        rows.append({'task': 'hvm', 'dataset': 'TableGPT-train', 'input': prompt, 'target': target, 'accept': [], 'meta': {'rows_used': used, 'truncated_over': over, 'src': d.get('metadata', '')}})
    return (rows, fail)

def build_hvm_test():
    df = pd.read_parquet(MMTU_PARQUET)
    sub = df[df.task == 'header-value-matching'].reset_index(drop=True)
    rows = []
    for (_, r) in sub.iterrows():
        gold = json.loads(r['metadata'])['label']
        rows.append({'task': 'hvm', 'dataset': 'MMTU', 'input': r['prompt'], 'target': json.dumps({'column_headers': gold}, ensure_ascii=False), 'accept': [], 'meta': {}})
    return rows

def write_split(slug, name, rows, seed):
    rng = random.Random(seed)
    rng.shuffle(rows)
    d = OUT / slug
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f'{name}.jsonl', 'w') as f:
        for (i, r) in enumerate(rows):
            r = dict(r, id=f'{slug}-{name}-{i}')
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return len(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=['cta', 'cpa', 'hvm'], choices=['cta', 'cpa', 'hvm'])
    ap.add_argument('--model', default=str(ROOT / 'Qwen3-4B-Instruct-2507'))
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--input_cap', type=int, default=2048)
    ap.add_argument('--max_cell', type=int, default=60)
    ap.add_argument('--start_rows', type=int, default=10)
    ap.add_argument('--min_rows', type=int, default=3)
    ap.add_argument('--train_size', type=int, default=8000)
    ap.add_argument('--val_size', type=int, default=600)
    ap.add_argument('--test_size', type=int, default=None)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    rend = Renderer(tok, args.input_cap, args.max_cell, args.start_rows, args.min_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {'seed': args.seed, 'input_cap': args.input_cap, 'max_cell': args.max_cell, 'start_rows': args.start_rows, 'min_rows': args.min_rows, 'train_size': args.train_size, 'val_size': args.val_size, 'test_size': args.test_size, 'model': args.model, 'tasks': {}}
    sotab_cfg = {'cta': dict(dir='CTA', gt='CTA_{}_gt.csv', cand="The JSON key must be exactly 'column type'. Do not use the target column ID as a JSON key.\nChoose the type from this candidate list:\n"), 'cpa': dict(dir='CPA', gt='CPA_{}_gt.csv', cand="The JSON key must be exactly 'property'.\nChoose the property from this candidate list:\n")}
    sotab_split_files = {'train_pool': ('train', 'training'), 'val': ('val', 'validation'), 'test': ('test', 'test')}
    for task in args.tasks:
        if task in ('cta', 'cpa'):
            cfg = sotab_cfg[task]
            base = SOTAB / cfg['dir']
            labels = None
            tinfo = {}
            for (split, (folder, gtkey)) in sotab_split_files.items():
                gt_path = base / folder / cfg['gt'].format(gtkey)
                table_dir = {'train': base / 'train' / 'Train', 'val': base / 'val' / 'Validation', 'test': base / 'test' / 'Test'}[folder]
                tgt_n = {'train_pool': args.train_size, 'val': args.val_size, 'test': args.test_size}[split]
                (rows, dropped, trunc, labels) = build_sotab(task, gt_path, table_dir, rend, args.max_cell, tgt_n, args.seed, labels=labels, cand_header=cfg['cand'])
                n = write_split(task, split, rows, args.seed)
                tinfo[split] = {'n': n, 'dropped_missing_tables': dropped, 'truncated_over_cap': trunc}
                print(f'{task} {split}: n={n} dropped={dropped} trunc_over_cap={trunc}')
            tinfo['n_labels'] = len(labels)
            stats['tasks'][task] = tinfo
        elif task == 'hvm':
            (train_all, fail) = build_hvm_train(rend, args.max_cell)
            rng = random.Random(args.seed)
            rng.shuffle(train_all)
            val = train_all[:args.val_size]
            train = train_all[args.val_size:]
            if args.train_size is not None:
                train = train[:args.train_size]
            test = build_hvm_test()
            n_tr = write_split('hvm', 'train_pool', train, args.seed)
            n_va = write_split('hvm', 'val', val, args.seed)
            n_te = write_split('hvm', 'test', test, args.seed)
            print(f'hvm train_pool={n_tr} val={n_va} test(MMTU)={n_te} | Table-GPT parse_fail={fail}/{len(train_all) + fail} usable={len(train_all)}')
            stats['tasks']['hvm'] = {'train_pool': {'n': n_tr}, 'val': {'n': n_va}, 'test': {'n': n_te, 'source': 'MMTU'}, 'tablegpt_parse_fail': fail}
    with open(OUT / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print('\nwrote', OUT / 'stats.json')
if __name__ == '__main__':
    main()
