import argparse
import collections
import glob
import json
import os
import random
import shutil
import model_load as ML
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import train_lora as TL
import structrel as STR
import table_only_lora as TOL
import recv_gate as RCV
ROOT = TL.ROOT
DATA = TL.DATA
QKV = ['q_proj', 'k_proj', 'v_proj']

def load_source_rows(tasks, source):
    out = {}
    for slug in tasks:
        p = f'{DATA}/{TL.data_dir(slug, source)}/train_pool.jsonl'
        if not os.path.exists(p):
            raise SystemExit(f'no train pool for task {slug} at {p}')
        out[slug] = TL.read_jsonl(p)
        print(f'[stage1] source {slug}: {len(out[slug])} train-split prompts <- {p}')
    return out

def heldout_table_keys(source):
    if source in ('t2d', 'shared', 'targetonly'):
        (keys, parts) = (set(), [])
        for slug in ('cta_mmtu', 'cpa_mmtu', 'cea_mmtu'):
            p = f'{DATA}/{slug}/test.jsonl'
            if not os.path.exists(p):
                raise SystemExit(f'[stage1] --extra_tables needs {p} to know which T2D tables are MMTU evaluation tables, and it is not there.')
            s = {r['meta']['csv'] for r in TL.read_jsonl(p)}
            keys |= s
            parts.append(f'{slug}={len(s)}')
        return (keys, 'csv', f"MMTU test tables, union over tasks ({', '.join(parts)})")
    if source in ('sotab', 'sotab_full', 'sotab_all'):
        (keys, parts) = (set(), [])
        for slug in ('cta', 'cpa'):
            for split in ('val', 'test'):
                p = f'{DATA}/{slug}/{split}.jsonl'
                if not os.path.exists(p):
                    raise SystemExit(f'[stage1] --extra_tables needs {p} to know which SOTAB tables are held out, and it is not there.')
                s = {r['meta']['table'] for r in TL.read_jsonl(p)}
                keys |= s
                parts.append(f'{slug}/{split}={len(s)}')
        return (keys, 'table', f"SOTAB val+test tables ({', '.join(parts)})")
    raise SystemExit(f'[stage1] --extra_tables does not know which tables are held out for --cta_cpa_source {source}. Refusing: drawing extra tables from a corpus whose evaluation split is unknown is how a test table gets pretrained on. Add it to heldout_table_keys().')

def table_key(row, field):
    k = row.get('meta', {}).get(field)
    if k is None:
        raise SystemExit(f"[stage1] source row {row.get('id')} has no meta.{field}, so it cannot be checked against the held-out tables.")
    return k

def interleave(pools, seed):
    shuffled = {}
    for (slug, rows) in pools.items():
        r = list(rows)
        random.Random(seed).shuffle(r)
        shuffled[slug] = r
    (order, idx) = ([], {s: 0 for s in shuffled})
    while any((idx[s] < len(shuffled[s]) for s in shuffled)):
        for s in pools:
            if idx[s] < len(shuffled[s]):
                order.append(shuffled[s][idx[s]])
                idx[s] += 1
    return order

def untrimmed_table_hash(src_row, max_rows):
    tables = STR.SR.find_tables(src_row['input'])
    if not tables:
        return None
    parsed = [STR.SR.parse_table(h, sep, d, max_rows) for (h, sep, d) in tables]
    parsed.sort(key=lambda t: -len(t[2]))
    return STR.SR.table_hash(parsed[0][0])

def build_structural_pool(args, tok):
    encode = lambda t: TL._encode_prompt(tok, t, args.model)
    reserve = STR.max_target_tokens(tok)
    need = args.struct_tables + args.val_tables
    pinned_train = pinned_val = None
    pin_by_src = False
    if args.tables_from:
        ref = json.load(open(args.tables_from))
        if ref.get('kind') == 'structrel_pin_list':
            pin_by_src = True
            pinned_train = [tuple(x) for x in ref['train_src_ids']]
            pinned_val = [tuple(x) for x in ref['val_src_ids']]
        else:
            pinned_train = list(ref['train_table_sha1'])
            pinned_val = list(ref['val_table_sha1'])
            print('[stage1][warn] --tables_from was given a struct_manifest.json, so tables are pinned by table_sha1. Those ids depend on the reference tokenizer; across models use scripts/make_struct_pin_list.py instead.')
        need = None
        print(f"[stage1] tables pinned to {args.tables_from} ({('source rows' if pin_by_src else 'table hashes')}): {len(pinned_train)} train + {len(pinned_val)} val (reference model={ref.get('reference_model') or ref.get('model')}, source={ref.get('cta_cpa_source')})")
        for (key, mine) in (('cta_cpa_source', args.cta_cpa_source), ('struct_source', list(args.struct_source))):
            theirs = ref.get(key) if key != 'struct_source' else list(ref.get(key) or ())
            if theirs != mine:
                raise SystemExit(f'[stage1] --tables_from disagrees on {key}: reference={theirs} this run={mine}. Refusing: the pinned tables would not be reachable from this source.')
    downstream_info = extra_info = row_trunc_info = None
    if args.downstream_budget:
        sel = TL.build_train_rows(args.downstream_regime, args.downstream_budget, args.seed, source=args.cta_cpa_source, select_mode=args.downstream_select_mode, drop_empty_target=args.downstream_drop_empty_target, drop_truncated_tables=args.downstream_drop_truncated_tables)
        if args.max_table_rows > 0:
            (_rt_method, _rt_max) = TL._row_trunc_plan(args.cta_cpa_source, args.max_table_rows, args.row_trunc_method)
            TL._prefetch_raw_tables(sel, args.cta_cpa_source, 'train', args.num_proc)
            _rt_stats = TL.apply_row_truncation(sel, args.cta_cpa_source, _rt_method, _rt_max, args.seed, tok, args.max_seq_len, args.model, False, split='train', hard_cap=True)
            row_trunc_info = {'max_table_rows': args.max_table_rows, 'row_trunc_method': args.row_trunc_method, 'plan': [_rt_method, _rt_max], 'hard_cap': True, 'num_proc': args.num_proc, 'stats': _rt_stats}
            print(f'[stage1][row_trunc] method={_rt_method} max_table_rows={_rt_max} (STATIC hard cap, raw-grounded, the same call stage 2 makes) {_rt_stats}', flush=True)
        want_tasks = set(args.struct_source)
        src = [r for r in sel if r.get('task') in want_tasks]
        if not src:
            raise SystemExit(f"[stage1] --downstream_budget selected {len(sel)} instances but none of them are {sorted(want_tasks)} (tasks seen: {sorted({r.get('task') for r in sel})}). --struct_source must name tasks the downstream regime actually trains on.")
        need = None
        downstream_info = {'regime': args.downstream_regime, 'budget': args.downstream_budget, 'seed': args.seed, 'select_mode': args.downstream_select_mode, 'drop_empty_target': bool(args.downstream_drop_empty_target), 'drop_truncated_tables': bool(args.downstream_drop_truncated_tables), 'cta_cpa_source': args.cta_cpa_source, 'n_instances_selected': len(sel), 'n_instances_used': len(src), 'instance_ids': sorted({str(r.get('id')) for r in src})}
        print(f'[stage1] source = THE DOWNSTREAM SELECTION: {args.downstream_regime} budget={args.downstream_budget} seed={args.seed} select_mode={args.downstream_select_mode} drop_empty_target={bool(args.downstream_drop_empty_target)} -> {len(sel)} instances, {len(src)} of them {sorted(want_tasks)}; every usable table among them is pretrained on (no --struct_tables cap)')
        if args.val_tables:
            print(f'[stage1][warn] --val_tables {args.val_tables} holds tables OUT of pretraining, so stage 1 would see fewer tables than stage 2. Pass --val_tables 0 to pretrain on the whole downstream selection.')
    else:
        src = interleave(load_source_rows(args.struct_source, args.cta_cpa_source), args.seed)
    if pinned_train is not None:
        want = set(pinned_train) | set(pinned_val)
        if pin_by_src:
            src = [r for r in src if (r.get('task'), r.get('id')) in want]
        else:
            src = [r for r in tqdm(src, desc='pin filter', unit='row') if untrimmed_table_hash(r, args.max_rows) in want]
        print(f'[stage1] pin filter: {len(src)} source prompts kept')
    seen = set()
    (rows, stats) = STR.synthesize_pool(src, max_rows=args.max_rows, encode=encode, max_len=args.max_seq_len, reserve=reserve, seen=seen, limit=need, tag=f"/{'+'.join(args.struct_source)}", progress=lambda it: tqdm(it, desc='structrel synthesize', unit='prompt'))
    if downstream_info is not None:
        used = {r['meta']['table_sha1'] for r in rows}
        skipped = {k[5:]: v for (k, v) in stats.items() if k.startswith('skip_')}
        n_dup = skipped.pop('duplicate_table', 0)
        n_inelig = sum(skipped.values())
        downstream_info.update(n_tables_pretrained=len(used), n_duplicate_tables=n_dup, n_ineligible_tables=n_inelig, ineligible_reasons=skipped)
        print(f"[stage1] COVERAGE of the downstream selection: {downstream_info['n_instances_used']} instances = {len(used)} tables pretrained + {n_dup} duplicates of them + {n_inelig} structurally ineligible {skipped}. " + ('Plus the --extra_tables draw below; nothing else.' if args.extra_tables else 'No table outside the selection is used.'))
    if args.extra_tables:
        if pinned_train is not None:
            raise SystemExit("[stage1] --extra_tables and --tables_from are mutually exclusive: --tables_from means 'exactly these tables, so a second model matches the first', and adding an unpinned draw on top would break that match.")
        if downstream_info is None:
            raise SystemExit("[stage1] --extra_tables is defined as 'the downstream selection PLUS n more', so it needs --downstream_budget. Without it the walk is already an open corpus draw and --struct_tables is the knob you want.")
        (held, field, held_desc) = heldout_table_keys(args.cta_cpa_source)
        print(f'[stage1] extra draw: excluding {len(held)} held-out tables by meta.{field} -- {held_desc}')
        pool = interleave(load_source_rows(args.struct_source, args.cta_cpa_source), args.seed)
        cand = [r for r in pool if table_key(r, field) not in held]
        print(f'[stage1] extra draw: {len(pool)} pool prompts -> {len(cand)} after the held-out filter ({len(pool) - len(cand)} dropped); walking for {args.extra_tables} tables not already taken from the b{args.downstream_budget} selection')
        n_before = len(seen)
        (extra_rows, extra_stats) = STR.synthesize_pool(cand, max_rows=args.max_rows, encode=encode, max_len=args.max_seq_len, reserve=reserve, seen=seen, limit=args.extra_tables, tag=f"/extra/{'+'.join(args.struct_source)}", progress=lambda it: tqdm(it, desc='structrel extra', unit='prompt'))
        n_extra = len(seen) - n_before
        if n_extra < args.extra_tables:
            print(f'[stage1][warn] only {n_extra} extra tables were available (asked for {args.extra_tables}); the pool ran out after the held-out and duplicate filters. Using all of them.')
        extra_info = {'asked': args.extra_tables, 'n_tables_added': n_extra, 'n_examples_added': len(extra_rows), 'heldout_field': field, 'n_heldout_excluded': len(held), 'heldout_description': held_desc, 'n_pool_prompts': len(pool), 'n_pool_prompts_after_filter': len(cand), 'stats': extra_stats}
        rows = rows + extra_rows
        print(f'[stage1] extra draw: +{n_extra} tables / +{len(extra_rows)} examples on top of the selection -> {len(rows)} structural examples in total')
    by_table = collections.OrderedDict()
    for r in rows:
        by_table.setdefault(r['meta']['table_sha1'], []).append(r)
    tables = list(by_table)
    if need is not None and len(tables) < need:
        print(f'[stage1][warn] only {len(tables)} eligible TRAIN tables available (asked for {need} = {args.struct_tables} train + {args.val_tables} val); using all of them and splitting proportionally.')
        n_val = max(1, round(args.val_tables * len(tables) / need))
    else:
        n_val = args.val_tables
    if pinned_train is not None:
        if pin_by_src:
            by_src = {(r['meta']['src_task'], r['meta']['src_id']): r['meta']['table_sha1'] for r in rows}
            to_tab = lambda ids: {by_src[i] for i in ids if i in by_src}
            (train_tables, val_tables) = (to_tab(pinned_train), to_tab(pinned_val))
        else:
            have = set(tables)
            (train_tables, val_tables) = (set(pinned_train) & have, set(pinned_val) & have)
        (lost_tr, lost_va) = (len(pinned_train) - len(train_tables), len(pinned_val) - len(val_tables))
        stats = dict(stats, pinned_from=args.tables_from, pinned_missing_train=lost_tr, pinned_missing_val=lost_va)
        if lost_tr or lost_va:
            print(f'[stage1][warn] {lost_tr}/{len(pinned_train)} train and {lost_va}/{len(pinned_val)} val pinned tables are not reproducible under this tokenizer and are DROPPED (not replaced).')
            frac = (lost_tr + lost_va) / max(1, len(pinned_train) + len(pinned_val))
            if frac > 0.05:
                raise SystemExit(f"[stage1] {frac:.1%} of the pinned tables were lost -- that is too far from 'the same tables' to call it a matched run.")
        else:
            print('[stage1] all pinned tables reproduced exactly.')
    else:
        keys = sorted(tables)
        random.Random(args.seed).shuffle(keys)
        (val_tables, train_tables) = (set(keys[:n_val]), set(keys[n_val:]))
    assert not val_tables & train_tables
    train_rows = [r for t in tables if t in train_tables for r in by_table[t]]
    val_rows = [r for t in tables if t in val_tables for r in by_table[t]]
    random.Random(args.seed).shuffle(train_rows)
    stats = dict(stats, downstream_selection=downstream_info, extra_corpus=extra_info, row_trunc=row_trunc_info, n_tables_total=len(tables), n_tables_train=len(train_tables), n_tables_val=len(val_tables), n_train_examples=len(train_rows), n_val_examples=len(val_rows), train_label_counts=dict(collections.Counter((json.loads(r['target'])['label'] for r in train_rows))), val_label_counts=dict(collections.Counter((json.loads(r['target'])['label'] for r in val_rows))), src_task_counts=dict(collections.Counter((r['meta']['src_task'] for r in rows))))
    return (train_rows, val_rows, stats)

class TableMaskTrainer(Trainer):

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        if getattr(self.args, 'remove_unused_columns', True):
            self.args.remove_unused_columns = False
            print(f'[{TOL.MECHANISM}] remove_unused_columns=False forced by TableMaskTrainer (region_ids must survive the dataloader)')

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        region_ids = inputs.pop('region_ids', None)
        if region_ids is None:
            raise ValueError('--lora_table_only is on but the batch carries no region_ids; the table-token scope is read from them, so the training dataset must be built with them.')
        TOL.set_table_mask_from_regions(region_ids)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Llama-3.2-3B-Instruct')
    ap.add_argument('--out', required=True)
    ap.add_argument('--struct_source', nargs='+', default=['cta', 'cpa'], choices=['cta', 'cpa', 'tabfact', 'wikiqa', 'finqa', 'tablebench'])
    ap.add_argument('--cta_cpa_source', default='t2d', choices=sorted(TL.CTA_CPA_DIRS))
    ap.add_argument('--struct_tables', type=int, default=1000)
    ap.add_argument('--downstream_budget', type=int, default=0)
    ap.add_argument('--downstream_regime', default='multi_ctacpa')
    ap.add_argument('--downstream_select_mode', default='random')
    ap.add_argument('--downstream_drop_empty_target', action='store_true')
    ap.add_argument('--downstream_drop_truncated_tables', action='store_true')
    ap.add_argument('--extra_tables', type=int, default=0)
    ap.add_argument('--tables_from', default=None)
    ap.add_argument('--val_tables', type=int, default=150)
    ap.add_argument('--max_table_rows', type=int, default=0)
    ap.add_argument('--row_trunc_method', choices=['random', 'tfidf', 'coverage', 'dedup', 'coverage_row', 'central_row', 'coverage_col'], default='coverage_row')
    ap.add_argument('--num_proc', type=int, default=1)
    ap.add_argument('--max_rows', type=int, default=20)
    ap.add_argument('--struct_modules', nargs='+', default=list(QKV), choices=list(QKV))
    ap.add_argument('--table_first', action='store_true')
    ap.add_argument('--lora_table_only', action='store_true')
    ap.add_argument('--epochs', type=float, default=3)
    ap.add_argument('--lr', type=float, default=0.0001)
    ap.add_argument('--micro_bs', type=int, default=4)
    ap.add_argument('--grad_accum', type=int, default=2)
    ap.add_argument('--max_seq_len', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log_steps', type=int, default=20)
    ap.add_argument('--attn_impl', default='sdpa', choices=['sdpa', 'flash_attention_2', 'eager'])
    ap.add_argument('--grad_checkpointing', action='store_true')
    ap.add_argument('--lora_r', type=int, default=16)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--dry_run_data', action='store_true')
    args = ap.parse_args()
    if args.max_table_rows > 0 and (not args.downstream_budget):
        raise SystemExit("[stage1] --max_table_rows needs --downstream_budget: row truncation is applied to the fine-tune's own selected tables, and there is no selection without it.")
    TL.set_all_seeds(args.seed)
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    STR.set_table_first(args.table_first)
    print(f"[stage1] prompt layout: {('TABLE FIRST (table + instruction + query)' if args.table_first else 'instruction + table + query (default)')}")
    (train_rows, val_rows, stats) = build_structural_pool(args, tok)
    vstats_tr = STR.verify_rows(train_rows)
    vstats_va = STR.verify_rows(val_rows)
    print(f'[stage1][verify] train {vstats_tr} | val {vstats_va}')
    lab_tr = stats['train_label_counts']
    spread = max(lab_tr.values()) - min(lab_tr.values()) if lab_tr else 0
    if len(lab_tr) != 3 or spread > 1:
        raise SystemExit(f'[stage1] structural classes are not balanced: {lab_tr}')
    print(f"[stage1] tables train={stats['n_tables_train']} val={stats['n_tables_val']} (disjoint) | examples train={len(train_rows)} val={len(val_rows)} | labels {lab_tr} | source tables {stats['src_task_counts']}")
    with open(f'{args.out}/struct_val.jsonl', 'w') as f:
        for r in val_rows:
            f.write(json.dumps(r) + '\n')
    (enc, region_ids, trimmed, dropped) = ([], [] if args.lora_table_only else None, 0, 0)
    for r in tqdm(train_rows, desc='tokenize', unit='ex'):
        e = TL.tokenize_example(tok, r, args.max_seq_len, args.model, want_regions=True if args.lora_table_only else None)
        if e is None:
            dropped += 1
            continue
        if args.lora_table_only:
            (ids, lab, was_trimmed, _roles, regs) = e
            if len(regs) != len(ids):
                raise SystemExit(f"[stage1] region ids ({len(regs)}) do not match input_ids ({len(ids)}) for {r['id']}")
            if not any((g == RCV.R_TABLE for g in regs)):
                raise SystemExit(f"[stage1] {r['id']} has NO table tokens under the shared span rule, so --lora_table_only would give it no trainable positions at all. Refusing: a structural example without a table is a bug, not a hard example.")
            region_ids.append(regs)
        else:
            (ids, lab, was_trimmed) = e
        if was_trimmed:
            raise SystemExit(f"[stage1] example {r['id']} was table-trimmed at tokenization; the pre-selection fit should have made that impossible.")
        trimmed += int(was_trimmed)
        enc.append((ids, lab))
    if dropped:
        raise SystemExit(f'[stage1] {dropped} structural examples did not fit max_seq_len={args.max_seq_len} even after trimming; refusing (the fit is supposed to guarantee they do).')
    print(f'[stage1] tokenized {len(enc)} examples, max_seq_len={args.max_seq_len}')
    if args.lora_table_only:
        n_tab = sum((sum((1 for g in r if g == RCV.R_TABLE)) for r in region_ids))
        n_tok = sum((len(r) for r in region_ids))
        print(f'[{TOL.MECHANISM}] {n_tab}/{n_tok} training tokens ({100.0 * n_tab / max(1, n_tok):.1f}%) are table tokens and carry the whole LoRA delta; the rest see the base projection')
    manifest = {'stage': 'structural_qkv_pretrain', 'task': STR.SLUG, 'labels': list(STR.LABELS), 'model': args.model, 'struct_source': list(args.struct_source), 'cta_cpa_source': args.cta_cpa_source, 'split_used': 'train_pool.jsonl (TRAIN SPLIT ONLY; no val/MMTU table can reach stage 1)', 'target_modules': list(args.struct_modules), 'table_first': bool(args.table_first), 'prompt_layout': 'table + instruction + query' if args.table_first else 'instruction + table + query', 'lora_table_only': bool(args.lora_table_only), 'build_stats': stats, 'verify_train': vstats_tr, 'verify_val': vstats_va, 'train_table_sha1': sorted({r['meta']['table_sha1'] for r in train_rows}), 'val_table_sha1': sorted({r['meta']['table_sha1'] for r in val_rows}), 'args': vars(args)}
    if args.dry_run_data:
        with open(f'{args.out}/struct_manifest.json', 'w') as f:
            json.dump({**manifest, 'dry_run_data': True}, f, indent=2)
        print(f'[dry_run_data] wrote {args.out}/struct_manifest.json and struct_val.jsonl (no model load)')
        return
    model = ML.load_causal_lm(args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn_impl, device_map={'': 0})
    TL.force_greedy_generation(model, tok)
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias='none', task_type='CAUSAL_LM', target_modules=list(args.struct_modules))
    model = get_peft_model(model, lora)
    trainable = [n for (n, p) in model.named_parameters() if p.requires_grad]
    leaked = [n for n in trainable if not any((f'.{m}.' in n for m in args.struct_modules))]
    if leaked:
        raise SystemExit(f'[stage1] parameters outside {list(args.struct_modules)} are trainable: {leaked[:5]}')
    non_lora = [n for n in trainable if 'lora_' not in n]
    if non_lora:
        raise SystemExit(f'[stage1] backbone parameters are trainable: {non_lora[:5]}')
    print(f'[stage1] trainable tensors: {len(trainable)} (all lora_A/lora_B on {list(args.struct_modules)})')
    model.print_trainable_parameters()
    table_only_info = None
    if args.lora_table_only:
        table_only_info = TOL.attach(model)
        print(f"[{TOL.MECHANISM}] hooked {table_only_info['n_lora_B_hooked']} lora_B modules")
        print(f"[{TOL.MECHANISM}] {table_only_info['formula']}")
        print(f"[{TOL.MECHANISM}] {table_only_info['token_scope']}")
        _g = torch.Generator().manual_seed(0)
        _probe = torch.randint(0, int(model.config.vocab_size), (2, 16), generator=_g).to(model.device)
        try:
            table_only_info['masking_check'] = TOL.verify_masking(model, _probe)
        except RuntimeError as e:
            import traceback as _tb
            _tb.print_exc()
            raise SystemExit(f'[{TOL.MECHANISM}] startup masking check FAILED: {e}')
        print(f"[{TOL.MECHANISM}] masking verified: max|logit diff| off the table = {table_only_info['masking_check']['max_abs_logit_diff_masked']:.3e}, on the table = {table_only_info['masking_check']['max_abs_logit_diff_table']:.3e}")
    targs = TrainingArguments(output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=args.micro_bs, gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, lr_scheduler_type='cosine', warmup_ratio=0.03, bf16=True, tf32=True, optim='adamw_torch_fused', logging_steps=args.log_steps, report_to=[], seed=args.seed, eval_strategy='no', save_strategy='no', label_names=['labels'], remove_unused_columns=False)
    if args.lora_table_only:
        print(f'[{TOL.MECHANISM}] remove_unused_columns=False (region_ids must reach compute_loss)')
    trainer_cls = TableMaskTrainer if args.lora_table_only else Trainer
    trainer = trainer_cls(model=model, args=targs, train_dataset=TL.SFTDataset(enc, region_ids=region_ids), data_collator=TL.PadCollator(tok.pad_token_id))
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    for ck in glob.glob(f'{args.out}/checkpoint-*'):
        shutil.rmtree(ck, ignore_errors=True)
    hist = [h for h in trainer.state.log_history if 'loss' in h]
    manifest['train_loss_history'] = hist
    manifest['final_train_loss'] = hist[-1]['loss'] if hist else None
    manifest['n_train_examples_used'] = len(enc)
    manifest['table_trimmed_over_len'] = trimmed
    manifest['table_only_lora'] = table_only_info
    with open(f'{args.out}/struct_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"[stage1] saved structural {'/'.join(args.struct_modules)} adapter -> {args.out}")
    if args.lora_table_only:
        print(f"[stage1] next: train_lora.py --init_qkv_adapter {args.out} (CONTINUES these {'/'.join(args.struct_modules)} matrices inside a fresh all-seven-module adapter; the base weights are NEVER touched and nothing is merged)")
    else:
        print(f'[stage1] next: scripts/eval_struct_relation.py --adapter {args.out}')
if __name__ == '__main__':
    main()
