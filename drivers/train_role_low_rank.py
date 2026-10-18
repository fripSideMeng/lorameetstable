import argparse
import glob
import json
import os
import random
import shutil
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
import evaluate as E
import train_lora as TL
import role_low_rank as RLR
ROOT = TL.ROOT
DATA = TL.DATA
ROLE_MODES = ('normal', 'no_role', 'role_swap', 'head_tail_swap')

def _render_prompt(tok, user_prompt, model_name):
    return TL.format_generation_prompt(tok, user_prompt, model_name)

def _trim_user_prompt(user_prompt):
    lines = user_prompt.split('\n')
    tbl = [i for (i, l) in enumerate(lines) if l.lstrip().startswith('|')]
    if len(tbl) <= 3:
        return (user_prompt, False)
    del lines[tbl[-1]]
    return ('\n'.join(lines), True)

def encode_example(tok, row, max_len, model_name):
    spec = RLR.role_spec(row['input'], row.get('task'))
    target_ids = tok(row['target'], add_special_tokens=False)['input_ids']
    if tok.eos_token_id is not None:
        target_ids = target_ids + [tok.eos_token_id]
    user_prompt = row['input']
    rendered = _render_prompt(tok, user_prompt, model_name)
    (p_ids, p_role) = RLR.tokens_role_ids(tok, rendered, spec)
    trimmed = False
    while len(p_ids) + len(target_ids) > max_len:
        (new_prompt, did) = _trim_user_prompt(user_prompt)
        if not did:
            break
        user_prompt = new_prompt
        rendered = _render_prompt(tok, user_prompt, model_name)
        (p_ids, p_role) = RLR.tokens_role_ids(tok, rendered, spec)
        trimmed = True
    input_ids = p_ids + target_ids
    if len(input_ids) > max_len:
        return None
    labels = [-100] * len(p_ids) + target_ids
    role_ids = p_role + [RLR.ROLE_NONE] * len(target_ids)
    if len({len(input_ids), len(labels), len(role_ids)}) != 1:
        raise RuntimeError('length mismatch in encode_example')
    if all((x == -100 for x in labels)):
        raise RuntimeError(f"no supervised tokens for id={row.get('id')}")
    if any((r not in spec.allowed for r in role_ids)):
        raise RuntimeError(f"invalid role id for id={row.get('id')} ({spec})")
    return (input_ids, labels, role_ids, spec, trimmed)

class RoleLowRankDataset(torch.utils.data.Dataset):

    def __init__(self, enc):
        self.enc = enc

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        (ids, lab, role) = self.enc[i]
        return {'input_ids': ids, 'labels': lab, 'attention_mask': [1] * len(ids), 'role_ids': role}

class RoleLowRankPadCollator:

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        m = max((len(b['input_ids']) for b in batch))
        (ii, ll, am, rr) = ([], [], [], [])
        for b in batch:
            n = m - len(b['input_ids'])
            ii.append(b['input_ids'] + [self.pad_id] * n)
            ll.append(b['labels'] + [-100] * n)
            am.append(b['attention_mask'] + [0] * n)
            rr.append(b['role_ids'] + [RLR.ROLE_NONE] * n)
        return {'input_ids': torch.tensor(ii), 'labels': torch.tensor(ll), 'attention_mask': torch.tensor(am), 'role_ids': torch.tensor(rr)}

@torch.no_grad()
def role_generate(model, tok, inputs, max_new_tokens, bs, model_name, task, role_mode='normal', length_bucket=True):
    if role_mode not in ROLE_MODES:
        raise ValueError(f'unknown role_mode={role_mode} (expected one of {sorted(ROLE_MODES)})')
    dev = model.device
    pad_id = tok.pad_token_id
    seqs = []
    for (orig_idx, c) in enumerate(inputs):
        spec = RLR.role_spec(c, task)
        rendered = _render_prompt(tok, c, model_name)
        (ids, role) = RLR.tokens_role_ids(tok, rendered, spec)
        if role_mode == 'no_role':
            role = [RLR.ROLE_NONE] * len(ids)
        elif role_mode != 'normal':
            role = RLR.permute_roles(role, role_mode)
        seqs.append({'orig_idx': orig_idx, 'ids': ids, 'role': role, 'length': len(ids)})
    if length_bucket:
        seqs.sort(key=lambda x: x['length'])
    outs = [None] * len(inputs)
    for i in tqdm(range(0, len(seqs), bs), desc=f'role-gen[{role_mode},bucket={length_bucket}]', leave=False):
        chunk = seqs[i:i + bs]
        m = max((x['length'] for x in chunk))
        (ii, am, rr) = ([], [], [])
        for x in chunk:
            n = m - len(x['ids'])
            ii.append([pad_id] * n + x['ids'])
            am.append([0] * n + [1] * len(x['ids']))
            rr.append([RLR.ROLE_NONE] * n + x['role'])
        input_ids = torch.tensor(ii, device=dev)
        attn = torch.tensor(am, device=dev)
        role_full = torch.tensor(rr, device=dev)
        RLR.set_role_ids(role_full)
        try:
            gen = model.generate(input_ids=input_ids, attention_mask=attn, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id, use_cache=True)
        finally:
            RLR.clear_role_ids()
        prompt_len = input_ids.shape[1]
        for (j, x) in enumerate(chunk):
            outs[x['orig_idx']] = tok.decode(gen[j][prompt_len:], skip_special_tokens=True)
    if any((o is None for o in outs)):
        raise RuntimeError('internal error: some generations were not filled')
    return outs

class RoleLowRankTrainer(Trainer):

    def __init__(self, *a, val_tasks=None, gen_tok=None, source='sotab', val_max=4096, val_rows=None, val_max_new_tokens=400, gen_bs=4, model_name=None, eval_role_mode='normal', defer_role_clear_until_after_backward=False, table_first=False, **kw):
        super().__init__(*a, **kw)
        self.defer_role_clear_until_after_backward = defer_role_clear_until_after_backward
        self.table_first = bool(table_first)
        self.gen_tok = gen_tok
        self.source = source
        self.val_max = val_max
        self.val_max_new_tokens = val_max_new_tokens
        self.gen_bs = gen_bs
        self.model_name = model_name
        self.eval_history = []
        self.best_val_score = None
        self.best_state = None
        if eval_role_mode not in set(ROLE_MODES) | {'all'}:
            raise ValueError(f'unknown eval_role_mode={eval_role_mode}')
        self.eval_role_mode = eval_role_mode
        self.val_data = {}
        self.val_counts = {}
        rng = random.Random(42)
        for slug in val_tasks or []:
            if val_rows and slug in val_rows:
                rows = TL.read_jsonl(val_rows[slug])
                n_total = len(rows)
            else:
                rows = TL.read_jsonl(f'{DATA}/{TL.data_dir(slug, source)}/val.jsonl')
                n_total = len(rows)
                if val_max is not None and val_max > 0 and (len(rows) > val_max):
                    rows = rng.sample(rows, val_max)
            if rows:
                TL.apply_table_first(rows, self.table_first, f'val:{slug}')
                self.val_data[slug] = rows
                self.val_counts[slug] = {'n_total': n_total, 'n_used': len(rows), 'val_max': val_max}
                print(f'[val] {slug}: using {len(rows)}/{n_total} (val_max={val_max})')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        role_ids = inputs.pop('role_ids')
        defer_clear = self.defer_role_clear_until_after_backward and model.training
        RLR.set_role_ids(role_ids)
        try:
            out = model(**inputs)
        except Exception:
            RLR.clear_role_ids()
            raise
        if not defer_clear:
            RLR.clear_role_ids()
        loss = out.loss
        return (loss, out) if return_outputs else loss

    def training_step(self, model, inputs, *args, **kwargs):
        try:
            return super().training_step(model, inputs, *args, **kwargs)
        finally:
            if self.defer_role_clear_until_after_backward:
                RLR.clear_role_ids()

    def _evaluate_one_role_mode(self, role_mode):
        (per_scores, per_details) = ({}, {})
        for (slug, rows) in self.val_data.items():
            gens = role_generate(self.model, self.gen_tok, [r['input'] for r in rows], self.val_max_new_tokens, self.gen_bs, self.model_name, slug, role_mode=role_mode)
            scorer = TL.resolve_scorer(slug, self.source)
            tc = tt = exact_count = parse_fail = 0
            (golds, preds) = ([], [])
            for (r, g) in zip(rows, gens):
                pj = E.extract_json(g)
                if pj is None:
                    parse_fail += 1
                gj = json.loads(r['target'])
                (c, t, ex) = scorer(pj, gj, r.get('accept'))
                tc += c
                tt += t
                exact_count += ex
                if slug in {'cta', 'cpa'}:
                    (gold, pred) = TL.get_gold_and_pred_for_f1(slug, pj, gj, self.source)
                    golds.append(gold)
                    preds.append(pred)
            acc = tc / max(1, tt)
            exact = exact_count / max(1, len(rows))
            if slug in {'cta', 'cpa'}:
                (macro_f1, weighted_f1) = E.macro_weighted_f1(golds, preds)
                per_scores[slug] = macro_f1
                per_details[slug] = {'selection_metric': 'macro_f1', 'selection_score': macro_f1, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1, 'acc': acc, 'exact': exact, 'parse_fail': parse_fail, 'n_val': len(rows), 'role_mode': role_mode}
            else:
                raise ValueError(f'unsupported val slug for role-low-rank pipeline: {slug}')
        val_score = float(np.mean(list(per_scores.values()))) if per_scores else 0.0
        return (val_score, per_details)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        tok = self.gen_tok
        was_training = self.model.training
        self.model.eval()
        prev_pad = tok.padding_side
        prev_cache = self.model.config.use_cache
        tok.padding_side = 'left'
        self.model.config.use_cache = True
        try:
            modes = list(ROLE_MODES) if self.eval_role_mode == 'all' else [self.eval_role_mode]
            (all_scores, all_details) = ({}, {})
            for mode in modes:
                (val_score, per_details) = self._evaluate_one_role_mode(mode)
                all_scores[mode] = val_score
                all_details[mode] = per_details
            main_mode = 'normal' if 'normal' in all_scores else modes[0]
            val_score = all_scores[main_mode]
            per_details = all_details[main_mode]
            self.eval_history.append((val_score, dict(per_details)))
            if self.best_val_score is None or val_score > self.best_val_score:
                self.best_val_score = val_score
                self.best_state = RLR.gather_trainable_state(self.model)
            metrics = {f'{metric_key_prefix}_val_score': val_score, f'{metric_key_prefix}_main_role_mode': main_mode}
            for mode in modes:
                metrics[f'{metric_key_prefix}_{mode}_val_score'] = all_scores[mode]
                for (slug, d) in all_details[mode].items():
                    prefix = f'{metric_key_prefix}_{mode}_{slug}'
                    metrics[f'{prefix}_selection_score'] = d['selection_score']
                    metrics[f'{prefix}_acc'] = d['acc']
                    metrics[f'{prefix}_exact'] = d['exact']
                    metrics[f'{prefix}_parse_fail'] = d['parse_fail']
                    metrics[f'{prefix}_macro_f1'] = d['macro_f1']
                    metrics[f'{prefix}_weighted_f1'] = d['weighted_f1']
            for (other, tag) in (('no_role', 'norole'), ('role_swap', 'swap'), ('head_tail_swap', 'htswap')):
                if 'normal' in all_scores and other in all_scores:
                    metrics[f'{metric_key_prefix}_normal_minus_{tag}_val_score'] = all_scores['normal'] - all_scores[other]
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics
        finally:
            tok.padding_side = prev_pad
            self.model.config.use_cache = prev_cache
            if was_training:
                self.model.train()

class RoleLowRankEvalCallback(TrainerCallback):

    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        self.trainer.evaluate()
        return control

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', required=True, choices=['single_cta', 'single_cpa', 'multi_ctacpa'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--budget', type=int, default=100)
    ap.add_argument('--cta_cpa_source', default='sotab', choices=['sotab', 'sotab_full', 't2d'])
    ap.add_argument('--select_mode', default='random')
    ap.add_argument('--drop_empty_target', action='store_true')
    ap.add_argument('--model', default=f'{ROOT}/Qwen3-4B-Instruct-2507')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_seq_len', type=int, default=2048)
    ap.add_argument('--epochs', type=float, default=3.0)
    ap.add_argument('--lr', type=float, default=0.0001)
    ap.add_argument('--micro_bs', type=int, default=2)
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--lora_r', type=int, default=16)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--target_modules', nargs='+', default=list(RLR.TARGET_MODULES), choices=list(RLR.TARGET_MODULES))
    ap.add_argument('--role_modules', nargs='+', default=list(RLR.ROLE_MODULES), choices=list(RLR.TARGET_MODULES))
    ap.add_argument('--init_qkv_adapter', default=None)
    ap.add_argument('--table_first', action='store_true')
    ap.add_argument('--max_table_rows', type=int, default=0)
    ap.add_argument('--row_trunc_method', default='coverage_row')
    ap.add_argument('--attn_impl', default='sdpa', choices=['sdpa', 'flash_attention_2', 'eager'])
    ap.add_argument('--val_max', type=int, default=4096)
    ap.add_argument('--train_rows', default=None)
    ap.add_argument('--val_rows', default=None)
    ap.add_argument('--val_max_new_tokens', type=int, default=400)
    ap.add_argument('--gen_bs', type=int, default=16)
    ap.add_argument('--log_steps', type=int, default=100)
    ap.add_argument('--grad_checkpointing', action='store_true')
    ap.add_argument('--eval_role_mode', default='normal', choices=['normal', 'no_role', 'role_swap', 'head_tail_swap', 'all'])
    args = ap.parse_args()
    TL.set_all_seeds(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if args.train_rows:
        rows = TL.read_jsonl(args.train_rows)
        print(f'[data] pre-selected training rows from {args.train_rows}: {len(rows)}')
    else:
        rows = TL.build_train_rows(args.regime, args.budget, args.seed, source=args.cta_cpa_source, select_mode=args.select_mode, drop_empty_target=args.drop_empty_target)
    if args.max_table_rows and args.max_table_rows > 0:
        TL._prefetch_raw_tables(rows, args.cta_cpa_source, 'train', 1)
        rt_stats = TL.apply_row_truncation(rows, args.cta_cpa_source, args.row_trunc_method, args.max_table_rows, args.seed, tok, args.max_seq_len, args.model, False, split='train', hard_cap=True)
        print(f'[row_trunc] method={args.row_trunc_method} max_table_rows={args.max_table_rows} (STATIC hard cap once pre-training, raw-grounded) {rt_stats}')
    TL.apply_table_first(rows, args.table_first, 'train')
    (enc, dropped, trimmed) = ([], 0, 0)
    role_totals = {n: 0 for n in RLR.ROLE_NAMES.values()}
    n_missing_designated = 0
    task_counts_used = {t: 0 for t in RLR.ROLE_TASKS}
    sanity_example = {}
    for r in tqdm(rows, desc='encode'):
        e = encode_example(tok, r, args.max_seq_len, args.model)
        if e is None:
            dropped += 1
            continue
        (ids, lab, role, spec, was) = e
        enc.append((ids, lab, role))
        trimmed += int(was)
        task_counts_used[spec.task] += 1
        sanity_example.setdefault(spec.task, (r['input'], spec))
        counts = RLR.role_id_counts(role)
        for k in role_totals:
            role_totals[k] += counts[k]
        if not any((counts[RLR.ROLE_NAMES[rid]] for rid in spec.name_to_role.values())):
            n_missing_designated += 1
    task_counts = {s: sum((1 for r in rows if r['task'] == s)) for s in TL.TASK_SLUGS}
    print(f'[{args.regime} seed={args.seed} source={args.cta_cpa_source}] requested={len(rows)} mix={task_counts} used={len(enc)} table_trimmed={trimmed} dropped_over_len={dropped}')
    tot_tok = max(1, sum(role_totals.values()))
    print('[roles] token share: ' + '  '.join((f'{k}={v / tot_tok:.4f}' for (k, v) in role_totals.items())))
    print(f'[roles] instances whose declared column(s) matched no table column: {n_missing_designated}/{len(enc)}')
    print(f'[roles] instances per task: {task_counts_used}')
    print('[roles] ---------- role assignment on one real example per task ----------')
    for t in RLR.ROLE_TASKS:
        if t not in sanity_example:
            continue
        (prompt, spec) = sanity_example[t]
        print(RLR.format_column_roles(prompt, spec))
    print('[roles] ' + '-' * 72)
    missing = [t for (t, n) in task_counts_used.items() if n == 0]
    if missing:
        raise RuntimeError(f'no training instances for task(s) {missing}: their role vectors could never receive a gradient and would stay at zero. This pipeline needs both of {RLR.ROLE_TASKS}.')
    if not enc:
        raise ValueError('no training examples after filtering')
    starved = [n for n in RLR.ROLE_VECTOR_NAMES if role_totals[n] == 0]
    if starved:
        raise RuntimeError(f'role(s) {starved} never occur in the training data, so their vectors could never receive a gradient and would stay at exactly zero. The task-shaped role vocabulary this pipeline tests does not exist in this data -- check --regime / prompt format.')
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn_impl, device_map={'': 0})
    TL.force_greedy_generation(model, tok)
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    for p in model.parameters():
        p.requires_grad = False
    patched = RLR.patch_model(model, args.lora_r, args.lora_alpha, args.lora_dropout, target_modules=args.target_modules, role_modules=args.role_modules)
    (summary, rep) = RLR.architecture_summary(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    print(summary)
    n_tr = rep['measured_trainable_params']
    assert rep['total_trainable'] == n_tr, (rep['total_trainable'], n_tr)
    facts = RLR.assert_role_architecture(model, expect_role_modules=args.role_modules)
    print(RLR.format_role_architecture(facts))
    qkv_init_info = None
    if args.init_qkv_adapter:
        qkv_init_info = RLR.init_from_struct_adapter(model, args.init_qkv_adapter, args.lora_r, args.lora_alpha)
    print(RLR.format_param_arithmetic(model))
    if args.grad_checkpointing:
        print('[role-low-rank] gradient checkpointing enabled; deferring clear_role_ids() until after backward.')
    val_tasks = TL.select_val_tasks(args.regime)
    targs = TrainingArguments(output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=args.micro_bs, gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, lr_scheduler_type='cosine', warmup_ratio=0.03, bf16=True, tf32=True, optim='adamw_torch_fused', group_by_length=args.regime in ('single_cta', 'single_cpa'), logging_steps=args.log_steps, report_to=[], seed=args.seed, eval_strategy='no', save_strategy='no', label_names=['labels'], remove_unused_columns=False)
    trainer = RoleLowRankTrainer(model=model, args=targs, train_dataset=RoleLowRankDataset(enc), data_collator=RoleLowRankPadCollator(tok.pad_token_id), val_tasks=val_tasks, gen_tok=tok, source=args.cta_cpa_source, val_max=args.val_max, table_first=args.table_first, val_rows=json.loads(args.val_rows) if args.val_rows else None, val_max_new_tokens=args.val_max_new_tokens, gen_bs=args.gen_bs, model_name=args.model, eval_role_mode=args.eval_role_mode, defer_role_clear_until_after_backward=bool(args.grad_checkpointing))
    trainer.add_callback(RoleLowRankEvalCallback(trainer))
    trainer.train()
    (best_idx, best_score, best_per_task) = (-1, None, {})
    if trainer.eval_history:
        best_idx = max(range(len(trainer.eval_history)), key=lambda i: trainer.eval_history[i][0])
        best_score = trainer.eval_history[best_idx][0]
        best_per_task = trainer.eval_history[best_idx][1]
    if trainer.best_state is not None:
        RLR.scatter_trainable_state(model, trainer.best_state)
        print(f'restored best epoch weights (val_score={trainer.best_val_score})')
    config = {'pipeline': 'role_low_rank', 'regime': args.regime, 'seed': args.seed, 'cta_cpa_source': args.cta_cpa_source, 'select_mode': args.select_mode, 'drop_empty_target': bool(args.drop_empty_target), 'lora_r': args.lora_r, 'lora_alpha': args.lora_alpha, 'lora_dropout': args.lora_dropout, 'scaling': args.lora_alpha / args.lora_r, 'target_modules': patched['target_modules'], 'role_modules': patched['role_modules'], 'role_scheme': 'cta_target_context + cpa_head_tail_context', 'role_vector_names': patched['role_vector_names'], 'n_role_vectors_per_module': patched['n_role_vectors_per_module'], 'roles': dict(RLR.ROLE_IDS), 'task_roles': {t: [RLR.ROLE_NAMES[i] for i in ids] for (t, ids) in RLR.TASK_ROLES.items()}, 'role_dim': args.lora_r, 'update_rule': 'delta_h = (alpha/r) * B ( A x + e_role )', 'role_up_proj_sharing': 'none -- the role offsets share the LoRA up-projection B', 'role_tasks': list(RLR.ROLE_TASKS), 'role_module_policy': 'role offsets attached to k_proj and v_proj only; q/o/gate/up/down carry ordinary shared LoRA with no role parameter', 'e_none': 'fixed zero (not a parameter), shared by both tasks', 'role_vector_init': 'zeros (identical to vanilla LoRA at init)', 'adapter_dtype': 'float32 (lora_A/lora_B/role_vec); frozen backbone stays bfloat16 -- matches standard PEFT autocast_adapter_dtype=True', 'no_role_semantics': 'ablation of THIS checkpoint with every role forced to none; NOT the standard-MTL baseline, which is the separately trained shared-LoRA model from train_lora.py', 'role_swap_semantics': 'every column handed a role it should not have (CTA context<->target; CPA context->head->tail->context); tests whether the offsets are aimed at the right columns, not merely non-zero', 'head_tail_swap_semantics': 'CPA head and tail exchanged and nothing else; the direction ablation this task-shaped vocabulary exists for', 'markdown_syntax_role': "none -- '|' delimiters, separator rows and padding whitespace never carry a role; only real header/data cell content does", 'cta_policy': 'target entity column -> cta_target; every other table column -> cta_context', 'cpa_policy': 'declared head column -> cpa_head; declared tail column -> cpa_tail; every other table column -> cpa_context. The direction is NOT collapsed.', 'role_source': 'structural column declarations in the prompt ("Target entity column:", "Head column:", "Tail column:"); never a label or gold answer', 'role_precedence': 'token role = highest-precedence role over the spanned chars (designated target/head/tail > context > none), leftmost among equals', 'non_table_role': 'none', 'model': args.model, 'table_first': bool(args.table_first), 'max_table_rows': int(args.max_table_rows or 0), 'row_trunc_method': args.row_trunc_method if args.max_table_rows else None, 'qkv_init': qkv_init_info}
    role_stats = RLR.role_separation_summary(model, per_module=True)
    print(RLR.format_role_separation(role_stats))
    dead = [n for n in RLR.ROLE_VECTOR_NAMES if role_stats['role_norms'][n].get('max', 0.0) == 0.0]
    if dead:
        raise RuntimeError(f'role vector(s) {dead} are still exactly zero in EVERY module after training (no gradients reached them). Check gradient checkpointing / role context lifetime.')
    for (a_, b_) in (('cta_target', 'cta_context'), ('cpa_head', 'cpa_context'), ('cpa_tail', 'cpa_context'), ('cpa_head', 'cpa_tail')):
        st = role_stats['role_pairs'][f'{a_}|{b_}']['e_diff']
        if st.get('max', 0.0) == 0.0:
            raise RuntimeError(f'e_{a_} == e_{b_} in EVERY module, so the distinction this pipeline exists to test is not being used.')
    RLR.save_role_adapter(model, args.out, config)
    tok.save_pretrained(args.out)
    for ck in glob.glob(f'{args.out}/checkpoint-*'):
        shutil.rmtree(ck, ignore_errors=True)
    with open(f'{args.out}/train_manifest.json', 'w') as f:
        json.dump({**config, 'budget': args.budget, 'n_requested': len(rows), 'mix': task_counts, 'n_used': len(enc), 'table_trimmed_over_len': trimmed, 'dropped_over_len': dropped, 'role_token_totals': role_totals, 'role_token_share': {k: v / tot_tok for (k, v) in role_totals.items()}, 'n_instances_without_designated_column': n_missing_designated, 'selection_tasks': val_tasks, 'param_report': rep, 'best_epoch_index': best_idx, 'best_val_score': best_score, 'best_val_per_task': best_per_task, 'val_counts': trainer.val_counts, 'role_separation_summary': role_stats, 'instances_per_role_task': task_counts_used, 'n_trainable': n_tr, 'train_ids': [r['id'] for r in rows], 'args': vars(args)}, f, indent=2)
    print(f'saved role-low-rank adapter -> {args.out}  (best_val_score={best_score} @ epoch_idx={best_idx})')
if __name__ == '__main__':
    main()
