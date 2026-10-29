import argparse
import collections
import glob
import hashlib
import json
import os
import random
import re
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import model_load as ML
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
from peft import LoraConfig, get_peft_model
import evaluate as E
from lora_internal_per_token import find_markdown_table_span
import table_trunc as TT
import samerow as SR
import grad_col_mask as GCM
import table_transpose as TTR
import table_reorder as TRO
import role_gate_mlp as RGM
import ffn_role as FRS
import rank_gate as RKG
import role_matrix as RM
import recv_gate as RCV
import split_gate_lora as SGL
import swiglu_thres as SWT
import ctx_relay as CRM
import tgt_sandwich as TSW
import soft_prompt as SP
import struct_role as SRB
import attn_train as ATM
import grad_tgt_b as GTB
ROOT = os.environ.get('LORA_GEN_ROOT') or '.'
DATA = f'{ROOT}/data_prepared'
CTA_CPA_DIRS = {'sotab': {'cta': 'cta', 'cpa': 'cpa'}, 'sotab_full': {'cta': 'cta_sotab_full', 'cpa': 'cpa_sotab_full'}, 'sotab_all': {'cta': 'cta_sotab_full', 'cpa': 'cpa_sotab_full'}, 't2d': {'cta': 'cta_t2d_full', 'cpa': 'cpa_t2d_full'}, 'shared': {'cta': 'cta_shared', 'cpa': 'cpa_shared'}, 'targetonly': {'cta': 'cta_t2d_full_targetonly', 'cpa': 'cpa_t2d_full_targetonly'}}
GOLD_KEY = {'sotab': {'cta': 'column type', 'cpa': 'property'}, 'sotab_full': {'cta': 'column type', 'cpa': 'property'}, 'sotab_all': {'cta': 'column type', 'cpa': 'property'}, 't2d': {'cta': 'DBpedia ontology class', 'cpa': 'DBpedia property'}, 'shared': {'cta': 'DBpedia ontology class', 'cpa': 'DBpedia property'}, 'targetonly': {'cta': 'DBpedia ontology class', 'cpa': 'DBpedia property'}}
PRED_KEYS = {'sotab': {'cta': ['column type'], 'cpa': ['property']}, 'sotab_full': {'cta': ['column type'], 'cpa': ['property']}, 'sotab_all': {'cta': ['column type'], 'cpa': ['property']}, 't2d': {'cta': ['DBpedia ontology class', 'column type'], 'cpa': ['DBpedia property', 'property']}, 'shared': {'cta': ['DBpedia ontology class', 'column type'], 'cpa': ['DBpedia property', 'property']}, 'targetonly': {'cta': ['DBpedia ontology class', 'column type'], 'cpa': ['DBpedia property', 'property']}}
SOTAB_SOURCES = ('sotab', 'sotab_full', 'sotab_all')
_SOTAB_TABLE_SUFFIX = re.compile('_(CTA|CPA)\\.json\\.gz$')

def _sotab_base(table):
    return _SOTAB_TABLE_SUFFIX.sub('', table)

def _shared_table_id(source, row):
    m = row['meta']
    return _sotab_base(m['table']) if source in SOTAB_SOURCES else m['csv']
REGIME_WEIGHTS = {'multi_tableqa': {'finqa': 2, 'tablebench': 4, 'wikiqa': 4}}

def weighted_budgets(tasks, weights, budget):
    total = sum((weights[t] for t in tasks))
    exact = {t: budget * weights[t] / total for t in tasks}
    out = {t: int(exact[t]) for t in tasks}
    rem = budget - sum(out.values())
    for t in sorted(tasks, key=lambda t: (-(exact[t] - int(exact[t])), tasks.index(t)))[:rem]:
        out[t] += 1
    return out

def data_dir(slug, source):
    if slug in ('cta', 'cpa'):
        return CTA_CPA_DIRS[source][slug]
    if slug == 'cea':
        return 'cea_t2d_full'
    return slug

def read_jsonl(p):
    return [json.loads(l) for l in open(p)]

def label_coverage_sample(pool, budget, rng, label_of, name):
    by_label = collections.defaultdict(list)
    for (i, r) in enumerate(pool):
        by_label[label_of(r)].append(i)
    n_labels = len(by_label)
    if budget < n_labels:
        raise ValueError(f'[{name}] budget={budget} < distinct labels in train pool ({n_labels}); cannot guarantee every label appears at least once. Increase the (per-task) budget to >= {n_labels}.')
    chosen = set()
    for (lab, idxs) in by_label.items():
        chosen.add(idxs[rng.randrange(len(idxs))])
    rest = [i for i in range(len(pool)) if i not in chosen]
    rng.shuffle(rest)
    for i in rest[:budget - len(chosen)]:
        chosen.add(i)
    out = [pool[i] for i in sorted(chosen)]
    rng.shuffle(out)
    return out

def set_all_seeds(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
TASK_SLUGS = ['cta', 'cpa', 'hvm']
REGIME_SINGLE = {'single_cta': 'cta', 'single_cpa': 'cpa', 'single_cea': 'cea', 'single_hvm': 'hvm', 'single_em': 'em', 'single_tabfact': 'tabfact', 'single_wikiqa': 'wikiqa', 'single_samerow': 'samerow'}
REGIME_MULTI = {'multi': ['cta', 'cpa', 'hvm'], 'multi_ctacpa': ['cta', 'cpa'], 'multi_shared_ctacpa': ['cta', 'cpa'], 'multi_paired_ctacpa': ['cta', 'cpa'], 'multi_em': ['cta', 'cpa', 'em'], 'multi_tabfact': ['cta', 'cpa', 'tabfact'], 'multi_cta_em_tabfact': ['cta', 'em', 'tabfact'], 'multi_cpa_em_tabfact': ['cpa', 'em', 'tabfact'], 'multi_tabfact_hvm': ['tabfact', 'hvm'], 'multi_tabfact_em': ['tabfact', 'em'], 'multi_cta_hvm': ['cta', 'hvm'], 'multi_cpa_hvm': ['cpa', 'hvm'], 'multi_cta_em': ['cta', 'em'], 'multi_cpa_em': ['cpa', 'em'], 'multi_cta_tabfact': ['cta', 'tabfact'], 'multi_cpa_tabfact': ['cpa', 'tabfact'], 'multi_tableqa': ['finqa', 'tablebench', 'wikiqa'], 'multi_all4': ['cta', 'cpa', 'hvm', 'tabfact'], 'multi_cta_samerow': ['cta', 'samerow'], 'multi_cpa_samerow': ['cpa', 'samerow'], 'multi_hvm_samerow': ['hvm', 'samerow'], 'multi_tabfact_samerow': ['tabfact', 'samerow']}
SAMEROW_REGIME_PARTNER = {'single_samerow': None, 'multi_cta_samerow': 'cta', 'multi_cpa_samerow': 'cpa', 'multi_hvm_samerow': 'hvm', 'multi_tabfact_samerow': 'tabfact'}

def _uses_samerow(regime):
    tasks = [REGIME_SINGLE[regime]] if regime in REGIME_SINGLE else list(REGIME_MULTI.get(regime, []))
    return SR.SLUG in tasks

def samerow_partner(regime, override):
    if override and override != 'auto':
        return override
    partner = SAMEROW_REGIME_PARTNER.get(regime)
    if partner is None:
        raise ValueError(f"regime {regime} does not name a table task to take tables from; pass --samerow_source {{{','.join(SR.SOURCE_TASKS)}}}.")
    return partner
_samerow_pool_cache = {}

def load_samerow_pool(split, src_slug, source, max_rows, tok=None, max_len=None, model_name=None):
    key = (split, src_slug, source, max_rows, None if tok is None else (getattr(tok, 'name_or_path', None), max_len, model_name))
    if key not in _samerow_pool_cache:
        fname = 'train_pool.jsonl' if split == 'train' else 'val.jsonl'
        src = read_jsonl(f'{DATA}/{data_dir(src_slug, source)}/{fname}')
        (encode, reserve) = (None, 0)
        if tok is not None and max_len is not None:
            encode = lambda t: _encode_prompt(tok, t, model_name)
            reserve = max((len(tok(json.dumps({'label': lab}), add_special_tokens=False)['input_ids']) for lab in (SR.LABEL_SAME, SR.LABEL_DIFF))) + (1 if tok.eos_token_id is not None else 0)
        (rows, _) = SR.synthesize_pool(src, max_rows=max_rows, tag=f'/{split}/{src_slug}', encode=encode, max_len=max_len, reserve=reserve)
        _samerow_pool_cache[key] = rows
    return list(_samerow_pool_cache[key])
SCORERS = {'cta': E.score_cta, 'cpa': E.score_cpa, 'hvm': E.score_hvm}

def resolve_scorer(slug, source):
    if source in ('t2d', 'shared', 'targetonly') and slug == 'cta':
        return E.score_cta_mmtu
    if source in ('t2d', 'shared', 'targetonly') and slug == 'cpa':
        return E.score_cpa_mmtu
    if slug == 'em':
        return E.score_em
    if slug == 'tabfact':
        return E.score_tabfact
    if slug in ('wikiqa', 'finqa', 'tablebench'):
        return E.score_wikiqa
    if slug == 'cea':
        return E.score_cea_mmtu
    if slug == 'samerow':
        return E.score_samerow
    return SCORERS[slug]
INVALID = '__invalid__'

def has_chat_template(tok):
    return getattr(tok, 'chat_template', None) is not None

def is_qwen3_model(model_name):
    name = str(model_name).lower()
    return 'qwen3' in name or 'qwen-3' in name

def is_qwen3_thinking_model(model_name):
    name = str(model_name).lower()
    return 'thinking' in name

def apply_model_chat_template(tok, messages, model_name, tokenize, add_generation_prompt):
    kwargs = {}
    if is_qwen3_model(model_name) and (not is_qwen3_thinking_model(model_name)):
        kwargs['enable_thinking'] = False
    return tok.apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, **kwargs)

def format_generation_prompt(tok, content, model_name):
    if not has_chat_template(tok):
        return content
    return apply_model_chat_template(tok, [{'role': 'user', 'content': content}], model_name=model_name, tokenize=False, add_generation_prompt=True)

def force_greedy_generation(model, tok):
    if not hasattr(model, 'generation_config'):
        return
    gen_cfg = model.generation_config
    gen_cfg.do_sample = False
    gen_cfg.num_beams = 1
    neutral_values = {'temperature': 1.0, 'top_p': 1.0, 'top_k': 50, 'typical_p': 1.0, 'epsilon_cutoff': 0.0, 'eta_cutoff': 0.0}
    for (attr, value) in neutral_values.items():
        if hasattr(gen_cfg, attr):
            setattr(gen_cfg, attr, value)
    if hasattr(gen_cfg, 'min_p'):
        gen_cfg.min_p = None
    gen_cfg.pad_token_id = tok.pad_token_id
    if tok.eos_token_id is not None:
        gen_cfg.eos_token_id = tok.eos_token_id
    _mt = str(getattr(getattr(model, 'config', None), 'model_type', ''))
    if (_mt.startswith('gemma3') or type(model).__name__.startswith('Gemma3')) and getattr(gen_cfg, 'cache_implementation', None):
        print(f"[gemma3] cache_implementation={gen_cfg.cache_implementation!r} -> None (dynamic cache); the hybrid static cache raises 'Max cache length is not consistent across layers' on the second val batch")
        gen_cfg.cache_implementation = None
    model.generation_config = gen_cfg

def _cta_cpa_label(slug, source, row):
    if source in SOTAB_SOURCES:
        return json.loads(row['target'])[GOLD_KEY[source][slug]]
    return row['meta']['label']

def _val_strat_label(store_slug, source, row):
    if store_slug in ('cta', 'cpa'):
        return _cta_cpa_label(store_slug, source, row)
    if store_slug in ('em', 'tabfact'):
        return json.loads(row['target'])['label']
    return None

def _table_key(source, row):
    m = row['meta']
    return m['table'] if source in SOTAB_SOURCES else m['csv']

def _select_by_target_rank(pools, tasks, budgets, source, seed, reverse=False):
    mode = 'target_rank_rev' if reverse else 'target_rank'
    non = [t for t in tasks if t not in ('cta', 'cpa')]
    if non:
        raise ValueError(f'select_mode={mode} supports only CTA/CPA tasks (one target column/pair per instance); regime tasks {non} have no such notion. Use a cta/cpa-only regime (single_cta, single_cpa, multi_ctacpa).')
    rows = []
    for slug in tasks:
        (pool, budget) = (pools[slug], budgets[slug])
        by_table = collections.defaultdict(list)
        for r in pool:
            by_table[_table_key(source, r)].append(r)
        keys = sorted(by_table.keys())
        random.Random(seed).shuffle(keys)
        rank_sign = 1 if reverse else -1
        keys.sort(key=lambda k: rank_sign * len(by_table[k]))
        order_desc = 'sparsest-first' if reverse else 'densest-first'
        by_label = collections.defaultdict(list)
        for r in pool:
            by_label[_cta_cpa_label(slug, source, r)].append(r)
        all_labels = set(by_label)
        if source in SOTAB_SOURCES:
            table_labels = {k: {_cta_cpa_label(slug, source, r) for r in by_table[k]} for k in keys}
            (covered, picked, picked_keys) = (set(), [], set())
            n_cover_tables = 0
            for k in keys:
                if table_labels[k] - covered:
                    picked.extend(by_table[k])
                    covered |= table_labels[k]
                    picked_keys.add(k)
                    n_cover_tables += 1
                    if covered == all_labels:
                        break
            n_cover = len(picked)
            n_fill_tables = 0
            for k in keys:
                if k in picked_keys:
                    continue
                sz = len(by_table[k])
                if len(picked) + sz > budget:
                    break
                picked.extend(by_table[k])
                picked_keys.add(k)
                n_fill_tables += 1
            over = '  [coverage alone exceeds budget]' if n_cover > budget else ''
            print(f'[{mode}/{slug}/{source}] budget={budget} -> coverage {n_cover} instances from {n_cover_tables} whole tables ({order_desc} greedy set-cover, all {len(all_labels)}/{len(all_labels)} labels) + {len(picked) - n_cover} density-fill instances from {n_fill_tables} tables -> total {len(picked)} instances{over}')
        else:
            (picked, n_tables) = ([], 0)
            for k in keys:
                sz = len(by_table[k])
                if len(picked) + sz > budget:
                    break
                picked.extend(by_table[k])
                n_tables += 1
            if not picked:
                raise ValueError(f"[{mode}/{slug}/{source}] budget={budget} < the first-ranked table's target-column count ({len(by_table[keys[0]])}); nothing selected. Increase --budget.")
            n_prefix = len(picked)
            covered = {_cta_cpa_label(slug, source, r) for r in picked}
            missing = sorted((lab for lab in all_labels if lab not in covered))
            if missing:
                cseed = int.from_bytes(hashlib.blake2b(f'{seed}:{slug}:target_rank_cover'.encode(), digest_size=8).digest(), 'big')
                crng = random.Random(cseed)
                picked_ids = {r['id'] for r in picked}
                for lab in missing:
                    cands = sorted((r for r in by_label[lab] if r['id'] not in picked_ids), key=lambda r: r['id'])
                    chosen = cands[crng.randrange(len(cands))]
                    picked.append(chosen)
                    picked_ids.add(chosen['id'])
            print(f'[{mode}/{slug}/{source}] budget={budget} -> {n_prefix} instances from {n_tables} {order_desc} tables (first-ranked has {len(by_table[keys[0]])} targets); +{len(missing)} coverage-completion draws for missing labels -> total {len(picked)} instances, labels {len(all_labels)}/{len(all_labels)} (complete)')
        rows.extend(picked)
    random.Random(seed).shuffle(rows)
    return rows

def _sample_task(slug, pool, n, rng, source):
    if len(pool) < n:
        raise ValueError(f'task {slug} ({source}) train_pool={len(pool)} < needed {n}')
    if slug == 'samerow':
        return SR.sample_balanced(pool, n, rng)
    if slug in ('cta', 'cpa'):
        return label_coverage_sample(pool, n, rng, label_of=lambda r: _cta_cpa_label(slug, source, r), name=f'{slug}/{source}')
    return rng.sample(pool, n)
_MD_SEP_RE = re.compile('^[\\s:|+-]+$')
_TOKEN_RE = re.compile('[a-z0-9]+')
_MH_PRIME = (1 << 61) - 1

def _table_tokens(text, cap=400):
    toks = set()
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith('|') or _MD_SEP_RE.match(s):
            continue
        for cell in s.strip('|').split('|'):
            for w in _TOKEN_RE.findall(cell.lower()):
                toks.add(w)
                if len(toks) >= cap:
                    return toks
    return toks

def _hash_token(t):
    return int.from_bytes(hashlib.blake2b(t.encode('utf-8', 'replace'), digest_size=4).digest(), 'big')

def _minhash_signatures(token_sets, num_perm, seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 1 << 32, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, 1 << 32, size=num_perm, dtype=np.uint64)
    sigs = np.zeros((len(token_sets), num_perm), dtype=np.uint64)
    for (i, toks) in enumerate(token_sets):
        if not toks:
            continue
        h = np.fromiter((_hash_token(t) for t in toks), dtype=np.uint64, count=len(toks))
        sigs[i] = ((a[None, :] * h[:, None] + b[None, :]) % _MH_PRIME).min(axis=0)
    return sigs

def _select_by_similarity(pools, tasks, budgets, source, mode, seed, num_perm=64, bands=16, rows_per_band=4):
    rng = random.Random(seed)
    (all_rows, task_of) = ([], [])
    for (ti, slug) in enumerate(tasks):
        for r in pools[slug]:
            all_rows.append(r)
            task_of.append(ti)
    N = len(all_rows)
    task_of = np.array(task_of, dtype=np.int64)

    def labelled(slug):
        return slug in ('cta', 'cpa')
    token_sets = [_table_tokens(r['input']) for r in all_rows]
    sigs = _minhash_signatures(token_sets, num_perm, seed)
    idx_bucket_members = [[] for _ in range(N)]
    for band in range(bands):
        cols = sigs[:, band * rows_per_band:(band + 1) * rows_per_band]
        keys = [cols[i].tobytes() for i in range(N)]
        buckets = collections.defaultdict(list)
        for (i, k) in enumerate(keys):
            buckets[k].append(i)
        arrs = {k: np.array(v, dtype=np.int64) for (k, v) in buckets.items()}
        for (i, k) in enumerate(keys):
            idx_bucket_members[i].append(arrs[k])
    if mode == 'similar':
        (w_sign, c_sign) = (-1.0, -1.0)
    elif mode == 'dissimilar':
        (w_sign, c_sign) = (1.0, 1.0)
    else:
        raise ValueError(f'unknown similarity mode: {mode}')
    selected_mask = np.zeros(N, dtype=bool)
    coll_within = np.zeros(N, dtype=np.int64)
    coll_cross = np.zeros(N, dtype=np.int64)
    task_count = [0] * len(tasks)
    selected = []

    def add(i):
        selected_mask[i] = True
        ti = task_of[i]
        task_count[ti] += 1
        selected.append(i)
        for arr in idx_bucket_members[i]:
            ta = task_of[arr]
            coll_within[arr[ta == ti]] += 1
            coll_cross[arr[ta != ti]] += 1
    for (ti, slug) in enumerate(tasks):
        if not labelled(slug):
            continue
        by_label = collections.defaultdict(list)
        for i in range(N):
            if task_of[i] == ti:
                by_label[_cta_cpa_label(slug, source, all_rows[i])].append(i)
        if budgets[slug] < len(by_label):
            raise ValueError(f'[{slug}/{source}] budget={budgets[slug]} < distinct labels in train pool ({len(by_label)}); cannot guarantee every label appears >=1 (select_mode={mode}). Increase the per-task budget to >= {len(by_label)}.')
        for (_lab, idxs) in by_label.items():
            add(idxs[rng.randrange(len(idxs))])
    prio = np.fromiter((rng.random() for _ in range(N)), dtype=np.float64, count=N) * 0.5
    budget_arr = np.array([budgets[t] for t in tasks], dtype=np.int64)
    total_target = int(budget_arr.sum())
    while len(selected) < total_target:
        counts = np.array(task_count, dtype=np.int64)
        task_full = (counts >= budget_arr)[task_of]
        avail = ~selected_mask & ~task_full
        if not avail.any():
            break
        key = w_sign * coll_within.astype(np.float64) + c_sign * coll_cross.astype(np.float64) + prio
        key[~avail] = np.inf
        add(int(np.argmin(key)))
    rows = [all_rows[i] for i in selected]
    rng.shuffle(rows)
    return rows

def _select_shared_ctacpa(pools, tasks, budgets, source, seed, select_mode):
    cta_ids = {_shared_table_id(source, r) for r in pools['cta']}
    cpa_ids = {_shared_table_id(source, r) for r in pools['cpa']}
    shared = cta_ids & cpa_ids
    if not shared:
        raise ValueError(f'[shared_ctacpa/{source}] no tables annotated by BOTH CTA and CPA; check that --cta_cpa_source ({source}) points at the full CTA/CPA pools.')
    rows = []
    for slug in tasks:
        full_pool = pools[slug]
        budget = budgets[slug]
        label_of = lambda r, s=slug: _cta_cpa_label(s, source, r)
        shared_pool = [r for r in full_pool if _shared_table_id(source, r) in shared]
        L_full = {label_of(r) for r in full_pool}
        shared_by_label = collections.defaultdict(list)
        for r in shared_pool:
            shared_by_label[label_of(r)].append(r)
        L_shared = set(shared_by_label)
        missing = sorted(L_full - L_shared)
        full_by_label = collections.defaultdict(list)
        for r in full_pool:
            full_by_label[label_of(r)].append(r)
        rng = random.Random(seed)
        (chosen, chosen_ids) = ([], set())

        def draw_one(cands):
            avail = [r for r in cands if r['id'] not in chosen_ids]
            if not avail:
                return
            pick = avail[rng.randrange(len(avail))]
            chosen.append(pick)
            chosen_ids.add(pick['id'])
        for lab in missing:
            draw_one(full_by_label[lab])
        for lab in sorted(L_shared):
            draw_one(shared_by_label[lab])
        n_cover = len(chosen)
        if select_mode == 'all':
            for r in shared_pool:
                if r['id'] not in chosen_ids:
                    chosen.append(r)
                    chosen_ids.add(r['id'])
        elif select_mode == 'random':
            if budget < len(L_full):
                raise ValueError(f'[shared_ctacpa/{source}/{slug}] budget {budget} < distinct labels {len(L_full)}; cannot keep full label coverage. Increase the per-task budget to >= {len(L_full)}.')
            rest = [r for r in shared_pool if r['id'] not in chosen_ids]
            rng.shuffle(rest)
            need = budget - len(chosen)
            fill = rest[:need]
            chosen.extend(fill)
            chosen_ids.update((r['id'] for r in fill))
            if len(chosen) < budget:
                extra = [r for r in full_pool if r['id'] not in chosen_ids]
                rng.shuffle(extra)
                chosen.extend(extra[:budget - len(chosen)])
        else:
            raise ValueError(f"[shared_ctacpa] select_mode '{select_mode}' is not supported with regime multi_shared_ctacpa; use --select_mode random or all.")
        n_from_shared = sum((1 for r in chosen if _shared_table_id(source, r) in shared))
        print(f'[shared_ctacpa/{source}/{slug}] shared_tables={len(shared)} shared_pool={len(shared_pool)}/{len(full_pool)}; labels={len(L_full)} (missing_from_shared={len(missing)} -> 1 each from non-shared); selected={len(chosen)} (from_shared={n_from_shared}, non_shared={len(chosen) - n_from_shared}); coverage-slots={n_cover}')
        rows.extend(chosen)
    random.Random(seed).shuffle(rows)
    return rows

def _dedicated_cover(labels, label_to_tables, available, rng):
    order = sorted(labels, key=lambda L: (len(label_to_tables.get(L, ())), L))
    assign = {}
    for L in order:
        cands = [t for t in label_to_tables.get(L, ()) if t in available]
        if not cands:
            raise ValueError(f'[paired_ctacpa] cannot cover label {L!r}: no unused shared table carries it (carriers={len(label_to_tables.get(L, ()))}). Increase budget or check the pool.')
        t = cands[rng.randrange(len(cands))]
        assign[L] = t
        available.discard(t)
    return assign

def _select_paired_ctacpa(pools, tasks, budgets, source, seed):
    if set(tasks) != {'cta', 'cpa'}:
        raise ValueError(f'[paired_ctacpa] regime task list must be exactly cta+cpa, got {tasks}')
    if budgets['cta'] != budgets['cpa']:
        raise ValueError(f"[paired_ctacpa] per-task budgets must be equal (even --budget); got cta={budgets['cta']} cpa={budgets['cpa']}.")
    n_groups = budgets['cta']
    cta_ids = {_shared_table_id(source, r) for r in pools['cta']}
    cpa_ids = {_shared_table_id(source, r) for r in pools['cpa']}
    shared = cta_ids & cpa_ids
    if not shared:
        raise ValueError(f'[paired_ctacpa/{source}] no tables annotated by BOTH CTA and CPA; check that --cta_cpa_source ({source}) points at the full CTA/CPA pools.')
    if len(shared) < n_groups:
        raise ValueError(f'[paired_ctacpa/{source}] shared tables ({len(shared)}) < needed groups ({n_groups}); lower --budget.')
    tbl_rows = {'cta': collections.defaultdict(list), 'cpa': collections.defaultdict(list)}
    lab_tbls = {'cta': collections.defaultdict(set), 'cpa': collections.defaultdict(set)}
    L_full = {}
    for slug in ('cta', 'cpa'):
        label_of = lambda r, s=slug: _cta_cpa_label(s, source, r)
        labs = set()
        for r in pools[slug]:
            tid = _shared_table_id(source, r)
            if tid in shared:
                tbl_rows[slug][tid].append(r)
                lab_tbls[slug][label_of(r)].add(tid)
            labs.add(label_of(r))
        L_full[slug] = labs
    n_cover = len(L_full['cta']) + len(L_full['cpa'])
    if n_groups < n_cover:
        raise ValueError(f"[paired_ctacpa/{source}] budget/2 ({n_groups}) < labels to cover ({len(L_full['cta'])} CTA + {len(L_full['cpa'])} CPA = {n_cover}); raise --budget to >= {2 * n_cover}.")
    rng = random.Random(seed)
    available = set(shared)
    sel = {}
    cta_cov = _dedicated_cover(L_full['cta'], lab_tbls['cta'], available, rng)
    for (L, tid) in cta_cov.items():
        rows_L = [r for r in tbl_rows['cta'][tid] if _cta_cpa_label('cta', source, r) == L]
        sel.setdefault(tid, {'cta': None, 'cpa': None})['cta'] = rows_L[rng.randrange(len(rows_L))]
    cpa_cov = _dedicated_cover(L_full['cpa'], lab_tbls['cpa'], available, rng)
    for (L, tid) in cpa_cov.items():
        rows_L = [r for r in tbl_rows['cpa'][tid] if _cta_cpa_label('cpa', source, r) == L]
        sel.setdefault(tid, {'cta': None, 'cpa': None})['cpa'] = rows_L[rng.randrange(len(rows_L))]
    fill_pool = list(available)
    rng.shuffle(fill_pool)
    for tid in fill_pool:
        if len(sel) >= n_groups:
            break
        sel.setdefault(tid, {'cta': None, 'cpa': None})
    (rows, missing_side) = ([], 0)
    for (tid, g) in sel.items():
        cta_r = g['cta'] if g['cta'] is not None else tbl_rows['cta'][tid][rng.randrange(len(tbl_rows['cta'][tid]))]
        cpa_r = g['cpa'] if g['cpa'] is not None else tbl_rows['cpa'][tid][rng.randrange(len(tbl_rows['cpa'][tid]))]
        for r in (cta_r, cpa_r):
            r = dict(r)
            r['meta'] = dict(r['meta'])
            r['meta']['pair_group'] = tid
            rows.append(r)
    for slug in ('cta', 'cpa'):
        got = {_cta_cpa_label(slug, source, r) for r in rows if r['task'] == slug}
        miss = L_full[slug] - got
        if miss:
            raise ValueError(f'[paired_ctacpa/{source}/{slug}] label coverage lost: {sorted(miss)}')
    n_from_shared = sum((1 for r in rows if _shared_table_id(source, r) in shared))
    print(f"[paired_ctacpa/{source}] shared_tables={len(shared)} groups={len(sel)} (cover={n_cover}: {len(L_full['cta'])} CTA + {len(L_full['cpa'])} CPA, fill={len(sel) - n_cover}); rows={len(rows)} (cta={sum((1 for r in rows if r['task'] == 'cta'))} cpa={sum((1 for r in rows if r['task'] == 'cpa'))}, from_shared={n_from_shared})")
    return rows
_QKV_MODULES = ('q_proj', 'k_proj', 'v_proj')

def _read_adapter_state_dict(adapter_dir):
    st = os.path.join(adapter_dir, 'adapter_model.safetensors')
    if os.path.exists(st):
        from safetensors.torch import load_file
        return load_file(st)
    bin_ = os.path.join(adapter_dir, 'adapter_model.bin')
    if os.path.exists(bin_):
        return torch.load(bin_, map_location='cpu')
    raise SystemExit(f'--init_qkv_adapter {adapter_dir} has no adapter_model.safetensors/.bin')

def init_qkv_from_struct_adapter(model, adapter_dir, lora_cfg):
    cfg_path = os.path.join(adapter_dir, 'adapter_config.json')
    if not os.path.exists(cfg_path):
        raise SystemExit(f'--init_qkv_adapter {adapter_dir} has no adapter_config.json; it does not look like a PEFT LoRA adapter directory.')
    cfg = json.load(open(cfg_path))
    if cfg.get('peft_type') not in (None, 'LORA'):
        raise SystemExit(f"--init_qkv_adapter is peft_type={cfg.get('peft_type')}, not LORA.")
    src_mods = set(cfg.get('target_modules') or ())
    if not src_mods or not src_mods <= set(_QKV_MODULES):
        raise SystemExit(f"--init_qkv_adapter target_modules={sorted(src_mods)}; the structural stage must adapt only {list(_QKV_MODULES)}. Refusing: importing o_proj/MLP weights would make the downstream run something other than 'fresh O/MLP + structural QKV'.")
    for (key, mine) in (('r', lora_cfg.r), ('lora_alpha', lora_cfg.lora_alpha), ('use_rslora', bool(getattr(lora_cfg, 'use_rslora', False)))):
        theirs = cfg.get(key, False if key == 'use_rslora' else None)
        if key == 'use_rslora':
            theirs = bool(theirs)
        if theirs != mine:
            raise SystemExit(f'--init_qkv_adapter {key}={theirs} but this run uses {key}={mine}. LoRA applies scaling=lora_alpha/r, so copying A/B across different scaling silently changes the delta the structural weights encode. Match them or retrain stage 1.')
    sd = _read_adapter_state_dict(adapter_dir)
    live = dict(model.named_parameters())
    (copied, nonzero_B, seen_B) = ([], 0, 0)
    with torch.no_grad():
        for (k, v) in sd.items():
            if 'lora_A' not in k and 'lora_B' not in k:
                raise SystemExit(f'--init_qkv_adapter holds a non-LoRA tensor {k}; refusing.')
            if not any((f'.{m}.' in k for m in _QKV_MODULES)):
                raise SystemExit(f'--init_qkv_adapter holds a non-QKV tensor {k}; refusing.')
            cands = [k] + [k.replace(f'{t}.weight', f'{t}.default.weight') for t in ('lora_A', 'lora_B') if f'{t}.weight' in k]
            name = next((c for c in cands if c in live), None)
            if name is None:
                raise SystemExit(f'--init_qkv_adapter tensor {k} has no counterpart in the downstream model (tried {cands}). Is it the same base model?')
            p = live[name]
            if tuple(p.shape) != tuple(v.shape):
                raise SystemExit(f'shape mismatch for {name}: downstream {tuple(p.shape)} vs structural {tuple(v.shape)}')
            p.copy_(v.to(dtype=p.dtype, device=p.device))
            copied.append(name)
            if 'lora_B' in k:
                seen_B += 1
                nonzero_B += int(bool(v.count_nonzero().item()))
    if seen_B == 0:
        raise SystemExit('--init_qkv_adapter contains no lora_B tensors; refusing.')
    if nonzero_B == 0:
        raise SystemExit("--init_qkv_adapter has ALL-ZERO lora_B: that is PEFT's fresh initialization, i.e. an adapter that never trained. It would be indistinguishable from the baseline. Refusing.")
    want = {n for n in live if 'lora_' in n and any((f'.{m}.' in n for m in src_mods))}
    missing = sorted(want - set(copied))
    if missing:
        raise SystemExit(f'{len(missing)} LoRA tensors of {sorted(src_mods)} were NOT initialized from the structural adapter (e.g. {missing[:3]}); they would still hold the fresh initialization.')
    sha = hashlib.sha256()
    for k in sorted(sd):
        sha.update(k.encode())
        sha.update(sd[k].float().cpu().numpy().tobytes())
    info = {'init_qkv_adapter': os.path.abspath(adapter_dir), 'structural_target_modules': sorted(src_mods), 'tensors_copied': len(copied), 'lora_B_tensors': seen_B, 'lora_B_nonzero': nonzero_B, 'structural_weights_sha256': sha.hexdigest(), 'structural_r': cfg.get('r'), 'structural_lora_alpha': cfg.get('lora_alpha'), 'structural_manifest': json.load(open(os.path.join(adapter_dir, 'struct_manifest.json'))).get('build_stats') if os.path.exists(os.path.join(adapter_dir, 'struct_manifest.json')) else None}
    fresh = sorted(set(_QKV_MODULES) - src_mods)
    print(f"[init_qkv_adapter] copied {len(copied)} tensors from {adapter_dir} ({seen_B} lora_B, {nonzero_B} of them non-zero) into {sorted(src_mods)}; {fresh + ['o_proj', 'gate_proj', 'up_proj', 'down_proj']} keep PEFT's fresh initialization and ALL seven modules stay trainable.")
    return info

def merge_adapter_into_base(model, adapter_dir):
    cfg_path = os.path.join(adapter_dir, 'adapter_config.json')
    if not os.path.exists(cfg_path):
        raise SystemExit(f'--merge_adapter {adapter_dir} has no adapter_config.json; it does not look like a PEFT LoRA adapter directory.')
    cfg = json.load(open(cfg_path))
    if cfg.get('peft_type') not in (None, 'LORA'):
        raise SystemExit(f"--merge_adapter is peft_type={cfg.get('peft_type')}, not LORA.")
    sd = _read_adapter_state_dict(adapter_dir)
    b_tensors = [k for k in sd if 'lora_B' in k]
    if not b_tensors:
        raise SystemExit(f'--merge_adapter {adapter_dir} contains no lora_B tensors; refusing.')
    nonzero_B = sum((int(bool(sd[k].count_nonzero().item())) for k in b_tensors))
    if nonzero_B == 0:
        raise SystemExit(f"--merge_adapter {adapter_dir} has ALL-ZERO lora_B: that is PEFT's fresh initialization, i.e. an adapter that never trained. Merging it is a no-op and the run would be indistinguishable from the baseline. Refusing.")
    sha = hashlib.sha256()
    for k in sorted(sd):
        sha.update(k.encode())
        sha.update(sd[k].float().cpu().numpy().tobytes())
    from peft import PeftModel
    tmods = set(cfg.get('target_modules') or ())
    before = {}
    for (n, prm) in model.named_parameters():
        parts = n.split('.')
        if parts[-1] != 'weight' or len(parts) < 2:
            continue
        owner = parts[-3] if parts[-2] == 'base_layer' and len(parts) >= 3 else parts[-2]
        if owner in tmods:
            before[n] = prm.detach().clone()
    if not before:
        print(f'[merge_adapter][warn] found no base weights named after {sorted(tmods)} on this model, so the post-merge change check is skipped.')
    merged = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    live = dict(merged.named_parameters())
    moved = sum((1 for (n, w) in before.items() if n in live and (not torch.equal(live[n].detach(), w.to(live[n].device)))))
    if before and moved == 0:
        raise SystemExit(f'--merge_adapter {adapter_dir}: merge_and_unload() changed NONE of the {len(before)} target-module weights. The adapter did not reach the base model; refusing to fine-tune on a base that is secretly untouched.')
    st_manifest = os.path.join(adapter_dir, 'struct_manifest.json')
    smf = json.load(open(st_manifest)) if os.path.exists(st_manifest) else {}
    info = {'merge_adapter': os.path.abspath(adapter_dir), 'merged_target_modules': sorted(cfg.get('target_modules') or ()), 'merged_r': cfg.get('r'), 'merged_lora_alpha': cfg.get('lora_alpha'), 'merged_use_rslora': bool(cfg.get('use_rslora', False)), 'merged_weights_sha256': sha.hexdigest(), 'lora_B_tensors': len(b_tensors), 'lora_B_nonzero': nonzero_B, 'base_weights_changed': moved, 'base_weights_checked': len(before), 'pretrain_stage': smf.get('stage'), 'pretrain_task': smf.get('task'), 'pretrain_table_first': smf.get('table_first'), 'pretrain_lora_table_only': smf.get('lora_table_only'), 'pretrain_build_stats': smf.get('build_stats'), 'pretrain_final_train_loss': smf.get('final_train_loss'), 'masked_delta_is_now_unmasked': bool(smf.get('lora_table_only'))}
    print(f"[merge_adapter] merged {adapter_dir} into the base weights: {sorted(cfg.get('target_modules') or ())} @ r={cfg.get('r')} alpha={cfg.get('lora_alpha')}, {nonzero_B}/{len(b_tensors)} lora_B non-zero, {moved}/{len(before)} base tensors changed.")
    if smf.get('lora_table_only'):
        print(f'[merge_adapter] NOTE: that adapter was TRAINED with --lora_table_only (delta only at table tokens). Merged into W0 it now applies at EVERY position -- that is the design of this pipeline, and it is recorded in the manifest.')
    print(f'[merge_adapter] stage 2 is now an ORDINARY vanilla LoRA run over this merged base.')
    return (merged, info)

def build_train_rows(regime, budget, seed, source='sotab', select_mode='random', drop_empty_target=False, drop_truncated_tables=False, samerow_source='auto', samerow_max_rows=20, samerow_tok=None, samerow_max_len=None, samerow_model=None):
    rng = random.Random(seed)
    pools = {}

    def _load_pool(slug):
        if slug == 'samerow':
            return load_samerow_pool('train', samerow_partner(regime, samerow_source), source, samerow_max_rows, tok=samerow_tok, max_len=samerow_max_len, model_name=samerow_model)
        p = read_jsonl(f'{DATA}/{data_dir(slug, source)}/train_pool.jsonl')
        if drop_empty_target:
            kept = [r for r in p if not E.target_is_empty(slug, r['input'])]
            if len(kept) != len(p):
                print(f'[drop_empty_target] pool/{slug}: dropped {len(p) - len(kept)}/{len(p)} empty-target examples before selection')
            p = kept
        if drop_truncated_tables:
            kept = [r for r in p if not r.get('meta', {}).get('truncated')]
            if len(kept) != len(p):
                print(f'[drop_truncated_tables] pool/{slug}: dropped {len(p) - len(kept)}/{len(p)} prep-truncated examples before selection')
            p = kept
        return p
    for slug in TASK_SLUGS:
        p = _load_pool(slug)
        rng.shuffle(p)
        pools[slug] = p
    post_canonical_state = rng.getstate()
    if regime in REGIME_SINGLE:
        tasks = [REGIME_SINGLE[regime]]
        budgets = {tasks[0]: budget}
    elif regime in REGIME_MULTI:
        tasks = list(REGIME_MULTI[regime])
        if regime in REGIME_WEIGHTS:
            budgets = weighted_budgets(tasks, REGIME_WEIGHTS[regime], budget)
            w = REGIME_WEIGHTS[regime]
            print(f'[{regime}] weighted budget split ' + '  '.join((f'{t}={budgets[t]}(w{w[t]})' for t in tasks)) + f'  total={sum(budgets.values())}')
        else:
            (base, rem) = divmod(budget, len(tasks))
            budgets = {slug: base + (1 if i < rem else 0) for (i, slug) in enumerate(tasks)}
    else:
        raise ValueError(regime)
    for slug in tasks:
        if slug not in pools:
            p = _load_pool(slug)
            rng.setstate(post_canonical_state)
            rng.shuffle(p)
            pools[slug] = p
    if regime == 'multi_shared_ctacpa':
        rows = _select_shared_ctacpa(pools, tasks, budgets, source, seed, select_mode)
    elif regime == 'multi_paired_ctacpa':
        rows = _select_paired_ctacpa(pools, tasks, budgets, source, seed)
    elif select_mode == 'random':
        rows = []
        for slug in tasks:
            trng = random.Random(seed)
            rows += _sample_task(slug, pools[slug], budgets[slug], trng, source)
        random.Random(seed).shuffle(rows)
    elif select_mode == 'all':
        rows = []
        for slug in tasks:
            rows += pools[slug]
        random.Random(seed).shuffle(rows)
    elif select_mode in ('target_rank', 'target_rank_rev'):
        rows = _select_by_target_rank(pools, tasks, budgets, source, seed, reverse=select_mode == 'target_rank_rev')
    elif select_mode in ('similar', 'dissimilar'):
        rows = _select_by_similarity(pools, tasks, budgets, source, select_mode, seed)
    else:
        raise ValueError(f'unknown select_mode: {select_mode}')
    return rows

def select_val_tasks(regime):
    tasks = [REGIME_SINGLE[regime]] if regime in REGIME_SINGLE else list(REGIME_MULTI[regime])
    val = [t for t in tasks if t != SR.SLUG]
    if not val:
        raise ValueError(f'regime {regime} trains only on the synthesized samerow task, which has no validation split, so there is no signal to select a checkpoint on. Use a multi_*_samerow regime (selected on its table task) instead.')
    return val

def _encode_prompt(tok, text, model_name):
    if has_chat_template(tok):
        return apply_model_chat_template(tok, [{'role': 'user', 'content': text}], model_name=model_name, tokenize=True, add_generation_prompt=True)
    return tok(text, add_special_tokens=True)['input_ids']

def _trim_table_rows(text, tok, target_len, max_len, model_name):
    lines = text.split('\n')
    prompt_ids = _encode_prompt(tok, text, model_name)
    while len(prompt_ids) + target_len > max_len:
        tbl = [i for (i, l) in enumerate(lines) if l.lstrip().startswith('|')]
        if len(tbl) <= 3:
            break
        del lines[tbl[-1]]
        text = '\n'.join(lines)
        prompt_ids = _encode_prompt(tok, text, model_name)
    return (text, prompt_ids)

def canonical_task(slug):
    s = str(slug).lower()
    for base in ('cta', 'cpa', 'cea', 'hvm', 'em', 'tabfact', 'samerow', 'wikiqa', 'finqa', 'tablebench'):
        if s == base or s.startswith(base + '_'):
            return base
    return s

def with_task_prefix(content, slug, enabled):
    if not enabled:
        return content
    label = canonical_task(slug)
    if not label:
        return content
    return f'[TASK: {label}]\n{content}'

def tokenize_example(tok, row, max_len, model_name, task_prefix=False, want_roles=False, want_regions=None, want_copy=False, sandwich=False, want_soft=False, want_struct=False):
    if want_struct:
        if want_copy or want_soft:
            raise ValueError('want_struct cannot be combined with want_copy / want_soft: those options change the prompt or the token sequence the coordinates were computed on.')
    target_ids = tok(row['target'], add_special_tokens=False)['input_ids']
    if tok.eos_token_id is not None:
        target_ids = target_ids + [tok.eos_token_id]
    content = with_task_prefix(row['input'], row.get('task'), task_prefix)
    prompt_ids = _encode_prompt(tok, content, model_name)
    trimmed = False
    if len(prompt_ids) + len(target_ids) > max_len:
        (content, prompt_ids) = _trim_table_rows(content, tok, len(target_ids), max_len, model_name)
        trimmed = True
    full_ids = prompt_ids + target_ids
    if len(full_ids) > max_len:
        return None
    if sandwich:
        (content, applied) = TSW.sandwich_row_text(row.get('task'), content)
        if applied:
            prompt_ids = _encode_prompt(tok, content, model_name)
            full_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if len(labels) != len(full_ids):
        raise RuntimeError(f'Label/input length mismatch: labels={len(labels)}, input_ids={len(full_ids)}')
    if all((x == -100 for x in labels)):
        raise RuntimeError(f"No supervised target tokens for row id={row.get('id')}")
    soft_ids = None
    soft_absorbed = 0
    if want_soft:
        _sp_placeholder = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        if _sp_placeholder is None:
            raise RuntimeError('--soft_role_prompt needs a pad or eos id to hold the inserted slots; this tokenizer has neither.')
        _sp_rendered = format_generation_prompt(tok, content, model_name)
        (_sp_ids, _sp_points, soft_absorbed) = SP.insertion_points(tok, row['input'], _sp_rendered, row.get('task'), expect_ids=prompt_ids)
        (full_ids, labels, soft_ids) = SP.insert(full_ids, labels, _sp_points, _sp_placeholder)
        if len(soft_ids) != len(full_ids) or len(labels) != len(full_ids):
            raise RuntimeError(f'soft-role expansion length mismatch: ids={len(full_ids)}, labels={len(labels)}, slots={len(soft_ids)}')
    if want_regions is None:
        want_regions = want_roles or want_copy
    if not (want_roles or want_regions or want_copy or want_struct):
        if want_soft:
            return (full_ids, labels, trimmed, None, None, None, (soft_ids, soft_absorbed))
        return (full_ids, labels, trimmed)
    rendered = format_generation_prompt(tok, content, model_name)
    role_ids = None
    if want_roles:
        prompt_roles = RGM.build_roles(tok, row['input'], rendered, row.get('task'), expect_ids=prompt_ids)
        role_ids = prompt_roles + [RGM.ROLE_NONE] * len(target_ids)
        if len(role_ids) != len(full_ids):
            raise RuntimeError(f'role/input length mismatch: roles={len(role_ids)}, input_ids={len(full_ids)}')
    region_ids = None
    if want_regions:
        prompt_regions = RCV.build_regions(tok, content, rendered, expect_ids=prompt_ids)
        region_ids = prompt_regions + [RCV.R_POST] * len(target_ids)
        if len(region_ids) != len(full_ids):
            raise RuntimeError(f'region/input length mismatch: regions={len(region_ids)}, input_ids={len(full_ids)}')
    struct_ids = None
    if want_struct:
        prompt_struct = SRB.build_struct(tok, rendered, expect_ids=prompt_ids)
        struct_ids = prompt_struct + [SRB.STRUCT_NONE] * len(target_ids)
        if len(struct_ids) != len(full_ids):
            raise RuntimeError(f'struct/input length mismatch: struct={len(struct_ids)}, input_ids={len(full_ids)}')
    if not want_copy:
        if want_soft:
            return (full_ids, labels, trimmed, role_ids, region_ids, None, (soft_ids, soft_absorbed))
        if want_struct:
            return (full_ids, labels, trimmed, role_ids, region_ids, struct_ids)
        return (full_ids, labels, trimmed, role_ids, region_ids)
    prompt_copies = TSW.build_copy_ids(tok, content, rendered, row.get('task'), expect_ids=prompt_ids)
    copy_ids = prompt_copies + [TSW.COPY_NONE] * len(target_ids)
    if len(copy_ids) != len(full_ids):
        raise RuntimeError(f'copy/input length mismatch: copies={len(copy_ids)}, input_ids={len(full_ids)}')
    if want_soft:
        return (full_ids, labels, trimmed, role_ids, region_ids, copy_ids, (soft_ids, soft_absorbed))
    return (full_ids, labels, trimmed, role_ids, region_ids, copy_ids)
_SOTAB_RAW_DIR = {'cta': 'CTA', 'cpa': 'CPA'}
_row_trunc_raw_cache = {}
_row_trunc_cache = {}
_covcol_orders_cache = {}

def _get_covcol_orders(oc_key, raw, seed, table_id, ncols, need_k):
    cached = _covcol_orders_cache.get(oc_key)
    if cached is not None and cached[1] >= need_k:
        return cached[0]
    cap = max(64, 1 << max(0, int(need_k) - 1).bit_length())
    orders = TT._coverage_col_orders(raw, seed, table_id, list(range(ncols)), max_keep=cap)
    _covcol_orders_cache[oc_key] = (orders, cap)
    return orders

def _galloping_max_fit(render_fn, hi_bound):
    (c1, ok1) = render_fn(1)
    if not ok1 or hi_bound <= 1:
        return (c1, 1)
    (best_c, best_k) = (c1, 1)
    k = 2
    while k <= hi_bound:
        (c, ok) = render_fn(k)
        if ok:
            (best_c, best_k) = (c, k)
            k *= 2
        else:
            break
    (lo, hi) = (best_k + 1, min(k, hi_bound + 1) - 1)
    while lo <= hi:
        mid = (lo + hi) // 2
        (c, ok) = render_fn(mid)
        if ok:
            (best_c, best_k) = (c, mid)
            lo = mid + 1
        else:
            hi = mid - 1
    return (best_c, best_k)
_HEAD_COL_RE = re.compile('Head column:\\s*col(\\d+)')
_TAIL_COL_RE = re.compile('Tail column:\\s*col(\\d+)')

def _reset_row_trunc_caches():
    _row_trunc_raw_cache.clear()
    _row_trunc_cache.clear()
    _covcol_orders_cache.clear()
    _render_block_cache.clear()
    TT._col_docs_cache.clear()
    _coverage_stats.clear()

def _target_positions(task, meta, col_names, prompt):

    def pos(colid):
        name = f'col{int(colid)}'
        return col_names.index(name) if name in col_names else None
    if task == 'cta':
        p = pos(meta['col'])
        return None if p is None else [p]
    if 'col_a' in meta and 'col_b' in meta:
        (head_id, tail_id) = (meta['col_a'], meta['col_b'])
    else:
        (mh, mt) = (_HEAD_COL_RE.search(prompt), _TAIL_COL_RE.search(prompt))
        if not (mh and mt):
            return None
        (head_id, tail_id) = (mh.group(1), mt.group(1))
    (ph, pt) = (pos(head_id), pos(tail_id))
    if ph is None or pt is None:
        return None
    return [ph, pt]

def _load_raw_table(source, task, meta, split='train'):
    import prepare_data as _PD
    if source in SOTAB_SOURCES:
        sub = {'train': ('train', 'Train'), 'val': ('val', 'Validation'), 'test': ('test', 'Test')}[split]
        path = _PD.SOTAB / _SOTAB_RAW_DIR[task] / sub[0] / sub[1] / meta['table']
        (cols, rows) = _PD.load_sotab_table(path, 60)
        return ('sotab', [f'col{c}' for c in cols], rows, None)
    import prepare_t2d as _PT
    df = _PT.load_table(meta['round'], meta['csv'])
    if df is None:
        raise FileNotFoundError(f"t2d raw table missing: r{meta['round']}/{meta['csv']}")
    return ('t2d', list(df.columns), df.values.tolist(), df)

def _render_table_block(kind, col_names, rows, idx, df, keep_cols=None, reindex=False):
    import prepare_data as _PD
    if kind == 'sotab':
        if keep_cols is None:
            sel = [rows[i] for i in idx]
            return _PD.render_md(sel, col_names, len(sel), 60)
        sub_cols = [f'col{i}' for i in range(len(keep_cols))] if reindex else [col_names[c] for c in keep_cols]
        sel = [[rows[i][c] for c in keep_cols] for i in idx]
        return _PD.render_md(sel, sub_cols, len(sel), 60)
    if keep_cols is None:
        return df.iloc[idx].to_markdown(index=False)
    sub = df.iloc[idx].iloc[:, keep_cols]
    if reindex:
        sub = sub.copy()
        sub.columns = [f'col{i}' for i in range(len(keep_cols))]
    return sub.to_markdown(index=False)

def _reindex_marker(task, text, keep_cols, target_pos):
    new_pos = {p: i for (i, p) in enumerate(keep_cols)}
    if task == 'cpa':
        text = E._sub_marker(text, E._CPA_HEAD_RE, f'col{new_pos[target_pos[0]]}')
        return E._sub_marker(text, E._CPA_TAIL_RE, f'col{new_pos[target_pos[1]]}')
    return E._sub_marker(text, E._CTA_TARGET_RE, f'col{new_pos[target_pos[0]]}')

def _render_block_cached(source, task, table_id, kind, col_names, rows, idx, df, split='train', keep_cols=None, reindex=False):
    key = (source, task, table_id, split, tuple(idx), tuple(keep_cols) if keep_cols is not None else None, bool(reindex))
    block = _render_block_cache.get(key)
    if block is None:
        block = _render_table_block(kind, col_names, rows, idx, df, keep_cols, reindex)
        _render_block_cache[key] = block
    return block
_coverage_stats = collections.Counter()
_render_block_cache = {}
_SOTAB_FULL_MAXROWS = 1000000000
_AUTO_HEADFIT_SOURCES = ('sotab_full', 't2d')

def _row_trunc_plan(source, max_table_rows, row_trunc_method, fill_budget=False):
    if max_table_rows and max_table_rows > 0:
        return (row_trunc_method, max_table_rows)
    if fill_budget:
        return (row_trunc_method, _SOTAB_FULL_MAXROWS)
    if source in _AUTO_HEADFIT_SOURCES:
        return ('head', _SOTAB_FULL_MAXROWS)
    return None
_MIN_PARALLEL_TABLES = 128

def _load_raw_worker(job):
    (raw_key, source, task, meta, split) = job
    try:
        return (raw_key, _load_raw_table(source, task, meta, split), None)
    except Exception as e:
        return (raw_key, None, f'{type(e).__name__}: {e}')

def _prefetch_raw_tables(rows, source, split, num_proc):
    if not num_proc or num_proc <= 1:
        return
    jobs = {}
    for r in rows:
        if r.get('task') not in ('cta', 'cpa'):
            continue
        raw_key = (split, r['task'], _shared_table_id(source, r))
        if raw_key in _row_trunc_raw_cache or raw_key in jobs:
            continue
        jobs[raw_key] = (raw_key, source, r['task'], r['meta'], split)
    if len(jobs) < _MIN_PARALLEL_TABLES:
        return
    import concurrent.futures as _cf
    import multiprocessing as _mp
    ctx = _mp.get_context('spawn')
    n_fail = 0
    with _cf.ProcessPoolExecutor(max_workers=min(num_proc, len(jobs)), mp_context=ctx) as ex:
        for (raw_key, loaded, err) in tqdm(ex.map(_load_raw_worker, list(jobs.values())), total=len(jobs), desc=f'raw_prefetch[{split},{num_proc}p]', unit='tbl'):
            _row_trunc_raw_cache[raw_key] = loaded
            if loaded is None and n_fail < 5:
                print(f'[row_trunc] WARN prefetch {raw_key}: {err}')
                n_fail += 1

def apply_row_truncation(rows, source, method, max_rows, seed, tok, max_len, model_name, task_prefix=False, split='train', hard_cap=False, max_context_cols=0, reindex_cols=False, drop_empty_context_cols=False, col_trunc_method='nearest'):
    n_trunc = n_shrink = n_fail = n_skip = n_full = 0
    drop_empty_on = bool(drop_empty_context_cols)
    n_empty_dropped = 0
    n_inst_empty_dropped = 0
    col_on = bool(max_context_cols and max_context_cols > 0 or drop_empty_on)
    reindex_on = bool(col_on and reindex_cols)
    tok_final = collections.Counter()
    tok_baseline = collections.Counter()
    col_n = collections.Counter()
    inst_n = collections.Counter()
    desc = f'row_trunc[{method} max={max_rows}]' + ('+coltrunc[' + (f'{col_trunc_method} <= {max_context_cols}' if max_context_cols and max_context_cols > 0 else 'no-cap') + (',dropempty' if drop_empty_on else '') + (',reindex' if reindex_on else '') + ']' if col_on else '')
    if method in ('coverage', 'coverage_col') or col_trunc_method == 'coverage_col':
        _coverage_stats.clear()
    for r in tqdm(rows, desc=desc, unit='ex'):
        task = r.get('task')
        if task not in ('cta', 'cpa'):
            n_skip += 1
            continue
        tid = _shared_table_id(source, r)
        raw_key = (split, task, tid)
        if raw_key not in _row_trunc_raw_cache:
            try:
                _row_trunc_raw_cache[raw_key] = _load_raw_table(source, task, r['meta'], split)
            except Exception as e:
                _row_trunc_raw_cache[raw_key] = None
                if n_fail < 5:
                    print(f'[row_trunc] WARN {source}/{task}/{tid}: {e}')
        loaded = _row_trunc_raw_cache[raw_key]
        if loaded is None:
            n_fail += 1
            continue
        (kind, col_names, raw, df) = loaded
        target_pos = None
        if method in ('tfidf', 'coverage', 'dedup') or col_on:
            target_pos = _target_positions(task, r['meta'], col_names, r['input'])
            if target_pos is None:
                n_fail += 1
                continue
        keep_cols = None
        if col_on:
            keep_cols = TT.column_keep_positions(task, target_pos, len(col_names), max_context_cols, rows=raw, drop_empty=drop_empty_on, method=col_trunc_method, seed=seed, table_id=tid, cache_key=raw_key)
        span = find_markdown_table_span(r['input'])
        if span is None:
            n_fail += 1
            continue
        n_tgt = len(tok(r['target'], add_special_tokens=False)['input_ids']) + (1 if tok.eos_token_id is not None else 0)
        n_all = len(raw)
        if method == 'coverage_col':
            oc_key = (split, task, tid)
            ncols = len(col_names)
            render_cols = keep_cols if keep_cols is not None else list(range(ncols))
            order = None
            n_keep = min(max_rows, n_all) if hard_cap else n_all

            def _render_topk(k, cols=render_cols):
                _orders = _get_covcol_orders(oc_key, raw, seed, tid, ncols, k)
                block = TT.build_coverage_col_block(kind, col_names, raw, df, _orders, k, cols, reindex=reindex_on and keep_cols is not None)
                content = r['input'][:span[0]] + block + r['input'][span[1]:]
                if reindex_on and keep_cols is not None:
                    content = _reindex_marker(task, content, keep_cols, target_pos)
                fit_ids = _encode_prompt(tok, with_task_prefix(content, task, task_prefix), model_name)
                return (content, len(fit_ids) + n_tgt <= max_len)
            if hard_cap:
                (content, ok_full) = _render_topk(n_keep)
                used = n_keep
                if not ok_full:
                    (content, used) = _galloping_max_fit(_render_topk, n_keep - 1)
            else:
                (content, used) = _galloping_max_fit(_render_topk, n_all)
                ok_full = used >= n_all
        else:
            sel_key = (split, task, tid, hard_cap, tuple(target_pos) if target_pos is not None else None)
            if sel_key not in _row_trunc_cache:
                if hard_cap:
                    _row_trunc_cache[sel_key] = TT._select_raw_rows(raw, method, max_rows, seed, tid, target_pos, full_rank=False)
                else:
                    _row_trunc_cache[sel_key] = TT._select_raw_rows(raw, method, len(raw), seed, tid, target_pos, full_rank=True)
            (sel, prio) = _row_trunc_cache[sel_key]
            cand = sel if hard_cap or method == 'dedup' else list(range(len(raw)))
            order = sorted(cand, key=lambda i: prio[i])

            def _render_topk(k, cols=keep_cols):
                keep = sorted(order[:k])
                block = _render_block_cached(source, task, tid, kind, col_names, raw, keep, df, split, keep_cols=cols, reindex=reindex_on and cols is not None)
                content = r['input'][:span[0]] + block + r['input'][span[1]:]
                if reindex_on and cols is not None:
                    content = _reindex_marker(task, content, cols, target_pos)
                fit_ids = _encode_prompt(tok, with_task_prefix(content, task, task_prefix), model_name)
                return (content, len(fit_ids) + n_tgt <= max_len)
            n_keep = len(order)
            (content, ok_full) = _render_topk(n_keep)
            used = n_keep
            if not ok_full:
                (best_content, _) = _render_topk(1)
                best = 1
                (lo, hi) = (2, n_keep - 1)
                while lo <= hi:
                    mid = (lo + hi) // 2
                    (c, ok) = _render_topk(mid)
                    if ok:
                        (best, best_content) = (mid, c)
                        lo = mid + 1
                    else:
                        hi = mid - 1
                (content, used) = (best_content, best)
            if drop_empty_on and order is not None:
                fin = sorted(order[:used])
                base_cols = list(keep_cols) if keep_cols is not None else list(range(len(col_names)))
                tset_pp = set(target_pos)
                emp = TT.empty_context_positions(raw, [c for c in base_cols if c not in tset_pp], fin)
                if emp:
                    cols2 = [c for c in base_cols if c not in emp]
                    block = _render_block_cached(source, task, tid, kind, col_names, raw, fin, df, split, keep_cols=cols2, reindex=reindex_on)
                    content = r['input'][:span[0]] + block + r['input'][span[1]:]
                    if reindex_on:
                        content = _reindex_marker(task, content, cols2, target_pos)
                    keep_cols = cols2
                    n_empty_dropped += len(emp)
                    n_inst_empty_dropped += 1
        r['input'] = content
        if used < n_all or not ok_full:
            n_trunc += 1
            n_shrink += 1
        else:
            n_full += 1
        if col_on:
            final_ntok = len(_encode_prompt(tok, with_task_prefix(content, task, task_prefix), model_name)) + n_tgt
            tok_final[task] += final_ntok
            inst_n[task] += 1
            if keep_cols is not None:
                if method == 'coverage_col':
                    base_orders = _get_covcol_orders(oc_key, raw, seed, tid, ncols, used)
                    base_block = TT.build_coverage_col_block(kind, col_names, raw, df, base_orders, used, list(range(len(col_names))))
                else:
                    keep_rows = sorted(order[:used])
                    base_block = _render_block_cached(source, task, tid, kind, col_names, raw, keep_rows, df, split, keep_cols=None)
                base_content = r['input'][:span[0]] + base_block + r['input'][span[1]:]
                base_ntok = len(_encode_prompt(tok, with_task_prefix(base_content, task, task_prefix), model_name)) + n_tgt
                tok_baseline[task] += base_ntok
                col_n[task] += 1
            else:
                tok_baseline[task] += final_ntok
    stats = {'truncated': n_trunc, 'shrunk_to_fit': n_shrink, 'rendered_full': n_full, 'failed': n_fail, 'skipped_non_ctacpa': n_skip}
    if method in ('coverage', 'coverage_col'):
        stats.update({f'coverage_{k}': int(v) for (k, v) in _coverage_stats.items()})
    if col_on:
        (tf, tb) = (int(sum(tok_final.values())), int(sum(tok_baseline.values())))
        stats['col_trunc'] = {'max_context_cols': int(max_context_cols), 'reindex_truncated_cols': bool(reindex_cols), 'drop_empty_context_cols': bool(drop_empty_context_cols), 'empty_cols_dropped_in_rendered_rows': int(n_empty_dropped), 'instances_with_empty_col_dropped': int(n_inst_empty_dropped), 'instances': int(sum(inst_n.values())), 'instances_columns_dropped': int(sum(col_n.values())), 'train_tokens_final': tf, 'train_tokens_baseline_all_cols': tb, 'train_tokens_saved': tb - tf, 'pct_saved': 100.0 * (tb - tf) / tb if tb else 0.0, 'per_task': {t: {'instances': int(inst_n[t]), 'instances_columns_dropped': int(col_n[t]), 'train_tokens_final': int(tok_final[t]), 'train_tokens_baseline_all_cols': int(tok_baseline[t]), 'train_tokens_saved': int(tok_baseline[t] - tok_final[t])} for t in sorted(inst_n)}}
    return stats

class SFTDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, enc_by_epoch=None, task_ids=None, role_ids=None, region_ids=None, gate_task_ids=None, recv_pos=None, copy_ids=None, soft_ids=None, struct_ids=None):
        self.enc_by_epoch = enc_by_epoch
        self.enc = enc_by_epoch[0] if enc_by_epoch else encodings
        self.epoch = 0
        self.task_ids = task_ids
        if task_ids is not None and len(task_ids) != len(self.enc):
            raise ValueError(f'task_ids length {len(task_ids)} != encodings length {len(self.enc)}; the task id of every training example must line up with its labels.')
        self.role_ids = role_ids
        if role_ids is not None:
            if len(role_ids) != len(self.enc):
                raise ValueError(f'role_ids length {len(role_ids)} != encodings length {len(self.enc)}')
            if enc_by_epoch is not None:
                raise ValueError('role ids cannot follow a per-epoch re-encode of the prompts')
        self.region_ids = region_ids
        if region_ids is not None:
            if len(region_ids) != len(self.enc):
                raise ValueError(f'region_ids length {len(region_ids)} != encodings length {len(self.enc)}')
            if enc_by_epoch is not None:
                raise ValueError('region ids cannot follow a per-epoch re-encode of the prompts')
        self.copy_ids = copy_ids
        if copy_ids is not None:
            if len(copy_ids) != len(self.enc):
                raise ValueError(f'copy_ids length {len(copy_ids)} != encodings length {len(self.enc)}')
            if enc_by_epoch is not None:
                raise ValueError('copy ids cannot follow a per-epoch re-encode of the prompts')
        self.struct_ids = struct_ids
        if struct_ids is not None:
            if len(struct_ids) != len(self.enc):
                raise ValueError(f'struct_ids length {len(struct_ids)} != encodings length {len(self.enc)}')
        self.soft_ids = soft_ids
        if soft_ids is not None:
            if len(soft_ids) != len(self.enc):
                raise ValueError(f'soft_ids length {len(soft_ids)} != encodings length {len(self.enc)}')
            if enc_by_epoch is not None:
                raise ValueError('soft-role slots cannot follow a per-epoch re-encode of the prompts')
        self.gate_task_ids = gate_task_ids
        if gate_task_ids is not None and len(gate_task_ids) != len(self.enc):
            raise ValueError(f'gate_task_ids length {len(gate_task_ids)} != encodings length {len(self.enc)}; the gate row of every example must line up with its labels.')
        self.recv_pos = recv_pos
        if recv_pos is not None:
            if len(recv_pos) != len(self.enc):
                raise ValueError(f'recv_pos length {len(recv_pos)} != encodings length {len(self.enc)}')
            if enc_by_epoch is not None:
                raise ValueError('semantic decision positions cannot follow a per-epoch re-encode of the prompts')

    def set_epoch(self, e):
        if self.enc_by_epoch:
            self.epoch = max(0, min(int(e), len(self.enc_by_epoch) - 1))
            self.enc = self.enc_by_epoch[self.epoch]

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        item = self.enc[i]
        (ids, lab) = (item[0], item[1])
        out = {'input_ids': ids, 'labels': lab, 'attention_mask': [1] * len(ids)}
        if self.task_ids is not None:
            out['task_id'] = self.task_ids[i]
        if self.role_ids is not None:
            r = self.role_ids[i]
            if len(r) != len(ids):
                raise RuntimeError(f'role ids ({len(r)}) do not match input_ids ({len(ids)}) for example {i}')
            out['role_ids'] = r
        if self.region_ids is not None:
            g = self.region_ids[i]
            if len(g) != len(ids):
                raise RuntimeError(f'region ids ({len(g)}) do not match input_ids ({len(ids)}) for example {i}')
            out['region_ids'] = g
        if self.copy_ids is not None:
            c = self.copy_ids[i]
            if len(c) != len(ids):
                raise RuntimeError(f'copy ids ({len(c)}) do not match input_ids ({len(ids)}) for example {i}')
            out['copy_ids'] = c
        if self.struct_ids is not None:
            s = self.struct_ids[i]
            if len(s) != len(ids):
                raise RuntimeError(f'struct ids ({len(s)}) do not match input_ids ({len(ids)}) for example {i}')
            out['struct_ids'] = s
        if self.soft_ids is not None:
            sp = self.soft_ids[i]
            if len(sp) != len(ids):
                raise RuntimeError(f'soft-role slots ({len(sp)}) do not match input_ids ({len(ids)}) for example {i}')
            out['soft_ids'] = sp
        if self.gate_task_ids is not None:
            out['gate_task_id'] = self.gate_task_ids[i]
        if self.recv_pos is not None:
            out['attn_recv'] = self.recv_pos[i]
        return out

class EpochRowDrawCallback(TrainerCallback):

    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dataset.set_epoch(int(round(state.epoch or 0)))

class RoleGateCheckpointCallback(TrainerCallback):

    def __init__(self, model):
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        ck = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
        if os.path.isdir(ck):
            RGM.save(self.model, ck)

class FFNRoleScaleCheckpointCallback(TrainerCallback):

    def __init__(self, model):
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        ck = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
        if os.path.isdir(ck):
            FRS.save(self.model, ck)

class RecvGateCheckpointCallback(TrainerCallback):

    def __init__(self, model):
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        ck = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
        if os.path.isdir(ck):
            RCV.save(self.model, ck)

class SplitGateCheckpointCallback(TrainerCallback):

    def __init__(self, model):
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        ck = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
        if os.path.isdir(ck):
            SGL.save(self.model, ck)

class StructRoleBiasCheckpointCallback(TrainerCallback):

    def __init__(self, model):
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        ck = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
        if os.path.isdir(ck):
            SRB.save(self.model, ck)

class SwigluThresCheckpointCallback(TrainerCallback):

    def __init__(self, model):
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        ck = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
        if os.path.isdir(ck):
            SWT.save(self.model, ck)

class RankGateCheckpointCallback(TrainerCallback):

    def __init__(self, model):
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        ck = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
        if os.path.isdir(ck):
            RKG.save(self.model, ck)

class SoftRolePromptCheckpointCallback(TrainerCallback):

    def __init__(self, model):
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        ck = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
        if os.path.isdir(ck):
            SP.save(self.model, ck)

class RoleMatrixCheckpointCallback(TrainerCallback):

    def __init__(self, model):
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        ck = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
        if os.path.isdir(ck):
            RM.save(self.model, ck)

@torch.no_grad()
def collect_rank_gate_diagnostics(model, dataset, pad_id, per_row=16, micro_bs=4, max_seq_len=2048):
    bank = RKG.get_bank(model)
    gate_rows = getattr(dataset, 'gate_task_ids', None)
    if not gate_rows or getattr(dataset, 'role_ids', None) is None:
        return None
    by_row = {}
    for (i, row) in enumerate(gate_rows):
        by_row.setdefault(int(row), []).append(i)
    coll = PadCollator(pad_id)
    lg_kw = RKG.minimal_logits_kwargs(model)
    was_training = model.training
    model.eval()
    bank.reset_stats()
    bank.collect = True
    n_seen = {}
    try:
        for (row, idxs) in sorted(by_row.items()):
            take = idxs[:per_row]
            n_seen[row] = len(take)
            for st in tqdm(range(0, len(take), micro_bs), desc=f'rank-gate diag row {row}', leave=False):
                batch = coll([dataset[j] for j in take[st:st + micro_bs]])
                gid = batch.pop('gate_task_id')
                roles = batch.pop('role_ids')
                batch.pop('region_ids', None)
                batch.pop('task_id', None)
                batch.pop('labels', None)
                batch = {k: v.to(model.device) for (k, v) in batch.items()}
                tm = RKG.gate_mask(model, roles).to(model.device)
                RKG.set_gate_context(gid.to(model.device), tm, valid_mask=batch.get('attention_mask'))
                try:
                    out = model(**batch, **lg_kw)
                finally:
                    RKG.clear_gate_context()
                del out, batch, gid, roles, tm
    finally:
        bank.collect = False
        if was_training:
            model.train()
    out = RKG.gate_activation_stats(model)
    row_name = {RKG.TASK_NONE: 'none'}
    row_name.update({RKG.TASK_INDEX[t]: t for t in RKG.GATE_TASKS})
    out['examples_per_row'] = {row_name.get(k, str(k)): v for (k, v) in sorted(n_seen.items())}
    out['max_seq_len'] = int(max_seq_len)
    bank.reset_stats()
    return out

@torch.no_grad()
def collect_recv_gate_diagnostics(model, dataset, pad_id, per_row=16, micro_bs=4, max_seq_len=2048):
    bank = RCV.get_bank(model)
    gate_rows = getattr(dataset, 'gate_task_ids', None)
    if not gate_rows or getattr(dataset, 'role_ids', None) is None or getattr(dataset, 'region_ids', None) is None:
        return None
    by_row = {}
    for (i, row) in enumerate(gate_rows):
        by_row.setdefault(int(row), []).append(i)
    coll = PadCollator(pad_id)
    lg_kw = RCV.minimal_logits_kwargs(model)
    was_training = model.training
    model.eval()
    bank.reset_stats()
    bank.collect = True
    n_seen = {}
    try:
        for (row, idxs) in sorted(by_row.items()):
            take = idxs[:per_row]
            n_seen[row] = len(take)
            for st in tqdm(range(0, len(take), micro_bs), desc=f'recv-gate diag row {row}', leave=False):
                batch = coll([dataset[j] for j in take[st:st + micro_bs]])
                gid = batch.pop('gate_task_id')
                roles = batch.pop('role_ids')
                regions = batch.pop('region_ids')
                batch.pop('task_id', None)
                labels = batch.pop('labels', None)
                batch = {k: v.to(model.device) for (k, v) in batch.items()}
                (pm, cm) = RCV.stream_masks(gid, roles, regions)
                rv = RCV.receiver_mask_from_regions(regions)
                RCV.set_gate_context(gid.to(model.device), pm.to(model.device), cm.to(model.device), rv.to(model.device), valid_mask=batch.get('attention_mask'), answer_mask=None if labels is None else (labels != -100).to(model.device))
                try:
                    out = model(**batch, **lg_kw)
                finally:
                    RCV.clear_gate_context()
                del out, batch, gid, roles, regions, labels, pm, cm, rv
    finally:
        bank.collect = False
        if was_training:
            model.train()
    out = RCV.gate_activation_stats(model)
    row_name = {RCV.TASK_NONE: 'none'}
    row_name.update({RCV.TASK_INDEX[t]: t for t in RCV.GATE_TASKS})
    out['examples_per_row'] = {row_name.get(k, str(k)): v for (k, v) in sorted(n_seen.items())}
    out['max_seq_len'] = int(max_seq_len)
    bank.reset_stats()
    return out

class GradColMaskCallback(TrainerCallback):

    def __init__(self, dataset, enc_rows, enc_orig, tok, model_name, args, out_dir):
        self.dataset = dataset
        self.enc_rows = enc_rows
        self.enc_orig = enc_orig
        self.tok = tok
        self.model_name = model_name
        self.args = args
        self.out_dir = out_dir
        self.done = False

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self.done:
            return
        self.done = True
        a = self.args
        method = a.col_mask_method
        max_len = a.max_seq_len
        model_name = self.model_name
        tok = self.tok
        device = next(model.parameters()).device
        step_before = int(state.global_step)
        with torch.no_grad():
            fp_before = float(sum((p.detach().float().sum().item() for p in model.parameters() if p.requires_grad)))
        ctacpa_idx = [i for (i, r) in enumerate(self.enc_rows) if canonical_task(r.get('task')) in ('cta', 'cpa')]
        items = []
        for i in ctacpa_idx:
            it = GCM.build_attrib_item(tok, self.enc_rows[i], max_len, model_name, False)
            it['_enc_idx'] = i
            items.append(it)
        attrib_items = [it for it in items if 'skip' not in it]
        print(f'[col_mask:{method}] attribution over {len(attrib_items)}/{len(items)} CTA/CPA instances (rest skipped/keep-all) on epoch-1 state (global_step={step_before})')
        attrib = GCM.attribution_sweep(model, tok, attrib_items, a.micro_bs, device)
        (masks, diag) = GCM.build_masks(items, attrib, method, a.seed)
        summary = GCM.summarize(diag)
        masked_enc = list(self.enc_orig)
        idx_by_id = {}
        for it in items:
            idx_by_id[it['instance_id']] = it['_enc_idx']
        n_applied = 0
        n_reencode_fail = 0
        tok_before = tok_after = 0
        for (iid, keep) in masks.items():
            i = idx_by_id[iid]
            row = self.enc_rows[i]
            (new_text, m) = GCM.mask_columns_prompt(row.get('task'), row['input'], keep)
            if m.get('skipped'):
                n_reencode_fail += 1
                continue
            e = tokenize_example(tok, {**row, 'input': new_text}, max_len, model_name, False)
            if e is None:
                n_reencode_fail += 1
                continue
            assert len(e[0]) <= max_len, 'masked prompt exceeds max_seq_len'
            masked_enc[i] = (e[0], e[1])
            tok_before += len(self.enc_orig[i][0])
            tok_after += len(e[0])
            n_applied += 1
        assert len(masked_enc) == len(self.enc_orig), 'masked encoding count drifted from epoch-1'
        self.dataset.enc = masked_enc
        with torch.no_grad():
            fp_after = float(sum((p.detach().float().sum().item() for p in model.parameters() if p.requires_grad)))
        assert int(state.global_step) == step_before, 'attribution advanced global_step'
        assert abs(fp_after - fp_before) < 0.001, 'attribution changed model parameters'
        model.zero_grad(set_to_none=True)
        model.train()
        red = 100.0 * (1 - tok_after / tok_before) if tok_before else 0.0
        print(f'[col_mask:{method}] applied masks to {n_applied} instances (reencode_fail={n_reencode_fail}); prompt-token reduction on masked instances {red:.1f}% ({tok_before}->{tok_after}); summary={summary}')
        out = {'method': method, 'attribution_batch_size': a.micro_bs, 'seed': a.seed, 'epoch1_global_step': step_before, 'n_ctacpa_instances': len(items), 'n_attributed': summary.get('n_attributed', 0), 'n_masks_applied': n_applied, 'n_reencode_fail': n_reencode_fail, 'masked_token_before': tok_before, 'masked_token_after': tok_after, 'masked_token_reduction_pct': red, 'summary': summary, 'per_instance': {str(k): v for (k, v) in diag.items()}}
        try:
            with open(f'{self.out_dir}/grad_col_mask.json', 'w') as f:
                json.dump(out, f, indent=2)
            print(f'[col_mask:{method}] wrote diagnostics -> {self.out_dir}/grad_col_mask.json')
        except Exception as e:
            print(f'[col_mask:{method}] WARN could not write diagnostics: {e}')

def augment_transposed(rows, tok, max_len, model_name):
    out = []
    stats = {'instances': len(rows), 'augmented': 0, 'no_table': 0, 'unparseable': 0, 'over_len': 0}
    for r in rows:
        out.append(r)
        (new_text, n_done, n_skip) = TTR.transpose_tables(r['input'])
        if n_done == 0:
            stats['unparseable' if n_skip else 'no_table'] += 1
            continue
        meta = dict(r.get('meta') or {})
        meta['transposed'] = True
        if meta.get('pair_group') is not None:
            meta['pair_group'] = f"{meta['pair_group']}::transposed"
        aug = {**r, 'input': new_text, 'id': f"{r['id']}::transposed", 'meta': meta}
        e = tokenize_example(tok, aug, max_len, model_name, False)
        if e is None or e[2]:
            stats['over_len'] += 1
            continue
        out.append(aug)
        stats['augmented'] += 1
    return (out, stats)
_SWAP_RNG = random.Random(0)

def augment_swapped_context(rows, tok, max_len, model_name):
    out = []
    stats = {'instances': len(rows), 'eligible': 0, 'augmented': 0, 'not_supported_task': 0, 'over_len': 0, 'per_task': collections.Counter(), 'skipped': collections.Counter()}
    for r in tqdm(rows, desc='swap_context_blocks', unit='ex'):
        out.append(r)
        slug = r['task']
        if canonical_task(slug) not in ('cta', 'cpa'):
            stats['not_supported_task'] += 1
            continue
        stats['eligible'] += 1
        (new_text, meta_sw) = E.shuffle_columns_prompt(slug, r['input'], _SWAP_RNG, mode='swap_blocks')
        if meta_sw.get('skipped'):
            stats['skipped'][f"{canonical_task(slug)}:{meta_sw.get('reason', 'unknown')}"] += 1
            continue
        meta = dict(r.get('meta') or {})
        meta['ctx_blocks_swapped'] = True
        meta['swap_old_target'] = list(meta_sw['old_target'])
        meta['swap_new_target'] = list(meta_sw['new_target'])
        if meta.get('pair_group') is not None:
            meta['pair_group'] = f"{meta['pair_group']}::ctxswap"
        aug = {**r, 'input': new_text, 'id': f"{r['id']}::ctxswap", 'meta': meta}
        e = tokenize_example(tok, aug, max_len, model_name, False)
        if e is None or e[2]:
            stats['over_len'] += 1
            continue
        out.append(aug)
        stats['augmented'] += 1
        stats['per_task'][canonical_task(slug)] += 1
    stats['skipped'] = dict(stats['skipped'])
    stats['per_task'] = dict(stats['per_task'])
    return (out, stats)
_TWO_VIEWS_RNG = random.Random(0)
TARGET_VIEWS = ('target_first', 'target_end')

def augment_target_views(rows, tok, max_len, model_name):
    out = []
    stats = {'instances': len(rows), 'eligible': 0, 'augmented': 0, 'already': 0, 'not_supported_task': 0, 'over_len': 0, 'passthrough': 0, 'views': list(TARGET_VIEWS), 'per_task': collections.Counter(), 'skipped': collections.Counter()}
    for r in tqdm(rows, desc='target_two_views', unit='ex'):
        slug = r['task']
        if canonical_task(slug) not in ('cta', 'cpa'):
            stats['not_supported_task'] += 1
            out.append(r)
            continue
        stats['eligible'] += 1
        (views, bad) = ([], None)
        for (view, mode) in (('target_first', 'target_begin'), ('target_end', 'target_end')):
            (new_text, m) = E.shuffle_columns_prompt(slug, r['input'], _TWO_VIEWS_RNG, mode=mode)
            if m.get('skipped'):
                bad = f"{canonical_task(slug)}:{view}:{m.get('reason', 'unknown')}"
                break
            meta = dict(r.get('meta') or {})
            meta['target_view'] = view
            meta['view_old_target'] = list(m['old_target'])
            meta['view_new_target'] = list(m['new_target'])
            if meta.get('pair_group') is not None:
                meta['pair_group'] = f"{meta['pair_group']}::{view}"
            views.append(({**r, 'input': new_text, 'id': f"{r['id']}::{view}", 'meta': meta}, new_text == r['input']))
        if bad is not None:
            stats['skipped'][bad] += 1
            stats['passthrough'] += 1
            out.append(r)
            continue
        fits = True
        for (v, _) in views:
            e = tokenize_example(tok, v, max_len, model_name, False)
            if e is None or e[2]:
                fits = False
                break
        if not fits:
            stats['over_len'] += 1
            stats['passthrough'] += 1
            out.append(r)
            continue
        for (v, same) in views:
            out.append(v)
            stats['already'] += int(same)
        stats['augmented'] += 1
        stats['per_task'][canonical_task(slug)] += 1
    stats['skipped'] = dict(stats['skipped'])
    stats['per_task'] = dict(stats['per_task'])
    return (out, stats)

def build_swapped_rows(rows):
    out = []
    stats = {'instances': len(rows), 'eligible': 0, 'swapped': 0, 'not_supported_task': 0, 'per_task': collections.Counter(), 'skipped': collections.Counter()}
    for r in tqdm(rows, desc='swap_context_epochs', unit='ex'):
        slug = r['task']
        if canonical_task(slug) not in ('cta', 'cpa'):
            stats['not_supported_task'] += 1
            out.append(r)
            continue
        stats['eligible'] += 1
        (new_text, meta_sw) = E.shuffle_columns_prompt(slug, r['input'], _SWAP_RNG, mode='swap_blocks')
        if meta_sw.get('skipped'):
            stats['skipped'][f"{canonical_task(slug)}:{meta_sw.get('reason', 'unknown')}"] += 1
            out.append(r)
            continue
        meta = dict(r.get('meta') or {})
        meta['ctx_blocks_swapped'] = True
        meta['swap_old_target'] = list(meta_sw['old_target'])
        meta['swap_new_target'] = list(meta_sw['new_target'])
        out.append({**r, 'input': new_text, 'meta': meta})
        stats['swapped'] += 1
        stats['per_task'][canonical_task(slug)] += 1
    stats['skipped'] = dict(stats['skipped'])
    stats['per_task'] = dict(stats['per_task'])
    return (out, stats)

def parse_col_major_tasks(spec):
    if not spec:
        return set()
    return {canonical_task(t.strip()) for t in str(spec).split(',') if t.strip()}

def parse_task_scope(spec):
    if not spec:
        return set()
    parts = spec.split(',') if isinstance(spec, str) else list(spec)
    return {canonical_task(str(t).strip()) for t in parts if str(t).strip()}

def task_in_scope(slug, scope):
    return not scope or canonical_task(slug) in scope

def transpose_to_fit(text, tok, target_len, max_len, model_name):
    (new, n_done, n_skip) = TTR.transpose_tables(text)
    if n_done == 0:
        return (text, 'unparseable' if n_skip else 'no_table')
    if len(_encode_prompt(tok, new, model_name)) + target_len <= max_len:
        return (new, 'ok')
    cur = text
    while True:
        lines = cur.split('\n')
        tbl = [i for (i, l) in enumerate(lines) if l.lstrip().startswith('|')]
        if len(tbl) <= 3:
            return (text, 'fallback')
        del lines[tbl[-1]]
        cur = '\n'.join(lines)
        (new, n_done, _) = TTR.transpose_tables(cur)
        if n_done == 0:
            return (text, 'fallback')
        if len(_encode_prompt(tok, new, model_name)) + target_len <= max_len:
            return (new, 'shrunk')

def apply_col_major(rows, col_tasks, tok, max_len, model_name, split):
    st = {'tasks': sorted(col_tasks), 'split': split, 'considered': 0, 'transposed': 0, 'shrunk': 0, 'no_table': 0, 'unparseable': 0, 'fallback_row_major': 0}
    if not col_tasks:
        return st
    for r in rows:
        if canonical_task(r.get('task')) not in col_tasks:
            continue
        st['considered'] += 1
        tl = len(tok(r['target'], add_special_tokens=False)['input_ids'])
        if tok.eos_token_id is not None:
            tl += 1
        (new, status) = transpose_to_fit(r['input'], tok, tl, max_len, model_name)
        if status in ('ok', 'shrunk'):
            r['input'] = new
            st['transposed'] += 1
            st['shrunk'] += int(status == 'shrunk')
        else:
            st['fallback_row_major' if status == 'fallback' else status] += 1
    if st['considered'] == 0:
        return st
    print(f"[col_major] {split}: tasks={st['tasks']} transposed {st['transposed']}/{st['considered']} prompts (row-shrunk first: {st['shrunk']}; no_table={st['no_table']} unparseable={st['unparseable']} kept_row_major={st['fallback_row_major']})")
    return st

def apply_target_sandwich(rows, on, split):
    st = {'enabled': bool(on), 'split': split, 'considered': 0, 'sandwiched': 0, 'not_supported_task': 0, 'skipped': collections.Counter()}
    if not on:
        st['skipped'] = {}
        return st
    for r in tqdm(rows, desc=f'target_sandwich/{split}', unit='ex'):
        slug = r['task']
        if canonical_task(slug) not in ('cta', 'cpa'):
            st['not_supported_task'] += 1
            continue
        st['considered'] += 1
        (new_text, meta) = TSW.sandwich_prompt(slug, r['input'])
        if meta.get('skipped'):
            st['skipped'][f"{canonical_task(slug)}:{meta.get('reason')}"] += 1
            continue
        r['input'] = new_text
        st['sandwiched'] += 1
    st['skipped'] = dict(st['skipped'])
    if st['considered']:
        print(f"[{TSW.MECHANISM}] {split}: sandwiched the target column in {st['sandwiched']}/{st['considered']} CTA/CPA prompts" + (f" (skipped {st['skipped']})" if st['skipped'] else ''))
    return st

def apply_drop_context_after_target(rows, on, split, tok=None):
    st = {'enabled': bool(on), 'split': split, 'considered': 0, 'truncated': 0, 'nothing_after': 0, 'not_supported_task': 0, 'cols_dropped': 0, 'cols_before': 0, 'cols_after': 0, 'tok_before': 0, 'tok_after': 0, 'per_task': collections.Counter(), 'skipped': collections.Counter()}
    if not on:
        (st['per_task'], st['skipped']) = ({}, {})
        return st
    for r in tqdm(rows, desc=f'drop_ctx_after_target/{split}', unit='ex'):
        slug = r['task']
        if canonical_task(slug) not in ('cta', 'cpa'):
            st['not_supported_task'] += 1
            continue
        st['considered'] += 1
        before = r['input']
        (new_text, meta) = E.drop_after_target_prompt(slug, before)
        if meta.get('skipped'):
            if meta.get('reason') == 'nothing_after_target':
                st['nothing_after'] += 1
            else:
                st['skipped'][f"{canonical_task(slug)}:{meta.get('reason', 'unknown')}"] += 1
            continue
        r['input'] = new_text
        st['truncated'] += 1
        st['cols_before'] += int(meta['n_cols'])
        st['cols_after'] += int(meta['n_kept'])
        st['cols_dropped'] += int(meta['n_dropped'])
        st['per_task'][canonical_task(slug)] += 1
        if tok is not None:
            st['tok_before'] += len(tok(before, add_special_tokens=False)['input_ids'])
            st['tok_after'] += len(tok(new_text, add_special_tokens=False)['input_ids'])
    (st['per_task'], st['skipped']) = (dict(st['per_task']), dict(st['skipped']))
    if st['considered']:
        saved = st['tok_before'] - st['tok_after']
        pct = 100.0 * saved / st['tok_before'] if st['tok_before'] else 0.0
        (st['tok_saved'], st['tok_saved_pct']) = (saved, round(pct, 2))
        print(f"[drop_ctx_after_target] {split}: truncated {st['truncated']}/{st['considered']} CTA/CPA prompts {st['per_task']} -- {st['cols_before']} -> {st['cols_after']} columns (dropped {st['cols_dropped']}); already_last={st['nothing_after']}" + (f" skipped={st['skipped']}" if st['skipped'] else '') + (f"; prompt tokens {st['tok_before']} -> {st['tok_after']} (saved {saved}, {pct:.1f}%)" if tok is not None else ''))
    return st

def apply_target_end(rows, on, split, tasks=(), mode='target_end'):
    if mode not in ('target_end', 'target_begin'):
        raise ValueError(f'unknown target-position mode {mode!r}')
    tag = 'target_end' if mode == 'target_end' else 'target_first'
    st = {'enabled': bool(on), 'mode': mode, 'split': split, 'tasks': sorted(tasks) or ['*'], 'considered': 0, 'moved': 0, 'already': 0, 'out_of_scope': 0, 'not_supported_task': 0, 'skipped': collections.Counter()}
    if not on:
        st['skipped'] = {}
        return st
    one = E.target_end_prompt if mode == 'target_end' else E.target_begin_prompt
    for r in tqdm(rows, desc=f'{tag}/{split}', unit='ex'):
        slug = r['task']
        if canonical_task(slug) not in ('cta', 'cpa'):
            st['not_supported_task'] += 1
            continue
        if not task_in_scope(slug, tasks):
            st['out_of_scope'] += 1
            continue
        st['considered'] += 1
        (new_text, meta) = one(slug, r['input'])
        if meta.get('skipped'):
            st['skipped'][f"{canonical_task(slug)}:{meta.get('reason', 'unknown')}"] += 1
            continue
        if new_text == r['input']:
            st['already'] += 1
            continue
        r['input'] = new_text
        st['moved'] += 1
    st['skipped'] = dict(st['skipped'])
    if st['considered']:
        print(f"[{tag}] {split}: tasks={st['tasks']} canonicalized {st['moved']}/{st['considered']} CTA/CPA prompts (already_canonical={st['already']}" + (f" out_of_scope={st['out_of_scope']}" if st['out_of_scope'] else '') + (f" skipped={st['skipped']}" if st['skipped'] else '') + ')')
    return st

def apply_table_first(rows, on, split, tasks=()):
    st = {'enabled': bool(on), 'split': split, 'tasks': sorted(tasks) or ['*'], 'considered': 0, 'moved': 0, 'out_of_scope': 0, 'already_first': 0, 'no_table': 0}
    if not on:
        return st
    for r in rows:
        if not task_in_scope(r.get('task'), tasks):
            st['out_of_scope'] += 1
            continue
        st['considered'] += 1
        (new, status) = TRO.move_tables_first(r['input'])
        if status == 'ok':
            r['input'] = new
            st['moved'] += 1
        else:
            st[status] += 1
    if st['considered'] == 0:
        return st
    print(f"[table_first] {split}: tasks={st['tasks']} moved the table to the top of {st['moved']}/{st['considered']} prompts (already_first={st['already_first']} no_table={st['no_table']} out_of_scope={st['out_of_scope']})")
    return st

def apply_table_last(rows, on, split):
    st = {'enabled': bool(on), 'split': split, 'considered': 0, 'moved': 0, 'already_last': 0, 'no_table': 0}
    if not on:
        return st
    for r in rows:
        st['considered'] += 1
        (new, status) = TRO.move_tables_last(r['input'])
        if status == 'ok':
            r['input'] = new
            st['moved'] += 1
        else:
            st[status] += 1
    if st['considered'] == 0:
        return st
    print(f"[table_last] {split}: moved the table to the end of {st['moved']}/{st['considered']} prompts (already_last={st['already_last']} no_table={st['no_table']})")
    return st

def build_task_ids(tasks):
    task_names = sorted(set(tasks))
    index = {t: i for (i, t) in enumerate(task_names)}
    return ([index[t] for t in tasks], task_names)

def supervised_token_stats(enc, task_ids, task_names):
    if len(enc) != len(task_ids):
        raise ValueError(f'enc length {len(enc)} != task_ids length {len(task_ids)}')
    counts = {t: [] for t in range(len(task_names))}
    for ((ids, lab), t) in zip(enc, task_ids):
        n = sum((1 for x in lab if x != -100))
        if n == 0:
            raise ValueError('training example with zero supervised tokens')
        if lab[0] != -100:
            raise ValueError('labels[0] != -100: the first position is supervised, so the shifted loss would drop a supervised token and the measured mean would not match the trained one.')
        counts[t].append(n)
    stats = {}
    for (t, name) in enumerate(task_names):
        c = np.asarray(counts[t], dtype=np.float64)
        if c.size == 0:
            raise ValueError(f'task {name!r} has no examples but appears in task_names')
        stats[name] = {'examples': int(c.size), 'mean': float(c.mean()), 'median': float(np.median(c)), 'min': int(c.min()), 'max': int(c.max()), 'total': int(c.sum())}
    return stats

def compute_task_loss_weights(stats):
    tok = {t: stats[t]['examples'] * stats[t]['mean'] for t in stats}
    n_examples = float(sum((stats[t]['examples'] for t in stats)))
    mu_global = float(sum(tok.values())) / n_examples
    weights = {t: mu_global / stats[t]['mean'] for t in stats}
    z_o = float(sum(tok.values()))
    z_w = float(sum((tok[t] * weights[t] for t in tok)))
    expected = {'mu_global': mu_global, 'original': {t: tok[t] / z_o for t in tok}, 'weighted': {t: tok[t] * weights[t] / z_w for t in tok}}
    return (weights, expected)

def weighted_supervised_ce(logits, labels, task_id, weight_vec):
    if labels.shape[0] != task_id.shape[0]:
        raise ValueError(f'task_id rows {task_id.shape[0]} != labels rows {labels.shape[0]}')
    if logits.shape[:2] != labels.shape[:2]:
        raise ValueError(f'logits {tuple(logits.shape[:2])} vs labels {tuple(labels.shape)}')
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    vocab = shift_logits.shape[-1]
    ce = F.cross_entropy(shift_logits.reshape(-1, vocab).float(), shift_labels.reshape(-1), ignore_index=-100, reduction='none').view(shift_labels.shape)
    valid = (shift_labels != -100).to(ce.dtype)
    w = weight_vec.to(device=ce.device, dtype=ce.dtype)[task_id].unsqueeze(1)
    num = (w * ce * valid).sum()
    den = valid.sum()
    if float(den) <= 0.0:
        raise ValueError('no supervised tokens in batch')
    return num / den

class PadCollator:

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        m = max((len(b['input_ids']) for b in batch))
        (ii, ll, am) = ([], [], [])
        for b in batch:
            n = m - len(b['input_ids'])
            ii.append(b['input_ids'] + [self.pad_id] * n)
            ll.append(b['labels'] + [-100] * n)
            am.append(b['attention_mask'] + [0] * n)
        out = {'input_ids': torch.tensor(ii), 'labels': torch.tensor(ll), 'attention_mask': torch.tensor(am)}
        if 'task_id' in batch[0]:
            tid = torch.tensor([b['task_id'] for b in batch], dtype=torch.long)
            assert tid.shape[0] == out['labels'].shape[0], f"task_id rows {tid.shape[0]} != labels rows {out['labels'].shape[0]}"
            out['task_id'] = tid
        if 'role_ids' in batch[0]:
            rr = [b['role_ids'] + [RGM.ROLE_NONE] * (m - len(b['role_ids'])) for b in batch]
            out['role_ids'] = torch.tensor(rr, dtype=torch.long)
            assert out['role_ids'].shape == out['input_ids'].shape, f"role ids {tuple(out['role_ids'].shape)} != input_ids {tuple(out['input_ids'].shape)}"
        if 'region_ids' in batch[0]:
            gg = [b['region_ids'] + [RCV.R_PAD] * (m - len(b['region_ids'])) for b in batch]
            out['region_ids'] = torch.tensor(gg, dtype=torch.long)
            assert out['region_ids'].shape == out['input_ids'].shape, f"region ids {tuple(out['region_ids'].shape)} != input_ids {tuple(out['input_ids'].shape)}"
        if 'copy_ids' in batch[0]:
            cc = [b['copy_ids'] + [TSW.COPY_NONE] * (m - len(b['copy_ids'])) for b in batch]
            out['copy_ids'] = torch.tensor(cc, dtype=torch.long)
            assert out['copy_ids'].shape == out['input_ids'].shape, f"copy ids {tuple(out['copy_ids'].shape)} != input_ids {tuple(out['input_ids'].shape)}"
        if 'struct_ids' in batch[0]:
            ss = [b['struct_ids'] + [SRB.STRUCT_NONE] * (m - len(b['struct_ids'])) for b in batch]
            out['struct_ids'] = torch.tensor(ss, dtype=torch.long)
            assert out['struct_ids'].shape == out['input_ids'].shape, f"struct ids {tuple(out['struct_ids'].shape)} != input_ids {tuple(out['input_ids'].shape)}"
        if 'soft_ids' in batch[0]:
            sp = [b['soft_ids'] + [SP.SLOT_NONE] * (m - len(b['soft_ids'])) for b in batch]
            out['soft_ids'] = torch.tensor(sp, dtype=torch.long)
            assert out['soft_ids'].shape == out['input_ids'].shape, f"soft-role slots {tuple(out['soft_ids'].shape)} != input_ids {tuple(out['input_ids'].shape)}"
        if 'gate_task_id' in batch[0]:
            gid = torch.tensor([b['gate_task_id'] for b in batch], dtype=torch.long)
            assert gid.shape[0] == out['labels'].shape[0], f"gate_task_id rows {gid.shape[0]} != labels rows {out['labels'].shape[0]}"
            out['gate_task_id'] = gid
        if 'attn_recv' in batch[0]:
            k = max(1, max((len(b['attn_recv']) for b in batch)))
            rr = [list(b['attn_recv']) + [-1] * (k - len(b['attn_recv'])) for b in batch]
            rt = torch.tensor(rr, dtype=torch.long)
            assert rt.shape[0] == out['labels'].shape[0], f"attn_recv rows {rt.shape[0]} != labels rows {out['labels'].shape[0]}"
            out['attn_recv'] = rt
        return out

class PairedGroupSampler(torch.utils.data.Sampler):

    def __init__(self, group_ids, window, seed):
        self.groups = collections.OrderedDict()
        for (i, g) in enumerate(group_ids):
            self.groups.setdefault(g, []).append(i)
        sizes = {len(v) for v in self.groups.values()}
        if len(sizes) != 1:
            raise ValueError(f'[PairedGroupSampler] non-uniform group sizes {sorted(sizes)}; multi_paired_ctacpa expects exactly one CTA + one CPA per table.')
        self.s = sizes.pop()
        if window < self.s or window % self.s != 0:
            raise ValueError(f'[PairedGroupSampler] window (micro_bs*grad_accum={window}) must be a multiple of the group size ({self.s}) and >= it, so groups tile steps without straddling.')
        self.window = window
        self.seed = seed
        self.epoch = 0
        self.n = len(group_ids)

    def set_epoch(self, e):
        self.epoch = int(e)

    def __len__(self):
        return self.n

    def __iter__(self):
        rng = random.Random((self.seed, self.epoch))
        self.epoch += 1
        gids = list(self.groups.keys())
        rng.shuffle(gids)
        order = []
        for g in gids:
            members = self.groups[g][:]
            rng.shuffle(members)
            order.extend(members)
        return iter(order)

@torch.no_grad()
def _generate(model, tok, inputs, max_new_tokens, bs, model_name, role_task=None, gate_task=None, gate_kind=None, table_mask_mods=(), ctx_relay_task=None, role_matrix_task=None, sandwich_task=None, soft_task=None, struct_task=None, ffn_role_task=None):
    outs = []
    dev = model.device
    add_special = not has_chat_template(tok)
    for i in tqdm(range(0, len(inputs), bs), desc='Generating validation', leave=False):
        chunk = inputs[i:i + bs]
        texts = [format_generation_prompt(tok, c, model_name) for c in chunk]
        if soft_task is not None:
            enc = SP.generation_batch(model, tok, chunk, texts, soft_task, device=dev)
        else:
            enc = tok(texts, return_tensors='pt', padding=True, add_special_tokens=add_special).to(dev)
        if role_task is not None or gate_task is not None or role_matrix_task is not None or (ffn_role_task is not None):
            _slug = role_task if role_task is not None else gate_task if gate_task is not None else role_matrix_task if role_matrix_task is not None else ffn_role_task
            per_ex = [RGM.build_roles(tok, chunk[j], texts[j], _slug) for j in range(len(chunk))]
            roles = RGM.roles_from_attention_mask(per_ex, enc['attention_mask'].cpu()).to(dev)
            if role_task is not None:
                RGM.set_role_ids(roles)
            if ffn_role_task is not None:
                FRS.set_role_ids(roles)
            if role_matrix_task is not None:
                RM.set_role_context(roles)
            if gate_task is not None and gate_kind == 'recv':
                regions = RCV.regions_from_attention_mask([RCV.build_regions(tok, chunk[j], texts[j]) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(dev)
                (_pm, _cm, _rv) = RCV.masks_for_slug(gate_task, roles, regions)
                RCV.set_gate_context_for_slug(gate_task, _pm, _cm, _rv, valid_mask=enc['attention_mask'], device=dev)
            elif gate_task is not None and gate_kind == 'rank':
                RKG.set_gate_context_for_slug(gate_task, RKG.gate_mask(model, roles), valid_mask=enc['attention_mask'], device=dev)
        if ctx_relay_task is not None:
            _cr_roles = RGM.roles_from_attention_mask([RGM.build_roles(tok, chunk[j], texts[j], ctx_relay_task) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(dev)
            _cr_regs = RCV.regions_from_attention_mask([RCV.build_regions(tok, chunk[j], texts[j]) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(dev)
            CRM.set_context_from_ids(_cr_roles, _cr_regs)
        if struct_task is not None:
            _sb_struct = SRB.struct_from_attention_mask([SRB.build_struct(tok, texts[j]) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(dev)
            _sb_roles = _sb_regs = None
            if SRB.needs_roles(model):
                _sb_roles = RGM.roles_from_attention_mask([RGM.build_roles(tok, chunk[j], texts[j], struct_task) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(dev)
                _sb_regs = RCV.regions_from_attention_mask([RCV.build_regions(tok, chunk[j], texts[j]) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(dev)
            SRB.set_context(_sb_struct, _sb_roles, _sb_regs)
        if sandwich_task is not None:
            _sw_copies = TSW.copy_ids_from_attention_mask([TSW.build_copy_ids(tok, chunk[j], texts[j], sandwich_task) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(dev)
            _sw_regs = RCV.regions_from_attention_mask([RCV.build_regions(tok, chunk[j], texts[j]) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(dev)
            TSW.set_context_from_ids(_sw_copies, _sw_regs)
        if table_mask_mods:
            _regions = RCV.regions_from_attention_mask([RCV.build_regions(tok, chunk[j], texts[j]) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(dev)
            for _M in table_mask_mods:
                _M.set_table_mask_from_regions(_regions)
        try:
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        finally:
            if role_task is not None:
                RGM.clear_role_ids()
            if ffn_role_task is not None:
                FRS.clear_role_ids()
            if gate_task is not None:
                (RCV if gate_kind == 'recv' else RKG).clear_gate_context()
            if role_matrix_task is not None:
                RM.clear_role_context()
            if ctx_relay_task is not None:
                CRM.clear_context()
            if struct_task is not None:
                SRB.clear_context()
            if sandwich_task is not None:
                TSW.clear_context()
            for _M in table_mask_mods:
                _M.clear_table_mask()
        for j in range(len(chunk)):
            new = gen[j] if soft_task is not None else gen[j][enc['input_ids'].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True))
    return outs

def get_gold_and_pred_for_f1(slug, pred_json, gold_json, source='sotab'):
    if slug not in ('cta', 'cpa'):
        raise ValueError(f'Macro-F1 is only defined here for CTA/CPA, got {slug}')
    gold_key = GOLD_KEY[source][slug]
    gold = E.strict(gold_json.get(gold_key))
    pred = INVALID
    if pred_json is not None:
        for k in PRED_KEYS[source][slug]:
            v = pred_json.get(k)
            if v is not None:
                pred = E.strict(v)
                break
    return (gold, pred)

class GradTgtBCallback(TrainerCallback):

    def __init__(self, diag):
        self.diag = diag

    def on_pre_optimizer_step(self, args, state, control, **kw):
        self.diag.on_optimizer_step(state.global_step + 1)
        return control

class GenEvalTrainer(Trainer):
    role_gate = False
    role_matrix = False
    recv_gate = False
    rank_gate = False
    rank_gate_lr = None
    grad_tgt = None
    table_first = False
    table_last = False
    target_first = False
    soft_role = False
    soft_role_lr = None
    ffn_role = False
    ffn_role_lr = None
    struct_role = False
    struct_role_lr = None
    ctx_relay = False
    target_sandwich = False
    sandwich_mask = False
    table_mask_mods = ()
    needs_table_mask = False
    attn_aux = None
    attn_aux_log = None
    attn_aux_skipped_batches = 0

    def __init__(self, *a, val_tasks=None, gen_tok=None, val_max=4096, val_max_new_tokens=768, gen_bs=4, model_name=None, source='sotab', task_prefix=False, drop_empty_target=False, max_table_rows=0, row_trunc_method='random', max_seq_len=2048, trunc_seed=42, num_proc=1, pair_group_ids=None, pair_window=None, pair_seed=None, task_loss_weight_vec=None, task_names=None, col_major_tasks=(), table_first=False, table_first_tasks=(), table_last=False, target_end=False, target_end_tasks=(), target_first=False, target_sandwich=False, sandwich_mask=False, val_rows=None, role_gate=False, role_matrix=False, recv_gate=False, rank_gate=False, table_mask_mods=(), ctx_relay=False, attn_aux=None, rank_gate_lr=None, grad_tgt=None, soft_role=False, soft_role_lr=None, struct_role=False, struct_role_lr=None, ffn_role=False, ffn_role_lr=None, **kw):
        super().__init__(*a, **kw)
        self.val_rows_files = dict(val_rows or {})
        self.task_loss_weight_vec = task_loss_weight_vec
        self.task_names = task_names
        if task_loss_weight_vec is not None:
            self.model_accepts_loss_kwargs = False
            if getattr(self.args, 'remove_unused_columns', True):
                self.args.remove_unused_columns = False
                print('[balance_supervised_tokens] remove_unused_columns=False (task_id must reach the collator)')
        self.role_gate = bool(role_gate)
        self.role_matrix = bool(role_matrix)
        self.grad_tgt = grad_tgt
        if (self.role_gate or self.role_matrix or self.grad_tgt is not None) and getattr(self.args, 'remove_unused_columns', True):
            self.args.remove_unused_columns = False
            who = 'role_gate_mlp' if self.role_gate else 'lora_role_matrix' if self.role_matrix else 'grad_tgt_b'
            print(f'[{who}] remove_unused_columns=False (role_ids must reach the collator)')
        self.soft_role = bool(soft_role)
        self.soft_role_lr = soft_role_lr
        if self.soft_role and getattr(self.args, 'remove_unused_columns', True):
            self.args.remove_unused_columns = False
            print(f'[{SP.MECHANISM}] remove_unused_columns=False (soft_ids must reach the collator)')
        self.struct_role = bool(struct_role)
        self.struct_role_lr = struct_role_lr
        if self.struct_role and getattr(self.args, 'remove_unused_columns', True):
            self.args.remove_unused_columns = False
            print(f'[{SRB.MECHANISM}] remove_unused_columns=False (struct_ids and role_ids must reach the collator)')
        self.ffn_role = bool(ffn_role)
        self.ffn_role_lr = ffn_role_lr
        if self.ffn_role and getattr(self.args, 'remove_unused_columns', True):
            self.args.remove_unused_columns = False
            print(f'[{FRS.MECHANISM}] remove_unused_columns=False (role_ids must reach the collator)')
        self.recv_gate = bool(recv_gate)
        self.rank_gate = bool(rank_gate)
        self.table_mask_mods = tuple(table_mask_mods)
        self.needs_table_mask = bool(self.table_mask_mods)
        self.ctx_relay = bool(ctx_relay)
        self.target_sandwich = bool(target_sandwich)
        self.sandwich_mask = bool(sandwich_mask)
        if self.sandwich_mask and getattr(self.args, 'remove_unused_columns', True):
            self.args.remove_unused_columns = False
            print(f'[{TSW.MECHANISM}] remove_unused_columns=False (copy_ids and region_ids must reach the collator)')
        if self.ctx_relay and getattr(self.args, 'remove_unused_columns', True):
            self.args.remove_unused_columns = False
            print(f'[{CRM.MECHANISM}] remove_unused_columns=False (role_ids and region_ids must reach the collator)')
        self.attn_aux = attn_aux
        if self.attn_aux and getattr(self.args, 'remove_unused_columns', True):
            self.args.remove_unused_columns = False
            print(f'[{ATM.MECHANISM}] remove_unused_columns=False (region_ids and attn_recv must reach the collator)')
        self.attn_aux_log = ATM.AuxLogger() if self.attn_aux else None
        self.attn_aux_skipped_batches = 0
        if self.table_mask_mods and getattr(self.args, 'remove_unused_columns', True):
            self.args.remove_unused_columns = False
            print(f"[{'/'.join((M.MECHANISM for M in self.table_mask_mods))}] remove_unused_columns=False (region_ids must reach the collator)")
        if self.recv_gate and self.rank_gate:
            raise ValueError("--lora_rank_gate and --lora_recv_gate are alternatives, never a stack: the sender gate rewrites k/v's own rank code while the receiver gate rewrites what a receiver retrieved, and a run carrying both would report neither mechanism.")
        self.rank_gate_lr = rank_gate_lr
        if (self.recv_gate or self.rank_gate) and getattr(self.args, 'remove_unused_columns', True):
            self.args.remove_unused_columns = False
            _who = 'lora_recv_gate' if self.recv_gate else 'lora_rank_gate'
            _need = 'gate_task_id and region_ids' if self.recv_gate else 'gate_task_id and role_ids'
            print(f'[{_who}] remove_unused_columns=False ({_need} must reach the collator)')
        self.pair_group_ids = pair_group_ids
        self.pair_window = pair_window
        self.pair_seed = pair_seed
        self.gen_tok = gen_tok
        self.val_max = val_max
        self.val_max_new_tokens = val_max_new_tokens
        self.gen_bs = gen_bs
        self.model_name = model_name
        self.source = source
        self.task_prefix = task_prefix
        self.drop_empty_target = drop_empty_target
        self.max_table_rows = max_table_rows
        self.row_trunc_method = row_trunc_method
        self.max_seq_len = max_seq_len
        self.trunc_seed = trunc_seed
        self.num_proc = num_proc
        self.col_major_tasks = set(col_major_tasks or ())
        self.table_first = bool(table_first)
        self.table_last = bool(table_last)
        self.table_first_tasks = set(table_first_tasks or ())
        self.target_end = bool(target_end)
        self.target_end_tasks = set(target_end_tasks or ())
        self.target_first = bool(target_first)
        self.eval_history = []
        self.val_data = {}
        self.val_counts = {}
        rng = random.Random(42)
        for slug in val_tasks or []:
            _preset = slug in self.val_rows_files
            if _preset:
                rows = read_jsonl(self.val_rows_files[slug])
                print(f'[val_rows] {slug}: {len(rows)} pre-sampled rows from {self.val_rows_files[slug]}')
            else:
                rows = read_jsonl(f'{DATA}/{data_dir(slug, source)}/val.jsonl')
            if self.drop_empty_target and (not _preset):
                kept = [r for r in rows if not E.target_is_empty(slug, r['input'])]
                if len(kept) != len(rows):
                    print(f'[drop_empty_target] val/{slug}: dropped {len(rows) - len(kept)}/{len(rows)} empty-target examples')
                rows = kept
            store_slug = slug
            n_total = len(rows)
            capped = 'offline' if _preset else 'none'
            if not _preset and val_max is not None and (val_max > 0) and (len(rows) > val_max):
                if _val_strat_label(store_slug, self.source, rows[0]) is not None:
                    rows = label_coverage_sample(rows, val_max, rng, lambda r: _val_strat_label(store_slug, self.source, r), name=f'val/{store_slug}')
                    capped = 'stratified'
                else:
                    rows = rng.sample(rows, val_max)
                    capped = 'random'
            n_vtrim = 0
            for r in rows:
                if len(_encode_prompt(self.gen_tok, r['input'], self.model_name)) > self.max_seq_len:
                    (r['input'], _) = _trim_table_rows(r['input'], self.gen_tok, 0, self.max_seq_len, self.model_name)
                    n_vtrim += 1
            if n_vtrim:
                print(f'[val] {store_slug}: trimmed {n_vtrim} over-length prompt(s) to max_seq_len={self.max_seq_len}')
            apply_target_end(rows, self.target_end, f'val/{store_slug}', tasks=self.target_end_tasks)
            apply_target_end(rows, self.target_first, f'val/{store_slug}', mode='target_begin')
            apply_col_major(rows, self.col_major_tasks, self.gen_tok, self.max_seq_len, self.model_name, f'val/{store_slug}')
            apply_table_first(rows, self.table_first, f'val/{store_slug}', tasks=self.table_first_tasks)
            apply_table_last(rows, self.table_last, f'val/{store_slug}')
            apply_target_sandwich(rows, self.target_sandwich, f'val/{store_slug}')
            if rows:
                self.val_data[store_slug] = rows
                self.val_counts[store_slug] = {'n_total': n_total, 'n_used': len(rows), 'val_max': val_max, 'cap': capped}
                print(f'[val] {store_slug}: using {len(rows)}/{n_total} examples (val_max={val_max}, cap={capped})')

    def _clear_forward_ctx(self):
        if self.role_gate:
            RGM.clear_role_ids()
        if self.recv_gate:
            RCV.clear_gate_context()
        if self.rank_gate:
            RKG.clear_gate_context()
        if self.role_matrix:
            RM.clear_role_context()
        if self.ctx_relay:
            CRM.clear_context()
        if self.struct_role:
            SRB.clear_context()
        if self.ffn_role:
            FRS.clear_role_ids()
        if self.sandwich_mask:
            TSW.clear_context()
        if self.attn_aux:
            ATM.clear_context()
        for M in self.table_mask_mods:
            M.clear_table_mask()

    def _add_attn_aux(self, out, batch_info):
        loss = out[0] if isinstance(out, tuple) else out
        (aux, info) = ATM.aux_loss()
        if aux is None:
            self.attn_aux_skipped_batches += 1
            return out
        lam = float(self.attn_aux['lam'])
        total = loss + lam * aux
        self.attn_aux_log.update(float(loss.detach().item()), float(aux.detach().item()), float(total.detach().item()), info, batch_info)
        self._maybe_log_attn_aux()
        return (total,) + out[1:] if isinstance(out, tuple) else total

    def _maybe_log_attn_aux(self):
        every = int(self.attn_aux.get('log_every', 0) or 0)
        if every <= 0 or self.attn_aux_log.n < every:
            return
        peak = None
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        line = self.attn_aux_log.line(peak_mem=peak)
        if self.attn_aux_skipped_batches:
            line += f' | batches_without_receivers={self.attn_aux_skipped_batches}'
        print(line, flush=True)
        self.attn_aux_log.reset()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        task_id = inputs.pop('task_id', None)
        gate_task_id = inputs.pop('gate_task_id', None)
        region_ids = inputs.pop('region_ids', None)
        role_ids = inputs.pop('role_ids', None)
        copy_ids = inputs.pop('copy_ids', None)
        struct_ids = inputs.pop('struct_ids', None)
        soft_ids = inputs.pop('soft_ids', None)
        if self.soft_role:
            if soft_ids is None:
                raise ValueError('--soft_role_prompt is on but the batch carries no soft_ids; the training dataset was built without them.')
            if 'input_ids' not in inputs:
                raise ValueError("--soft_role_prompt needs the batch's input_ids to build inputs_embeds.")
            inputs['inputs_embeds'] = SP.embed_inputs(self.model, inputs.pop('input_ids'), soft_ids)
        if self.recv_gate:
            if gate_task_id is None:
                raise ValueError('--lora_recv_gate is on but the batch carries no gate_task_id; the training dataset was built without gate rows.')
            if role_ids is None:
                raise ValueError('--lora_recv_gate is on but the batch carries no role_ids; the primary and context SENDER streams are read from them, so the training dataset must be built with them.')
            if region_ids is None:
                raise ValueError('--lora_recv_gate is on but the batch carries no region_ids; the RECEIVER positions (query + answer) are read from them. They cannot be inferred from role_ids, whose `none` also covers the instruction, the Markdown syntax and the padding.')
            (_pm, _cm) = RCV.stream_masks(gate_task_id, role_ids, region_ids)
            _labels = inputs.get('labels')
            RCV.set_gate_context(gate_task_id, _pm, _cm, RCV.receiver_mask_from_regions(region_ids), valid_mask=inputs.get('attention_mask'), answer_mask=None if _labels is None else _labels != -100)
        if self.struct_role:
            if struct_ids is None:
                raise ValueError('--struct_role_bias is on but the batch carries no struct_ids; the same-row/same-column relations come from those table coordinates, so the training dataset must be built with them.')
            SRB.set_context(struct_ids, role_ids, region_ids)
        if self.ctx_relay:
            if role_ids is None or region_ids is None:
                raise ValueError(f"--ctx_relay_mask is on but the batch carries no {('role_ids' if role_ids is None else 'region_ids')}; the context keys come from the role ids and the post-table receivers from the region ids, so the training dataset must be built with both.")
            CRM.set_context_from_ids(role_ids, region_ids)
        if self.sandwich_mask:
            if copy_ids is None or region_ids is None:
                raise ValueError(f"--target_sandwich is on but the batch carries no {('copy_ids' if copy_ids is None else 'region_ids')}; the cut keys come from the copy ids and the post-table cutting queries from the region ids, so the training dataset must be built with both.")
            TSW.set_context_from_ids(copy_ids, region_ids)
        attn_recv = inputs.pop('attn_recv', None)
        attn_aux_batch = None
        if self.attn_aux is not None and model.training and torch.is_grad_enabled():
            if region_ids is None:
                raise ValueError('--attn_table_aux is on but the batch carries no region_ids; the serialized table span the auxiliary term sums attention over is read from them, so the training dataset must be built with them.')
            if attn_recv is None:
                raise ValueError('--attn_table_aux is on but the batch carries no attn_recv; the semantic decision positions are computed once when the training set is tokenized, so the dataset must be built with them.')
            _valid = attn_recv >= 0
            attn_aux_batch = ATM.set_context(attn_recv.clamp(min=0), _valid, region_ids, inputs.get('attention_mask'))
        if self.table_mask_mods:
            if region_ids is None:
                raise ValueError(f"{'/'.join((M.MECHANISM for M in self.table_mask_mods))} is on but the batch carries no region_ids; the table / non-table token split is read from them, so the training dataset must be built with them.")
            for M in self.table_mask_mods:
                M.set_table_mask_from_regions(region_ids)
        if self.rank_gate:
            if gate_task_id is None:
                raise ValueError('--lora_rank_gate is on but the batch carries no gate_task_id; the training dataset was built without gate rows.')
            if role_ids is None:
                raise ValueError("--lora_rank_gate is on but the batch carries no role_ids; the gate's table-cell token scope is read from them, so the training dataset must be built with them.")
            RKG.set_gate_context(gate_task_id, RKG.gate_mask(self.model, role_ids), valid_mask=inputs.get('attention_mask'))
        if self.role_gate:
            if role_ids is None:
                raise ValueError('--role_gate_mlp is on but the batch carries no role_ids; the training dataset was built without them.')
            RGM.set_role_ids(role_ids)
        if self.ffn_role:
            if role_ids is None:
                raise ValueError('--ffn_role_scale is on but the batch carries no role_ids; the four task/role scaling vectors are selected from them, so the training dataset must be built with them.')
            FRS.set_role_ids(role_ids)
        if self.role_matrix:
            if role_ids is None:
                raise ValueError('--lora_role_matrix is on but the batch carries no role_ids; the target/context/none routing is read from them, so the training dataset must be built with them.')
            RM.set_role_context(role_ids)
        if self.grad_tgt is not None and model.training:
            if role_ids is None:
                raise ValueError('--grad_tgt_b is on but the batch carries no role_ids; the training dataset was built without them.')
            self.grad_tgt.set_batch_mask(role_ids)
        defer_clear = (self.role_gate or self.ffn_role or self.recv_gate or self.rank_gate or self.role_matrix or self.needs_table_mask or self.ctx_relay or self.sandwich_mask or self.struct_role) and model.training
        try:
            if self.task_loss_weight_vec is None:
                out = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
                if attn_aux_batch is not None:
                    out = self._add_attn_aux(out, attn_aux_batch)
                if not defer_clear:
                    self._clear_forward_ctx()
                return out
        except Exception:
            self._clear_forward_ctx()
            raise
        if task_id is None:
            raise ValueError('--balance_supervised_tokens is on but the batch carries no task_id; the training dataset was built without task ids.')
        labels = inputs.pop('labels')
        try:
            outputs = model(**inputs)
        except Exception:
            self._clear_forward_ctx()
            raise
        loss = weighted_supervised_ce(outputs.logits, labels, task_id, self.task_loss_weight_vec)
        if attn_aux_batch is not None:
            loss = self._add_attn_aux(loss, attn_aux_batch)
        if not defer_clear:
            self._clear_forward_ctx()
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, *args, **kwargs):
        if self.grad_tgt is not None:
            self.grad_tgt.begin_microbatch(self.state.global_step + 1)
        try:
            return super().training_step(model, inputs, *args, **kwargs)
        finally:
            self._clear_forward_ctx()
            if self.grad_tgt is not None:
                self.grad_tgt.clear_batch_mask()

    def _side_banks(self):
        out = []
        gate = getattr(self.model, 'recv_gate_bank', None) or getattr(self.model, 'rank_gate_bank', None)
        if gate is not None and self.rank_gate_lr is not None:
            out.append(('receiver gate' if self.recv_gate else 'K/V rank gate', gate, float(self.rank_gate_lr)))
        srb = getattr(self.model, 'struct_role_bias', None)
        if srb is not None and self.struct_role_lr is not None:
            out.append((SRB.MECHANISM, srb, float(self.struct_role_lr)))
        frs = getattr(self.model, 'ffn_role_bank', None)
        if frs is not None and self.ffn_role_lr is not None:
            out.append((FRS.MECHANISM, frs, float(self.ffn_role_lr)))
        srp = getattr(self.model, 'soft_role_bank', None)
        if srp is not None and self.soft_role_lr is not None:
            out.append((SP.MECHANISM, srp, float(self.soft_role_lr)))
        return out

    def create_optimizer(self):
        banks = self._side_banks()
        if self.optimizer is not None or not banks:
            return super().create_optimizer()
        bank_ids = [{id(p) for p in m.parameters()} for (_l, m, _lr) in banks]
        decay = set(self.get_decay_parameter_names(self.model))
        named = [(n, p) for (n, p) in self.model.named_parameters() if p.requires_grad]

        def owner(p):
            for (i, ids) in enumerate(bank_ids):
                if id(p) in ids:
                    return i
            return -1

        def pick(which, is_decay):
            return [p for (n, p) in named if owner(p) == which and (n in decay) is is_decay]
        groups = [{'params': pick(-1, True), 'weight_decay': self.args.weight_decay}, {'params': pick(-1, False), 'weight_decay': 0.0}]
        tags = [-1, -1]
        for (i, (_label, _m, lr)) in enumerate(banks):
            groups.append({'params': pick(i, True), 'weight_decay': self.args.weight_decay, 'lr': lr})
            groups.append({'params': pick(i, False), 'weight_decay': 0.0, 'lr': lr})
            tags += [i, i]
        keep = [j for (j, g) in enumerate(groups) if g['params']]
        groups = [groups[j] for j in keep]
        tags = [tags[j] for j in keep]
        for (i, (label, m, _lr)) in enumerate(banks):
            got = sum((p.numel() for (g, t) in zip(groups, tags) if t == i for p in g['params']))
            want = sum((p.numel() for p in m.parameters() if p.requires_grad))
            if got != want:
                raise RuntimeError(f'the {label} parameter group does not hold every trainable tensor: grouped {got} of {want}')
        if self.optimizer_cls_and_kwargs is not None:
            (optimizer_cls, optimizer_kwargs) = self.optimizer_cls_and_kwargs
        else:
            (optimizer_cls, optimizer_kwargs) = self.get_optimizer_cls_and_kwargs(self.args, self.model)
        optimizer_kwargs.pop('params', None)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        n_lora = sum((p.numel() for (g, t) in zip(groups, tags) if t < 0 for p in g['params']))
        self.optimizer_group_summary = [{'n_params': sum((p.numel() for p in g['params'])), 'n_tensors': len(g['params']), 'lr': float(self.args.learning_rate if t < 0 else banks[t][2]), 'weight_decay': float(g['weight_decay']), 'bank': None if t < 0 else banks[t][0], 'gate': bool(t >= 0)} for (g, t) in zip(groups, tags)]
        print(f'[optim] {len(groups)} parameter groups (weight_decay={self.args.weight_decay:g}, scheduler unchanged): LoRA {n_lora:,} params @ lr={self.args.learning_rate:g}')
        for (i, (label, _m, lr)) in enumerate(banks):
            n_b = sum((p.numel() for (g, t) in zip(groups, tags) if t == i for p in g['params']))
            print(f'[optim]   {label} {n_b:,} params @ lr={lr:g}')
        return super().create_optimizer()

    def _get_train_sampler(self, train_dataset=None):
        if self.pair_group_ids is not None:
            return PairedGroupSampler(self.pair_group_ids, self.pair_window, self.pair_seed)
        return super()._get_train_sampler(train_dataset)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        tok = self.gen_tok
        was_training = self.model.training
        self.model.eval()
        prev_pad = tok.padding_side
        prev_cache = self.model.config.use_cache
        tok.padding_side = 'left'
        self.model.config.use_cache = True
        per_task_scores = {}
        per_task_details = {}
        for (slug, rows) in self.val_data.items():
            gens = _generate(self.model, tok, [with_task_prefix(r['input'], slug, self.task_prefix) for r in rows], self.val_max_new_tokens, self.gen_bs, self.model_name, role_task=slug if self.role_gate else None, gate_task=slug if self.recv_gate or self.rank_gate else None, gate_kind='recv' if self.recv_gate else 'rank' if self.rank_gate else None, role_matrix_task=slug if self.role_matrix else None, table_mask_mods=self.table_mask_mods, ctx_relay_task=slug if self.ctx_relay else None, sandwich_task=slug if self.sandwich_mask else None, soft_task=slug if self.soft_role else None, struct_task=slug if self.struct_role else None, ffn_role_task=slug if self.ffn_role else None)
            scorer = resolve_scorer(slug, self.source)
            tc = 0
            tt = 0
            exact_count = 0
            parse_fail = 0
            golds = []
            preds = []
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
                    (gold, pred) = get_gold_and_pred_for_f1(slug, pj, gj, self.source)
                    golds.append(gold)
                    preds.append(pred)
                elif slug == 'em':
                    golds.append(E.strict(gj['label']))
                    pv = E._read_value(pj, ['label']) if pj is not None else None
                    preds.append(E.strict(pv) if pv is not None else INVALID)
            acc = tc / max(1, tt)
            exact = exact_count / max(1, len(rows))
            if slug in {'cta', 'cpa'}:
                (macro_f1, weighted_f1) = E.macro_weighted_f1(golds, preds)
                selected_score = macro_f1
                per_task_scores[slug] = selected_score
                per_task_details[slug] = {'selection_metric': 'macro_f1', 'selection_score': selected_score, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1, 'acc': acc, 'exact': exact, 'parse_fail': parse_fail, 'n_val': len(rows)}
            elif slug in ('hvm', 'tabfact', 'samerow', 'wikiqa', 'finqa', 'tablebench', 'cea'):
                selected_score = acc
                per_task_scores[slug] = selected_score
                per_task_details[slug] = {'selection_metric': 'accuracy', 'selection_score': selected_score, 'acc': acc, 'exact': exact, 'parse_fail': parse_fail, 'n_val': len(rows)}
            elif slug == 'em':
                prf = E.binary_prf(golds, preds, E.strict('match'))
                selected_score = prf['f1']
                per_task_scores[slug] = selected_score
                per_task_details[slug] = {'selection_metric': 'match_f1', 'selection_score': selected_score, 'match_f1': prf['f1'], 'match_precision': prf['precision'], 'match_recall': prf['recall'], 'acc': acc, 'exact': exact, 'parse_fail': parse_fail, 'n_val': len(rows)}
            else:
                raise ValueError(f'Unknown validation task: {slug}')
        tok.padding_side = prev_pad
        self.model.config.use_cache = prev_cache
        if was_training:
            self.model.train()
        val_score = float(np.mean(list(per_task_scores.values()))) if per_task_scores else 0.0
        self.eval_history.append((val_score, dict(per_task_details)))
        metrics = {f'{metric_key_prefix}_val_score': val_score}
        for (slug, detail) in per_task_details.items():
            metrics[f'{metric_key_prefix}_{slug}_selection_score'] = detail['selection_score']
            metrics[f'{metric_key_prefix}_{slug}_acc'] = detail['acc']
            metrics[f'{metric_key_prefix}_{slug}_exact'] = detail['exact']
            metrics[f'{metric_key_prefix}_{slug}_parse_fail'] = detail['parse_fail']
            if 'macro_f1' in detail:
                metrics[f'{metric_key_prefix}_{slug}_macro_f1'] = detail['macro_f1']
            if 'weighted_f1' in detail:
                metrics[f'{metric_key_prefix}_{slug}_weighted_f1'] = detail['weighted_f1']
            if 'match_f1' in detail:
                metrics[f'{metric_key_prefix}_{slug}_match_f1'] = detail['match_f1']
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

def _const_ce_logits(ce, vocab=4):
    z = float(np.log(np.expm1(ce)))
    row = [0.0, z] + [-1e+30] * (vocab - 2)
    return row

def selftest_balance():
    ok = [True]

    def check(name, cond, detail=''):
        status = 'PASS' if cond else 'FAIL'
        print(f"  [{status}] {name}{(' -- ' + detail if detail else '')}")
        if not cond:
            ok[0] = False
    torch.manual_seed(0)
    V = 4
    print('statistics and fixed weights')
    enc = [([1, 2, 3, 4], [-100, -100, 7, 7]), ([1, 2, 3, 4, 5, 6], [-100, -100, 7, 7, 7, 7]), ([1, 2, 3], [-100, -100, 7]), ([1, 2, 3], [-100, -100, 7])]
    (tids, names) = build_task_ids(['A', 'A', 'B', 'B'])
    check('build_task_ids names', names == ['A', 'B'], str(names))
    check('build_task_ids ids', tids == [0, 0, 1, 1], str(tids))
    st = supervised_token_stats(enc, tids, names)
    check('mu_A', abs(st['A']['mean'] - 3.0) < 1e-12, f"mean={st['A']['mean']}")
    check('mu_B', abs(st['B']['mean'] - 1.0) < 1e-12, f"mean={st['B']['mean']}")
    check('median/min/max A', (st['A']['median'], st['A']['min'], st['A']['max']) == (3.0, 2, 4))
    (w, exp) = compute_task_loss_weights(st)
    check('mu_global', abs(exp['mu_global'] - 2.0) < 1e-12, f"{exp['mu_global']}")
    check('w_A == mu_global/mu_A == 2/3', abs(w['A'] - 2.0 / 3.0) < 1e-12, f"{w['A']}")
    check('w_B == mu_global/mu_B == 2', abs(w['B'] - 2.0) < 1e-12, f"{w['B']}")
    mass_w = sum((st[t]['examples'] * st[t]['mean'] * w[t] for t in st))
    mass_o = sum((st[t]['examples'] * st[t]['mean'] for t in st))
    check('dataset-scale preserved (sum N*mu*w == sum N*mu)', abs(mass_w - mass_o) < 1e-12, f'{mass_w} vs {mass_o}')
    ma = st['A']['examples'] * st['A']['mean'] * w['A']
    mb = st['B']['examples'] * st['B']['mean'] * w['B']
    check('equal examples -> equal weighted token mass', abs(ma - mb) < 1e-12, f'{ma} vs {mb}')
    check('q_A original 6/8', abs(exp['original']['A'] - 0.75) < 1e-12, f"{exp['original']['A']}")
    check('q_A weighted 1/2', abs(exp['weighted']['A'] - 0.5) < 1e-12, f"{exp['weighted']['A']}")
    check('q_B weighted 1/2', abs(exp['weighted']['B'] - 0.5) < 1e-12, f"{exp['weighted']['B']}")
    st_solo = supervised_token_stats(enc[:2], [0, 0], ['A'])
    (w_solo, exp_solo) = compute_task_loss_weights(st_solo)
    check('single-task dataset -> w == 1.0', w_solo['A'] == 1.0, f"{w_solo['A']}")
    check('single-task mu_global == mu_A', abs(exp_solo['mu_global'] - st_solo['A']['mean']) < 1e-12)
    st_unequal = {'A': {'examples': 100, 'mean': 3.0}, 'B': {'examples': 2, 'mean': 1.0}}
    (w_un, exp_un) = compute_task_loss_weights(st_unequal)
    mu_g_un = (100 * 3.0 + 2 * 1.0) / 102.0
    check('unequal example counts: w still mu_global/mu_t', abs(w_un['A'] - mu_g_un / 3.0) < 1e-12 and abs(w_un['B'] - mu_g_un / 1.0) < 1e-12)
    check('unequal example counts are NOT equalized', abs(exp_un['weighted']['A'] - exp_un['weighted']['B']) > 0.1, f"A={exp_un['weighted']['A']:.4f} B={exp_un['weighted']['B']:.4f}")
    check('supervised_token_stats rejects supervised position 0', _raises(lambda : supervised_token_stats([([1, 2], [5, 5])], [0], ['A'])))
    check('supervised_token_stats rejects length mismatch', _raises(lambda : supervised_token_stats(enc, tids[:2], names)))
    print('data path (dataset -> collator -> batch)')
    ds = SFTDataset(enc, task_ids=tids)
    coll = PadCollator(pad_id=0)
    batch = coll([ds[i] for i in range(len(enc))])
    check('batch carries task_id', 'task_id' in batch)
    check('task_id rows == labels rows', batch['task_id'].shape[0] == batch['labels'].shape[0])
    check('task_id values survive batching', batch['task_id'].tolist() == tids)
    per_row = (batch['labels'] != -100).sum(1).tolist()
    check('padding adds no supervised tokens', per_row == [2, 4, 1, 1], str(per_row))
    check('row->task alignment holds after padding', [names[t] for t in batch['task_id'].tolist()] == ['A', 'A', 'B', 'B'])
    check('SFTDataset rejects misaligned task_ids', _raises(lambda : SFTDataset(enc, task_ids=tids[:3])))
    check('no task_id when disabled', 'task_id' not in SFTDataset(enc)[0] and 'task_id' not in coll([SFTDataset(enc)[i] for i in range(2)]))
    print('control 2: single-task invariance')
    (B, L) = (3, 9)
    logits = torch.randn(B, L, V, dtype=torch.float64)
    labels = torch.full((B, L), -100, dtype=torch.long)
    labels[0, 5:] = torch.randint(0, V, (L - 5,))
    labels[1, 6:] = torch.randint(0, V, (L - 6,))
    labels[2, 4:] = torch.randint(0, V, (L - 4,))
    tid_same = torch.zeros(B, dtype=torch.long)
    plain = F.cross_entropy(logits[:, :-1, :].reshape(-1, V).float(), labels[:, 1:].reshape(-1), ignore_index=-100, reduction='mean')
    enc_solo = [([0] * L, labels[i].tolist()) for i in range(B)]
    st_s = supervised_token_stats(enc_solo, [0] * B, ['only'])
    (w_s, _) = compute_task_loss_weights(st_s)
    check('single-task run derives w == 1.0', w_s['only'] == 1.0, f"{w_s['only']}")
    got = weighted_supervised_ce(logits, labels, tid_same, torch.tensor([w_s['only']], dtype=torch.float64))
    check('single task at its derived w == ordinary token-mean CE', torch.allclose(got, plain, atol=1e-12), f'{float(got):.12f} vs {float(plain):.12f}')
    print('control 3: synthetic 3x length imbalance')
    (ce_a, ce_b) = (1.0, 4.0)
    (nA, nB) = (3, 1)
    (B2, L2) = (4, 8)
    lg = torch.zeros(B2, L2, V, dtype=torch.float64)
    lb = torch.full((B2, L2), -100, dtype=torch.long)
    for i in range(B2):
        (n, ce) = (nA, ce_a) if i < 2 else (nB, ce_b)
        for j in range(L2 - n, L2):
            lg[i, j - 1] = torch.tensor(_const_ce_logits(ce, V), dtype=torch.float64)
            lb[i, j] = 0
    tid = torch.tensor([0, 0, 1, 1])
    counts = (lb != -100).sum(1).tolist()
    check('synthetic counts 3:1', counts == [nA, nA, nB, nB], str(counts))
    n_tok = 2 * nA + 2 * nB
    unweighted = weighted_supervised_ce(lg, lb, tid, torch.tensor([1.0, 1.0]))
    exp_unw = (2 * nA * ce_a + 2 * nB * ce_b) / n_tok
    check('ordinary loss == token-count-weighted mean', _close32(unweighted, exp_unw), f'{float(unweighted):.9f} vs {exp_unw:.9f}')
    share_a = 2 * nA / n_tok
    check('ordinary token share A == 3x B', abs(share_a - 0.75) < 1e-12, f'{share_a}')
    st2 = supervised_token_stats([([0] * L2, lb[i].tolist()) for i in range(B2)], tid.tolist(), ['A', 'B'])
    (w2, exp2) = compute_task_loss_weights(st2)
    check('mu_global == 2', abs(exp2['mu_global'] - 2.0) < 1e-12, f"{exp2['mu_global']}")
    check('derived weights (2/3, 2)', abs(w2['A'] - 2.0 / 3.0) < 1e-12 and abs(w2['B'] - 2.0) < 1e-12, f"A={w2['A']:.6f} B={w2['B']:.6f}")
    check('weighted token mass equal per equal example count', abs(2 * nA * w2['A'] - 2 * nB * w2['B']) < 1e-12, f"A={2 * nA * w2['A']} B={2 * nB * w2['B']}")
    wv2 = torch.tensor([w2['A'], w2['B']], dtype=torch.float64)
    balanced = weighted_supervised_ce(lg, lb, tid, wv2)
    exp_bal = (2 * nA * w2['A'] * ce_a + 2 * nB * w2['B'] * ce_b) / n_tok
    check('weighted loss == weighted numerator / ordinary token count', _close32(balanced, exp_bal), f'{float(balanced):.9f} vs {exp_bal:.9f}')
    check('weighted expected contributions equal', abs(exp2['weighted']['A'] - exp2['weighted']['B']) < 1e-12, f"A={exp2['weighted']['A']:.6f} B={exp2['weighted']['B']:.6f}")
    check('unweighted expected contributions are 3:1', abs(exp2['original']['A'] - 0.75) < 1e-12, f"A={exp2['original']['A']:.6f}")
    print('control 3b: task-pure batch keeps the fixed multiplier (regression)')
    for (name, rows, wt, ce_t) in (('A', (0, 1), w2['A'], ce_a), ('B', (2, 3), w2['B'], ce_b)):
        sl = list(rows)
        pure_o = weighted_supervised_ce(lg[sl], lb[sl], tid[sl] * 0 + tid[sl[0]], torch.tensor([1.0, 1.0], dtype=torch.float64))
        pure_w = weighted_supervised_ce(lg[sl], lb[sl], tid[sl], wv2)
        ratio = float(pure_w) / float(pure_o)
        check(f'task-pure {name}: L_weighted == w_{name} * L_ordinary', _close32(ratio, wt), f'ratio={ratio:.9f} w_{name}={wt:.9f}')
        check(f'task-pure {name}: w does NOT cancel', abs(ratio - 1.0) > 0.1, f'ratio={ratio:.6f}')
        check(f'task-pure {name}: L_weighted == w * mean(CE)', _close32(pure_w, wt * ce_t), f'{float(pure_w):.9f} vs {wt * ce_t:.9f}')
    print('control 4: -100 contributes neither CE nor weight')
    lg4 = lg.clone()
    mask_pos = lb[:, 1:] == -100
    noise = torch.randn(B2, L2 - 1, V, dtype=torch.float64) * 50.0
    lg4[:, :-1, :][mask_pos] = noise[mask_pos]
    check('loss unchanged when ignored logits are scrambled', torch.allclose(weighted_supervised_ce(lg4, lb, tid, wv2), balanced, atol=1e-12))
    den = float((lb[:, 1:] != -100).double().sum())
    check('denominator == number of supervised tokens (no w)', abs(den - n_tok) < 1e-12, f'{den} vs {n_tok}')
    scaled = weighted_supervised_ce(lg, lb, tid, wv2 * 3.0)
    check('scaling every weight by k scales the loss by k (denominator is w-free)', _close32(scaled, 3.0 * float(balanced)), f'{float(scaled):.9f} vs {3.0 * float(balanced):.9f}')
    all_masked = torch.full((2, 5), -100, dtype=torch.long)
    check('all-masked batch raises rather than dividing by zero', _raises(lambda : weighted_supervised_ce(torch.randn(2, 5, V, dtype=torch.float64), all_masked, torch.zeros(2, dtype=torch.long), torch.tensor([1.0]))))
    print('control 5: task ids stay aligned with examples after the shift')
    swapped = weighted_supervised_ce(lg, lb, torch.tensor([1, 1, 0, 0]), wv2)
    exp_sw = (2 * nA * w2['B'] * ce_a + 2 * nB * w2['A'] * ce_b) / n_tok
    check('swapping task ids moves exactly the per-row weights', _close32(swapped, exp_sw), f'{float(swapped):.9f} vs {exp_sw:.9f}')
    check('swapped != aligned', abs(float(swapped) - float(balanced)) > 1e-06)
    doubled = weighted_supervised_ce(lg, lb, tid, torch.tensor([2 * w2['A'], w2['B']], dtype=torch.float64))
    exp_d = (2 * nA * 2 * w2['A'] * ce_a + 2 * nB * w2['B'] * ce_b) / n_tok
    check('weight applies to all shifted targets of its row', _close32(doubled, exp_d), f'{float(doubled):.9f} vs {exp_d:.9f}')
    check('rejects task_id/labels row mismatch', _raises(lambda : weighted_supervised_ce(lg, lb, tid[:2], wv2)))
    print('control 1: disabled mode uses the stock Trainer loss path')
    (stock_ok, detail) = _selftest_stock_loss_path()
    check('GenEvalTrainer.compute_loss delegates to Trainer.compute_loss when disabled', stock_ok, detail)
    print('control 6: bf16 / autocast')
    lg_bf = lg.to(torch.bfloat16)
    got_bf = weighted_supervised_ce(lg_bf, lb, tid, wv2.float())
    check('bf16 logits accepted, fp32 loss', got_bf.dtype == torch.float32)
    check('bf16 result close to fp64 result', abs(float(got_bf) - float(balanced)) < 0.05, f'{float(got_bf):.6f} vs {float(balanced):.6f}')
    with torch.autocast('cpu', dtype=torch.bfloat16):
        got_ac = weighted_supervised_ce(lg_bf, lb, tid, wv2.float())
    check('runs under autocast', torch.isfinite(got_ac).item())
    lg_g = lg.clone().float().requires_grad_(True)
    weighted_supervised_ce(lg_g, lb, tid, wv2.float()).backward()
    check('gradient flows to logits', lg_g.grad is not None and torch.isfinite(lg_g.grad).all())
    check('no gradient at ignored positions', float(lg_g.grad[:, :-1, :][mask_pos].abs().max()) == 0.0)
    print('ALL PASS' if ok[0] else 'FAILURES PRESENT')
    return ok[0]

def _close32(a, b, rtol=5e-06):
    return abs(float(a) - float(b)) <= rtol * max(1.0, abs(float(b)))

def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False

def _selftest_stock_loss_path():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(1234)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    model = LlamaForCausalLM(cfg).eval()
    ids = torch.randint(0, 64, (3, 12))
    labels = torch.full((3, 12), -100, dtype=torch.long)
    labels[:, 8:] = ids[:, 8:]
    am = torch.ones_like(ids)
    reference = model(input_ids=ids, attention_mask=am, labels=labels).loss
    trainer = GenEvalTrainer.__new__(GenEvalTrainer)
    trainer.task_loss_weight_vec = None
    trainer.label_smoother = None
    trainer.compute_loss_func = None
    trainer.model_accepts_loss_kwargs = False
    trainer.args = argparse.Namespace(past_index=-1, average_tokens_across_devices=False)
    inputs = {'input_ids': ids, 'attention_mask': am, 'labels': labels, 'task_id': torch.tensor([0, 1, 0])}
    got = GenEvalTrainer.compute_loss(trainer, model, inputs)
    same = bool(torch.equal(got, reference))
    stripped = 'task_id' not in inputs
    return (same and stripped, f'stock={float(reference):.12f} got={float(got):.12f} task_id_stripped={stripped}')
FULL_FINETUNE_CONFLICTS = (('--init_qkv_adapter', lambda a: a.init_qkv_adapter is not None), ('--merge_adapter', lambda a: a.merge_adapter is not None), ('--lora_role_matrix', lambda a: bool(a.lora_role_matrix)), ('--lora_rank_gate', lambda a: a.lora_rank_gate != 'off'), ('--lora_recv_gate', lambda a: a.lora_recv_gate != 'off'), ('--role_gate_mlp', lambda a: bool(a.role_gate_mlp)), ('--ffn_role_scale', lambda a: bool(a.ffn_role_scale)), ('--split_gate_proj', lambda a: bool(a.split_gate_proj)), ('--swiglu_thres', lambda a: bool(a.swiglu_thres)), ('--soft_role_prompt', lambda a: bool(a.soft_role_prompt)), ('--struct_role_bias', lambda a: bool(a.struct_role_bias)), ('--ctx_relay_mask', lambda a: bool(a.ctx_relay_mask)), ('--target_sandwich', lambda a: bool(a.target_sandwich)), ('--attn_table_aux', lambda a: bool(a.attn_table_aux)), ('--grad_col_mask', lambda a: bool(a.grad_col_mask)), ('--grad_tgt_b', lambda a: bool(a.grad_tgt_b)))

def refuse_full_finetune_conflicts(args):
    if not args.full_finetune:
        return
    bad = [n for (n, on) in FULL_FINETUNE_CONFLICTS if on(args)]
    if bad:
        raise SystemExit(f"--full_finetune trains the backbone itself, with no LoRA adapter anywhere, so {', '.join(bad)} would have nothing to act on. Drop them, or drop --full_finetune.")

def setup_full_finetune(model, grad_checkpointing):
    for p in model.parameters():
        p.requires_grad_(True)
    frozen = [n for (n, p) in model.named_parameters() if not p.requires_grad]
    if frozen:
        raise SystemExit(f'--full_finetune left {len(frozen)} tensors frozen (e.g. {frozen[:3]}); every backbone parameter must train.')
    peft = [n for (n, _p) in model.named_parameters() if 'lora_' in n]
    if peft:
        raise SystemExit(f'--full_finetune found {len(peft)} LoRA tensors on the model (e.g. {peft[:3]}); this run must carry no adapter at all.')
    n_all = sum((p.numel() for p in model.parameters()))
    n_tr = sum((p.numel() for p in model.parameters() if p.requires_grad))
    info = {'n_parameters': int(n_all), 'n_trainable': int(n_tr), 'trainable_pct': 100.0 * n_tr / max(1, n_all), 'n_tensors': sum((1 for _ in model.parameters())), 'param_dtypes': sorted({str(p.dtype) for p in model.parameters()}), 'peft_attached': False, 'optim': 'adamw_bnb_8bit', 'grad_checkpointing_requested': bool(grad_checkpointing), 'gradient_checkpointing': bool(getattr(model, 'is_gradient_checkpointing', False)), 'use_cache': bool(model.config.use_cache)}
    print(f"[full_finetune] NO PEFT: {info['n_tensors']} tensors, {n_tr:,} trainable of {n_all:,} ({info['trainable_pct']:.2f}%), dtypes {info['param_dtypes']}, optimizer adamw_bnb_8bit, gradient_checkpointing={info['gradient_checkpointing']}, use_cache={info['use_cache']}")
    return info

def main():
    import sys
    if '--selftest' in sys.argv[1:]:
        raise SystemExit(0 if selftest_balance() else 1)
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', required=True, choices=['single_cta', 'single_cpa', 'single_cea', 'single_hvm', 'single_em', 'single_tabfact', 'single_wikiqa', 'single_samerow', 'multi', 'multi_ctacpa', 'multi_shared_ctacpa', 'multi_paired_ctacpa', 'multi_em', 'multi_tabfact', 'multi_cta_em_tabfact', 'multi_cpa_em_tabfact', 'multi_tabfact_hvm', 'multi_tabfact_em', 'multi_cta_hvm', 'multi_cpa_hvm', 'multi_cta_em', 'multi_cpa_em', 'multi_cta_tabfact', 'multi_cpa_tabfact', 'multi_tableqa', 'multi_cta_samerow', 'multi_cpa_samerow', 'multi_hvm_samerow', 'multi_tabfact_samerow'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--budget', type=int, default=100)
    ap.add_argument('--samerow_source', default='auto', choices=('auto',) + tuple(SR.SOURCE_TASKS))
    ap.add_argument('--samerow_max_rows', type=int, default=20)
    ap.add_argument('--model', default=f'{ROOT}/Qwen3-4B-Instruct-2507')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cta_cpa_source', default='t2d', choices=['sotab', 'sotab_full', 'sotab_all', 't2d', 'shared', 'targetonly'])
    ap.add_argument('--select_mode', default='random', choices=['random', 'similar', 'dissimilar', 'all', 'target_rank', 'target_rank_rev'])
    ap.add_argument('--val_gen_batch_size', type=int, default=16)
    ap.add_argument('--drop_empty_target', action='store_true')
    ap.add_argument('--drop_truncated_tables', action='store_true')
    ap.add_argument('--col_mask_method', default='off', choices=['off', 'gradient', 'random'])
    ap.add_argument('--grad_col_mask', action='store_true')
    ap.add_argument('--max_table_rows', type=int, default=0)
    ap.add_argument('--row_trunc_method', choices=['random', 'tfidf', 'coverage', 'dedup', 'coverage_row', 'central_row', 'coverage_col'], default='random')
    ap.add_argument('--fill_budget', action='store_true')
    ap.add_argument('--num_proc', type=int, default=4)
    ap.add_argument('--drop_empty_context_cols', action='store_true')
    ap.add_argument('--reindex_truncated_cols', action='store_true')
    ap.add_argument('--max_context_cols', type=int, default=0)
    ap.add_argument('--col_trunc_method', choices=['nearest', 'random', 'coverage_col'], default='nearest')
    ap.add_argument('--dry_run_data', action='store_true')
    ap.add_argument('--train_rows', default=None)
    ap.add_argument('--val_rows', default=None)
    ap.add_argument('--max_seq_len', type=int, default=2048)
    ap.add_argument('--epochs', type=float, default=3.0)
    ap.add_argument('--lr', type=float, default=0.0001)
    ap.add_argument('--micro_bs', type=int, default=8)
    ap.add_argument('--grad_accum', type=int, default=1)
    ap.add_argument('--group_by_length', action='store_true')
    ap.add_argument('--lora_r', type=int, default=16)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--target_modules', nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], choices=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    ap.add_argument('--init_qkv_adapter', default=None)
    ap.add_argument('--merge_adapter', default=None)
    ap.add_argument('--full_finetune', action='store_true')
    ap.add_argument('--attn_impl', default='sdpa', choices=['sdpa', 'flash_attention_2', 'eager'])
    ap.add_argument('--grad_checkpointing', action='store_true')
    ap.add_argument('--val_max', type=int, default=4096)
    ap.add_argument('--log_steps', type=int, default=100)
    ap.add_argument('--val_max_new_tokens', type=int, default=400)
    ap.add_argument('--balance-supervised-tokens', '--balance_supervised_tokens', dest='balance_supervised_tokens', action='store_true')
    ap.add_argument('--no_val', action='store_true')
    ap.add_argument('--grad_tgt_b', type=str, default='')
    ap.add_argument('--split_gate_proj', action='store_true')
    ap.add_argument('--swiglu_thres', action='store_true')
    ap.add_argument('--role_gate_mlp', action='store_true')
    ap.add_argument('--ffn_role_scale', action='store_true')
    ap.add_argument('--ffn_role_scale_lr', type=float, default=0.001)
    ap.add_argument('--lora_role_matrix', action='store_true')
    ap.add_argument('--lora_rank_gate', choices=list(RKG.MODES), default=RKG.DEFAULT_MODE)
    ap.add_argument('--lora_recv_gate', choices=list(RCV.MODES), default=RCV.DEFAULT_MODE)
    ap.add_argument('--rank_gate_exclude_target', nargs='+', default=[], choices=list(RCV.PRIMARY_TASKS))
    ap.add_argument('--rank_gate_diag_n', type=int, default=16)
    ap.add_argument('--rank_gate_lr', type=float, default=None)
    ap.add_argument('--transpose_augment', action='store_true')
    ap.add_argument('--swap_context_epochs', action='store_true')
    ap.add_argument('--swap_context_blocks', action='store_true')
    ap.add_argument('--target_sandwich', action='store_true')
    ap.add_argument('--sandwich_mask', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--soft_role_prompt', action='store_true')
    ap.add_argument('--struct_role_bias', action='store_true')
    ap.add_argument('--struct_role_lr', type=float, default=0.001)
    ap.add_argument('--struct_role_relations', choices=sorted(SRB.RELATION_SETS), default=SRB.DEFAULT_RELATION_SET)
    ap.add_argument('--ctx_relay_mask', action='store_true')
    ap.add_argument('--attn_table_aux', action='store_true')
    ap.add_argument('--attn_aux_lambda', type=float, default=ATM.DEFAULT_LAMBDA)
    ap.add_argument('--attn_aux_beta', type=float, default=ATM.DEFAULT_BETA)
    ap.add_argument('--drop_context_after_target', action='store_true')
    ap.add_argument('--target_end', action='store_true')
    ap.add_argument('--target_end_tasks', type=str, default='')
    ap.add_argument('--soft_role_param', choices=list(SP.PARAMETERIZATIONS), default=SP.DEFAULT_PARAM)
    ap.add_argument('--soft_role_lr', type=float, default=None)
    ap.add_argument('--target_first', action='store_true')
    ap.add_argument('--target_two_views', action='store_true')
    ap.add_argument('--table_first', action='store_true')
    ap.add_argument('--table_first_tasks', type=str, default='')
    ap.add_argument('--table_last', action='store_true')
    ap.add_argument('--col_major_tasks', type=str, default='')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    target_end_tasks = parse_task_scope(args.target_end_tasks)
    table_first_tasks = parse_task_scope(args.table_first_tasks)
    args.target_end = bool(args.target_end or target_end_tasks)
    args.table_first = bool(args.table_first or table_first_tasks)
    if args.role_gate_mlp and (args.col_mask_method != 'off' or args.grad_col_mask):
        raise SystemExit('--role_gate_mlp cannot be combined with the column mask: the mask re-encodes every CTA/CPA prompt after epoch 1, which would leave the per-token role ids aligned to the epoch-1 tokenization.')
    if args.split_gate_proj and (args.col_mask_method != 'off' or args.grad_col_mask):
        raise SystemExit('--split_gate_proj cannot be combined with the column mask: the mask re-encodes every CTA/CPA prompt after epoch 1, which would leave the table / non-table token split aligned to the epoch-1 tokenization.')
    if args.swiglu_thres and (args.col_mask_method != 'off' or args.grad_col_mask):
        raise SystemExit('--swiglu_thres cannot be combined with the column mask: the mask re-encodes every CTA/CPA prompt after epoch 1, which would leave the table / non-table token split aligned to the epoch-1 tokenization.')
    if args.split_gate_proj and args.role_gate_mlp:
        raise SystemExit("--split_gate_proj and --role_gate_mlp both act on the SwiGLU gate: the split flag replaces gate_proj's LoRA with two region-selected pairs, while role_gate_mlp adds its role vector to the gate pre-activation those pairs produce. Run them separately.")
    if args.lora_recv_gate != 'off' and (args.col_mask_method != 'off' or args.grad_col_mask):
        raise SystemExit("--lora_recv_gate cannot be combined with the column mask: the mask re-encodes every CTA/CPA prompt after epoch 1, which would leave the gate's sender streams and receiver positions aligned to the epoch-1 tokenization.")
    if args.lora_recv_gate != 'off':
        missing = [m for m in (RCV.CONTROLLER_SOURCE_MODULE, RCV.CONTROLLED_MODULE) if m not in set(args.target_modules)]
        if missing:
            raise SystemExit(f'--lora_recv_gate {args.lora_recv_gate} needs LoRA on {missing} ({RCV.CONTROLLER_SOURCE_MODULE} is the controller input, {RCV.CONTROLLED_MODULE} the controlled residual), but --target_modules is {list(args.target_modules)}.')
        if args.attn_impl not in RCV.ALLOWED_BASE:
            raise SystemExit(f'--lora_recv_gate needs --attn_impl in {list(RCV.ALLOWED_BASE)} (got {args.attn_impl!r}): the retrieved streams must ride on the SAME attention probabilities as the ordinary message, which flash_attention_2 cannot guarantee for an arbitrary additive mask.')
    if args.lora_rank_gate != 'off' and (args.col_mask_method != 'off' or args.grad_col_mask):
        raise SystemExit("--lora_rank_gate cannot be combined with the column mask: the mask re-encodes every CTA/CPA prompt after epoch 1, which would leave the gate's table-cell token scope aligned to the epoch-1 tokenization.")
    if args.lora_rank_gate != 'off' and args.lora_recv_gate != 'off':
        raise SystemExit(f"--lora_rank_gate {args.lora_rank_gate} and --lora_recv_gate {args.lora_recv_gate} are alternatives, not a stack. The sender gate recombines k/v's OWN rank code on table cells; the receiver gate re-weights what a receiver RETRIEVED. Pick one.")
    if args.lora_rank_gate != 'off':
        missing = [m for m in RKG.MODE_MODULES[args.lora_rank_gate] if m not in set(args.target_modules)]
        if missing:
            raise SystemExit(f'--lora_rank_gate {args.lora_rank_gate} gates the LoRA rank code of {list(RKG.MODE_MODULES[args.lora_rank_gate])}, but --target_modules is {list(args.target_modules)} and is missing {missing}.')
    _gate_on = args.lora_recv_gate != 'off' or args.lora_rank_gate != 'off'
    if args.rank_gate_lr is not None and (not _gate_on):
        raise SystemExit('--rank_gate_lr was given but neither --lora_rank_gate nor --lora_recv_gate is on; there are no gate parameters for that learning rate to apply to.')
    if args.rank_gate_exclude_target and (not _gate_on):
        raise SystemExit('--rank_gate_exclude_target was given but neither --lora_rank_gate nor --lora_recv_gate is on; there is no gate for it to make the target unavailable to.')
    if args.rank_gate_exclude_target:
        _regime_tasks = {canonical_task(t) for t in select_val_tasks(args.regime)} | {canonical_task(t) for t in args.regime.split('_')[1:] or []}
        _named = {RCV.canonical_task(t) for t in args.rank_gate_exclude_target}
        if not _named & _regime_tasks:
            _tag = 'lora_recv_gate' if args.lora_recv_gate != 'off' else 'lora_rank_gate'
            print(f'[{_tag}] WARNING: --rank_gate_exclude_target {sorted(_named)} names no task this regime appears to train ({sorted(_regime_tasks)}); the scope may be unchanged.')
    if args.regime == 'multi_shared_ctacpa' and args.cta_cpa_source not in ('sotab_all', 't2d'):
        raise SystemExit('--regime multi_shared_ctacpa requires --cta_cpa_source sotab_all (SOTAB) or t2d (train only on tables annotated by BOTH CTA and CPA).')
    if args.cta_cpa_source == 'sotab_all' and args.regime not in ('multi_shared_ctacpa', 'multi_paired_ctacpa'):
        raise SystemExit('--cta_cpa_source sotab_all is only valid with regime multi_shared_ctacpa or multi_paired_ctacpa.')
    if args.regime == 'multi_shared_ctacpa' and args.select_mode not in ('random', 'all'):
        raise SystemExit('--regime multi_shared_ctacpa supports only --select_mode random or all (the coverage-preserving shared-table selector).')
    if args.regime == 'multi_paired_ctacpa':
        if args.cta_cpa_source not in ('sotab_all', 'sotab_full', 't2d'):
            raise SystemExit('--regime multi_paired_ctacpa requires --cta_cpa_source sotab_all/sotab_full (SOTAB) or t2d (needs the FULL pool to form shared tables).')
        if args.select_mode != 'random':
            raise SystemExit('--regime multi_paired_ctacpa uses its own paired selector; run with --select_mode random (the default).')
        if args.budget % 2 != 0:
            raise SystemExit(f'--regime multi_paired_ctacpa needs an EVEN --budget so the two tasks pair 1:1 (budget/2 tables, one CTA + one CPA each); got --budget {args.budget}.')
        _win = args.micro_bs * args.grad_accum
        if _win % 2 != 0:
            raise SystemExit(f'--regime multi_paired_ctacpa needs an EVEN optimizer-step window (micro_bs*grad_accum), so each 2-example pair stays in one step without straddling a boundary; got micro_bs={args.micro_bs} * grad_accum={args.grad_accum} = {_win}.')
        if args.group_by_length:
            raise SystemExit('--regime multi_paired_ctacpa is incompatible with --group_by_length (the length sampler would break same-table pairs). Drop --group_by_length.')
        if int(os.environ.get('WORLD_SIZE', '1')) > 1:
            raise SystemExit('--regime multi_paired_ctacpa is single-GPU only for now (the same-step guarantee has no DistributedSampler sharding). Launch on one device.')
    if args.fill_budget and args.max_table_rows and (args.max_table_rows > 0):
        raise SystemExit('--fill_budget is incompatible with --max_table_rows > 0: the former fills the token budget (no row cap), the latter hard-caps to N rows. Use one or the other.')
    if args.max_table_rows and args.max_table_rows > 0 or args.fill_budget:
        if args.cta_cpa_source not in SOTAB_SOURCES + ('t2d',):
            raise SystemExit('--max_table_rows / --fill_budget require --cta_cpa_source in {sotab, sotab_full, sotab_all, t2d} (raw-grounded row selection is only defined for these).')
    SHARED_OK_REGIMES = {'single_cta', 'single_cpa'}
    if args.cta_cpa_source == 'shared' and args.regime not in SHARED_OK_REGIMES:
        raise SystemExit('--cta_cpa_source shared is only valid with regime single_cta or single_cpa')
    if args.select_mode in ('target_rank', 'target_rank_rev'):
        rtasks = [REGIME_SINGLE[args.regime]] if args.regime in REGIME_SINGLE else list(REGIME_MULTI.get(args.regime, []))
        bad = [t for t in rtasks if t not in ('cta', 'cpa')]
        if not rtasks or bad:
            raise SystemExit(f"--select_mode {args.select_mode} requires a CTA/CPA-only regime (single_cta, single_cpa, multi_ctacpa); got --regime {args.regime} with tasks {rtasks or '[]'}.")
    if args.grad_col_mask and args.col_mask_method == 'off':
        args.col_mask_method = 'gradient'
    col_mask_on = args.col_mask_method != 'off'
    col_major_tasks = parse_col_major_tasks(args.col_major_tasks)
    if col_major_tasks and args.transpose_augment:
        raise SystemExit('--col_major_tasks and --transpose_augment cannot be combined: the augmentation transposes an already-transposed prompt, which restores (approximately) the row-major layout and makes the copy a near-duplicate of the original.')
    if col_major_tasks and col_mask_on:
        raise SystemExit('--col_major_tasks is incompatible with --col_mask_method/--grad_col_mask: the column attribution locates a column by its POSITION within a row-major table line, which a column-major prompt does not have.')
    if col_mask_on and args.transpose_augment:
        raise SystemExit('--transpose_augment is incompatible with --col_mask_method/--grad_col_mask: the column attribution locates a column by its POSITION within a row-major table line, which a transposed prompt does not have.')
    if args.train_rows:
        _explicit_trunc = [n for (n, v) in (('--max_table_rows', args.max_table_rows > 0), ('--fill_budget', args.fill_budget), ('--max_context_cols', args.max_context_cols > 0), ('--drop_empty_context_cols', args.drop_empty_context_cols)) if v]
        if _explicit_trunc:
            raise SystemExit(f"--train_rows carries prompts that are ALREADY rendered and truncated; {' '.join(_explicit_trunc)} would truncate them a second time. Bake the truncation into the pack instead (prepare_ctxswap_pack.py passes these through to apply_row_truncation).")
    if args.val_rows:
        if not args.train_rows:
            raise SystemExit('--val_rows is only meaningful alongside --train_rows (both come from one pack).')
        _vr = json.loads(args.val_rows)
        _missing = [t for t in select_val_tasks(args.regime) if t not in _vr]
        if _missing:
            raise SystemExit(f'--val_rows must name EVERY validation slug of this run; {_missing} is/are missing. A partial mapping would leave those slugs sampling from a Random(42) stream the pre-sampled slugs never advanced, so they would see a different val subset than the same run without --val_rows.')
        for (_t, _f) in _vr.items():
            if not os.path.exists(_f):
                raise SystemExit(f'--val_rows: {_t} -> {_f} does not exist')
    if args.soft_role_prompt:
        _sp_readers = [n for (n, on) in (('--role_gate_mlp', args.role_gate_mlp), ('--lora_role_matrix', args.lora_role_matrix), ('--grad_tgt_b', bool(args.grad_tgt_b)), ('--lora_recv_gate', args.lora_recv_gate != 'off'), ('--lora_rank_gate', args.lora_rank_gate != 'off'), ('--split_gate_proj', args.split_gate_proj), ('--swiglu_thres', args.swiglu_thres), ('--ctx_relay_mask', args.ctx_relay_mask), ('--attn_table_aux', bool(args.attn_table_aux)), ('--ffn_role_scale', args.ffn_role_scale), ('--target_sandwich', args.target_sandwich)) if on]
        if _sp_readers:
            raise SystemExit(f"--soft_role_prompt cannot be combined with {'/'.join(_sp_readers)}: it INSERTS positions into the token sequence, and those mechanisms read per-token ids (roles / regions / copies / receiver positions) built against the sequence before the insertion. Every id after the first marked cell would name the wrong token.")
        if args.col_mask_method != 'off' or args.grad_col_mask or args.swap_context_epochs:
            raise SystemExit("--soft_role_prompt cannot be combined with --col_mask_method / --grad_col_mask / --swap_context_epochs: all three RE-ENCODE the prompts after tokenization, and the inserted role slots are built once, during that tokenization. The re-encoded prompts would carry another example's slots.")
        if args.col_major_tasks or args.transpose_augment:
            raise SystemExit('--soft_role_prompt cannot be combined with --col_major_tasks/--transpose_augment: the audited cell parser reads a ROW-major table (line 1 = header, each vertical slice = a column); on a transposed table it would mark rows as columns.')
    if args.ffn_role_scale:
        _fr_bad = [n for (n, on) in (('--struct_role_bias', args.struct_role_bias), ('--soft_role_prompt', args.soft_role_prompt), ('--target_sandwich', args.target_sandwich), ('--ctx_relay_mask', args.ctx_relay_mask), ('--attn_table_aux', bool(args.attn_table_aux)), ('--split_gate_proj', args.split_gate_proj), ('--swiglu_thres', args.swiglu_thres), ('--role_gate_mlp', args.role_gate_mlp), ('--lora_role_matrix', args.lora_role_matrix), ('--lora_recv_gate', args.lora_recv_gate != 'off'), ('--lora_rank_gate', args.lora_rank_gate != 'off')) if on]
        if _fr_bad:
            raise SystemExit(f"--ffn_role_scale cannot be combined with {'/'.join(_fr_bad)}: this first experiment is deliberately isolated, and --split_gate_proj / --swiglu_thres / --role_gate_mlp act on the same SwiGLU it modulates.")
        if args.col_mask_method != 'off' or args.grad_col_mask or args.swap_context_epochs:
            raise SystemExit('--ffn_role_scale cannot be combined with --col_mask_method / --grad_col_mask / --swap_context_epochs: all three RE-ENCODE the prompts after tokenization, and the per-token role ids are built once, during that tokenization.')
        if args.col_major_tasks or args.transpose_augment:
            raise SystemExit('--ffn_role_scale cannot be combined with --col_major_tasks/--transpose_augment: the audited role parser reads a ROW-major table (line 1 = header, each vertical slice = a column); on a transposed table it would mark rows as columns.')
    if args.struct_role_bias:
        _sb_bad = [n for (n, on) in (('--soft_role_prompt', args.soft_role_prompt), ('--target_sandwich', args.target_sandwich), ('--ctx_relay_mask', args.ctx_relay_mask), ('--attn_table_aux', bool(args.attn_table_aux))) if on]
        if _sb_bad:
            raise SystemExit(f"--struct_role_bias cannot be combined with {'/'.join(_sb_bad)}: those options insert positions into the sequence or install their own attention kernel/4D mask, and the second mechanism to touch attention silently wins.")
        if args.col_mask_method != 'off' or args.grad_col_mask or args.swap_context_epochs:
            raise SystemExit('--struct_role_bias cannot be combined with --col_mask_method / --grad_col_mask / --swap_context_epochs: all three RE-ENCODE the prompts after tokenization, and the per-token table coordinates are built once, during that tokenization.')
        if args.col_major_tasks or args.transpose_augment:
            raise SystemExit("--struct_role_bias cannot be combined with --col_major_tasks/--transpose_augment: the table parse reads a ROW-major table (line 1 = header, each vertical slice = a column); on a transposed table same_row and same_column would name each other's edges.")
        if args.attn_impl not in SRB.ALLOWED_BASE:
            raise SystemExit(f'--struct_role_bias needs --attn_impl in {list(SRB.ALLOWED_BASE)} (got {args.attn_impl!r}): flash_attention_2 cannot take an additive attention bias.')
    if args.target_sandwich:
        if args.sandwich_mask and (args.table_first or args.table_last):
            raise SystemExit('--target_sandwich cannot be combined with --table_first/--table_last: the query positions the mask cuts are the POST-TABLE region, and both move the query out of it. evaluate.py refuses the same pair. (--no-sandwich_mask builds no mask and has no such constraint, exactly as evaluate.py allows it.)')
        if args.ctx_relay_mask and args.sandwich_mask:
            raise SystemExit('--target_sandwich and --ctx_relay_mask cannot be combined: both install their own 4D attention mask through the same argument, and the second one silently wins. (--no-sandwich_mask installs none, so that pair is allowed.)')
        if args.sandwich_mask and args.attn_impl not in TSW.ALLOWED_BASE:
            raise SystemExit(f'--target_sandwich needs --attn_impl in {list(TSW.ALLOWED_BASE)} (got {args.attn_impl!r}): flash_attention_2 cannot take a custom 4D attention mask. --no-sandwich_mask builds no mask and has no such constraint.')
        if args.col_major_tasks or args.transpose_augment:
            raise SystemExit('--target_sandwich cannot be combined with --col_major_tasks/--transpose_augment: the sandwich duplicates a COLUMN of a ROW-major table (line 1 = header, each vertical slice = a column); on a transposed table that reading is false.')
        if args.attn_table_aux and args.sandwich_mask:
            raise SystemExit('--attn_table_aux cannot be combined with --target_sandwich: the objective sums attention over the table span under a mask it did not build.')
        if args.col_mask_method != 'off' or args.grad_col_mask or args.swap_context_epochs:
            raise SystemExit("--target_sandwich cannot be combined with --col_mask_method / --grad_col_mask / --swap_context_epochs: all three RE-ENCODE the prompts after tokenization, and the per-token copy ids the mask reads are built once, during that tokenization. The re-encoded prompts would be served under another example's mask.")
    if args.target_end:
        if args.transpose_augment:
            raise SystemExit('--target_end and --transpose_augment cannot be combined: the transposed copy has no columns to reorder, so the two halves of the training set would carry different layouts.')
        if args.swap_context_epochs:
            raise SystemExit('--target_end and --swap_context_epochs cannot be combined: --target_end leaves every CTA target last, so C_right is empty and the swapped epochs would put every target FIRST -- the run would alternate between two extremes instead of training the column order --target_end declares canonical.')
        if args.swap_context_blocks:
            raise SystemExit('--target_end and --swap_context_blocks cannot be combined: the swap moves the context blocks around the target and --target_end then pins the target to the end, which collapses both copies of every instance onto the SAME canonical column order -- the augmentation would silently become a duplication.')
        if col_mask_on:
            raise SystemExit('--target_end is incompatible with --col_mask_method/--grad_col_mask: the mask is recorded against column POSITIONS, which --target_end changes.')
    if args.target_first:
        if args.target_end:
            raise SystemExit('--target_first and --target_end cannot be combined: each pins the SAME column(s) to a different end of the table, so whichever ran second would silently decide the layout.')
        if args.target_two_views:
            raise SystemExit('--target_first and --target_two_views cannot be combined: the two views ARE target-first and target-last, so fixing the layout first would collapse both copies of every instance onto the same prompt.')
        if args.ctx_relay_mask:
            raise SystemExit('--target_first and --ctx_relay_mask cannot be combined: the relay path needs the designated column(s) LAST (that is what --ctx_relay_mask requires --target_end for); with them FIRST the context would be cut off from the answer and from the target rather than relayed.')
        if args.transpose_augment:
            raise SystemExit('--target_first and --transpose_augment cannot be combined: the transposed copy has no columns to reorder, so the two halves of the training set would carry different layouts.')
        if args.swap_context_epochs:
            raise SystemExit('--target_first and --swap_context_epochs cannot be combined: --target_first leaves every CTA target first, so C_left is empty and the swapped epochs would put every target LAST -- the run would alternate between two extremes instead of training the column order --target_first declares canonical.')
        if args.swap_context_blocks:
            raise SystemExit('--target_first and --swap_context_blocks cannot be combined: the swap moves the context blocks around the target and --target_first then pins the target to the front, which collapses both copies of every instance onto the SAME canonical column order -- the augmentation would silently become a duplication.')
        if col_mask_on:
            raise SystemExit('--target_first is incompatible with --col_mask_method/--grad_col_mask: the mask is recorded against column POSITIONS, which --target_first changes.')
        if args.target_sandwich:
            raise SystemExit('--target_first and --target_sandwich cannot be combined: the sandwich duplicates the target column around the context and reads the target by its position, which --target_first has already pinned to the front.')
    if args.ctx_relay_mask:
        if not args.target_end:
            raise SystemExit("--ctx_relay_mask requires --target_end. The mask routes the table's context to the answer THROUGH the designated column(s), which only exists as a path if those columns come last (CTA [.. context .., TARGET]; CPA [.. context .., HEAD, TAIL]). With the corpus column order the context of every prompt whose target sits first would be deleted, not relayed.")
        if args.attn_impl not in CRM.ALLOWED_BASE:
            raise SystemExit(f'--ctx_relay_mask needs --attn_impl in {list(CRM.ALLOWED_BASE)} (got {args.attn_impl!r}): flash_attention_2 cannot take a custom 4D attention mask.')
        if args.table_first or args.table_last:
            raise SystemExit('--ctx_relay_mask cannot be combined with --table_first/--table_last: the receiver set is the POST-TABLE region, and both flags move the query out of it, so the mask would silently stop applying to the query.')
        if args.lora_recv_gate != 'off' or args.lora_rank_gate != 'off':
            raise SystemExit("--ctx_relay_mask cannot be combined with --lora_recv_gate/--lora_rank_gate: the gates register their own attention kernel and mask builder, and this mask replaces the batch's attention mask, so the two would fight over the same argument.")
    if args.attn_table_aux:
        if args.attn_impl not in ATM.ALLOWED_BASE:
            raise SystemExit(f'--attn_table_aux needs --attn_impl in {list(ATM.ALLOWED_BASE)} (got {args.attn_impl!r}).')
        if args.grad_checkpointing:
            raise SystemExit('--attn_table_aux cannot be combined with --grad_checkpointing: the auxiliary term is collected during the forward, which checkpointing re-runs in no-grad, so it would silently receive no gradient.')
        if args.ctx_relay_mask:
            raise SystemExit("--attn_table_aux cannot be combined with --ctx_relay_mask: the relay mask rewrites the batch's attention mask, so the receiver rows measured here would no longer be the attention the model actually computes.")
        if args.lora_recv_gate != 'off' or args.lora_rank_gate != 'off':
            raise SystemExit('--attn_table_aux cannot be combined with --lora_recv_gate/--lora_rank_gate: they register their own attention kernel, which this objective replaces.')
        if args.col_mask_method != 'off' or args.grad_col_mask or args.swap_context_blocks:
            raise SystemExit('--attn_table_aux cannot be combined with --col_mask_method/--grad_col_mask/--swap_context_blocks: they re-encode the prompts mid-run, and the precomputed semantic decision positions are absolute token positions of the ORIGINAL tokenization.')
    if args.lora_role_matrix:
        if args.lora_rank_gate != 'off':
            raise SystemExit('--lora_role_matrix and --lora_rank_gate cannot be combined: both transform the LoRA rank code of k_proj/v_proj between A and B, so the two would compose into a third mechanism that is neither.')
        if args.lora_recv_gate != 'off':
            raise SystemExit('--lora_role_matrix and --lora_recv_gate cannot be combined: the receiver gate also rewrites the retrieved K/V LoRA contribution.')
        _missing = [m for m in RM.ROLE_MATRIX_MODULES if m not in set(args.target_modules)]
        if _missing:
            raise SystemExit(f'--lora_role_matrix needs LoRA on {list(RM.ROLE_MATRIX_MODULES)}, but --target_modules is {list(args.target_modules)} (missing {_missing}). There would be no rank code to transform.')
    if args.drop_context_after_target:
        if args.col_major_tasks:
            raise SystemExit('--drop_context_after_target and --col_major_tasks cannot be combined: the pass removes COLUMNS of a ROW-major table (line 1 = header, each vertical slice = a column); on a transposed table that reading is false.')
        if args.target_sandwich:
            raise SystemExit('--drop_context_after_target and --target_sandwich cannot be combined: the truncation cuts the columns to the RIGHT of the target, which is exactly where the sandwich puts its back copy.')
        if args.target_two_views or args.transpose_augment or args.swap_context_blocks or args.swap_context_epochs:
            raise SystemExit('--drop_context_after_target cannot be combined with an augmentation that rewrites the same table (--target_two_views / --transpose_augment / --swap_context_blocks / --swap_context_epochs).')
        if args.col_mask_method != 'off' or args.grad_col_mask:
            raise SystemExit('--drop_context_after_target is incompatible with --col_mask_method/--grad_col_mask: the mask is recorded against column POSITIONS, which the truncation changes.')
    if args.target_two_views:
        if args.target_end:
            raise SystemExit('--target_two_views and --target_end cannot be combined: target_end pins every prompt to ONE canonical layout, which is exactly one of the two views the augmentation emits, so the pair would collapse to a duplicate.')
        for (flag_name, on) in (('--transpose_augment', args.transpose_augment), ('--swap_context_blocks', args.swap_context_blocks), ('--swap_context_epochs', args.swap_context_epochs)):
            if on:
                raise SystemExit(f'--target_two_views and {flag_name} cannot be combined: both rewrite the training set, and composing them would emit copies of copies whose layout is neither view.')
        if args.col_major_tasks:
            raise SystemExit('--target_two_views and --col_major_tasks cannot be combined: the views are COLUMN permutations of a ROW-major table (line 1 = header, each vertical slice = a column); on a transposed table that reading is false.')
        if args.target_sandwich:
            raise SystemExit('--target_two_views and --target_sandwich cannot be combined: the two views are target-first and target-last layouts, and the sandwich is both at once.')
        if args.col_mask_method != 'off' or args.grad_col_mask:
            raise SystemExit('--target_two_views is incompatible with --col_mask_method/--grad_col_mask: the mask is recorded against column POSITIONS and per-example ids, both of which the two views change.')
    if args.merge_adapter and args.init_qkv_adapter:
        raise SystemExit("--merge_adapter and --init_qkv_adapter cannot be combined: they are two different ways to inherit the SAME pretrained q/k/v stage -- one folds its delta into the base weights, the other copies its matrices into this run's fresh adapter. Together the pretrained delta would be applied TWICE, once from W0 and once from the trainable adapter. Pick one.")
    if args.table_first and args.table_last:
        raise SystemExit('--table_first and --table_last cannot be combined: they name opposite positions for the same block, and they occupy the same slot in the pipeline, so the second would simply undo the first.')
    if args.swap_context_blocks and args.swap_context_epochs:
        raise SystemExit("--swap_context_blocks and --swap_context_epochs cannot be combined: they are two ways to show the same instance in both column orders -- one appends a second example, the other alternates the single example's layout across epochs. Together the appended copy would itself be swapped on even epochs, putting BOTH copies of every instance in the same order in every step. Pick one.")
    if args.swap_context_epochs:
        if args.role_gate_mlp or args.grad_tgt_b:
            raise SystemExit('--swap_context_epochs cannot be combined with --role_gate_mlp/--grad_tgt_b: the per-token role ids are built once against a single tokenization, and the swap re-tokenizes the prompt every other epoch.')
        if args.group_by_length:
            raise SystemExit('--swap_context_epochs cannot be combined with --group_by_length: the length sampler reads the lengths of the epoch-0 encodings once, and the swapped prompts are not the same length (the column markers change index).')
    if args.swap_context_blocks or args.swap_context_epochs:
        if col_major_tasks:
            raise SystemExit('--swap_context_blocks/--swap_context_epochs and --col_major_tasks cannot be combined: the swap reorders the COLUMNS of a row-major Markdown table, which a column-major prompt does not have.')
        if args.transpose_augment:
            raise SystemExit('--swap_context_blocks/--swap_context_epochs and --transpose_augment cannot be combined: the transposed copy is column-major, so it has no columns for the swap to reorder and the two halves of the training set would carry different layouts.')
        if col_mask_on:
            raise SystemExit('--swap_context_blocks/--swap_context_epochs is incompatible with --col_mask_method/--grad_col_mask: the column attribution locates a column by its POSITION, which the swapped copy changes.')
    if col_mask_on:
        if not ('cta' in args.regime or 'cpa' in args.regime):
            raise SystemExit(f'--col_mask_method {args.col_mask_method} needs a regime containing CTA/CPA (the mask is a CTA/CPA per-instance column mask); got --regime {args.regime}.')
        if args.epochs < 2:
            raise SystemExit(f'--col_mask_method {args.col_mask_method} needs --epochs >= 2 (epoch 1 trains full, the attribution sweep runs after it, epochs 2+ train masked); got --epochs {args.epochs}.')
        if args.max_table_rows and args.max_table_rows > 0:
            raise SystemExit('--col_mask_method is incompatible with --max_table_rows > 0 (explicit row selection). Use one or the other.')
        if args.fill_budget:
            raise SystemExit('--col_mask_method is incompatible with --fill_budget (explicit row selection). Use one or the other.')
        if args.regime == 'multi_paired_ctacpa':
            raise SystemExit('--col_mask_method is incompatible with --regime multi_paired_ctacpa (masking would rewrite the paired prompts the same-step sampler assumes).')
    refuse_full_finetune_conflicts(args)
    set_all_seeds(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    _out = os.path.abspath(args.out)
    if os.path.dirname(_out) == _out or _out.count(os.sep) < 2:
        raise SystemExit(f'--out {args.out!r} resolves to {_out!r}, which is a filesystem root; refusing to delete it.')
    if os.path.commonpath([_out, os.path.abspath(os.getcwd())]) == _out:
        raise SystemExit(f'--out {args.out!r} resolves to {_out!r}, which CONTAINS the working directory; refusing to delete it.')
    if os.path.islink(_out) or (os.path.exists(_out) and (not os.path.isdir(_out))):
        os.remove(_out)
        print(f'[out] removed the existing non-directory at {_out}')
    elif os.path.isdir(_out):
        _n = len(os.listdir(_out))
        shutil.rmtree(_out)
        print(f'[out] wiped the existing output directory {_out} ({_n} entries) -- this run does not merge with a previous one')
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if args.train_rows:
        rows = read_jsonl(args.train_rows)
        print(f'[train_rows] {len(rows)} pre-selected, pre-rendered rows from {args.train_rows} (build_train_rows and the raw row re-render are both skipped)')
    else:
        rows = build_train_rows(args.regime, args.budget, args.seed, args.cta_cpa_source, select_mode=args.select_mode, drop_empty_target=args.drop_empty_target, drop_truncated_tables=args.drop_truncated_tables, samerow_source=args.samerow_source, samerow_max_rows=args.samerow_max_rows, samerow_tok=tok if _uses_samerow(args.regime) else None, samerow_max_len=args.max_seq_len, samerow_model=args.model)
    _rt_plan = _row_trunc_plan(args.cta_cpa_source, args.max_table_rows, args.row_trunc_method, args.fill_budget)
    if args.train_rows and _rt_plan is not None:
        print(f'[train_rows] skipping the {args.cta_cpa_source} raw-grounded row re-render ({_rt_plan[0]}): --train_rows prompts are already rendered.')
        _rt_plan = None
    _hard_cap = bool(args.max_table_rows and args.max_table_rows > 0)
    _col_on = bool(args.max_context_cols and args.max_context_cols > 0 or args.drop_empty_context_cols)
    if args.col_trunc_method != 'nearest' and (not (args.max_context_cols and args.max_context_cols > 0)):
        print(f'[col_trunc] WARNING: --col_trunc_method {args.col_trunc_method} has NO EFFECT without --max_context_cols > 0 (no columns are being dropped, so there is nothing to choose). Ignoring it.')
    if args.reindex_truncated_cols and (not _col_on):
        print('[col_trunc] WARNING: --reindex_truncated_cols has NO EFFECT without --max_context_cols > 0 (nothing is being truncated, so there is nothing to renumber). Ignoring it.')
    if _col_on:
        if _rt_plan is None:
            _rt_plan = ('head', _SOTAB_FULL_MAXROWS)
        _cm_desc = {'nearest': 'nearest-neighbour by POSITION: CTA around the target, CPA between head/tail then outward', 'random': 'uniform RANDOM draw per table, seeded md5(seed|table#cols)', 'coverage_col': 'greedy facility-location COVERAGE over the context columns as documents (column-space analogue of coverage_row)'}[args.col_trunc_method]
        print(f'[col_trunc] max_context_cols={args.max_context_cols} method={args.col_trunc_method} (deterministic; {_cm_desc})')
        print(f'[col_trunc] reindex_truncated_cols={args.reindex_truncated_cols} ' + ('(kept columns relabelled col0..colK-1 and the CTA/CPA marker rewritten to match, so truncated TRAIN prompts carry the same contiguous header layout as untruncated EVAL prompts)' if args.reindex_truncated_cols else '(ORIGINAL colN labels kept -> non-contiguous headers: CPA interior gaps, CTA a window that need not start at col0; this is the pre-2026-09-08 behaviour)'))
    instance_mix = dict(collections.Counter((r['task'] for r in rows)))
    enc_by_epoch = None
    rt_stats = None
    _post_trunc_dropped = 0
    pair_group_ids = None
    paired_dropped_groups = 0
    enc_rows = None
    if _rt_plan is not None:
        (_rt_method, _rt_max) = _rt_plan
        _prefetch_raw_tables(rows, args.cta_cpa_source, 'train', args.num_proc)
        rt_stats = apply_row_truncation(rows, args.cta_cpa_source, _rt_method, _rt_max, args.seed, tok, args.max_seq_len, args.model, False, split='train', hard_cap=_hard_cap, max_context_cols=args.max_context_cols, reindex_cols=args.reindex_truncated_cols, drop_empty_context_cols=args.drop_empty_context_cols, col_trunc_method=args.col_trunc_method)
        if _hard_cap:
            print(f'[row_trunc] method={_rt_method} max_table_rows={args.max_table_rows} (STATIC hard cap once pre-training; dedup drops duplicate-target rows) (train, raw-grounded, num_proc={args.num_proc}) {rt_stats}')
        else:
            _mode = 'fill-budget selection' if args.fill_budget else 'auto head-fit'
            print(f'[row_trunc] method={_rt_method} {args.cta_cpa_source}: {_mode} to max_seq_len={args.max_seq_len} (raw rows, run tokenizer; full table kept if it fits, else as many {_rt_method}-ranked rows as fit) (train, raw-grounded, num_proc={args.num_proc}) {rt_stats}')
        if args.drop_empty_target:
            _before = len(rows)
            rows = [r for r in rows if not E.target_is_empty(r.get('task', ''), r['input'])]
            _post_trunc_dropped = _before - len(rows)
            if _post_trunc_dropped:
                print(f'[drop_empty_target] post-truncation: dropped {_post_trunc_dropped}/{_before} examples whose TARGET column is blank in the RE-RENDERED rows (the pool-load filter ran on the prepared prompt, before the re-render); {len(rows)} training instances remain')
                instance_mix = dict(collections.Counter((r['task'] for r in rows)))
        if _col_on and isinstance(rt_stats, dict) and ('col_trunc' in rt_stats):
            ct = rt_stats['col_trunc']
            print(f"[col_trunc] context columns <= {ct['max_context_cols']}: {ct['instances_columns_dropped']}/{ct['instances']} instances truncated; train tokens {ct['train_tokens_baseline_all_cols']} -> {ct['train_tokens_final']} (saved {ct['train_tokens_saved']}, {ct['pct_saved']:.1f}%) with {args.model}")
    train_instance_ids = [r['id'] for r in rows]
    dca_stats = apply_drop_context_after_target(rows, args.drop_context_after_target, 'train', tok)
    te_stats = apply_target_end(rows, args.target_end, 'train', tasks=target_end_tasks)
    tfirst_stats = apply_target_end(rows, args.target_first, 'train', mode='target_begin')
    cm_stats = apply_col_major(rows, col_major_tasks, tok, args.max_seq_len, args.model, 'train')
    tf_stats = apply_table_first(rows, args.table_first, 'train', tasks=table_first_tasks)
    tl_stats = apply_table_last(rows, args.table_last, 'train')
    if args.target_sandwich:
        TSW.reset_transform_stats()
    ta_stats = None
    if args.transpose_augment:
        (rows, ta_stats) = augment_transposed(rows, tok, args.max_seq_len, args.model)
        print(f"[transpose_augment] {ta_stats['instances']} instances -> {len(rows)} examples (+{ta_stats['augmented']} column-major copies; no_table={ta_stats['no_table']} unparseable={ta_stats['unparseable']} over_len_dropped={ta_stats['over_len']})")
    tv_stats = None
    if args.target_two_views:
        (rows, tv_stats) = augment_target_views(rows, tok, args.max_seq_len, args.model)
        print(f"[target_two_views] {tv_stats['eligible']} CTA/CPA instances (of {tv_stats['instances']}) -> {len(rows)} examples ({tv_stats['augmented']} x 2 views {tv_stats['per_task']}; emitted_unchanged_by_its_view={tv_stats['already']}; original kept once for {tv_stats['passthrough']} unviewable instances: over_len={tv_stats['over_len']} declined={tv_stats['skipped'] or 'none'})")
        if tv_stats['eligible'] == 0:
            raise SystemExit(f'--target_two_views is a CTA/CPA-only augmentation, but --regime {args.regime} selected no CTA or CPA instances, so the flag would be a silent no-op.')
        if tv_stats['augmented'] == 0:
            raise SystemExit(f"--target_two_views produced no view pair at all: every CTA/CPA instance was declined or over-length (over_len={tv_stats['over_len']} skipped={tv_stats['skipped']}). Refusing to train the unaugmented baseline under an augmented tag.")
    sc_stats = None
    if args.swap_context_blocks:
        (rows, sc_stats) = augment_swapped_context(rows, tok, args.max_seq_len, args.model)
        print(f"[swap_context_blocks] {sc_stats['eligible']} CTA/CPA instances (of {sc_stats['instances']}) -> {len(rows)} examples (+{sc_stats['augmented']} swapped copies {sc_stats['per_task']}; over_len_dropped={sc_stats['over_len']} skipped={sc_stats['skipped']})")
        if sc_stats['eligible'] == 0:
            raise SystemExit(f'--swap_context_blocks is a CTA/CPA-only augmentation, but --regime {args.regime} selected no CTA or CPA instances, so the flag would be a silent no-op.')
    se_stats = None
    rows_swapped = None
    if args.swap_context_epochs:
        (rows_swapped, se_stats) = build_swapped_rows(rows)
        print(f"[swap_context_epochs] {se_stats['eligible']} CTA/CPA instances (of {se_stats['instances']}) -> {se_stats['swapped']} swapped {se_stats['per_task']} for the even epochs (unchanged_in_every_epoch={se_stats['skipped'] or 'none'})")
        if se_stats['eligible'] == 0:
            raise SystemExit(f'--swap_context_epochs is a CTA/CPA-only transform, but --regime {args.regime} selected no CTA or CPA instances, so the flag would be a silent no-op.')
    enc = []
    enc_alt = [] if args.swap_context_epochs else None
    dropped = 0
    trimmed = 0
    dropped_trimmed = 0
    enc_groups = []
    enc_rows = []
    enc_tasks = []
    struct_relations = SRB.resolve_relations(args.struct_role_relations)
    struct_needs_roles = bool(args.struct_role_bias and SRB.needs_roles_for(struct_relations))
    _roles_with_regions = bool(args.role_gate_mlp or args.grad_tgt_b or args.lora_recv_gate != 'off' or (args.lora_rank_gate != 'off') or args.split_gate_proj or args.swiglu_thres or args.ctx_relay_mask or struct_needs_roles or args.lora_role_matrix)
    need_roles = bool(_roles_with_regions or args.ffn_role_scale)
    need_copy = bool(args.target_sandwich)
    need_regions = bool(_roles_with_regions or args.attn_table_aux or need_copy)
    need_soft = bool(args.soft_role_prompt)
    need_struct = bool(args.struct_role_bias)
    enc_struct = [] if need_struct else None
    enc_roles = [] if need_roles else None
    enc_regions = [] if need_regions else None
    enc_copy = [] if need_copy else None
    enc_soft = [] if need_soft else None
    soft_marks = collections.Counter()
    soft_absorbed_total = 0
    for (_ri, r) in enumerate(rows):
        e = tokenize_example(tok, r, args.max_seq_len, args.model, False, want_roles=need_roles, want_regions=need_regions, want_copy=need_copy, sandwich=args.target_sandwich, want_soft=need_soft, want_struct=need_struct)
        if e is None:
            dropped += 1
            continue
        copies = None
        softs = None
        structs = None
        if need_struct:
            (ids, lab, was_trimmed, roles, regions, structs) = e
        elif need_soft:
            (ids, lab, was_trimmed, roles, regions, copies, (softs, _ab)) = e
            soft_absorbed_total += _ab
        elif need_copy:
            (ids, lab, was_trimmed, roles, regions, copies) = e
        elif need_regions or need_roles:
            (ids, lab, was_trimmed, roles, regions) = e
        else:
            (ids, lab, was_trimmed) = e
        if was_trimmed and r['task'] == SR.SLUG:
            raise RuntimeError(f"samerow prompt {r['id']} was row-trimmed at max_seq_len={args.max_seq_len}, which invalidates its same-row label. This is a bug: the samerow pool should have been fitted to this budget at synthesis time.")
        if args.drop_truncated_tables and was_trimmed:
            dropped += 1
            dropped_trimmed += 1
            continue
        if enc_alt is not None:
            ea = tokenize_example(tok, rows_swapped[_ri], args.max_seq_len, args.model, False)
            if ea is None or ea[2]:
                dropped += 1
                continue
            enc_alt.append((ea[0], ea[1]))
        enc.append((ids, lab))
        enc_groups.append(r['meta'].get('pair_group'))
        enc_rows.append(r)
        enc_tasks.append(r['task'])
        if need_roles:
            enc_roles.append(roles)
        if need_regions:
            enc_regions.append(regions)
        if need_copy:
            enc_copy.append(copies)
        if need_struct:
            enc_struct.append(structs)
        if need_soft:
            enc_soft.append(softs)
            for _sl in softs:
                if _sl != SP.SLOT_NONE:
                    soft_marks[SP.SLOT_NAMES[_sl]] += 1
        trimmed += int(was_trimmed)
    assert len(enc_tasks) == len(enc), 'task slugs drifted from encodings'
    if enc_alt is not None:
        assert len(enc_alt) == len(enc), 'swapped encodings drifted from the originals'
    if need_roles:
        assert len(enc_roles) == len(enc), 'role ids drifted from encodings'
    if need_regions:
        assert len(enc_regions) == len(enc), 'region ids drifted from encodings'
    if need_struct:
        assert len(enc_struct) == len(enc), 'struct ids drifted from encodings'
    if need_soft:
        assert len(enc_soft) == len(enc), 'soft-role slots drifted from encodings'
        _n_marks = sum(soft_marks.values())
        if _n_marks == 0:
            raise SystemExit(f'[{SP.MECHANISM}] not one target/context cell was marked across {len(enc)} training examples -- this run is the plain baseline. --soft_role_prompt needs CTA/CPA data (regime={args.regime!r} selected none).')
        print(f'[{SP.MECHANISM}] inserted {_n_marks:,} role positions over {len(enc):,} examples ({_n_marks / max(1, len(enc)):.1f}/example): ' + ', '.join((f'{k}={soft_marks[k]:,}' for k in SP.SLOT_NAMES if soft_marks[k])))
        print(f"[{SP.MECHANISM}] tokens now {sum((len(a) for (a, _) in enc)):,} (+{_n_marks:,} inserted, max_seq_len={args.max_seq_len} was applied BEFORE the insertion and nothing was re-truncated); cells absorbed into a neighbour's token: {soft_absorbed_total:,}")
    task_counts = dict(collections.Counter((r['task'] for r in rows)))
    pair_group_ids = None
    paired_dropped_groups = 0
    if args.regime == 'multi_paired_ctacpa':
        by_g = collections.defaultdict(list)
        for (i, g) in enumerate(enc_groups):
            by_g[g].append(i)
        keep_idx = [i for (g, idxs) in by_g.items() if len(idxs) == 2 for i in idxs]
        paired_dropped_groups = sum((1 for (g, idxs) in by_g.items() if len(idxs) != 2))
        keep_idx.sort()
        enc = [enc[i] for i in keep_idx]
        if enc_alt is not None:
            enc_alt = [enc_alt[i] for i in keep_idx]
        pair_group_ids = [enc_groups[i] for i in keep_idx]
        enc_tasks = [enc_tasks[i] for i in keep_idx]
        if need_roles:
            enc_roles = [enc_roles[i] for i in keep_idx]
        if need_regions:
            enc_regions = [enc_regions[i] for i in keep_idx]
        if need_soft:
            enc_soft = [enc_soft[i] for i in keep_idx]
        if paired_dropped_groups:
            print(f'[paired_ctacpa] dropped {paired_dropped_groups} partial group(s) (a member exceeded max_seq_len); {len(pair_group_ids) // 2} pairs remain.')
    role_audit = None
    if args.role_gate_mlp:
        by_task = collections.defaultdict(lambda : collections.Counter())
        no_designated = collections.Counter()
        for (roles, t) in zip(enc_roles, enc_tasks):
            base = canonical_task(t)
            c = RGM.role_id_counts(roles)
            for (k, v) in c.items():
                by_task[base][k] += v
            desig = sum((v for (k, v) in c.items() if k != 'none' and (not k.endswith('_context'))))
            if desig == 0:
                no_designated[base] += 1
        role_audit = {'token_counts_by_task': {k: dict(v) for (k, v) in by_task.items()}, 'examples_without_a_designated_column': dict(no_designated), 'n_examples': len(enc_roles)}
        for (t, c) in role_audit['token_counts_by_task'].items():
            tot = sum(c.values())
            named = {k: v for (k, v) in c.items() if v}
            print(f'[role_gate_mlp] {t}: {tot} tokens, roles {named}')
        print(f"[role_gate_mlp] examples with NO designated (target/head/tail) token: {dict(no_designated) or 'none'} of {len(enc_roles)}")
        bad = sum(no_designated.values())
        if bad > 0.05 * max(1, len(enc_roles)):
            raise SystemExit(f'[role_gate_mlp] {bad}/{len(enc_roles)} training examples carry no designated target/head/tail token, so the role vocabulary has collapsed towards `context` for them. Refusing to run an experiment whose intervention is mostly absent.')
    ffn_role_audit = None
    if args.ffn_role_scale:
        by_task = collections.defaultdict(lambda : collections.Counter())
        no_target = collections.Counter()
        for (roles, t) in zip(enc_roles, enc_tasks):
            base = canonical_task(t)
            c = FRS.slot_counts(torch.tensor([roles], dtype=torch.long))
            for (k, v) in c.items():
                by_task[base][k] += v
            if c['cta_target'] + c['cpa_target'] == 0:
                no_target[base] += 1
        ffn_role_audit = {'token_counts_by_task': {k: dict(v) for (k, v) in by_task.items()}, 'examples_without_a_target_token': dict(no_target), 'n_examples': len(enc_roles)}
        for (t, c) in ffn_role_audit['token_counts_by_task'].items():
            named = {k: v for (k, v) in c.items() if v}
            print(f'[{FRS.MECHANISM}] {t}: {sum(c.values())} tokens, conditions {named}')
        print(f"[{FRS.MECHANISM}] examples with NO target token: {dict(no_target) or 'none'} of {len(enc_roles)}")
        bad = sum(no_target.values())
        if bad > 0.05 * max(1, len(enc_roles)):
            raise SystemExit(f'[{FRS.MECHANISM}] {bad}/{len(enc_roles)} training examples carry no target token, so the four-condition vocabulary has collapsed towards `context` for them. Refusing to run an experiment whose intervention is mostly absent.')
    print(f'[{args.regime} seed={args.seed}] instances={len(train_instance_ids)} instance_mix={instance_mix} examples={len(rows)} example_mix={task_counts} used={len(enc)} table_trimmed={trimmed} dropped_over_len={dropped}' + (f' dropped_truncated_tables={dropped_trimmed}' if args.drop_truncated_tables else ''))
    if len(enc) == 0:
        raise ValueError('No training examples left after max_seq_len filtering. Increase --max_seq_len or check prepared data.')
    if enc_alt is not None:
        enc_by_epoch = [enc if e % 2 == 0 else enc_alt for e in range(max(1, int(args.epochs)))]
        print(f'[swap_context_epochs] {len(enc)} examples x {len(enc_by_epoch)} epochs: ' + ' '.join((f"e{e + 1}={('orig' if e % 2 == 0 else 'swapped')}" for e in range(len(enc_by_epoch)))))
    task_ids = task_names = task_weights = tw_expected = None
    tok_stats = None
    if args.balance_supervised_tokens:
        (task_ids, task_names) = build_task_ids(enc_tasks)
        tok_stats = supervised_token_stats(enc, task_ids, task_names)
        (task_weights, tw_expected) = compute_task_loss_weights(tok_stats)
        print('\nTask supervised-token statistics:')
        for t in task_names:
            v = tok_stats[t]
            print(f"{t}: examples={v['examples']}, mean={v['mean']:.4f}, median={v['median']:.1f}, min={v['min']}, max={v['max']}")
        print(f"\nGlobal mean supervised-token length: mu_global={tw_expected['mu_global']:.4f} (sets the fixed weight scale: w_t = mu_global / mu_t)")
        print('\nFixed task weights:')
        for t in task_names:
            print(f'{t}: {task_weights[t]:.6f}')
        print('\nExpected supervised-token contribution (q_t = N_t*mu_t / sum_u N_u*mu_u):')
        for t in task_names:
            print(f"{t}: original={tw_expected['original'][t]:.6f}  weighted={tw_expected['weighted'][t]:.6f}")
        print()
    tgt_sandwich_audit = None
    if args.target_sandwich:
        tsw_stats = TSW.transform_stats()
        print(f"[{TSW.MECHANISM}] train: sandwiched the target column in {tsw_stats['sandwiched']}/{tsw_stats['considered']} CTA/CPA prompts, AFTER the max_seq_len={args.max_seq_len} budget was spent on the original prompt" + (f" (skipped {tsw_stats['skipped']})" if tsw_stats['skipped'] else ''))
        _over = [len(i) for (i, _l) in enc if len(i) > args.max_seq_len]
        print(f'[{TSW.MECHANISM}] {len(_over):,}/{len(enc):,} tokenized examples exceed max_seq_len after the second copy' + (f' (max {max(_over):,} tokens, mean {sum(_over) / len(_over):.0f})' if _over else ''))
        tgt_sandwich_audit = TSW.training_audit(enc_copy, enc_regions, enc_tasks=enc_tasks)
        print(TSW.format_audit(tgt_sandwich_audit) if args.sandwich_mask else f'[{TSW.MECHANISM}] serialization-only arm (--no-sandwich_mask): the counts below say what a mask WOULD cut; nothing is cut in this run.\n' + TSW.format_audit(tgt_sandwich_audit))
        _inert = sum((d['inert'] for d in tgt_sandwich_audit['per_task'].values()))
        if _inert == tgt_sandwich_audit['n_examples']:
            raise SystemExit(f'[{TSW.MECHANISM}] every training example is inert -- no example carries a T_front column, so the mask would cut nothing and this run would be the ordinary baseline under a method name. Check the regime and the task mix.')
        print(f"[{TSW.MECHANISM}] {tgt_sandwich_audit['n_examples'] - _inert:,}/{tgt_sandwich_audit['n_examples']:,} training examples are actually masked")
    ctx_relay_audit = None
    if args.ctx_relay_mask:
        ctx_relay_audit = CRM.training_audit(enc_roles, enc_regions, enc_tasks=enc_tasks)
        print(CRM.format_audit(ctx_relay_audit))
        if ctx_relay_audit['n_no_relay']:
            print(f"[{CRM.MECHANISM}] NOTE: {ctx_relay_audit['n_no_relay']} example(s) have no designated-column token (declared target/head/tail is not a header cell of their table), so for those the mask cuts the whole table rather than relaying it. They are trained on unchanged.")
        if ctx_relay_audit['n_no_receiver']:
            print(f"[{CRM.MECHANISM}] NOTE: {ctx_relay_audit['n_no_receiver']} example(s) have no post-table region (table span not located), so the mask is inert on them. They are trained on unchanged.")
        _inert = sum((d['inert'] for d in ctx_relay_audit['per_task'].values()))
        if _inert == ctx_relay_audit['n_examples']:
            raise SystemExit(f"[{CRM.MECHANISM}] the mask is inert on EVERY training example; this run would be the plain baseline wearing the mechanism's name.")
        print(f"[{CRM.MECHANISM}] {ctx_relay_audit['n_examples'] - _inert:,}/{ctx_relay_audit['n_examples']:,} training examples are actually masked")
    attn_aux_candidates = None
    attn_aux_audit = None
    enc_recv = None
    if args.attn_table_aux:
        attn_aux_candidates = ATM.build_task_candidates(tok, enc_tasks, [r['target'] for r in enc_rows])
        for (_slug, _c) in attn_aux_candidates.items():
            print(f'[{ATM.MECHANISM}] {_slug}: {len(_c):,} distinct valid outputs in the label space of this run')
        enc_recv = []
        (n_recv_tot, n_no_recv, reasons) = (0, 0, collections.Counter())
        per_task = collections.defaultdict(lambda : {'examples': 0, 'receivers': 0, 'none': 0})
        for ((_ids, _lab), _t) in tqdm(list(zip(enc, enc_tasks)), desc=f'{ATM.MECHANISM}: decision positions', unit='ex'):
            (pos, reason) = ATM.receivers_for_example(_lab, attn_aux_candidates[_t].root)
            enc_recv.append(pos)
            per_task[_t]['examples'] += 1
            per_task[_t]['receivers'] += len(pos)
            n_recv_tot += len(pos)
            if not pos:
                n_no_recv += 1
                per_task[_t]['none'] += 1
                reasons[reason or 'unknown'] += 1
        assert len(enc_recv) == len(enc), 'receiver positions drifted from the encodings'
        attn_aux_audit = {'n_examples': len(enc), 'n_receivers': int(n_recv_tot), 'n_no_receiver': int(n_no_recv), 'mean_receivers_per_example': n_recv_tot / len(enc) if enc else 0.0, 'skip_reasons': dict(reasons), 'labels_per_task': {k: len(v) for (k, v) in attn_aux_candidates.items()}, 'per_task': {k: dict(v) for (k, v) in sorted(per_task.items())}}
        print(f"[{ATM.MECHANISM}] {n_recv_tot:,} semantic decision positions over {len(enc):,} training examples ({attn_aux_audit['mean_receivers_per_example']:.2f}/example); {n_no_recv:,} example(s) have none" + (f' {dict(reasons)}' if reasons else ''))
        for (_t, _d) in attn_aux_audit['per_task'].items():
            print(f"[{ATM.MECHANISM}]   {_t}: {_d['receivers']:,} positions over {_d['examples']:,} examples, {_d['none']:,} with none")
        if n_recv_tot == 0:
            raise SystemExit(f'[{ATM.MECHANISM}] the training set has ZERO semantic decision positions, so the auxiliary term could never receive a gradient and the run would be the plain baseline.')
    data_manifest = {'ctx_relay_mask': bool(args.ctx_relay_mask), 'ctx_relay_audit': ctx_relay_audit, 'attn_table_aux': bool(args.attn_table_aux), 'attn_aux_lambda': float(args.attn_aux_lambda) if args.attn_table_aux else None, 'attn_aux_objective': 'depth_variance' if args.attn_table_aux else None, 'attn_aux_beta_ignored': float(args.attn_aux_beta) if args.attn_table_aux else None, 'attn_aux_audit': attn_aux_audit, 'split_gate_proj': bool(args.split_gate_proj), 'role_gate_mlp': bool(args.role_gate_mlp), 'ffn_role_scale': bool(args.ffn_role_scale), 'ffn_role_scale_lr': args.ffn_role_scale_lr if args.ffn_role_scale else None, 'lora_recv_gate': args.lora_recv_gate, 'lora_rank_gate': args.lora_rank_gate, 'rank_gate_exclude_target': list(args.rank_gate_exclude_target), 'role_audit': role_audit, 'balance_supervised_tokens': bool(args.balance_supervised_tokens), 'mean_supervised_tokens_by_task': {t: tok_stats[t]['mean'] for t in tok_stats} if tok_stats else None, 'supervised_token_stats_by_task': tok_stats, 'task_loss_weights': task_weights, 'mu_global_supervised_tokens': tw_expected['mu_global'] if tw_expected else None, 'original_expected_token_contribution_by_task': tw_expected['original'] if tw_expected else None, 'weighted_expected_contribution_by_task': tw_expected['weighted'] if tw_expected else None, 'regime': args.regime, 'seed': args.seed, 'budget': args.budget, 'cta_cpa_source': args.cta_cpa_source, 'select_mode': args.select_mode, 'drop_empty_target': args.drop_empty_target, 'drop_empty_target_post_truncation_dropped': int(_post_trunc_dropped), 'samerow_source': samerow_partner(args.regime, args.samerow_source) if _uses_samerow(args.regime) else None, 'samerow_max_rows': args.samerow_max_rows if _uses_samerow(args.regime) else None, 'col_mask_method': args.col_mask_method, 'max_table_rows': args.max_table_rows, 'row_trunc_method': _rt_plan[0] if _rt_plan is not None else None, 'row_trunc_max_rows': _rt_plan[1] if _rt_plan is not None else None, 'val_truncated': False, 'row_trunc_num_proc': args.num_proc, 'static_row_cap': _hard_cap, 'row_trunc_stats': rt_stats, 'max_context_cols': args.max_context_cols, 'col_trunc_method': args.col_trunc_method, 'reindex_truncated_cols': bool(args.reindex_truncated_cols), 'drop_empty_context_cols': bool(args.drop_empty_context_cols), 'fill_budget_select': bool(args.fill_budget), 'auto_headfit': bool(_rt_plan is not None and (not _hard_cap) and (not args.fill_budget)), 'sotab_full_uncapped': bool(_rt_plan is not None and (not _hard_cap) and (not args.fill_budget)), 'paired_ctacpa': args.regime == 'multi_paired_ctacpa', 'paired_window': args.micro_bs * args.grad_accum if args.regime == 'multi_paired_ctacpa' else None, 'paired_groups': len(pair_group_ids) // 2 if pair_group_ids is not None else None, 'paired_dropped_groups': paired_dropped_groups, 'transpose_augment': bool(args.transpose_augment), 'transpose_augment_stats': ta_stats, 'target_two_views': bool(args.target_two_views), 'target_two_views_stats': tv_stats, 'drop_context_after_target': bool(args.drop_context_after_target), 'drop_context_after_target_stats': dca_stats, 'lora_role_matrix': bool(args.lora_role_matrix), 'struct_role_bias': bool(args.struct_role_bias), 'struct_role_relations': args.struct_role_relations if args.struct_role_bias else None, 'struct_role_lr': args.struct_role_lr if args.struct_role_bias else None, 'soft_role_prompt': bool(args.soft_role_prompt), 'soft_role_param': args.soft_role_param if args.soft_role_prompt else None, 'soft_role_lr': args.soft_role_lr if args.soft_role_prompt else None, 'soft_role_prompt_marks': {k: int(soft_marks[k]) for k in SP.SLOT_NAMES} if args.soft_role_prompt else None, 'soft_role_prompt_cells_absorbed': int(soft_absorbed_total) if args.soft_role_prompt else None, 'train_rows_file': args.train_rows, 'val_rows_files': json.loads(args.val_rows) if args.val_rows else None, 'swap_context_blocks': bool(args.swap_context_blocks), 'swap_context_blocks_stats': sc_stats, 'swap_context_epochs': bool(args.swap_context_epochs), 'swap_context_epochs_stats': se_stats, 'swap_context_epochs_schedule': None if enc_by_epoch is None else ['orig' if e % 2 == 0 else 'swapped' for e in range(len(enc_by_epoch))], 'col_major_tasks': sorted(col_major_tasks), 'col_major_stats': cm_stats, 'table_first': bool(args.table_first), 'table_first_tasks': sorted(table_first_tasks), 'table_last': bool(args.table_last), 'table_last_stats': tl_stats, 'table_first_stats': tf_stats, 'target_end': bool(args.target_end), 'target_end_tasks': sorted(target_end_tasks), 'target_sandwich': bool(args.target_sandwich), 'target_sandwich_stats': TSW.transform_stats() if args.target_sandwich else None, 'target_end_stats': te_stats, 'target_first': bool(args.target_first), 'target_first_stats': tfirst_stats, 'n_requested': len(rows), 'mix': instance_mix, 'mix_examples': task_counts, 'n_used': len(enc), 'train_tokens_total': int(sum((len(ids) for (ids, _lab) in enc))), 'train_supervised_tokens_total': int(sum((sum((1 for t in lab if t != -100)) for (_ids, lab) in enc))), 'table_trimmed_over_len': trimmed, 'dropped_over_len': dropped, 'drop_truncated_tables': bool(args.drop_truncated_tables), 'dropped_truncated_tables': int(dropped_trimmed), 'train_ids': [r['id'] for r in rows], 'train_instance_ids': train_instance_ids}
    if args.dry_run_data:
        os.makedirs(args.out, exist_ok=True)
        with open(f'{args.out}/train_manifest.json', 'w') as f:
            json.dump({**data_manifest, 'dry_run_data': True, 'args': vars(args)}, f, indent=2)
        print(f'[dry_run_data] wrote data-only manifest -> {args.out}/train_manifest.json (no model load)')
        return
    model = ML.load_causal_lm(args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn_impl, device_map={'': 0})
    merged_adapter_info = None
    if args.merge_adapter:
        (model, merged_adapter_info) = merge_adapter_into_base(model, args.merge_adapter)
    force_greedy_generation(model, tok)
    model.config.use_cache = False
    if args.grad_checkpointing:
        _mt = str(getattr(getattr(model, 'config', None), 'model_type', ''))
        if _mt.startswith('gemma3') or type(model).__name__.startswith('Gemma3'):
            import torch._dynamo as _dynamo
            _dynamo.config.disable = True
            _dynamo.config.suppress_errors = True
            print(f"[grad_ckpt] {type(model).__name__} (model_type={_mt or '?'}): torch._dynamo DISABLED + suppress_errors -- the enable_input_require_grads hook is untraceable under compile")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora_targets = list(args.target_modules)
    split_r = split_alpha = None
    if args.split_gate_proj:
        missing = [m for m in SGL.SPLIT_MODULES if m not in lora_targets]
        if missing:
            raise SystemExit(f'--split_gate_proj replaces the {list(SGL.SPLIT_MODULES)} adapters, but {missing} are not in --target_modules ({lora_targets}); the arm would not be a partition of the vanilla adapter.')
        lora_targets = [m for m in lora_targets if m not in SGL.SPLIT_MODULES]
        (split_r, split_alpha) = SGL.split_rank(args.lora_r, args.lora_alpha)
        print(f'[split_gate_proj] PEFT target_modules {lora_targets} @ r={args.lora_r} alpha={args.lora_alpha}; {list(SGL.SPLIT_MODULES)} each served by two {list(SGL.BRANCHES)} pairs @ r={split_r} alpha={split_alpha:g} (scaling {split_alpha / split_r:g} == {args.lora_alpha / args.lora_r:g})')
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias='none', task_type='CAUSAL_LM', target_modules=lora_targets)
    full_ft_info = None
    if args.full_finetune:
        lora = None
        lora_targets = []
        full_ft_info = setup_full_finetune(model, args.grad_checkpointing)
    else:
        model = get_peft_model(model, lora)
    qkv_init_info = None
    if args.init_qkv_adapter:
        qkv_init_info = init_qkv_from_struct_adapter(model, args.init_qkv_adapter, lora)
    split_gate_info = None
    if args.split_gate_proj:
        split_gate_info = SGL.attach(model, r=split_r, alpha=split_alpha, dropout=args.lora_dropout)
        print(f"[split_gate_proj] {split_gate_info['n_modules']} MLP modules {split_gate_info['n_modules_per_family']} x {len(SGL.BRANCHES)} branches @ r={split_gate_info['r']} -> {split_gate_info['n_split_params']:,} parameters (dtype {split_gate_info['param_dtype']}, B zero-init)")
        print(f"[split_gate_proj] {split_gate_info['formula']}")
        print(f"[split_gate_proj] {split_gate_info['token_scope']}")
        _g = torch.Generator().manual_seed(0)
        _probe = torch.randint(0, int(model.config.vocab_size), (2, 16), generator=_g).to(model.device)
        try:
            split_gate_info['zero_init_check'] = SGL.verify_zero_init(model, _probe)
        except RuntimeError as e:
            import traceback as _tb
            _tb.print_exc()
            raise SystemExit(f'[split_gate_proj] startup identity check FAILED: {e} The mechanism is mis-inserted; refusing to train.')
        print(f"[split_gate_proj] zero-init identity: max|logit diff| = {split_gate_info['zero_init_check']['max_abs_logit_diff']:.3e}")
    swiglu_info = None
    if args.swiglu_thres:
        swiglu_info = SWT.attach(model)
        print(f"[swiglu_thres] {swiglu_info['n_layers']} gated MLPs x {len(SWT.BRANCHES)} temperatures -> {swiglu_info['n_params']} parameters (dtype {swiglu_info['param_dtype']}, beta init {swiglu_info['init_beta']})")
        print(f"[swiglu_thres] {swiglu_info['formula']}")
        print(f"[swiglu_thres] {swiglu_info['token_scope']}")
        print(f"[swiglu_thres] activation probe: max|act(x) - silu(x)| = {swiglu_info['activation_probe_max_diff']:.3e}")
        _ma = swiglu_info['memory_audit']
        print(f"[swiglu_thres] memory audit ({_ma['shape']} {_ma['dtype']}): baseline SiLU keeps {_ma['baseline_silu_saved_bytes']:,} B for backward, this mechanism {_ma['swiglu_thres_saved_bytes']:,} B (+{_ma['overhead_bytes']:,} B = {100 * _ma['overhead_ratio_of_one_activation']:.4f}% of one activation); a plain op chain would have kept +{_ma['naive_op_chain_overhead_bytes']:,} B per layer")
        if _ma['overhead_ratio_of_one_activation'] > 0.01:
            raise SystemExit(f"[swiglu_thres] memory audit FAILED: the mechanism keeps {_ma['overhead_bytes']:,} extra bytes per layer alive for backward, more than 1% of one full-size activation. Refusing to train a mechanism that is not memory-neutral against the SiLU path it replaces.")
        _g = torch.Generator().manual_seed(0)
        _probe = torch.randint(0, int(model.config.vocab_size), (2, 16), generator=_g).to(model.device)
        try:
            swiglu_info['init_identity_check'] = SWT.verify_identity(model, _probe)
        except RuntimeError as e:
            import traceback as _tb
            _tb.print_exc()
            raise SystemExit(f'[swiglu_thres] startup identity check FAILED: {e} The mechanism is mis-inserted; refusing to train.')
        print(f"[swiglu_thres] init identity: max|logit diff| = {swiglu_info['init_identity_check']['max_abs_logit_diff']:.3e}")
    role_matrix_info = None
    role_matrix_init = None
    if args.lora_role_matrix:
        role_matrix_info = RM.attach(model)
        role_matrix_init = RM.snapshot(model)
        print(f"[lora_role_matrix] {role_matrix_info['n_modules']} patched modules ({role_matrix_info['modules_per_family']}) x {RM.N_SLOTS} roles {list(RM.SLOT_NAMES)} x r={role_matrix_info['lora_r']} -> {role_matrix_info['n_role_matrix_params']:,} parameters ({role_matrix_info['param_count_formula']}, identity-init)")
        print(f"[lora_role_matrix] {role_matrix_info['formula']}")
        for (_slot, _src) in role_matrix_info['role_sources'].items():
            print(f'[lora_role_matrix] slot {_slot:<12} <- {_src}')
        print(f"[lora_role_matrix] bypassed (ordinary LoRA): {role_matrix_info['bypassed_roles']}")
        print(f"[lora_role_matrix] untouched families (ordinary LoRA): {role_matrix_info['untouched_families']}")
    role_gate_info = None
    role_gate_init = None
    if args.role_gate_mlp:
        role_gate_info = RGM.attach(model)
        role_gate_init = RGM.snapshot(model)
        print(f"[role_gate_mlp] {role_gate_info['n_gated_mlps']} gated MLPs, d_intermediate={role_gate_info['d_intermediate']}, {role_gate_info['n_role_params']:,} role parameters ({len(RGM.ROLE_VECTOR_NAMES)} x d per layer, zero-init)")
        print(f"[role_gate_mlp] placement check: {role_gate_info['placement_check']}")
    ffn_role_info = None
    if args.ffn_role_scale:
        ffn_role_info = FRS.attach(model)
        print(f"[{FRS.MECHANISM}] {ffn_role_info['n_gated_mlps']} gated MLPs, d_intermediate={ffn_role_info['d_intermediate']}, {ffn_role_info['n_scale_params']:,} scale parameters ({FRS.N_SCALE} x d per layer, zero-init -> scale 1)")
        print(f"[{FRS.MECHANISM}] placement check: {ffn_role_info['placement_check']}")
    soft_role_info = None
    soft_role_init = None
    if args.soft_role_prompt:
        soft_slots = SP.slots_for_tasks(set(enc_tasks))
        _unmarked = [k for k in soft_marks if soft_marks[k] and k not in soft_slots]
        if _unmarked:
            raise SystemExit(f'[{SP.MECHANISM}] the training set carries markers for {_unmarked}, which the task set {sorted({canonical_task(t) for t in set(enc_tasks)})} does not allocate. Refusing to train with a slot the bank does not hold.')
        soft_role_info = SP.attach(model, slots=soft_slots, seed=args.seed, param=args.soft_role_param)
        print(f"[{SP.MECHANISM}] tasks {sorted({canonical_task(t) for t in set(enc_tasks)})} -> {soft_role_info['n_slots']} slot(s) allocated {list(soft_role_info['slots'])}" + (f"; not allocated: {soft_role_info['slots_not_allocated']}" if soft_role_info['slots_not_allocated'] else ''))
        soft_role_init = SP.snapshot(model)
        if args.soft_role_param == SP.PARAM_DIRECT:
            print(f"[{SP.MECHANISM}] PROMPT TUNING (direct vectors, no encoder): {soft_role_info['n_slots']} x d_model={soft_role_info['d_model']} = {soft_role_info['n_soft_params']:,} trainable ({soft_role_info['param_count_formula']})")
        else:
            print(f"[{SP.MECHANISM}] P-tuning special-token encoder: {soft_role_info['n_slots']} role embedding(s) x d_role={soft_role_info['role_dim']} ({soft_role_info['n_role_embed_params']:,} params) -> one SHARED linear {soft_role_info['role_dim']} -> {soft_role_info['d_model']} ({soft_role_info['n_encoder_params']:,} params) = {soft_role_info['n_soft_params']:,} trainable ({soft_role_info['param_count_formula']})")
        print(f"[{SP.MECHANISM}] init: {soft_role_info['init']} (emb_std={soft_role_info['init_std']:.5f}, seed={soft_role_info['init_seed']})")
        print(f"[{SP.MECHANISM}] {soft_role_info['formula']}")
        for (_slot, _src) in soft_role_info['slot_sources'].items():
            print(f"[{SP.MECHANISM}] slot {_slot:<12} <- {', '.join(_src)}")
        print(f"[{SP.MECHANISM}] {soft_role_info['insertion']}")
    struct_role_info = None
    struct_role_init = None
    struct_role_audit = None
    if args.struct_role_bias:
        struct_role_info = SRB.attach(model, base_impl=args.attn_impl, relations=struct_relations)
        struct_role_init = SRB.snapshot(model)
        print(f"[{SRB.MECHANISM}] relation set '{struct_role_info['relation_set']}': {', '.join(struct_role_info['relations_active'])}" + (f"   (pinned at 0, never in the graph: {', '.join(struct_role_info['relations_pinned_at_zero'])})" if struct_role_info['relations_pinned_at_zero'] else ''))
        print(f"[{SRB.MECHANISM}] {struct_role_info['param_count_formula']} ({struct_role_info['init']})")
        for (_rel, _scope) in struct_role_info['scopes'].items():
            print(f'[{SRB.MECHANISM}] {_rel:<12} {_scope}')
        if struct_role_info['roles_active']:
            print(f"[{SRB.MECHANISM}] target roles {struct_role_info['target_roles']}, context roles {struct_role_info['context_roles']}")
        print(f"[{SRB.MECHANISM}] per-token metadata: {', '.join(struct_role_info['metadata_needed'])}")
        print(f"[{SRB.MECHANISM}] {struct_role_info['indexing']}")
        print(f"[{SRB.MECHANISM}] bias tensor: {struct_role_info['bias_tensor']} (base kernel {struct_role_info['base_attn_impl']}, {struct_role_info['attention_modules_patched']} attention modules patched)")
        struct_role_audit = SRB.training_audit(enc_struct, enc_roles if struct_needs_roles else None, enc_regions if struct_needs_roles else None, enc_tasks)
        print(SRB.format_audit(struct_role_audit, active=struct_relations))
        struct_role_info['train_audit'] = struct_role_audit
    rank_gate_info = None
    rank_gate_init = None
    if args.lora_rank_gate != 'off':
        rank_gate_info = RKG.attach(model, mode=args.lora_rank_gate, exclude_target_tasks=args.rank_gate_exclude_target)
        rank_gate_init = RKG.snapshot(model)
        print(f"[lora_rank_gate] mode={args.lora_rank_gate}: {rank_gate_info['n_gated_modules']} gated modules ({rank_gate_info['modules_per_family']}) x {len(rank_gate_info['gate_tasks'])} tasks x r={rank_gate_info['lora_r']} -> {rank_gate_info['n_gate_params']:,} gate parameters (zero-init)")
        print(f"[lora_rank_gate] {rank_gate_info['formula']}")
        _sg = rank_gate_info['gate_stop_gradient']
        print(f'[lora_rank_gate] gate_stop_gradient={_sg} ' + ('(instr_mds/gated_ctx_lora_iso.md): the controller reads ' if _sg else '-- PRE-ISO COUPLED routing: the controller back-propagates into ') + '; '.join(rank_gate_info['stop_gradient_inputs']) + (' through .detach(), so d(z_out)/dz is the identity and the gate trains the matrices ONLY; ordinary LoRA keeps every gradient it had.' if _sg else '.'))
        print(f"[lora_rank_gate] task rows: {rank_gate_info['task_rows']} (row 0 is a FIXED ZERO matrix -> ordinary LoRA)")
        print(f"[lora_rank_gate] token scope per task: {rank_gate_info['token_scope_by_task']}   (exclude_target_tasks={rank_gate_info['exclude_target_tasks']}; a task whose target is UNAVAILABLE leaves its designated columns UNGATED, g == 1 exactly)")
        print(f"[lora_rank_gate] gated roles: {rank_gate_info['gated_roles']}")
        print(f"[lora_rank_gate] ungated roles: {rank_gate_info['ungated_roles']}")
        _g = torch.Generator().manual_seed(0)
        _rows = torch.arange(RKG.N_TASK_ROWS, dtype=torch.long)
        _probe = torch.randint(0, int(model.config.vocab_size), (int(_rows.shape[0]), 16), generator=_g).to(model.device)
        _tm = torch.zeros(int(_rows.shape[0]), 16, dtype=torch.bool, device=model.device)
        _tm[:, 8:] = True
        try:
            rank_gate_info['zero_gate_check'] = RKG.verify_zero_gate(model, _probe, attention_mask=torch.ones_like(_probe), task_id=_rows, table_mask=_tm)
        except RuntimeError as e:
            import traceback as _tb
            _tb.print_exc()
            raise SystemExit(f'[lora_rank_gate] startup identity check FAILED: {e} The mechanism is mis-inserted; refusing to train.')
        _zc = rank_gate_info['zero_gate_check']
        print(f"[lora_rank_gate] zero-gate identity: max|logit diff| = {_zc['max_abs_logit_diff']:.3e}, mean(g) = {_zc['mean_g']:.8f}, max|g-1| = {_zc['max_abs_g_minus_1']:.3e}, rows exercised {_zc['task_rows_exercised']}, frac tokens gated " + ('n/a' if _zc.get('frac_tokens_gated') is None else f"{_zc['frac_tokens_gated']:.3f}"))
        print(f"[lora_rank_gate] mask pre-hook on: {rank_gate_info['prehook_on']}")
    ctx_relay_info = None
    if args.ctx_relay_mask:
        ctx_relay_info = CRM.attach(model, base_impl=args.attn_impl)
        print(f'[{CRM.MECHANISM}] attached: ' + '; '.join((f'{k.upper()} {v}' for (k, v) in CRM.RELAY.items())))
        print(f"[{CRM.MECHANISM}] cut keys: roles {ctx_relay_info['context_roles']} (real cell content of the non-designated columns -- header text and data cells; Markdown syntax is ROLE_NONE and stays visible)")
        print(f"[{CRM.MECHANISM}] kept keys: roles {ctx_relay_info['relay_roles']}, plus every non-table token (instruction, candidate list, chat/control, query)")
        print(f"[{CRM.MECHANISM}] receivers: {ctx_relay_info['receiver_scope']}")
        print(f"[{CRM.MECHANISM}] {ctx_relay_info['mask_kind']} 4D mask on {ctx_relay_info['base_attn_impl']}, pre-hook on {ctx_relay_info['prehook_on']}, {ctx_relay_info['n_new_parameters']} new parameters")
    tgt_sandwich_info = None
    if args.target_sandwich and (not args.sandwich_mask):
        print(f"[{TSW.MECHANISM}] --no-sandwich_mask: SERIALIZATION ONLY. The target column is written at both ends of the table and attention is the model's ordinary causal pattern -- no hook attached, no 4D mask built, nothing restricted.")
    if args.target_sandwich and args.sandwich_mask:
        tgt_sandwich_info = TSW.attach(model, base_impl=args.attn_impl)
        print(f'[{TSW.MECHANISM}] attached: ' + '; '.join((f'{k.upper()} {v}' for (k, v) in TSW.SANDWICH.items())))
        print(f"[{TSW.MECHANISM}] cut keys: {tgt_sandwich_info['cut_keys']} (header text and data cells of the FRONT copy; Markdown syntax carries no copy id and stays visible)")
        print(f"[{TSW.MECHANISM}] cut queries: {tgt_sandwich_info['cut_queries']}")
        print(f'[{TSW.MECHANISM}] kept: T_front -> context -> T_back -> answer, and every non-table token (instruction, candidate list, chat/control, query)')
        print(f"[{TSW.MECHANISM}] {tgt_sandwich_info['mask_kind']} 4D mask on {tgt_sandwich_info['base_attn_impl']}, pre-hook on {tgt_sandwich_info['prehook_on']}, {tgt_sandwich_info['n_new_parameters']} new parameters")
    attn_aux_cfg = None
    if args.attn_table_aux:
        n_patched = ATM.attach(model, base_impl=args.attn_impl)
        attn_aux_cfg = {'lam': float(args.attn_aux_lambda), 'log_every': int(getattr(args, 'log_steps', 0) or 0) or 20}
        print(f'[{ATM.MECHANISM}] attached to {n_patched} attention modules over {args.attn_impl} (delegated), 0 new parameters')
        print(f'[{ATM.MECHANISM}] L_total = L_CE + {args.attn_aux_lambda} * R_var, R_var = mean_b sum_l (m_l - mean_l m_l)^2 (deviations SUMMED over layers, = L x the population variance), m_l = mean table attention share of the semantic decision positions at layer l (raw masses; no softmax layer weighting, no normalization across layers)')
        print(f'[{ATM.MECHANISM}] receivers: the position BEFORE each still-unresolved output token (the state that predicts it), same causal shift as analyze_first_answer_attention.py')
    recv_gate_info = None
    recv_gate_init = None
    if args.lora_recv_gate != 'off':
        recv_gate_info = RCV.attach(model, mode=args.lora_recv_gate, exclude_target_tasks=args.rank_gate_exclude_target, base_impl=args.attn_impl)
        recv_gate_init = RCV.snapshot(model)
        print(f"[lora_recv_gate] mode={args.lora_recv_gate}: {recv_gate_info['n_layers']} layers x {len(RCV.STREAMS)} streams x {len(recv_gate_info['gate_tasks'])} tasks x r={recv_gate_info['lora_r']} -> {recv_gate_info['n_gate_params']:,} controller parameters (zero-init)")
        print(f"[lora_recv_gate] {recv_gate_info['formula']}")
        _sg = recv_gate_info['gate_stop_gradient']
        print(f'[lora_recv_gate] gate_stop_gradient={_sg} ' + ('(instr_mds/gated_ctx_lora_iso.md): the correction branch reads each of ' if _sg else '-- PRE-ISO COUPLED routing: the correction branch back-propagates into ') + '; '.join(recv_gate_info['stop_gradient_tensors']) + (' through .detach(), so it trains the controllers ONLY; ordinary LoRA keeps every gradient it had.' if _sg else '.'))
        print(f"[lora_recv_gate] receiver scope: {recv_gate_info['receiver_scope']}")
        print(f"[lora_recv_gate] controller input: {recv_gate_info['controller_input']}")
        print(f"[lora_recv_gate] controlled: {recv_gate_info['controlled_quantity']}")
        print(f"[lora_recv_gate] task rows: {recv_gate_info['task_rows']} (row 0 is a FIXED ZERO matrix -> ordinary LoRA)")
        print(f"[lora_recv_gate] target available per task: {recv_gate_info['target_available_by_task']}   (exclude_target_tasks={recv_gate_info['exclude_target_tasks']}; a task whose target is UNAVAILABLE keeps gP == 1 and gates context only)")
        print(f"[lora_recv_gate] attention kernel: {recv_gate_info['attn_impl_registered']} delegating to {recv_gate_info['base_attn_impl']} on {recv_gate_info['n_attention_modules_patched']} modules")
        _g = torch.Generator().manual_seed(0)
        _rows = torch.tensor([RCV.TASK_NONE] + [RCV.TASK_INDEX[t] for t in RCV.GATE_TASKS], dtype=torch.long)
        _probe = torch.randint(0, int(model.config.vocab_size), (int(_rows.shape[0]), 16), generator=_g).to(model.device)
        (_pm, _cm, _rv, _proles, _pregions) = RCV.probe_masks(_rows, 16)
        _table = _pregions == RCV.R_TABLE
        for (_i, _row) in enumerate(_rows.tolist()):
            _t = {v: k for (k, v) in RCV.TASK_INDEX.items()}.get(_row)
            if _t is None:
                assert not bool(_pm[_i].any()) and (not bool(_cm[_i].any())), 'gate row `none` must stream nothing'
                continue
            if RCV.has_primary_stream(_t):
                assert bool(_pm[_i].any()), f'{_t}: probe built an empty primary stream'
            else:
                assert int(_pm[_i].sum()) == 0, f'{_t} has no primary span, but the probe gave it {int(_pm[_i].sum())} primary tokens'
                assert bool(torch.equal(_cm[_i], _table[_i])), f'{_t}: context must cover exactly the table region'
            assert not bool((_pm[_i] & _cm[_i]).any()), f'{_t}: streams overlap'
            assert not bool((_rv[_i] & (_pm[_i] | _cm[_i])).any()), f'{_t}: a sender was also marked a receiver'
        print(f'[lora_recv_gate] probe streams per row: ' + ', '.join((f"{ {v: k for (k, v) in RCV.TASK_INDEX.items()}.get(r, 'none')}(P={int(_pm[i].sum())},C={int(_cm[i].sum())},R={int(_rv[i].sum())})" for (i, r) in enumerate(_rows.tolist()))))
        (_pmd, _cmd, _rvd) = (_pm.to(model.device), _cm.to(model.device), _rv.to(model.device))
        try:
            recv_gate_info['zero_gate_check'] = RCV.verify_zero_gate(model, _probe, attention_mask=torch.ones_like(_probe), task_id=_rows, primary_mask=_pmd, context_mask=_cmd, recv_mask=_rvd)
            recv_gate_info['zero_gate_check_generation'] = RCV.verify_zero_gate_generation(model, _probe, attention_mask=torch.ones_like(_probe), task_id=_rows, primary_mask=_pmd, context_mask=_cmd, recv_mask=_rvd, max_new_tokens=3)
        except RuntimeError as e:
            import traceback as _tb
            _tb.print_exc()
            raise SystemExit(f'[lora_recv_gate] startup identity check FAILED: {e} The mechanism is mis-inserted; refusing to train.')
        _zc = recv_gate_info['zero_gate_check']
        _zg = recv_gate_info['zero_gate_check_generation']
        print(f"[lora_recv_gate] zero-gate identity: max|logit diff| = {_zc['max_abs_logit_diff']:.3e}, mean(g) = {_zc['mean_g']:.8f}, max|g-1| = {_zc['max_abs_g_minus_1']:.3e}, rows exercised {_zc['rows_exercised']}, {_zc['n_values']:,} gate values")
        print(f"[lora_recv_gate] zero-gate identity through CACHED decoding: {_zg['n_generated']} generated tokens identical = {_zg['tokens_identical']}")
        print(f"[lora_recv_gate] mask pre-hook on: {recv_gate_info['prehook_on']}")
    if not args.full_finetune:
        model.print_trainable_parameters()
    if args.lora_rank_gate != 'off':
        n_lora = sum((p.numel() for (n, p) in model.named_parameters() if 'lora_' in n and p.requires_grad))
        frozen = [n for (n, prm) in model.named_parameters() if 'lora_' in n and (not prm.requires_grad)]
        if frozen:
            raise SystemExit(f'[lora_rank_gate] {len(frozen)} LoRA tensors are frozen (e.g. {frozen[:3]}); the baseline LoRA must stay fully trainable.')
        fams = {m for m in args.target_modules if any((f'.{m}.' in n and 'lora_' in n for (n, _) in model.named_parameters()))}
        if fams != set(args.target_modules):
            raise SystemExit(f'[lora_rank_gate] LoRA is missing from {sorted(set(args.target_modules) - fams)}; the all-linear baseline must not be weakened for this experiment.')
        n_gate = rank_gate_info['n_gate_params']
        print(f'[lora_rank_gate] trainable: {n_lora:,} LoRA + {n_gate:,} gate = {n_lora + n_gate:,}  (gate = {100.0 * n_gate / n_lora:.3f}% of LoRA)')
    if args.lora_recv_gate != 'off':
        n_lora = sum((p.numel() for (n, p) in model.named_parameters() if 'lora_' in n and p.requires_grad))
        frozen = [n for (n, prm) in model.named_parameters() if 'lora_' in n and (not prm.requires_grad)]
        if frozen:
            raise SystemExit(f'[lora_recv_gate] {len(frozen)} LoRA tensors are frozen (e.g. {frozen[:3]}); the baseline LoRA must stay fully trainable.')
        fams = {m for m in args.target_modules if any((f'.{m}.' in n and 'lora_' in n for (n, _) in model.named_parameters()))}
        if fams != set(args.target_modules):
            raise SystemExit(f'[lora_recv_gate] LoRA is missing from {sorted(set(args.target_modules) - fams)}; the all-linear baseline must not be weakened for this experiment.')
        n_gate = recv_gate_info['n_gate_params']
        print(f'[lora_recv_gate] trainable: {n_lora:,} LoRA + {n_gate:,} gate = {n_lora + n_gate:,}  (gate = {100.0 * n_gate / n_lora:.3f}% of LoRA)')
    if args.role_gate_mlp:
        n_lora = sum((p.numel() for (n, p) in model.named_parameters() if 'lora_' in n and p.requires_grad))
        frozen = [n for (n, prm) in model.named_parameters() if 'lora_' in n and (not prm.requires_grad)]
        if frozen:
            raise SystemExit(f'[role_gate_mlp] {len(frozen)} LoRA tensors are frozen (e.g. {frozen[:3]}); the baseline LoRA must stay fully trainable.')
        fams = {m for m in args.target_modules if any((f'.{m}.' in n and 'lora_' in n for (n, _) in model.named_parameters()))}
        if fams != set(args.target_modules):
            raise SystemExit(f'[role_gate_mlp] LoRA is missing from {sorted(set(args.target_modules) - fams)}; the baseline must not be weakened for this experiment.')
        n_role = role_gate_info['n_role_params']
        print(f'[role_gate_mlp] trainable: {n_lora:,} LoRA + {n_role:,} role = {n_lora + n_role:,}  (role = {100.0 * n_role / n_lora:.2f}% of LoRA)')
    if args.ffn_role_scale:
        n_lora = sum((p.numel() for (n, p) in model.named_parameters() if 'lora_' in n and p.requires_grad))
        frozen = [n for (n, prm) in model.named_parameters() if 'lora_' in n and (not prm.requires_grad)]
        if frozen:
            raise SystemExit(f'[{FRS.MECHANISM}] {len(frozen)} LoRA tensors are frozen (e.g. {frozen[:3]}); the baseline LoRA must stay fully trainable.')
        fams = {m for m in args.target_modules if any((f'.{m}.' in n and 'lora_' in n for (n, _) in model.named_parameters()))}
        if fams != set(args.target_modules):
            raise SystemExit(f'[{FRS.MECHANISM}] LoRA is missing from {sorted(set(args.target_modules) - fams)}; the baseline must not be weakened for this experiment.')
        n_scale = ffn_role_info['n_scale_params']
        print(f'[{FRS.MECHANISM}] trainable: {n_lora:,} LoRA + {n_scale:,} scale = {n_lora + n_scale:,}  (scale = {100.0 * n_scale / n_lora:.2f}% of LoRA)')
    if args.init_qkv_adapter:
        fams = {m for m in args.target_modules if any((f'.{m}.' in n and 'lora_' in n for (n, _) in model.named_parameters()))}
        if fams != set(args.target_modules):
            raise SystemExit(f'[init_qkv_adapter] LoRA is missing from {sorted(set(args.target_modules) - fams)}')
        frozen = [n for (n, prm) in model.named_parameters() if 'lora_' in n and (not prm.requires_grad)]
        if frozen:
            raise SystemExit(f'[init_qkv_adapter] {len(frozen)} LoRA tensors are frozen (e.g. {frozen[:3]}); stage 2 must train all seven modules.')
        print(f'[init_qkv_adapter] stage-2 check OK: LoRA on {sorted(fams)}, all trainable.')
    val_tasks = [] if args.no_val else select_val_tasks(args.regime)
    if args.no_val:
        print('[no_val] validation disabled: no val split, no per-epoch generation, load_best_model_at_end=False -- the saved adapter is the LAST epoch')
    _val_rows_map = json.loads(args.val_rows) if args.val_rows else None
    targs = TrainingArguments(output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=args.micro_bs, gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, lr_scheduler_type='cosine', warmup_ratio=0.03, bf16=True, tf32=True, optim='adamw_bnb_8bit' if args.full_finetune else 'adamw_torch_fused', group_by_length=args.group_by_length, logging_steps=args.log_steps, report_to=[], seed=args.seed, eval_strategy='no' if args.no_val else 'epoch', save_strategy='no' if args.no_val else 'epoch', save_total_limit=1, load_best_model_at_end=not args.no_val, metric_for_best_model=None if args.no_val else 'eval_val_score', greater_is_better=True, label_names=['labels'])
    gate_task_ids = None
    if args.lora_rank_gate != 'off':
        gate_task_ids = [RKG.task_row(t) for t in enc_tasks]
        rg_counts = collections.Counter(gate_task_ids)
        row_name = {RKG.TASK_NONE: 'none'}
        row_name.update({RKG.TASK_INDEX[t]: t for t in RKG.GATE_TASKS})
        print('[lora_rank_gate] training examples per gate row: ' + ', '.join((f'{row_name[k]}={rg_counts.get(k, 0)}' for k in sorted(row_name))))
        if rg_counts.get(RKG.TASK_NONE, 0) == len(gate_task_ids):
            raise SystemExit(f'[lora_rank_gate] no training example maps to a gate row: this regime carries none of {list(RKG.GATE_TASKS)}, so every gate matrix would stay at its zero initialization and the run would be the plain baseline under another name.')
        _scope_tot = collections.Counter()
        _per_row = {r: {'examples': 0, 'gated': 0, 'seen': 0} for r in set(gate_task_ids)}
        for (_roles, _row) in tqdm(list(zip(enc_roles, gate_task_ids)), desc='rank-gate token-scope audit', leave=False):
            _rt = torch.as_tensor(_roles)
            _m = RKG.table_mask_from_roles(_rt, args.rank_gate_exclude_target)
            for (_rid, _n) in zip(*[t.tolist() for t in _rt.unique(return_counts=True)]):
                _scope_tot[RGM.RLR.ROLE_NAMES[int(_rid)]] += int(_n)
            _rec = _per_row[_row]
            _rec['examples'] += 1
            _rec['gated'] += int(_m.sum())
            _rec['seen'] += int(_rt.numel())
        print('[lora_rank_gate] training tokens by role: ' + ', '.join((f'{k}={v:,}' for (k, v) in sorted(_scope_tot.items()))))
        for _t in RKG.GATE_TASKS:
            _r = RKG.TASK_INDEX[_t]
            if _r not in _per_row or _per_row[_r]['examples'] == 0:
                print(f'[lora_rank_gate] {_t}: no training example on this gate row (its matrices will stay at zero)')
                continue
            _rec = _per_row[_r]
            print(f"[lora_rank_gate] {_t}: {_rec['examples']:,} examples, {_rec['gated']:,} gated tokens ({100.0 * _rec['gated'] / max(1, _rec['seen']):.1f}% of their tokens)")
            if _rec['gated'] == 0:
                raise SystemExit(f'[lora_rank_gate] {_t} contributes ZERO gated tokens, so its gate matrices could never receive a gradient and the run would be the plain baseline. Fix the role parse rather than training on an inert mechanism.')
    elif args.lora_recv_gate != 'off':
        gate_task_ids = [RCV.task_row(t) for t in enc_tasks]
        rg_counts = collections.Counter(gate_task_ids)
        row_name = {RCV.TASK_NONE: 'none'}
        row_name.update({RCV.TASK_INDEX[t]: t for t in RCV.GATE_TASKS})
        print('[lora_recv_gate] training examples per gate row: ' + ', '.join((f'{row_name[k]}={rg_counts.get(k, 0)}' for k in sorted(row_name))))
        if rg_counts.get(RCV.TASK_NONE, 0) == len(gate_task_ids):
            raise SystemExit(f'[lora_recv_gate] no training example maps to a gate row: this regime carries none of {list(RCV.GATE_TASKS)}, so every controller would stay at its zero initialization and the run would be the plain baseline under another name.')
        _role_tot = collections.Counter()
        for _roles in enc_roles:
            for (_k, _v) in RGM.role_id_counts(_roles).items():
                _role_tot[_k] += _v
        _stream_tot = {r: {'primary': 0, 'context': 0, 'examples': 0} for r in sorted(set(gate_task_ids))}
        for (_roles, _regs, _row) in tqdm(list(zip(enc_roles, enc_regions, gate_task_ids)), desc='recv-gate stream audit', leave=False):
            (_p1, _c1) = RCV.stream_masks(torch.tensor([_row]), torch.tensor([_roles]), torch.tensor([_regs]))
            _stream_tot[_row]['primary'] += int(_p1.sum())
            _stream_tot[_row]['context'] += int(_c1.sum())
            _stream_tot[_row]['examples'] += 1
        _n_primary = sum((v['primary'] for v in _stream_tot.values()))
        _n_context = sum((v['context'] for v in _stream_tot.values()))
        print('[lora_recv_gate] training tokens by role: ' + ', '.join((f'{k}={_role_tot.get(k, 0)}' for k in ['none'] + list(RGM.ROLE_VECTOR_NAMES))))
        print(f'[lora_recv_gate] sender streams: primary={_n_primary:,} tokens, context={_n_context:,} tokens')
        if _n_context == 0:
            raise SystemExit('[lora_recv_gate] not one training token reaches the CONTEXT stream; that controller could never receive a gradient.')
        for _t in RCV.GATE_TASKS:
            _r = RCV.TASK_INDEX[_t]
            _rec = _stream_tot.get(_r)
            if _rec is None or _rec['examples'] == 0:
                print(f'[lora_recv_gate] {_t}: no training example on this gate row (its controllers stay at zero)')
                continue
            _live = RCV.primary_gated(_t, args.rank_gate_exclude_target)
            print(f"[lora_recv_gate] {_t}: {_rec['examples']:,} examples, primary={_rec['primary']:,} tokens (gated={_live}), context={_rec['context']:,} tokens")
            if _live and _rec['primary'] == 0:
                raise SystemExit(f'[lora_recv_gate] {_t} has a primary stream that is supposed to be gated, but NOT ONE of its training tokens lands in it. The declared column is not being found in the table, so this run would silently be the context-only gate for {_t}.')
            if not RCV.has_primary_stream(_t) and _rec['primary'] != 0:
                raise SystemExit(f"[lora_recv_gate] {_t} has NO primary span, but {_rec['primary']:,} of its tokens were routed into the primary stream. A primary was invented somewhere; refusing to train.")
            if _rec['context'] == 0:
                raise SystemExit(f'[lora_recv_gate] {_t} contributes zero CONTEXT tokens; its context controller could never receive a gradient.')
        _reg_tot = collections.Counter()
        _no_recv = 0
        for _regs in enc_regions:
            c = collections.Counter(_regs)
            for (_k, _v) in c.items():
                _reg_tot[_k] += _v
            if c.get(RCV.R_POST, 0) == 0:
                _no_recv += 1
        print('[lora_recv_gate] training tokens by region: ' + ', '.join((f'{RCV.REGION_NAME[k]}={_reg_tot.get(k, 0)}' for k in sorted(_reg_tot))))
        print(f'[lora_recv_gate] receivers: {_reg_tot.get(RCV.R_POST, 0):,} tokens ({100.0 * _reg_tot.get(RCV.R_POST, 0) / max(1, sum(_reg_tot.values())):.1f}% of training tokens); {_no_recv} example(s) have none')
        if _reg_tot.get(RCV.R_POST, 0) == 0:
            raise SystemExit('[lora_recv_gate] the training set has ZERO receiver positions; the controllers could never receive a gradient and the run would be the plain baseline.')
        if _no_recv:
            raise SystemExit(f'[lora_recv_gate] {_no_recv} training example(s) have no post-table region, so the Markdown table span could not be located in them. The receiver set would be silently empty for those prompts; fix the span rule rather than training on a partly inert mechanism.')
    train_dataset = SFTDataset(enc, enc_by_epoch=enc_by_epoch, task_ids=task_ids, role_ids=enc_roles, region_ids=enc_regions, gate_task_ids=gate_task_ids, recv_pos=enc_recv, copy_ids=enc_copy, soft_ids=enc_soft, struct_ids=enc_struct)
    grad_tgt_diag = None
    if args.grad_tgt_b:
        grad_tgt_diag = GTB.TargetGradDiagnostic(model, target_role_id=RGM.RLR.ROLE_CTA_TARGET, out=args.grad_tgt_b, meta={'model': args.model, 'regime': args.regime, 'seed': args.seed, 'budget': args.budget, 'epochs': args.epochs, 'cta_cpa_source': args.cta_cpa_source, 'max_seq_len': args.max_seq_len, 'micro_bs': args.micro_bs, 'grad_accum': args.grad_accum, 'lora_r': args.lora_r, 'lora_alpha': args.lora_alpha, 'target_role': 'cta_target (real cell content: header-cell text + data cells)'})
        print(f'[grad_tgt_b] instrumented {len(grad_tgt_diag.layers)} LoRA layers; report -> {args.grad_tgt_b}')
    task_loss_weight_vec = torch.tensor([task_weights[t] for t in task_names], dtype=torch.float32) if task_weights else None
    trainer = GenEvalTrainer(model=model, args=targs, train_dataset=train_dataset, eval_dataset=SFTDataset(enc[:1]), data_collator=PadCollator(tok.pad_token_id), val_tasks=val_tasks, gen_tok=tok, val_max=args.val_max, val_rows=_val_rows_map, val_max_new_tokens=args.val_max_new_tokens, gen_bs=args.val_gen_batch_size, model_name=args.model, source=args.cta_cpa_source, task_prefix=False, drop_empty_target=args.drop_empty_target, max_table_rows=args.max_table_rows, row_trunc_method=args.row_trunc_method, max_seq_len=args.max_seq_len, trunc_seed=args.seed, num_proc=args.num_proc, pair_group_ids=pair_group_ids, pair_window=args.micro_bs * args.grad_accum, pair_seed=args.seed, task_loss_weight_vec=task_loss_weight_vec, task_names=task_names, col_major_tasks=col_major_tasks, table_first=args.table_first, table_first_tasks=table_first_tasks, table_last=args.table_last, target_end=args.target_end, target_end_tasks=target_end_tasks, target_first=args.target_first, target_sandwich=args.target_sandwich, sandwich_mask=bool(args.target_sandwich and args.sandwich_mask), role_gate=args.role_gate_mlp, ffn_role=args.ffn_role_scale, ffn_role_lr=args.ffn_role_scale_lr if args.ffn_role_scale else None, role_matrix=args.lora_role_matrix, recv_gate=args.lora_recv_gate != 'off', rank_gate=args.lora_rank_gate != 'off', table_mask_mods=tuple([M for (M, on) in ((SGL, args.split_gate_proj), (SWT, args.swiglu_thres)) if on]), ctx_relay=args.ctx_relay_mask, soft_role=args.soft_role_prompt, soft_role_lr=args.soft_role_lr if args.soft_role_prompt else None, struct_role=args.struct_role_bias, struct_role_lr=args.struct_role_lr if args.struct_role_bias else None, attn_aux=attn_aux_cfg, rank_gate_lr=args.rank_gate_lr, grad_tgt=grad_tgt_diag)
    if enc_by_epoch is not None:
        trainer.add_callback(EpochRowDrawCallback(train_dataset))
    if col_mask_on:
        trainer.add_callback(GradColMaskCallback(train_dataset, enc_rows, enc, tok, args.model, args, args.out))
    if args.role_gate_mlp:
        trainer.add_callback(RoleGateCheckpointCallback(model))
    if args.ffn_role_scale:
        trainer.add_callback(FFNRoleScaleCheckpointCallback(model))
    if args.lora_recv_gate != 'off':
        trainer.add_callback(RecvGateCheckpointCallback(model))
    if args.lora_rank_gate != 'off':
        trainer.add_callback(RankGateCheckpointCallback(model))
    if args.lora_role_matrix:
        trainer.add_callback(RoleMatrixCheckpointCallback(model))
    if args.soft_role_prompt:
        trainer.add_callback(SoftRolePromptCheckpointCallback(model))
    if args.struct_role_bias:
        trainer.add_callback(StructRoleBiasCheckpointCallback(model))
    if args.split_gate_proj:
        trainer.add_callback(SplitGateCheckpointCallback(model))
    if args.swiglu_thres:
        trainer.add_callback(SwigluThresCheckpointCallback(model))
    if grad_tgt_diag is not None:
        trainer.add_callback(GradTgtBCallback(grad_tgt_diag))
        grad_tgt_diag.install_clip_probe(trainer.accelerator, trainer.args.max_grad_norm)
    trainer.train()
    if grad_tgt_diag is not None:
        grad_tgt_diag.remove()
        _rep = grad_tgt_diag.finish()
        GTB.print_report(_rep)
        print(f'[grad_tgt_b] wrote {args.grad_tgt_b}')
    if args.role_gate_mlp:
        ck = trainer.state.best_model_checkpoint
        if ck and os.path.exists(os.path.join(ck, RGM.STATE_FILE)):
            RGM.load(model, ck)
            print(f'[role_gate_mlp] restored the role vectors of the best checkpoint ({ck})')
        elif ck:
            raise SystemExit(f'[role_gate_mlp] best checkpoint {ck} carries no {RGM.STATE_FILE}; refusing to save a model whose LoRA and role vectors come from different epochs.')
    if args.ffn_role_scale:
        ck = trainer.state.best_model_checkpoint
        if ck and os.path.exists(os.path.join(ck, FRS.STATE_FILE)):
            FRS.load(model, ck)
            print(f'[{FRS.MECHANISM}] restored the scale deltas of the best checkpoint ({ck})')
        elif ck:
            raise SystemExit(f'[{FRS.MECHANISM}] best checkpoint {ck} carries no {FRS.STATE_FILE}; refusing to save a model whose LoRA and scale deltas come from different epochs.')
    if args.split_gate_proj:
        ck = trainer.state.best_model_checkpoint
        if ck and os.path.exists(os.path.join(ck, SGL.STATE_FILE)):
            SGL.load(model, ck)
            print(f'[split_gate_proj] restored the split gate_proj pairs of the best checkpoint ({ck})')
        elif ck:
            raise SystemExit(f'[split_gate_proj] best checkpoint {ck} carries no {SGL.STATE_FILE}; refusing to save a model whose LoRA and gate_proj pairs come from different epochs.')
    if args.swiglu_thres:
        ck = trainer.state.best_model_checkpoint
        if ck and os.path.exists(os.path.join(ck, SWT.STATE_FILE)):
            SWT.load(model, ck)
            print(f'[swiglu_thres] restored the temperatures of the best checkpoint ({ck})')
        elif ck:
            raise SystemExit(f'[swiglu_thres] best checkpoint {ck} carries no {SWT.STATE_FILE}; refusing to save a model whose LoRA and temperatures come from different epochs.')
    if args.lora_role_matrix:
        ck = trainer.state.best_model_checkpoint
        if ck and os.path.exists(os.path.join(ck, RM.STATE_FILE)):
            RM.load(model, ck)
            print(f'[lora_role_matrix] restored the role matrices of the best checkpoint ({ck})')
        elif ck:
            raise SystemExit(f'[lora_role_matrix] best checkpoint {ck} carries no {RM.STATE_FILE}; refusing to save a model whose LoRA and role matrices come from different epochs.')
    if args.soft_role_prompt:
        ck = trainer.state.best_model_checkpoint
        if ck and os.path.exists(os.path.join(ck, SP.STATE_FILE)):
            SP.load(model, ck)
            print(f'[{SP.MECHANISM}] restored the role vectors of the best checkpoint ({ck})')
        elif ck:
            raise SystemExit(f'[{SP.MECHANISM}] best checkpoint {ck} carries no {SP.STATE_FILE}; refusing to save a model whose LoRA and role vectors come from different epochs.')
    if args.struct_role_bias:
        ck = trainer.state.best_model_checkpoint
        if ck and os.path.exists(os.path.join(ck, SRB.STATE_FILE)):
            SRB.load(model, ck)
            print(f'[{SRB.MECHANISM}] restored beta of the best checkpoint ({ck})')
        elif ck:
            raise SystemExit(f'[{SRB.MECHANISM}] best checkpoint {ck} carries no {SRB.STATE_FILE}; refusing to save a model whose LoRA and bias come from different epochs.')
    if args.lora_rank_gate != 'off':
        ck = trainer.state.best_model_checkpoint
        if ck and os.path.exists(os.path.join(ck, RKG.STATE_FILE)):
            RKG.load(model, ck)
            print(f'[lora_rank_gate] restored the gate matrices of the best checkpoint ({ck})')
        elif ck:
            raise SystemExit(f'[lora_rank_gate] best checkpoint {ck} carries no {RKG.STATE_FILE}; refusing to save a model whose LoRA and gate matrices come from different epochs.')
    if args.lora_recv_gate != 'off':
        ck = trainer.state.best_model_checkpoint
        if ck and os.path.exists(os.path.join(ck, RCV.STATE_FILE)):
            RCV.load(model, ck)
            print(f'[lora_recv_gate] restored the controllers of the best checkpoint ({ck})')
        elif ck:
            raise SystemExit(f'[lora_recv_gate] best checkpoint {ck} carries no {RCV.STATE_FILE}; refusing to save a model whose LoRA and controllers come from different epochs.')
    best_score = trainer.state.best_metric
    best_val_per_task = {}
    if trainer.eval_history:
        best_idx = max(range(len(trainer.eval_history)), key=lambda i: trainer.eval_history[i][0])
        best_val_per_task = trainer.eval_history[best_idx][1]
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    if args.attn_table_aux:
        _aux_summary = trainer.attn_aux_log.summary() if trainer.attn_aux_log else {}
        _aux_info = {'objective': 'depth_variance', 'formula': 'L_CE + lambda * mean_b sum_l (m_l - mean_l m_l)^2', 'lambda': float(args.attn_aux_lambda), 'beta_ignored': float(args.attn_aux_beta), 'train_audit': attn_aux_audit, 'forward_stats': ATM.stats(), 'last_window': _aux_summary, 'batches_without_receivers': int(trainer.attn_aux_skipped_batches), 'affects_inference': False}
        _p = ATM.write_marker(args.out, _aux_info)
        print(f'[{ATM.MECHANISM}] wrote marker -> {_p}')
        _st = ATM.stats()
        print(f"[{ATM.MECHANISM}] measured attention forwards={_st['forwards_measured']:,} (pass-through={_st['forwards_skipped']:,}); batches without receivers={trainer.attn_aux_skipped_batches:,}")
        if _st['forwards_measured'] == 0:
            raise SystemExit(f"[{ATM.MECHANISM}] the auxiliary term was never computed, so this run is the plain baseline wearing the mechanism's name.")
    if args.struct_role_bias:
        struct_role_info['final_stats'] = SRB.stats(model, init_snapshot=struct_role_init)
        struct_role_info['forward_stats'] = SRB.forward_stats()
        SRB.save(model, args.out, extra={'model': args.model, 'seed': args.seed, 'attn_impl': args.attn_impl, 'tasks': sorted(set(enc_tasks))})
        print(f'[{SRB.MECHANISM}] saved beta -> {args.out}/{SRB.STATE_FILE}')
        _bs = struct_role_info['final_stats']
        for _rel in struct_role_info['relations_active']:
            print(f"[{SRB.MECHANISM}] {_rel:<12} mean={_bs['per_relation_mean'][_rel]:+.5f}  |mean|={_bs['per_relation_absmean'][_rel]:.5f}  max|beta|={_bs['per_relation_absmax'][_rel]:.5f}" + (f"  moved={_bs['move_from_init'][_rel]:.5f}" if 'move_from_init' in _bs else ''))
        _fs = struct_role_info['forward_stats']
        print(f"[{SRB.MECHANISM}] layer forwards with the bias added={_fs['layer_forwards_biased']:,}, pass-through={_fs['layer_forwards_passthrough']:,}")
    if args.ctx_relay_mask:
        ctx_relay_info['train_audit'] = ctx_relay_audit
        ctx_relay_info['forward_stats'] = CRM.stats()
        _p = CRM.write_marker(args.out, ctx_relay_info)
        print(f'[{CRM.MECHANISM}] wrote marker -> {_p}')
        _st = CRM.stats()
        print(f"[{CRM.MECHANISM}] forwards masked={_st['forwards_masked']:,} unmasked={_st['forwards_unmasked']:,}")
        if _st['forwards_masked'] == 0:
            raise SystemExit(f"[{CRM.MECHANISM}] the mask was never installed on a single forward -- this run is the plain baseline. Refusing to save it under the mechanism's name.")
    if args.target_sandwich:
        info = dict(tgt_sandwich_info or {'mechanism': TSW.MECHANISM, 'version': TSW.VERSION})
        info[TSW.MASK_KEY] = bool(args.sandwich_mask)
        info['train_audit'] = tgt_sandwich_audit
        if args.sandwich_mask:
            info['forward_stats'] = TSW.stats()
        _p = TSW.write_marker(args.out, info)
        print(f'[{TSW.MECHANISM}] wrote marker -> {_p} (mask={bool(args.sandwich_mask)})')
        if args.sandwich_mask:
            _st = TSW.stats()
            print(f"[{TSW.MECHANISM}] forwards masked={_st['forwards_masked']:,} unmasked={_st['forwards_unmasked']:,}")
            if _st['forwards_masked'] == 0:
                raise SystemExit(f"[{TSW.MECHANISM}] the mask was never installed on a single forward -- this run is the plain baseline on a duplicated column. Refusing to save it under the mechanism's name.")
    if args.role_gate_mlp:
        role_gate_info['final_stats'] = RGM.stats(model, init_snapshot=role_gate_init)
        RGM.save(model, args.out, extra={'model': args.model, 'seed': args.seed})
        print(f'[role_gate_mlp] saved role vectors -> {args.out}/{RGM.STATE_FILE}')
        for (name, rec) in role_gate_info['final_stats'].items():
            print(f"[role_gate_mlp] {name:<12} mean|r|={rec['mean_norm']:.5f} moved={rec.get('mean_move_from_init', 0.0):.5f}")
        if all((r['mean_norm'] == 0.0 for r in role_gate_info['final_stats'].values())):
            raise SystemExit('[role_gate_mlp] every role vector is still exactly zero after training -- the mechanism never received a gradient. This run is the baseline, not the method.')
    if args.ffn_role_scale:
        ffn_role_info['train_audit'] = ffn_role_audit
        ffn_role_info['final_stats'] = FRS.stats(model)
        ffn_role_info['per_layer'] = FRS.per_layer_table(model)
        FRS.save(model, args.out, extra={'model': args.model, 'seed': args.seed, 'lora_lr': args.lr, 'ffn_role_scale_lr': args.ffn_role_scale_lr, 'regime': args.regime})
        print(f'[{FRS.MECHANISM}] saved scale deltas -> {args.out}/{FRS.STATE_FILE}')
        print(FRS.format_stats(model))
        if all((r['max_abs_delta'] == 0.0 for r in ffn_role_info['final_stats'].values())):
            raise SystemExit(f'[{FRS.MECHANISM}] every scale vector is still exactly the identity after training -- the mechanism never received a gradient. This run is the baseline, not the method.')
    if args.soft_role_prompt:
        soft_role_info['final_stats'] = SP.stats(model, init_snapshot=soft_role_init)
        SP.save(model, args.out, extra={'model': args.model, 'seed': args.seed, 'lora_lr': args.lr, 'regime': args.regime, 'run_lora_r': args.lora_r, 'run_lora_alpha': args.lora_alpha, 'max_seq_len': args.max_seq_len})
        print(f'[{SP.MECHANISM}] saved role vectors -> {args.out}/{SP.STATE_FILE}')
        try:
            _ss = soft_role_info['final_stats']
            _alloc = list(soft_role_info['slots'])
            _direct = args.soft_role_param == SP.PARAM_DIRECT
            for _slot in _alloc:
                _r = _ss[_slot]
                if _direct:
                    print(f"[{SP.MECHANISM}] {_slot:<12} ||v||={_r['norm']:.5f} moved={_r.get('move_from_init', 0.0):.5f}")
                else:
                    print(f"[{SP.MECHANISM}] {_slot:<12} latent ||e||={_r['latent_norm']:.5f} moved={_r.get('latent_move_from_init', 0.0):.5f}  |  produced ||v||={_r['norm']:.5f} moved={_r.get('move_from_init', 0.0):.5f}")
            _enc = _ss.get('encoder', {})
            if _direct:
                print(f"[{SP.MECHANISM}] no encoder (prompt tuning): {len(_alloc)} x d_model={_enc.get('d_model', '?')} vectors are the trainable state")
            else:
                print(f"[{SP.MECHANISM}] encoder  d_role={_enc['d_role']} -> d_model={_enc['d_model']}  ||W||={_enc['weight_norm']:.5f} moved={_enc.get('weight_move_from_init', 0.0):.5f}  ||b||={_enc['bias_norm']:.5f} moved={_enc.get('bias_move_from_init', 0.0):.5f}")
            for _t in SP.ROLE_TASKS:
                if f'{_t}_target_context_cos' in _ss:
                    _line = f"[{SP.MECHANISM}] cos({_t}_target, {_t}_context) = {_ss[f'{_t}_target_context_cos']:+.4f} (produced)"
                    _lat = _ss.get(f'{_t}_latent_target_context_cos')
                    print(_line + (f', {_lat:+.4f} (latent)' if _lat is not None else ''))
            if all((_ss[_slot].get('move_from_init', 0.0) == 0.0 for _slot in _alloc)):
                raise SystemExit(f'[{SP.MECHANISM}] not one PRODUCED role vector moved from its initialization -- neither the latents nor the encoder received a gradient, so the inserted representation is the one training started with. This run is the baseline, not the method.')
        except Exception:
            import traceback as _tb
            print(f'[{SP.MECHANISM}] WARNING: the end-of-run report failed. The trained adapter and its role vectors are already saved and the run continues; only this report is missing.')
            _tb.print_exc()
            soft_role_info['report_error'] = _tb.format_exc().strip().splitlines()[-1]
    if args.split_gate_proj:
        split_gate_info['final_weight_stats'] = SGL.weight_stats(model)
        SGL.save(model, args.out, extra={'model': args.model, 'seed': args.seed, 'lora_lr': args.lr, 'regime': args.regime, 'run_lora_r': args.lora_r, 'run_lora_alpha': args.lora_alpha, 'peft_target_modules': lora_targets})
        print(f'[split_gate_proj] saved the split pairs -> {args.out}/{SGL.STATE_FILE}')
        for (br, rec) in split_gate_info['final_weight_stats'].items():
            print(f"[split_gate_proj] {br:<9} mean||A||={rec['mean_norm_A']:.5f} mean||B||={rec['mean_norm_B']:.5f} max||B||={rec['max_norm_B']:.5f}")
            for (fam, frec) in rec['by_family'].items():
                print(f"[split_gate_proj]   {br:<9} {fam:<10} mean||A||={frec['mean_norm_A']:.5f} mean||B||={frec['mean_norm_B']:.5f} max||B||={frec['max_norm_B']:.5f}")
        if all((rec['max_norm_B'] == 0.0 for rec in split_gate_info['final_weight_stats'].values())):
            raise SystemExit('[split_gate_proj] every B is still exactly zero after training -- the mechanism never received a gradient. This run is the baseline, not the method.')
    if args.swiglu_thres:
        swiglu_info['final_betas'] = SWT.beta_summary(model)
        SWT.save(model, args.out, extra={'model': args.model, 'seed': args.seed, 'lora_lr': args.lr, 'regime': args.regime, 'run_lora_r': args.lora_r, 'run_lora_alpha': args.lora_alpha, 'peft_target_modules': lora_targets})
        print(f'[swiglu_thres] saved the temperatures -> {args.out}/{SWT.STATE_FILE}')
        _fb = swiglu_info['final_betas']
        for b in SWT.BRANCHES:
            print(f"[swiglu_thres] beta_{b:<9} min={_fb[b]['min']:.4f} mean={_fb[b]['mean']:.4f} max={_fb[b]['max']:.4f}")
        _gap = _fb['table_minus_nontable']
        print(f"[swiglu_thres] beta_table - beta_nontable: mean={_gap['mean']:+.4f} min={_gap['min']:+.4f} max={_gap['max']:+.4f}; {_gap['n_layers_table_sharper']}/{swiglu_info['n_layers']} layers sharper on table tokens")
        print('[swiglu_thres] per-layer betas (table): ' + ', '.join((f'{v:.3f}' for v in _fb['by_layer']['table'])))
        print('[swiglu_thres] per-layer betas (nontable): ' + ', '.join((f'{v:.3f}' for v in _fb['by_layer']['nontable'])))
        if all((abs(v - 1.0) < 1e-06 for br in SWT.BRANCHES for v in _fb['by_layer'][br])):
            raise SystemExit('[swiglu_thres] every beta is still exactly 1.0 after training -- the temperatures never received a gradient. This run is the baseline, not the method.')
    if args.lora_role_matrix:
        diag = RM.get_bank(model).diagnostics()
        role_matrix_info['final_diagnostics'] = diag
        RM.save(model, args.out, extra={'model': args.model, 'seed': args.seed, 'lora_lr': args.lr, 'regime': args.regime})
        print(f'[lora_role_matrix] saved role matrices -> {args.out}/{RM.STATE_FILE}')
        _sm = diag['summary']
        for _slot in RM.SLOT_NAMES:
            print(f"[lora_role_matrix] ||R_{_slot} - I||_F mean={_sm[f'{_slot}_minus_I_fro_mean']:.5f} max={_sm[f'{_slot}_minus_I_fro_max']:.5f}  diag share of ||R - I||^2 = {_sm[f'{_slot}_diag_energy_frac_mean']:.3f}")
        print('[lora_role_matrix] pairwise ||.||_F: ' + '  '.join((f"{_n}={_sm[f'{_n}_fro_mean']:.5f}" for (_n, _a, _b) in RM.DIAG_PAIRS)))
        print('[lora_role_matrix] (diag share 1.0 = pure coordinate gating, small = real mixing)')
        with open(os.path.join(args.out, 'role_matrix_diag.json'), 'w') as f:
            json.dump(diag, f, indent=2)
        if role_matrix_init is not None:
            _moved = float((RM.snapshot(model) - role_matrix_init).abs().max().item())
            print(f'[lora_role_matrix] max|R - R_init| = {_moved:.3e}')
            if _moved == 0.0:
                raise SystemExit('[lora_role_matrix] the role matrices never moved from their identity initialization: they received no gradient, so this run is the ordinary-LoRA baseline, not the method.')
    if args.lora_rank_gate != 'off':
        rank_gate_info['final_weight_stats'] = RKG.gate_weight_stats(model, init_snapshot=rank_gate_init)
        RKG.save(model, args.out, extra={'model': args.model, 'seed': args.seed, 'rank_gate_lr': args.rank_gate_lr, 'lora_lr': args.lr, 'regime': args.regime})
        print(f'[lora_rank_gate] saved gate matrices -> {args.out}/{RKG.STATE_FILE}')
        for (name, rec) in sorted(rank_gate_info['final_weight_stats'].items()):
            if not isinstance(rec, dict) or 'mean_fro' not in rec:
                continue
            print(f"[lora_rank_gate] {name:<16} mean||W||_F={rec['mean_fro']:.5f} max={rec.get('max_fro', float('nan')):.5f} moved={rec.get('mean_move_from_init', 0.0):.5f}")
        _fro = [rec['mean_fro'] for rec in rank_gate_info['final_weight_stats'].values() if isinstance(rec, dict) and 'mean_fro' in rec]
        if _fro and all((v == 0.0 for v in _fro)):
            raise SystemExit('[lora_rank_gate] every gate matrix is still exactly zero after training -- the mechanism never received a gradient. This run is the baseline, not the method.')
    if args.lora_recv_gate != 'off':
        recv_gate_info['final_weight_stats'] = RCV.gate_weight_stats(model, init_snapshot=recv_gate_init)
        RCV.save(model, args.out, extra={'model': args.model, 'seed': args.seed, 'rank_gate_lr': args.rank_gate_lr, 'lora_lr': args.lr, 'regime': args.regime, 'base_attn_impl': args.attn_impl})
        print(f'[lora_recv_gate] saved controllers -> {args.out}/{RCV.STATE_FILE}')
        for (name, rec) in sorted(recv_gate_info['final_weight_stats'].items()):
            if not isinstance(rec, dict) or 'mean_fro' not in rec:
                continue
            print(f"[lora_recv_gate] {name:<16} mean||W||_F={rec['mean_fro']:.5f} max={rec.get('max_fro', float('nan')):.5f} moved={rec.get('mean_move_from_init', 0.0):.5f} zero_layers={rec.get('n_exactly_zero_layers', 'n/a')}")

        def _is_live(key):
            if not isinstance(key, str) or key.count('/') != 1:
                return False
            (_t, _stream) = key.split('/')
            return _stream != 'primary' or RCV.primary_gated(_t, args.rank_gate_exclude_target)
        _live = [k for k in recv_gate_info['final_weight_stats'] if _is_live(k)]
        if all((recv_gate_info['final_weight_stats'][k]['mean_fro'] == 0.0 for k in _live)):
            raise SystemExit('[lora_recv_gate] every trainable controller is still exactly zero after training -- the mechanism never received a gradient. This run is the baseline, not the method.')
        for (k, rec) in recv_gate_info['final_weight_stats'].items():
            if k in _live or not isinstance(rec, dict) or 'mean_fro' not in rec:
                continue
            if rec['mean_fro'] != 0.0:
                raise SystemExit(f'[lora_recv_gate] {k} is inert (no primary span, or the target is unavailable), so its controller must stay exactly zero, but it moved.')
    (_gate_mod, _gate_tag) = (RKG, 'lora_rank_gate') if args.lora_rank_gate != 'off' else (RCV, 'lora_recv_gate') if args.lora_recv_gate != 'off' else (None, None)
    if _gate_mod is not None:
        try:
            _verify = _gate_mod.load(model, args.out, strict=True)
        except Exception as e:
            import traceback as _tb
            _tb.print_exc()
            raise SystemExit(f'[{_gate_tag}] the gate side-car just written to {args.out} does NOT reload strictly: {type(e).__name__}: {e}\nThe checkpoint-* directories have deliberately NOT been deleted -- the per-epoch gate state is still in {args.out}/checkpoint-*. Fix the save/load path and rebuild from those rather than retraining.')
        print(f"[{_gate_tag}] final reload verified: {args.out}/{_gate_mod.STATE_FILE} loads strictly against this model (trained_with_gate_stop_gradient={(_verify.get('trained_with_gate_stop_gradient') if isinstance(_verify, dict) else 'n/a')})")
    if args.lora_rank_gate != 'off' and args.rank_gate_diag_n > 0:
        try:
            rank_gate_diag = collect_rank_gate_diagnostics(model, train_dataset, tok.pad_token_id, per_row=args.rank_gate_diag_n, micro_bs=args.micro_bs, max_seq_len=args.max_seq_len)
            if rank_gate_diag is not None:
                with open(f'{args.out}/rank_gate_diag.json', 'w') as f:
                    json.dump(rank_gate_diag, f, indent=2)
                rank_gate_info['activation_diag'] = rank_gate_diag
                for (key, rec) in rank_gate_diag.get('by_group', {}).items():
                    if not rec.get('n_values'):
                        print(f"[lora_rank_gate] {key:<16} no gated token on this row/family ({rec.get('n_tokens_seen', 0):,} tokens seen)")
                        continue
                    _rel = rec.get('rank_space_rel_change')
                    print(f"[lora_rank_gate] {key:<16} mean(g)={rec['mean_g']:.5f} mean|g-1|={rec['mean_abs_dev']:.5f} rms(g-1)={rec['rms_dev']:.5f} max|g-1|={rec['max_abs_dev']:.5f} gated {rec['n_tokens_gated']:,} tokens rank-space rel change=" + ('n/a' if _rel is None else f'{_rel:.5f}'))
                print(f'[lora_rank_gate] wrote {args.out}/rank_gate_diag.json')
        except Exception:
            import traceback as _tb
            print('[lora_rank_gate] WARNING: the activation diagnostics failed. The trained adapter and its gate matrices are already saved and the run continues; only this report is missing.')
            _tb.print_exc()
            rank_gate_info['activation_diag_error'] = _tb.format_exc().strip().splitlines()[-1]
    if args.lora_recv_gate != 'off' and args.rank_gate_diag_n > 0:
        try:
            recv_gate_diag = collect_recv_gate_diagnostics(model, train_dataset, tok.pad_token_id, per_row=args.rank_gate_diag_n, micro_bs=args.micro_bs, max_seq_len=args.max_seq_len)
            if recv_gate_diag is not None:
                with open(f'{args.out}/recv_gate_diag.json', 'w') as f:
                    json.dump(recv_gate_diag, f, indent=2)
                recv_gate_info['activation_diag'] = recv_gate_diag
                for (key, rec) in recv_gate_diag.get('by_task_stream', {}).items():
                    if not rec.get('applicable', True):
                        print(f"[lora_recv_gate] {key:<16} not applicable ({rec.get('reason', 'inert stream')})")
                        continue
                    print(f"[lora_recv_gate] {key:<16} mean(g)={rec['mean_g']:.5f} mean|g-1|={rec['mean_abs_dev']:.5f} rms(g-1)={rec['rms_dev']:.5f} max|g-1|={rec['max_abs_dev']:.5f} g<1 {rec['frac_lt_1']:.2f} / g>1 {rec['frac_gt_1']:.2f} rank-space rel change={rec['rank_space_rel_change']:.5f}")
                for (key, rec) in recv_gate_diag.get('by_receiver_half', {}).items():
                    if 'mean_abs_dev' not in rec:
                        continue
                    print(f"[lora_recv_gate] receiver half {key:<7} mean|g-1|={rec['mean_abs_dev']:.5f} over {rec.get('n_receivers', 0):,} positions")
                print(f'[lora_recv_gate] wrote {args.out}/recv_gate_diag.json')
        except Exception:
            import traceback as _tb
            print('[lora_recv_gate] WARNING: the activation diagnostics failed. The trained adapter and its controllers are already saved and the run continues; only this report is missing.')
            _tb.print_exc()
            recv_gate_info['activation_diag_error'] = _tb.format_exc().strip().splitlines()[-1]
    with open(f'{args.out}/prompt_format.json', 'w') as f:
        json.dump({'drop_empty_target': args.drop_empty_target, 'ctx_relay_mask': bool(args.ctx_relay_mask), 'split_gate_proj': bool(args.split_gate_proj), 'swiglu_thres': bool(args.swiglu_thres), 'role_gate_mlp': bool(args.role_gate_mlp), 'ffn_role_scale': bool(args.ffn_role_scale), 'lora_recv_gate': args.lora_recv_gate}, f, indent=2)
    with open(f'{args.out}/train_manifest.json', 'w') as f:
        json.dump({**data_manifest, 'selection_tasks': val_tasks, 'selection_rule': 'CTA/CPA use strict macro-F1; HVM/TabFact/samerow use accuracy; EM uses match-class F1; multi uses mean over selected task scores.', 'best_val_score': best_score, 'best_val_per_task': best_val_per_task, 'val_counts': trainer.val_counts, 'grad_checkpointing': args.grad_checkpointing, 'structural_qkv_init': qkv_init_info, 'merge_adapter': args.merge_adapter, 'merged_adapter': merged_adapter_info, 'split_gate_proj': bool(args.split_gate_proj), 'split_gate_info': split_gate_info, 'swiglu_thres': bool(args.swiglu_thres), 'swiglu_thres_info': swiglu_info, 'peft_target_modules': lora_targets, 'full_finetune': bool(args.full_finetune), 'full_finetune_info': full_ft_info, 'role_gate_mlp': bool(args.role_gate_mlp), 'role_gate_info': role_gate_info, 'ffn_role_scale': bool(args.ffn_role_scale), 'ffn_role_scale_lr': args.ffn_role_scale_lr if args.ffn_role_scale else None, 'ffn_role_info': ffn_role_info, 'lora_recv_gate': args.lora_recv_gate, 'recv_gate_info': recv_gate_info, 'lora_rank_gate': args.lora_rank_gate, 'rank_gate_info': rank_gate_info, 'rank_gate_lr': args.rank_gate_lr, 'lora_role_matrix': bool(args.lora_role_matrix), 'role_matrix_info': role_matrix_info, 'soft_role_prompt': bool(args.soft_role_prompt), 'soft_role_param': args.soft_role_param if args.soft_role_prompt else None, 'soft_role_lr': args.soft_role_lr if args.soft_role_prompt else None, 'soft_role_prompt_info': soft_role_info, 'struct_role_bias': bool(args.struct_role_bias), 'struct_role_relations': args.struct_role_relations if args.struct_role_bias else None, 'struct_role_lr': args.struct_role_lr if args.struct_role_bias else None, 'struct_role_bias_info': struct_role_info, 'ctx_relay_mask': bool(args.ctx_relay_mask), 'ctx_relay_info': ctx_relay_info, 'ctx_relay_audit': ctx_relay_audit, 'optimizer_param_groups': getattr(trainer, 'optimizer_group_summary', None), 'args': vars(args)}, f, indent=2)
    _cks = sorted(glob.glob(f'{args.out}/checkpoint-*'))
    for ck in _cks:
        shutil.rmtree(ck, ignore_errors=True)
    if _cks:
        print(f'[out] the run completed; removed {len(_cks)} intermediate checkpoint dir(s)')
    print(f'saved best adapter -> {args.out}  (best_val_score={best_score})')
if __name__ == '__main__':
    main()
