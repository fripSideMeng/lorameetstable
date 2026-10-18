import argparse
import collections
import json
import os
import random
import re
import model_load as ML
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import table_transpose as TTR
import table_reorder as TRO
import lora_token_ablation as LTA
import role_gate_mlp as RGM
import ffn_role as FRS
import tgt_col_mask as TCM
import rank_gate as RKG
import role_matrix as RM
import recv_gate as RCV
import soft_prompt as SP
import ctx_relay as CRM
import struct_role as SRB
import tgt_sandwich as TSW
import split_gate_lora as SGL
import swiglu_thres as SWT
ROOT = os.environ.get('LORA_GEN_ROOT') or '.'
DATA = f'{ROOT}/data_prepared'

def read_jsonl(p):
    return [json.loads(l) for l in open(p)]

def strip_thinking(text):
    if text is None:
        return ''
    if '</think>' in text:
        return text.split('</think>', 1)[1].strip()
    return text.strip()

def extract_json(text):
    text = strip_thinking(text)
    start = text.find('{')
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None

def strict(x):
    return str(x).strip()

def loose(x):
    return re.sub('[^a-z0-9]+', '', str(x).lower())

def _read_value(pred_obj, primary_keys):
    if not isinstance(pred_obj, dict):
        return None
    for k in primary_keys:
        if k in pred_obj:
            return pred_obj[k]
    vals = list(pred_obj.values())
    return vals[0] if len(vals) == 1 else None

def score_cta(pred_obj, gold_obj, accept, n=strict):
    accepted = {n(a) for a in accept or []} or {n(gold_obj['column type'])}
    pred = _read_value(pred_obj, ['column type'])
    pred = n(pred) if pred is not None else None
    ok = int(pred in accepted)
    return (ok, 1, ok)

def score_cpa(pred_obj, gold_obj, accept, n=strict):
    accepted = {n(a) for a in accept or []} or {n(gold_obj['property'])}
    pred = _read_value(pred_obj, ['property'])
    pred = n(pred) if pred is not None else None
    ok = int(pred in accepted)
    return (ok, 1, ok)

def score_hvm(pred_obj, gold_obj, accept=None, n=strict):
    gold = [n(x) for x in gold_obj['column_headers']]
    pred_val = _read_value(pred_obj, ['column_headers'])
    pred = []
    if isinstance(pred_val, list):
        pred = [n(x) for x in pred_val]
    correct = sum((1 for (i, g) in enumerate(gold) if i < len(pred) and pred[i] == g))
    total = len(gold)
    exact = int(pred == gold)
    return (correct, total, exact)

def score_cta_mmtu(pred_obj, gold_obj, accept, n=strict):
    accepted = {n(a) for a in accept or []} or {n(gold_obj['DBpedia ontology class'])}
    pred = _read_value(pred_obj, ['DBpedia ontology class', 'column type'])
    pred = n(pred) if pred is not None else None
    ok = int(pred in accepted)
    return (ok, 1, ok)

def score_cpa_mmtu(pred_obj, gold_obj, accept, n=strict):
    accepted = {n(a) for a in accept or []} or {n(gold_obj['DBpedia property'])}
    pred = _read_value(pred_obj, ['DBpedia property', 'property'])
    pred = n(pred) if pred is not None else None
    ok = int(pred in accepted)
    return (ok, 1, ok)

def score_cea_mmtu(pred_obj, gold_obj, accept, n=strict):
    accepted = {n(a) for a in accept or []} or {n(gold_obj['DBpedia entity'])}
    pred = _read_value(pred_obj, ['DBpedia entity', 'entity'])
    pred = n(pred) if pred is not None else None
    ok = int(pred in accepted)
    return (ok, 1, ok)

def score_em(pred_obj, gold_obj, accept, n=strict):
    accepted = {n(a) for a in accept or []} or {n(gold_obj['label'])}
    pred = _read_value(pred_obj, ['label'])
    pred = n(pred) if pred is not None else None
    ok = int(pred in accepted)
    return (ok, 1, ok)

def score_cea_bin(pred_obj, gold_obj, accept, n=strict):
    accepted = {n(a) for a in accept or []} or {n(gold_obj['answer'])}
    pred = _read_value(pred_obj, ['answer'])
    pred = n(pred) if pred is not None else None
    ok = int(pred in accepted)
    return (ok, 1, ok)
BIN_CEA_INSTRUCTION = 'You are given a table and a target cell in it. You are also given a candidate DBpedia resource URI. Decide whether the candidate URI correctly identifies the entity referenced in the target cell.\n \nNo explanation, return your answer in JSON format: {"answer": "yes"} if the candidate is correct, or {"answer": "no"} if it is not. Please note that DBpedia entity URIs are in the format of "http://dbpedia.org/resource/<entity>".'

def cea_table_block(prompt):
    cut = prompt.find('Input Table:')
    return prompt[cut:] if cut != -1 else prompt

def binary_cea_prompt(table_block, candidate):
    return f'{BIN_CEA_INSTRUCTION}\n\n{table_block}\n\nCandidate DBpedia entity: {candidate}\n\nIs this candidate the correct DBpedia entity for the target cell?'

def load_cea_mmtu_binary(data_root):
    rows = read_jsonl(f'{data_root}/cea_mmtu/test.jsonl')
    out = []
    for (i, r) in enumerate(rows):
        gold = json.loads(r['target'])['DBpedia entity']
        out.append({'id': f'cea_mmtu_bin-test-{i}', 'task': 'cea_mmtu_bin', 'dataset': r.get('dataset'), 'input': binary_cea_prompt(cea_table_block(r['input']), gold), 'target': json.dumps({'answer': 'yes'}, ensure_ascii=False), 'accept': ['yes'], 'meta': {**r.get('meta', {}), 'candidate': gold, 'polarity': 'yes', 'orig_target': r['target']}})
    return out

def score_tabfact(pred_obj, gold_obj, accept, n=strict):
    accepted = {n(a) for a in accept or []} or {n(gold_obj['label'])}
    pred = _read_value(pred_obj, ['label'])
    pred = n(pred) if pred is not None else None
    ok = int(pred in accepted)
    return (ok, 1, ok)

def score_samerow(pred_obj, gold_obj, accept, n=strict):
    accepted = {n(a) for a in accept or []} or {n(gold_obj['label'])}
    pred = _read_value(pred_obj, ['label'])
    pred = n(pred) if pred is not None else None
    ok = int(pred in accepted)
    return (ok, 1, ok)

def _qa_norm(x):
    t = str(x).strip()
    if len(t) >= 2 and t[0] == t[-1] and (t[0] in '"\''):
        t = t[1:-1].strip()
    t = re.sub('\\s+', ' ', t).strip().rstrip('.').strip().lower()
    num = t.replace(',', '').lstrip('$£€').rstrip('%').strip()
    if num not in ('', '-', '+', '.') and re.fullmatch('[+-]?(\\d+\\.?\\d*|\\.\\d+)', num):
        v = float(num)
        return str(int(v)) if v == int(v) else repr(v)
    return t

def _qa_answer_set(x):
    if isinstance(x, (list, tuple)):
        return {_qa_norm(i) for i in x if str(i).strip() != ''}
    t = str(x)
    whole = _qa_norm(t)
    if re.fullmatch('[+-]?\\d+(\\.\\d+)?', whole):
        return {whole}
    parts = re.split('\\s*\\|\\s*', t) if '|' in t else re.split('\\s*[;,]\\s*', t) if re.search('[;,]', t) else [t]
    return {_qa_norm(p) for p in parts if str(p).strip() != ''}

def score_wikiqa(pred_obj, gold_obj, accept, n=strict):
    golds = [a for a in accept or [] if a is not None] or [gold_obj.get('answer')]
    accepted = [_qa_answer_set(g) for g in golds]
    pred = _read_value(pred_obj, ['answer'])
    ok = int(pred is not None and _qa_answer_set(pred) in accepted)
    return (ok, 1, ok)
SCORERS = {'cta': score_cta, 'cta_viznet': score_cta, 'cpa': score_cpa, 'hvm': score_hvm, 'cta_mmtu': score_cta_mmtu, 'cpa_mmtu': score_cpa_mmtu, 'cta_mmtu_ctx': score_cta_mmtu, 'cpa_mmtu_ctx': score_cpa_mmtu, 'cea_mmtu': score_cea_mmtu, 'em_mmtu': score_em, 'tabfact_mmtu': score_tabfact, 'wikiqa': score_wikiqa, 'finqa': score_wikiqa, 'tablebench': score_wikiqa, 'wikiqa_mmtu': score_wikiqa, 'cea_bin': score_cea_bin, 'cea_mmtu_bin': score_cea_bin}

def pred_repr(slug, pj):
    if pj is None:
        return None
    if slug in ('cta', 'cta_viznet'):
        return _read_value(pj, ['column type'])
    if slug == 'cpa':
        return _read_value(pj, ['property'])
    if slug in ('cta_mmtu', 'cta_mmtu_ctx'):
        return _read_value(pj, ['DBpedia ontology class', 'column type'])
    if slug in ('cpa_mmtu', 'cpa_mmtu_ctx'):
        return _read_value(pj, ['DBpedia property', 'property'])
    if slug == 'cea_mmtu':
        return _read_value(pj, ['DBpedia entity', 'entity'])
    if slug in ('em_mmtu', 'tabfact_mmtu'):
        return _read_value(pj, ['label'])
    if slug in ('cea_bin', 'cea_mmtu_bin', 'wikiqa', 'wikiqa_mmtu', 'finqa', 'tablebench'):
        return _read_value(pj, ['answer'])
    if slug == 'hvm':
        return _read_value(pj, ['column_headers'])
    return None
CLASSIF = {'cta', 'cta_viznet', 'cpa', 'cta_mmtu', 'cpa_mmtu', 'cta_mmtu_ctx', 'cpa_mmtu_ctx', 'em_mmtu', 'tabfact_mmtu'}
GOLD_KEYS = {'cta': 'column type', 'cta_viznet': 'column type', 'cpa': 'property', 'cta_mmtu': 'DBpedia ontology class', 'cpa_mmtu': 'DBpedia property', 'cta_mmtu_ctx': 'DBpedia ontology class', 'cpa_mmtu_ctx': 'DBpedia property', 'em_mmtu': 'label', 'tabfact_mmtu': 'label'}
INVALID = '__invalid__'

def gold_repr(slug, gj):
    return gj.get(GOLD_KEYS[slug])

def binary_prf(golds, preds, positive):
    tp = sum((1 for (g, p) in zip(golds, preds) if g == positive and p == positive))
    fp = sum((1 for (g, p) in zip(golds, preds) if g != positive and p == positive))
    fn = sum((1 for (g, p) in zip(golds, preds) if g == positive and p != positive))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {'precision': prec, 'recall': rec, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'support': tp + fn}

def macro_weighted_f1(golds, preds):
    from collections import Counter
    (tp, fp, fn) = (Counter(), Counter(), Counter())
    support = Counter(golds)
    for (g, p) in zip(golds, preds):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    gold_classes = [c for c in support if support[c] > 0]
    if not gold_classes:
        return (0.0, 0.0)
    f1 = {}
    for c in gold_classes:
        prec = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1[c] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    macro = sum((f1[c] for c in gold_classes)) / len(gold_classes)
    tot = sum((support[c] for c in gold_classes))
    weighted = sum((f1[c] * support[c] for c in gold_classes)) / tot
    return (macro, weighted)

def force_greedy_generation(model, tok):
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
    is_gemma3 = str(getattr(model.config, 'model_type', '')).startswith('gemma3')
    declared_eos = None
    if is_gemma3:
        d = gen_cfg.eos_token_id
        declared_eos = [d] if isinstance(d, int) else list(d) if d is not None else []
    if tok.eos_token_id is not None:
        if declared_eos is None:
            gen_cfg.eos_token_id = tok.eos_token_id
        else:
            gen_cfg.eos_token_id = list(dict.fromkeys(declared_eos + [tok.eos_token_id]))
    if is_gemma3 and getattr(gen_cfg, 'cache_implementation', None):
        print(f'[gemma3] cache_implementation={gen_cfg.cache_implementation!r} -> None (dynamic cache); the hybrid static cache raises on the second batch and forces a torch.compile')
        gen_cfg.cache_implementation = None
    model.generation_config = gen_cfg

def is_qwen3_model(model_name):
    name = str(model_name).lower()
    return 'qwen3' in name or 'qwen-3' in name

def is_qwen3_thinking_model(model_name):
    name = str(model_name).lower()
    return 'thinking' in name

def apply_model_chat_template(tok, content, model_name):
    messages = [{'role': 'user', 'content': content}]
    if getattr(tok, 'chat_template', None) is None:
        return content
    if is_qwen3_model(model_name) and (not is_qwen3_thinking_model(model_name)):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def generate_batch(model, tok, inputs, max_new_tokens, bs, model_name, masker=None, ablate='none', role_task=None, gate_task=None, recv_task=None, split_gate=False, swiglu_thres=False, ctx_relay_task=None, role_matrix_task=None, sandwich_task=None, soft_task=None, struct_task=None, ffn_role_task=None, tgt_col_task=None):
    from tqdm import tqdm
    outs = []
    do_mask = masker is not None and ablate != 'none'
    do_roles = role_task is not None
    do_ffn_role = ffn_role_task is not None
    do_gate = gate_task is not None
    do_recv = recv_task is not None
    do_rm = role_matrix_task is not None
    do_split = bool(split_gate)
    do_swt = bool(swiglu_thres)
    do_relay = ctx_relay_task is not None
    do_tgtcol = tgt_col_task is not None
    do_sandwich = sandwich_task is not None
    do_soft = soft_task is not None
    do_struct = struct_task is not None
    add_special = getattr(tok, 'chat_template', None) is None
    for i in tqdm(range(0, len(inputs), bs), desc='generating', unit='batch'):
        chunk = inputs[i:i + bs]
        texts = [apply_model_chat_template(tok, c, model_name) for c in chunk]
        if do_soft:
            enc = SP.generation_batch(model, tok, chunk, texts, soft_task, device=model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, generation_config=model.generation_config)
            for j in range(len(chunk)):
                outs.append(tok.decode(gen[j], skip_special_tokens=True))
            continue
        enc = tok(texts, return_tensors='pt', padding=True, add_special_tokens=add_special, return_offsets_mapping=do_mask)
        if do_mask:
            offsets = enc.pop('offset_mapping').tolist()
            attn = enc['attention_mask'].tolist()
            regions = [LTA.region_ids_for_example(offsets[j], attn[j], LTA.table_char_span(chunk[j], texts[j])) for j in range(len(chunk))]
            masker.note_regions(regions)
            enc = enc.to(model.device)
            masker.set_prompt_mask(LTA.keep_mask_from_regions(regions, ablate, device=model.device))
        else:
            enc = enc.to(model.device)
        if do_roles or do_gate or do_recv or do_rm or do_ffn_role:
            _slug = role_task if do_roles else gate_task if do_gate else recv_task if do_recv else role_matrix_task if do_rm else ffn_role_task
            per_ex = [RGM.build_roles(tok, chunk[j], texts[j], _slug) for j in range(len(chunk))]
            roles = RGM.roles_from_attention_mask(per_ex, enc['attention_mask'].cpu()).to(model.device)
            if do_roles:
                RGM.set_role_ids(roles)
            if do_ffn_role:
                FRS.set_role_ids(roles)
            if do_gate:
                RKG.set_gate_context_for_slug(gate_task, RKG.gate_mask(model, roles), valid_mask=enc['attention_mask'], device=model.device)
            if do_rm:
                RM.set_role_context(roles)
            if do_recv:
                _regs = RCV.regions_from_attention_mask([RCV.build_regions(tok, chunk[j], texts[j]) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(model.device)
                (_pm, _cm, _rv) = RCV.masks_for_slug(recv_task, roles, _regs)
                RCV.set_gate_context_for_slug(recv_task, _pm, _cm, _rv, valid_mask=enc['attention_mask'], device=model.device)
        if do_relay:
            _cr_roles = RGM.roles_from_attention_mask([RGM.build_roles(tok, chunk[j], texts[j], ctx_relay_task) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(model.device)
            _cr_regs = RCV.regions_from_attention_mask([RCV.build_regions(tok, chunk[j], texts[j]) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(model.device)
            CRM.set_context_from_ids(_cr_roles, _cr_regs)
        if do_tgtcol:
            _tc_struct = SRB.struct_from_attention_mask([SRB.build_struct(tok, texts[j]) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(model.device)
            _tc_roles = RGM.roles_from_attention_mask([RGM.build_roles(tok, chunk[j], texts[j], tgt_col_task) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(model.device)
            TCM.set_context_from_ids(_tc_struct, _tc_roles)
        if do_struct:
            _sb_struct = SRB.struct_from_attention_mask([SRB.build_struct(tok, texts[j]) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(model.device)
            _sb_roles = _sb_regs = None
            if SRB.needs_roles(model):
                _sb_roles = RGM.roles_from_attention_mask([RGM.build_roles(tok, chunk[j], texts[j], struct_task) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(model.device)
                _sb_regs = RCV.regions_from_attention_mask([RCV.build_regions(tok, chunk[j], texts[j]) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(model.device)
            SRB.set_context(_sb_struct, _sb_roles, _sb_regs)
        if do_sandwich:
            _sw_copies = TSW.copy_ids_from_attention_mask([TSW.build_copy_ids(tok, chunk[j], texts[j], sandwich_task) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(model.device)
            _sw_regs = RCV.regions_from_attention_mask([RCV.build_regions(tok, chunk[j], texts[j]) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(model.device)
            TSW.set_context_from_ids(_sw_copies, _sw_regs)
        if do_split or do_swt:
            _treg = RCV.regions_from_attention_mask([RCV.build_regions(tok, chunk[j], texts[j]) for j in range(len(chunk))], enc['attention_mask'].cpu()).to(model.device)
            if do_split:
                SGL.set_table_mask_from_regions(_treg)
            if do_swt:
                SWT.set_table_mask_from_regions(_treg)
        try:
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, generation_config=model.generation_config)
        finally:
            if do_roles:
                RGM.clear_role_ids()
            if do_ffn_role:
                FRS.clear_role_ids()
            if do_gate:
                RKG.clear_gate_context()
            if do_recv:
                RCV.clear_gate_context()
            if do_rm:
                RM.clear_role_context()
            if do_split:
                SGL.clear_table_mask()
            if do_swt:
                SWT.clear_table_mask()
            if do_relay:
                CRM.clear_context()
            if do_tgtcol:
                TCM.clear_context()
            if do_struct:
                SRB.clear_context()
            if do_sandwich:
                TSW.clear_context()
        for j in range(len(chunk)):
            new = gen[j][enc['input_ids'].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True))
    if do_mask:
        masker.clear()
    return outs
TABFACT_LABELS = ('ENTAILED', 'REFUTED')
MARGIN_TASKS = ('tabfact', 'tabfact_mmtu')

@torch.no_grad()
def decision_margin(model, tok, slug, prompts, test, bs, model_name, masker=None, ablate='none', gate_task=None, recv_task=None):
    from tqdm import tqdm
    cand_txt = [json.dumps({'label': l}) for l in TABFACT_LABELS]
    cand_ids = [tok(c, add_special_tokens=False)['input_ids'] for c in cand_txt]
    keep_k = max((len(c) for c in cand_ids)) + 1
    do_mask = masker is not None and ablate != 'none'
    pad_id = tok.pad_token_id
    rows = []
    for (i, prompt) in enumerate(prompts):
        text = apply_model_chat_template(tok, prompt, model_name)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=do_mask)
        pids = enc['input_ids']
        regs = None
        if do_mask:
            regs = LTA.region_ids_for_example(enc['offset_mapping'], [1] * len(pids), LTA.table_char_span(prompt, text))
        for ci in range(len(TABFACT_LABELS)):
            rows.append((i, ci, pids, regs))
    div = 0
    while div < min(len(cand_ids[0]), len(cand_ids[1])) and cand_ids[0][div] == cand_ids[1][div]:
        div += 1
    div_ok = div < min(len(cand_ids[0]), len(cand_ids[1]))
    div_toks = [cand_ids[0][div], cand_ids[1][div]] if div_ok else None
    logps = [[None] * len(TABFACT_LABELS) for _ in prompts]
    first = [None] * len(prompts)
    for st in tqdm(range(0, len(rows), bs), desc=f'margin[{slug}]', unit='batch'):
        chunk = rows[st:st + bs]
        seqs = [r[2] + cand_ids[r[1]] for r in chunk]
        width = max((len(x) for x in seqs))
        (ids, att, regions) = ([], [], [])
        for (r, seq) in zip(chunk, seqs):
            npad = width - len(seq)
            ids.append([pad_id] * npad + seq)
            att.append([0] * npad + [1] * len(seq))
            if do_mask:
                regions.append([LTA.R_PAD] * npad + list(r[3]) + [LTA.R_GEN] * len(cand_ids[r[1]]))
        ids_t = torch.tensor(ids, device=model.device)
        att_t = torch.tensor(att, device=model.device)
        pos_t = (att_t.cumsum(-1) - 1).clamp(min=0)
        if do_mask:
            masker.set_prompt_mask(LTA.keep_mask_from_regions(regions, ablate, device=model.device))
        if gate_task is not None:
            RKG.set_gate_context(torch.full((ids_t.shape[0],), RKG.task_row(gate_task), dtype=torch.long, device=model.device), torch.zeros_like(ids_t, dtype=torch.bool))
        if recv_task is not None:
            _z = torch.zeros_like(ids_t, dtype=torch.bool)
            RCV.set_gate_context(torch.full((ids_t.shape[0],), RCV.task_row(recv_task), dtype=torch.long, device=model.device), _z, _z, _z)
        try:
            out = model(input_ids=ids_t, attention_mask=att_t, position_ids=pos_t, logits_to_keep=keep_k)
        finally:
            if gate_task is not None:
                RKG.clear_gate_context()
            if recv_task is not None:
                RCV.clear_gate_context()
        raw = out.logits.float()
        lp = raw.log_softmax(-1)
        for (b, r) in enumerate(chunk):
            ct = cand_ids[r[1]]
            n = len(ct)
            tot = 0.0
            for (j, t) in enumerate(ct):
                tot += float(lp[b, keep_k - n + j - 1, t])
            logps[r[0]][r[1]] = tot
            if r[1] == 0 and div_ok:
                k = keep_k - n + div - 1
                first[r[0]] = (float(raw[b, k, div_toks[0]]), float(raw[b, k, div_toks[1]]))
    if do_mask:
        masker.clear()
    recs = []
    n_correct = n_ent = 0
    n_first_correct = n_first_agree = n_first = 0
    (m_all, m_gold_ent, m_gold_ref, f_all) = ([], [], [], [])
    for (i, r) in enumerate(test):
        gold = json.loads(r['target']).get('label')
        lo = logps[i]
        margin = lo[0] - lo[1]
        pred = TABFACT_LABELS[0] if margin >= 0 else TABFACT_LABELS[1]
        n_correct += int(pred == gold)
        n_ent += int(pred == TABFACT_LABELS[0])
        m_all.append(margin)
        (m_gold_ent if gold == TABFACT_LABELS[0] else m_gold_ref).append(margin)
        rec = {'id': r['id'], 'gold': gold, 'logp_entailed': lo[0], 'logp_refuted': lo[1], 'margin': margin, 'argmax_label': pred, 'argmax_correct': int(pred == gold)}
        if first[i] is not None:
            (fe, fr) = first[i]
            fm = fe - fr
            fpred = TABFACT_LABELS[0] if fm >= 0 else TABFACT_LABELS[1]
            n_first += 1
            n_first_correct += int(fpred == gold)
            n_first_agree += int(fpred == pred)
            f_all.append(fm)
            rec.update({'first_div_logit_entailed': fe, 'first_div_logit_refuted': fr, 'first_div_margin': fm, 'first_div_label': fpred})
        recs.append(rec)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None
    summary = {'n': len(test), 'labels': list(TABFACT_LABELS), 'argmax_accuracy': n_correct / max(1, len(test)), 'argmax_entailed_rate': n_ent / max(1, len(test)), 'mean_margin': _mean(m_all), 'mean_abs_margin': _mean([abs(x) for x in m_all]), 'mean_margin_gold_entailed': _mean(m_gold_ent), 'mean_margin_gold_refuted': _mean(m_gold_ref), 'n_gold_entailed': len(m_gold_ent), 'n_gold_refuted': len(m_gold_ref)}
    summary['first_div'] = {'answer_token_index': div if div_ok else None, 'entailed_token': tok.decode([div_toks[0]]) if div_ok else None, 'refuted_token': tok.decode([div_toks[1]]) if div_ok else None, 'shared_prefix': tok.decode(cand_ids[0][:div]) if div_ok else None, 'n': n_first, 'mean_logit_margin': _mean(f_all), 'mean_abs_logit_margin': _mean([abs(x) for x in f_all]), 'argmax_accuracy': n_first_correct / max(1, n_first), 'agreement_with_full_margin': n_first_agree / max(1, n_first)}
    return (recs, summary)

def canonical_task(slug):
    s = str(slug).lower()
    for base in ('cta', 'cpa', 'cea', 'hvm', 'em', 'tabfact', 'wikiqa', 'finqa', 'tablebench'):
        if s == base or s.startswith(base + '_'):
            return base
    return s
_SINGLE_COL_TASKS = {'cta_mmtu', 'cpa_mmtu'}
_CTA_TARGET_RE = re.compile('Target entity column:\\s*(\\S+)')
_CPA_HEAD_RE = re.compile('Head column:\\s*(\\S+)')
_CPA_TAIL_RE = re.compile('Tail column:\\s*(\\S+)')
_MD_TABLE_RE = re.compile('^\\|.*(?:\\n\\|.*)*', re.M)

def _md_cells(row):
    s = row
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    return s.split('|')

def single_column_prompt(slug, text):
    if slug not in _SINGLE_COL_TASKS:
        return text
    if slug == 'cta_mmtu':
        m = _CTA_TARGET_RE.search(text)
        if not m:
            raise ValueError("cta_mmtu: no 'Target entity column:' marker in prompt")
        keep_names = [m.group(1)]
    else:
        (mh, mt) = (_CPA_HEAD_RE.search(text), _CPA_TAIL_RE.search(text))
        if not (mh and mt):
            raise ValueError("cpa_mmtu: no 'Head column:'/'Tail column:' markers in prompt")
        keep_names = [mh.group(1), mt.group(1)]
    tm = _MD_TABLE_RE.search(text)
    if not tm:
        raise ValueError(f'{slug}: no Markdown table found in prompt')
    rows = tm.group(0).split('\n')
    header = [c.strip() for c in _md_cells(rows[0])]
    idxs = []
    for nm in keep_names:
        if nm not in header:
            raise ValueError(f'{slug}: target column {nm!r} not in table header {header}')
        idxs.append(header.index(nm))
    idxs = sorted(set(idxs))

    def rebuild(row):
        cs = _md_cells(row)
        return '|' + '|'.join((cs[i] for i in idxs)) + '|'
    new_block = '\n'.join((rebuild(r) for r in rows))
    return text[:tm.start()] + new_block + text[tm.end():]
_MD_SEP_CELL_RE = re.compile('^:?-{1,}:?$')

def _is_md_separator(row):
    cells = [c.strip() for c in _md_cells(row)]
    return bool(cells) and all((_MD_SEP_CELL_RE.match(c) for c in cells))
_SHUFFLE_TASKS = {'cta', 'cpa', 'cta_mmtu', 'cpa_mmtu', 'cta_viznet', 'cta_mmtu_ctx', 'cpa_mmtu_ctx'}
_NO_MARKER_SHUFFLE_TASKS = {'tabfact', 'tabfact_mmtu'}
_COL_SHUFFLE_TASKS = _SHUFFLE_TASKS | _NO_MARKER_SHUFFLE_TASKS
_COL_NUM_RE = re.compile('col(\\d+)')

def _perm_moving_targets(n, targets, rng):
    for _ in range(200):
        perm = list(range(n))
        rng.shuffle(perm)
        inv = [0] * n
        for (p, o) in enumerate(perm):
            inv[o] = p
        if all((inv[t] != t for t in targets)):
            return (perm, inv)
    perm = [(p - 1) % n for p in range(n)]
    inv = [(o + 1) % n for o in range(n)]
    return (perm, inv)

def _relabel_header_cell(raw_cell, new_idx):
    new = _COL_NUM_RE.sub(f'col{new_idx}', raw_cell, count=1)
    if len(new) < len(raw_cell):
        new = new + ' ' * (len(raw_cell) - len(new))
    return new

def _sub_marker(text, pattern, new_name):
    return pattern.sub(lambda m: m.group(0).replace(m.group(1), new_name, 1), text, count=1)

def _perm_plain(n, rng):
    perm = list(range(n))
    rng.shuffle(perm)
    inv = [0] * n
    for (p, o) in enumerate(perm):
        inv[o] = p
    return (perm, inv)
_COL_SHUFFLE_MODES = ('full', 'after_target', 'before_target', 'target_last', 'target_first', 'swap_blocks', 'target_end', 'target_begin')

def _perm_segment(n, lo, hi, rng, max_tries=200):
    seg = list(range(lo, hi))
    if len(seg) < 2:
        return None
    s = seg[:]
    for _ in range(max_tries):
        rng.shuffle(s)
        if s != seg:
            break
    else:
        s = seg[1:] + seg[:1]
    perm = list(range(n))
    for (k, pos) in enumerate(range(lo, hi)):
        perm[pos] = s[k]
    inv = [0] * n
    for (pos, old) in enumerate(perm):
        inv[old] = pos
    return (perm, inv)

def _perm_target_first(n, t):
    if t == 0:
        return None
    perm = [t] + [o for o in range(n) if o != t]
    inv = [0] * n
    for (pos, old) in enumerate(perm):
        inv[old] = pos
    return (perm, inv)

def _perm_target_last(n, t):
    if t == n - 1:
        return None
    perm = [o for o in range(n) if o != t] + [t]
    inv = [0] * n
    for (pos, old) in enumerate(perm):
        inv[old] = pos
    return (perm, inv)

def _inv_of(perm):
    inv = [0] * len(perm)
    for (pos, old) in enumerate(perm):
        inv[old] = pos
    return inv

def _perm_swap_blocks(n, t):
    if n < 2:
        return None
    perm = list(range(t + 1, n)) + [t] + list(range(0, t))
    if perm == list(range(n)):
        return None
    return (perm, _inv_of(perm))

def _perm_swap_blocks_pair(n, h, t):
    rest = [c for c in range(n) if c != h]
    ti = rest.index(t)
    (P, S) = (rest[:ti], rest[ti + 1:])
    if not P and (not S):
        return None
    perm = [h] + S + [t] + P
    if perm == list(range(n)):
        return None
    return (perm, _inv_of(perm))

def _perm_target_end(n, t):
    perm = [o for o in range(n) if o != t] + [t]
    return (perm, _inv_of(perm))

def _perm_target_end_pair(n, h, t):
    perm = [o for o in range(n) if o not in (h, t)] + [h, t]
    return (perm, _inv_of(perm))

def _perm_target_begin(n, t):
    perm = [t] + [o for o in range(n) if o != t]
    return (perm, _inv_of(perm))

def _perm_target_begin_pair(n, h, t):
    perm = [h, t] + [o for o in range(n) if o not in (h, t)]
    return (perm, _inv_of(perm))
_TARGET_END_RNG = random.Random(0)

def target_end_prompt(slug, text):
    return shuffle_columns_prompt(slug, text, _TARGET_END_RNG, mode='target_end')

def target_begin_prompt(slug, text):
    return shuffle_columns_prompt(slug, text, _TARGET_END_RNG, mode='target_begin')

def shuffle_columns_prompt(slug, text, rng, enforce_move=True, mode='full'):
    if slug not in _COL_SHUFFLE_TASKS:
        return (text, {'skipped': True, 'reason': 'task_not_supported'})
    no_marker = slug in _NO_MARKER_SHUFFLE_TASKS
    if no_marker:
        tgt_names = []
    elif slug.startswith('cpa'):
        (mh, mt) = (_CPA_HEAD_RE.search(text), _CPA_TAIL_RE.search(text))
        if not (mh and mt):
            return (text, {'skipped': True, 'reason': 'no_markers'})
        tgt_names = [mh.group(1), mt.group(1)]
    else:
        m = _CTA_TARGET_RE.search(text)
        if not m:
            return (text, {'skipped': True, 'reason': 'no_marker'})
        tgt_names = [m.group(1)]
    tm = _MD_TABLE_RE.search(text)
    if not tm:
        return (text, {'skipped': True, 'reason': 'no_table'})
    rows = tm.group(0).split('\n')
    header = [c.strip() for c in _md_cells(rows[0])]
    n = len(header)
    if n < 2:
        return (text, {'skipped': True, 'reason': 'too_few_columns', 'n_cols': n})
    try:
        targets = [header.index(nm) for nm in tgt_names]
    except ValueError:
        return (text, {'skipped': True, 'reason': 'target_not_in_header'})
    if mode not in _COL_SHUFFLE_MODES:
        raise ValueError(f'unknown shuffle mode {mode!r}; expected one of {_COL_SHUFFLE_MODES}')
    if mode in ('swap_blocks', 'target_end', 'target_begin'):
        if no_marker:
            return (text, {'skipped': True, 'reason': 'mode_requires_markers', 'n_cols': n, 'mode': mode})
        if mode == 'swap_blocks':
            built = _perm_swap_blocks(n, targets[0]) if len(targets) == 1 else _perm_swap_blocks_pair(n, targets[0], targets[1])
            why = 'no_context_to_swap'
        elif mode == 'target_begin':
            built = _perm_target_begin(n, targets[0]) if len(targets) == 1 else _perm_target_begin_pair(n, targets[0], targets[1])
            why = 'already_canonical'
        else:
            built = _perm_target_end(n, targets[0]) if len(targets) == 1 else _perm_target_end_pair(n, targets[0], targets[1])
            why = 'already_canonical'
        if built is None:
            return (text, {'skipped': True, 'reason': why, 'n_cols': n, 'mode': mode})
        (perm, inv) = built
    elif mode != 'full':
        if no_marker or len(targets) != 1:
            return (text, {'skipped': True, 'reason': 'mode_requires_single_target', 'n_cols': n, 'mode': mode})
        t = targets[0]
        if mode == 'after_target':
            (built, why) = (_perm_segment(n, t + 1, n, rng), 'fewer_than_2_columns_after_target')
        elif mode == 'before_target':
            (built, why) = (_perm_segment(n, 0, t, rng), 'fewer_than_2_columns_before_target')
        elif mode == 'target_first':
            (built, why) = (_perm_target_first(n, t), 'target_already_first')
        else:
            (built, why) = (_perm_target_last(n, t), 'target_already_last')
        if built is None:
            return (text, {'skipped': True, 'reason': why, 'n_cols': n, 'mode': mode})
        (perm, inv) = built
    elif no_marker:
        (perm, inv) = _perm_moving_targets(n, list(range(n)), rng) if enforce_move else _perm_plain(n, rng)
    else:
        (perm, inv) = _perm_moving_targets(n, targets, rng) if enforce_move else _perm_plain(n, rng)
    new_rows = []
    for (ri, row) in enumerate(rows):
        cs = _md_cells(row)
        if len(cs) != n:
            return (text, {'skipped': True, 'reason': 'ragged_row'})
        if ri == 0:
            if no_marker:
                new_rows.append('|' + '|'.join((cs[perm[p]] for p in range(n))) + '|')
            else:
                new_rows.append('|' + '|'.join((_relabel_header_cell(cs[perm[p]], p) for p in range(n))) + '|')
        else:
            new_rows.append('|' + '|'.join((cs[perm[p]] for p in range(n))) + '|')
    new_text = text[:tm.start()] + '\n'.join(new_rows) + text[tm.end():]
    if no_marker:
        pass
    elif slug.startswith('cpa'):
        new_text = _sub_marker(new_text, _CPA_HEAD_RE, f'col{inv[targets[0]]}')
        new_text = _sub_marker(new_text, _CPA_TAIL_RE, f'col{inv[targets[1]]}')
    else:
        new_text = _sub_marker(new_text, _CTA_TARGET_RE, f'col{inv[targets[0]]}')
    return (new_text, {'skipped': False, 'n_cols': n, 'mode': mode, 'old_target': targets, 'new_target': [inv[t] for t in targets], 'perm': perm})

def drop_after_target_prompt(slug, text):
    if slug not in _COL_SHUFFLE_TASKS or slug in _NO_MARKER_SHUFFLE_TASKS:
        return (text, {'skipped': True, 'reason': 'task_not_supported'})
    if slug.startswith('cpa'):
        (mh, mt) = (_CPA_HEAD_RE.search(text), _CPA_TAIL_RE.search(text))
        if not (mh and mt):
            return (text, {'skipped': True, 'reason': 'no_markers'})
        tgt_names = [mh.group(1), mt.group(1)]
    else:
        m = _CTA_TARGET_RE.search(text)
        if not m:
            return (text, {'skipped': True, 'reason': 'no_marker'})
        tgt_names = [m.group(1)]
    tm = _MD_TABLE_RE.search(text)
    if not tm:
        return (text, {'skipped': True, 'reason': 'no_table'})
    rows = tm.group(0).split('\n')
    header = [c.strip() for c in _md_cells(rows[0])]
    n = len(header)
    if n < 2:
        return (text, {'skipped': True, 'reason': 'too_few_columns', 'n_cols': n})
    try:
        targets = [header.index(nm) for nm in tgt_names]
    except ValueError:
        return (text, {'skipped': True, 'reason': 'target_not_in_header'})
    keep = list(range(max(targets) + 1))
    if len(keep) == n:
        return (text, {'skipped': True, 'reason': 'nothing_after_target', 'n_cols': n, 'n_kept': n, 'n_dropped': 0})
    new_rows = []
    for (ri, row) in enumerate(rows):
        cs = _md_cells(row)
        if len(cs) != n:
            return (text, {'skipped': True, 'reason': 'ragged_row'})
        if ri == 0:
            new_rows.append('|' + '|'.join((_relabel_header_cell(cs[c], p) for (p, c) in enumerate(keep))) + '|')
        else:
            new_rows.append('|' + '|'.join((cs[c] for c in keep)) + '|')
    new_text = text[:tm.start()] + '\n'.join(new_rows) + text[tm.end():]
    new_pos = {c: p for (p, c) in enumerate(keep)}
    if slug.startswith('cpa'):
        new_text = _sub_marker(new_text, _CPA_HEAD_RE, f'col{new_pos[targets[0]]}')
        new_text = _sub_marker(new_text, _CPA_TAIL_RE, f'col{new_pos[targets[1]]}')
    else:
        new_text = _sub_marker(new_text, _CTA_TARGET_RE, f'col{new_pos[targets[0]]}')
    return (new_text, {'skipped': False, 'n_cols': n, 'n_kept': len(keep), 'n_dropped': n - len(keep), 'old_target': targets, 'new_target': [new_pos[t] for t in targets]})
_ROW_SHUFFLE_TASKS = set(_COL_SHUFFLE_TASKS)

def shuffle_rows_prompt(slug, text, rng):
    if slug not in _ROW_SHUFFLE_TASKS:
        return (text, {'skipped': True, 'reason': 'task_not_supported'})
    tm = _MD_TABLE_RE.search(text)
    if not tm:
        return (text, {'skipped': True, 'reason': 'no_table'})
    rows = tm.group(0).split('\n')
    if len(rows) < 2 or not _is_sep_cells([c.strip() for c in _md_cells(rows[1])]):
        return (text, {'skipped': True, 'reason': 'no_separator'})
    data = rows[2:]
    n = len(data)
    if n < 2:
        return (text, {'skipped': True, 'reason': 'too_few_rows', 'n_rows': n})
    cells = [_md_cells(r) for r in data]
    ncols = len(cells[0])
    if any((len(c) != ncols for c in cells)):
        return (text, {'skipped': True, 'reason': 'ragged_row', 'n_rows': n})
    cols = [[cells[r][j] for r in range(n)] for j in range(ncols)]

    def _reconstruct(perms):
        return [[cols[j][perms[j][r]] for j in range(ncols)] for r in range(n)]
    col_perms = None
    for _ in range(200):
        perms = []
        for _j in range(ncols):
            p = list(range(n))
            rng.shuffle(p)
            perms.append(p)
        if ncols >= 2 and all((p == perms[0] for p in perms)):
            continue
        if _reconstruct(perms) != cells:
            col_perms = perms
            break
    if col_perms is None:
        return (text, {'skipped': True, 'reason': 'no_effect', 'n_rows': n, 'n_cols': ncols})
    new_cells = _reconstruct(col_perms)
    new_data = ['|' + '|'.join(row) + '|' for row in new_cells]
    new_rows = rows[:2] + new_data
    new_text = text[:tm.start()] + '\n'.join(new_rows) + text[tm.end():]
    return (new_text, {'skipped': False, 'n_rows': n, 'n_cols': ncols, 'col_perms': col_perms})
_TARGET_TASKS = {'cta', 'cpa', 'cta_viznet', 'cta_mmtu', 'cpa_mmtu', 'cta_mmtu_ctx', 'cpa_mmtu_ctx'}
_SEP_CELL_RE = re.compile('^:?-+:?$')
_NULL_TOKENS = {'undefined', 'null', 'n/a', 'none'}

def _is_sep_cells(cells):
    return bool(cells) and all((_SEP_CELL_RE.match(c.strip()) for c in cells))

def _cell_is_blank(v):
    s = str(v).strip()
    return not s or s.lower() in _NULL_TOKENS

def _named_col_empty(text, colname):
    tm = _MD_TABLE_RE.search(text)
    if not tm:
        return None
    rows = tm.group(0).split('\n')
    header = [c.strip() for c in _md_cells(rows[0])]
    if colname not in header:
        return None
    ci = header.index(colname)
    for r in rows[1:]:
        cs = _md_cells(r)
        if _is_sep_cells([c.strip() for c in cs]):
            continue
        if ci < len(cs) and (not _cell_is_blank(cs[ci])):
            return False
    return True

def target_is_empty(slug, text):
    if slug not in _TARGET_TASKS:
        return False
    if slug in ('cpa', 'cpa_mmtu', 'cpa_mmtu_ctx'):
        (mh, mt) = (_CPA_HEAD_RE.search(text), _CPA_TAIL_RE.search(text))
        if not (mh and mt):
            return False
        return _named_col_empty(text, mh.group(1)) is True or _named_col_empty(text, mt.group(1)) is True
    m = _CTA_TARGET_RE.search(text)
    return bool(m) and _named_col_empty(text, m.group(1)) is True

def with_task_prefix(content, slug, enabled):
    if not enabled:
        return content
    label = canonical_task(slug)
    if not label:
        return content
    return f'[TASK: {label}]\n{content}'

def _read_train_task_prefix(adapter_dir):
    if not adapter_dir:
        return None
    for fn in ('prompt_format.json', 'train_manifest.json'):
        p = os.path.join(adapter_dir, fn)
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if 'task_prefix' in d:
            return bool(d['task_prefix'])
        if isinstance(d.get('args'), dict) and 'task_prefix' in d['args']:
            return bool(d['args']['task_prefix'])
    return None

def _read_train_flag(adapter_dir, key):
    if not adapter_dir:
        return None
    for fn in ('prompt_format.json', 'train_manifest.json'):
        p = os.path.join(adapter_dir, fn)
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if key in d:
            return bool(d[key])
        if isinstance(d.get('args'), dict) and key in d['args']:
            return bool(d['args'][key])
    return None

def _read_train_value(adapter_dir, key):
    if not adapter_dir:
        return None
    for fn in ('prompt_format.json', 'train_manifest.json'):
        p = os.path.join(adapter_dir, fn)
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if key in d:
            return d[key]
        if isinstance(d.get('args'), dict) and key in d['args']:
            return d['args'][key]
    return None

def parse_col_major_tasks(spec):
    if not spec:
        return set()
    parts = spec.split(',') if isinstance(spec, str) else list(spec)
    return {canonical_task(str(t).strip()) for t in parts if str(t).strip()}

def parse_task_scope(spec):
    if not spec:
        return set()
    parts = spec.split(',') if isinstance(spec, str) else list(spec)
    return {canonical_task(str(t).strip()) for t in parts if str(t).strip()}

def task_in_scope(slug, scope):
    return not scope or canonical_task(slug) in scope

def resolve_layout_tasks(adapter, adapter2, arg, key, label):
    if arg is None:
        detected = None
        for ad in (adapter, adapter2):
            v = _read_train_value(ad, key)
            if v is not None:
                detected = v
        scope = parse_task_scope(detected)
        where = 'auto-detected from adapter manifest'
    else:
        scope = parse_task_scope(arg)
        where = 'explicitly set'
    print(f'[{label}] task scope {where}: ' + (f'{sorted(scope)}' if scope else 'every supported task'))
    return scope

def resolve_col_major(adapter, adapter2, arg):
    if arg is None:
        detected = None
        for ad in (adapter, adapter2):
            v = _read_train_value(ad, 'col_major_tasks')
            if v is not None:
                detected = v
        col_major = parse_col_major_tasks(detected)
        print(f'[col_major] auto-detected from adapter manifest: {sorted(col_major)}' if detected is not None else '[col_major] no adapter marker found; defaulting to row-major for every task')
    else:
        col_major = parse_col_major_tasks(arg)
        print(f'[col_major] explicitly set: {sorted(col_major)}')
    return col_major

def resolve_target_sandwich(adapter, adapter2, arg):
    if arg is None:
        detected = None
        for ad in (adapter, adapter2):
            v = TSW.read_marker(ad)
            if v is not None:
                detected = v
                break
        on = bool(detected)
        print(f'[{TSW.MECHANISM}] auto-detected from the adapter: {on}' if detected is not None else f'[{TSW.MECHANISM}] no adapter marker found; defaulting to OFF')
    else:
        on = bool(arg)
        print(f'[{TSW.MECHANISM}] explicitly set: {on}')
    return on

def resolve_sandwich_mask(adapter, adapter2, arg):
    if arg is None:
        detected = None
        for ad in (adapter, adapter2):
            v = TSW.read_mask_marker(ad)
            if v is not None:
                detected = v
                break
        on = True if detected is None else bool(detected)
        print(f'[{TSW.MECHANISM}] mask auto-detected from the adapter: {on}' if detected is not None else f'[{TSW.MECHANISM}] adapter records no mask setting; defaulting to ON')
    else:
        on = bool(arg)
        print(f'[{TSW.MECHANISM}] mask explicitly set: {on}')
    return on

def resolve_target_end(adapter, adapter2, arg):
    if arg is None:
        detected = None
        for ad in (adapter, adapter2):
            v = _read_train_flag(ad, 'target_end')
            if v is not None:
                detected = v if detected is None else detected or v
        target_end = bool(detected)
        print(f'[target_end] auto-detected from adapter manifest: {target_end}' if detected is not None else '[target_end] no adapter marker found; defaulting to OFF')
    else:
        target_end = bool(arg)
        print(f'[target_end] explicitly set: {target_end}')
    return target_end

def resolve_target_first(adapter, adapter2, arg):
    if arg is None:
        detected = None
        for ad in (adapter, adapter2):
            v = _read_train_flag(ad, 'target_first')
            if v is not None:
                detected = v if detected is None else detected or v
        target_first = bool(detected)
        print(f'[target_first] auto-detected from adapter manifest: {target_first}' if detected is not None else '[target_first] no adapter marker found; defaulting to OFF')
    else:
        target_first = bool(arg)
        print(f'[target_first] explicitly set: {target_first}')
    return target_first

def resolve_ctx_relay(adapter, adapter2, arg):
    if arg is None:
        detected = None
        for ad in (adapter, adapter2):
            v = CRM.read_marker(ad)
            if v is not None:
                detected = v if detected is None else detected or v
        on = bool(detected)
        print(f'[{CRM.MECHANISM}] auto-detected from the adapter: {on}' if detected is not None else f'[{CRM.MECHANISM}] no adapter marker found; defaulting to OFF')
    else:
        on = bool(arg)
        print(f'[{CRM.MECHANISM}] explicitly set: {on}')
    return on

def resolve_table_first(adapter, adapter2, arg):
    if arg is None:
        detected = None
        for ad in (adapter, adapter2):
            v = _read_train_flag(ad, 'table_first')
            if v is not None:
                detected = v
        table_first = bool(detected)
        print(f'[table_first] auto-detected from adapter manifest: {table_first}' if detected is not None else '[table_first] no adapter marker found; defaulting to OFF')
    else:
        table_first = bool(arg)
        print(f'[table_first] explicitly set: {table_first}')
    return table_first

def resolve_merge_adapter(adapter, adapter2, arg):
    if arg is not None:
        path = arg.strip()
        print(f"[merge_adapter] explicitly set: {path or '(none -- plain base)'}")
    else:
        path = None
        for ad in (adapter, adapter2):
            v = _read_train_value(ad, 'merge_adapter')
            if v:
                path = str(v)
        print(f'[merge_adapter] auto-detected from adapter manifest: {path}' if path else '[merge_adapter] no adapter marker found; serving the plain base')
    if not path:
        return None
    for cand in (path, os.path.join(ROOT, path)):
        if os.path.isdir(cand):
            if cand != path:
                print(f'[merge_adapter] resolved against ROOT -> {cand}')
            return cand
    raise SystemExit(f"[merge_adapter] {path} does not exist (tried it as given and under ROOT={ROOT}). This adapter was TRAINED over a base with that adapter merged in, so scoring it against the plain base would silently evaluate a different model. Provide the path explicitly, or pass --merge_adapter '' if you really mean the plain base.")

def resolve_table_last(adapter, adapter2, arg):
    if arg is None:
        detected = None
        for ad in (adapter, adapter2):
            v = _read_train_flag(ad, 'table_last')
            if v is not None:
                detected = v
        table_last = bool(detected)
        print(f'[table_last] auto-detected from adapter manifest: {table_last}' if detected is not None else '[table_last] no adapter marker found; defaulting to OFF')
    else:
        table_last = bool(arg)
        print(f'[table_last] explicitly set: {table_last}')
    return table_last

def resolve_manifest_flags(adapter, adapter2, task_prefix_arg, drop_empty_arg):
    if task_prefix_arg is None:
        detected = None
        for ad in (adapter, adapter2):
            v = _read_train_task_prefix(ad)
            if v is not None:
                detected = v
        task_prefix = bool(detected)
        print(f'[task_prefix] auto-detected from adapter manifest: {task_prefix}' if detected is not None else '[task_prefix] no adapter marker found; defaulting to OFF')
    else:
        task_prefix = task_prefix_arg
        print(f'[task_prefix] explicitly set: {task_prefix}')
    if drop_empty_arg is None:
        detected = None
        for ad in (adapter, adapter2):
            v = _read_train_flag(ad, 'drop_empty_target')
            if v is not None:
                detected = v
        drop_empty = bool(detected)
        print(f'[drop_empty_target] auto-detected from adapter manifest: {drop_empty}' if detected is not None else '[drop_empty_target] no adapter marker found; defaulting to OFF')
    else:
        drop_empty = drop_empty_arg
        print(f'[drop_empty_target] explicitly set: {drop_empty}')
    return (task_prefix, drop_empty)

def warn_flag_scope(tasks, single_column, shuffle_columns, shuffle_rows=False):
    if single_column:
        skipped = [t for t in tasks if t not in _SINGLE_COL_TASKS]
        if skipped:
            print(f'[single_column] applies only to {sorted(_SINGLE_COL_TASKS)}; ignored for {skipped}')
    if shuffle_columns:
        skipped = [t for t in tasks if t not in _COL_SHUFFLE_TASKS]
        if skipped:
            print(f'[shuffle_columns] applies only to {sorted(_COL_SHUFFLE_TASKS)}; ignored for {skipped}')
    if shuffle_rows:
        skipped = [t for t in tasks if t not in _ROW_SHUFFLE_TASKS]
        if skipped:
            print(f'[shuffle_rows] applies only to {sorted(_ROW_SHUFFLE_TASKS)}; ignored for {skipped}')

def load_and_prepare(slug, *, limit=None, drop_empty=False, single_column=False, shuffle_columns=False, shuffle_rows=False, shuffle_seed=0, shuffle_mode='full', task_prefix=False, col_major_tasks=(), table_first=False, table_last=False, target_end=False, target_end_tasks=(), target_sandwich=False, table_first_tasks=(), target_first=False):
    if slug == 'cea_mmtu_bin':
        test = load_cea_mmtu_binary(DATA)
    else:
        test = read_jsonl(f'{DATA}/{slug}/test.jsonl')
    if drop_empty:
        kept = [r for r in test if not target_is_empty(slug, r['input'])]
        n_dropped = len(test) - len(kept)
        if n_dropped:
            print(f'[drop_empty_target] {slug}: dropped {n_dropped}/{len(test)} examples with an empty target column')
        test = kept
    if limit:
        test = test[:limit]
    do_single = single_column and slug in _SINGLE_COL_TASKS
    do_shuffle = shuffle_columns and slug in _COL_SHUFFLE_TASKS
    shuf_metas = [None] * len(test)
    n_shuffled = 0
    shuffle_skipped = {}
    if do_single:
        inputs = [single_column_prompt(slug, r['input']) for r in test]
        print(f'[single_column] {slug}: reduced tables to target column(s) for {len(inputs)} prompts')
    elif do_shuffle:
        inputs = []
        reasons = collections.Counter()
        for (i, r) in enumerate(test):
            rng = random.Random((shuffle_seed, i))
            (new_text, meta) = shuffle_columns_prompt(slug, r['input'], rng, mode=shuffle_mode)
            inputs.append(new_text)
            shuf_metas[i] = meta
            if meta.get('skipped'):
                reasons[meta.get('reason')] += 1
            else:
                n_shuffled += 1
        shuffle_skipped = dict(reasons)
        print(f'[shuffle_columns mode={shuffle_mode}] {slug}: shuffled {n_shuffled}/{len(test)} prompts' + (f'; skipped {shuffle_skipped}' if shuffle_skipped else ''))
    else:
        inputs = [r['input'] for r in test]
    do_rowshuf = shuffle_rows and slug in _ROW_SHUFFLE_TASKS
    row_metas = [None] * len(test)
    n_row_shuffled = 0
    row_shuffle_skipped = {}
    if do_rowshuf:
        new_inputs = []
        reasons = collections.Counter()
        for (i, x) in enumerate(inputs):
            rng = random.Random((shuffle_seed, i, 'rows'))
            (new_text, meta) = shuffle_rows_prompt(slug, x, rng)
            new_inputs.append(new_text)
            row_metas[i] = meta
            if meta.get('skipped'):
                reasons[meta.get('reason')] += 1
            else:
                n_row_shuffled += 1
        inputs = new_inputs
        row_shuffle_skipped = dict(reasons)
        print(f'[shuffle_rows] {slug}: broke row integrity (per-column cell shuffle) for {n_row_shuffled}/{len(test)} prompts (header/columns/markers unchanged)' + (f'; skipped {row_shuffle_skipped}' if row_shuffle_skipped else ''))
    te_moved = te_already = 0
    te_skipped = {}
    if target_end and slug in _COL_SHUFFLE_TASKS and task_in_scope(slug, target_end_tasks):
        reasons = collections.Counter()
        new_inputs = []
        for x in inputs:
            (nx, meta) = target_end_prompt(slug, x)
            new_inputs.append(nx)
            if meta.get('skipped'):
                reasons[meta.get('reason')] += 1
            elif nx == x:
                te_already += 1
            else:
                te_moved += 1
        inputs = new_inputs
        te_skipped = dict(reasons)
        print(f'[target_end] {slug}: canonicalized {te_moved}/{len(inputs)} prompts (already_canonical={te_already})' + (f'; skipped {te_skipped}' if te_skipped else ''))
    elif target_end and slug in _COL_SHUFFLE_TASKS:
        print(f"[target_end] {slug}: outside the flag's task scope {sorted(target_end_tasks)} -- prompts left unchanged")
    elif target_end:
        print(f'[target_end] {slug}: not a CTA/CPA task -- prompts left unchanged')
    tb_moved = tb_already = 0
    tb_skipped = {}
    if target_first and slug in _COL_SHUFFLE_TASKS:
        reasons = collections.Counter()
        new_inputs = []
        for x in inputs:
            (nx, meta) = target_begin_prompt(slug, x)
            new_inputs.append(nx)
            if meta.get('skipped'):
                reasons[meta.get('reason')] += 1
            elif nx == x:
                tb_already += 1
            else:
                tb_moved += 1
        inputs = new_inputs
        tb_skipped = dict(reasons)
        print(f'[target_first] {slug}: canonicalized {tb_moved}/{len(inputs)} prompts (already_canonical={tb_already})' + (f'; skipped {tb_skipped}' if tb_skipped else ''))
    elif target_first:
        print(f'[target_first] {slug}: not a CTA/CPA task -- prompts left unchanged')
    cm_tasks = col_major_tasks if isinstance(col_major_tasks, set) else set(col_major_tasks or ())
    do_colmajor = canonical_task(slug) in cm_tasks
    n_colmajor = colmajor_skipped = 0
    if do_colmajor:
        new_inputs = []
        for x in inputs:
            (nt, n_done, _) = TTR.transpose_tables(x)
            if n_done:
                new_inputs.append(nt)
                n_colmajor += 1
            else:
                new_inputs.append(x)
                colmajor_skipped += 1
        inputs = new_inputs
        print(f'[col_major] {slug}: re-serialized {n_colmajor}/{len(inputs)} prompts column-major' + (f'; left row-major {colmajor_skipped}' if colmajor_skipped else ''))
    tf_moved = tf_skipped = 0
    if table_first and (not task_in_scope(slug, table_first_tasks)):
        print(f"[table_first] {slug}: outside the flag's task scope {sorted(table_first_tasks)} -- prompts left unchanged")
    elif table_first:
        new_inputs = []
        for x in inputs:
            (nx, status) = TRO.move_tables_first(x)
            new_inputs.append(nx)
            if status == 'ok':
                tf_moved += 1
            else:
                tf_skipped += 1
        inputs = new_inputs
        print(f'[table_first] {slug}: moved the table to the top of {tf_moved}/{len(inputs)} prompts' + (f'; left unchanged {tf_skipped}' if tf_skipped else ''))
    tl_moved = tl_skipped = 0
    if table_last:
        new_inputs = []
        for x in inputs:
            (nx, status) = TRO.move_tables_last(x)
            new_inputs.append(nx)
            if status == 'ok':
                tl_moved += 1
            else:
                tl_skipped += 1
        inputs = new_inputs
        print(f'[table_last] {slug}: moved the table to the end of {tl_moved}/{len(inputs)} prompts' + (f'; left unchanged {tl_skipped}' if tl_skipped else ''))
    sw_stats = {'target_sandwich': bool(target_sandwich), 'sandwiched': 0, 'sandwich_skipped': {}}
    if target_sandwich:
        (reasons, new_inputs, n_sw) = (collections.Counter(), [], 0)
        for x in inputs:
            (nx, meta) = TSW.sandwich_prompt(slug, x)
            new_inputs.append(nx)
            if meta.get('skipped'):
                reasons[meta.get('reason')] += 1
            else:
                n_sw += 1
        inputs = new_inputs
        sw_stats = {'target_sandwich': True, 'sandwiched': n_sw, 'sandwich_skipped': dict(reasons)}
        print(f'[{TSW.MECHANISM}] {slug}: sandwiched the target column in {n_sw}/{len(inputs)} prompts' + (f"; left unchanged {sw_stats['sandwich_skipped']}" if sw_stats['sandwich_skipped'] else ''))
    prompts = [with_task_prefix(x, slug, task_prefix) for x in inputs]
    prep = {'do_single': do_single, 'do_shuffle': do_shuffle, **sw_stats, 'target_end': bool(target_end), 'target_end_tasks': sorted(target_end_tasks), 'table_first_tasks': sorted(table_first_tasks), 'target_end_moved': te_moved, 'target_end_already': te_already, 'target_end_skipped': te_skipped, 'target_first': bool(target_first), 'target_first_moved': tb_moved, 'target_first_already': tb_already, 'target_first_skipped': tb_skipped, 'shuffle_mode': shuffle_mode, 'n_shuffled': n_shuffled, 'shuffle_skipped': shuffle_skipped, 'shuf_metas': shuf_metas, 'do_rowshuf': do_rowshuf, 'n_row_shuffled': n_row_shuffled, 'row_shuffle_skipped': row_shuffle_skipped, 'row_metas': row_metas, 'do_colmajor': do_colmajor, 'n_colmajor': n_colmajor, 'colmajor_skipped': colmajor_skipped}
    return (test, prompts, prep)

def score_and_dump(slug, test, gens, prep, out_dir, base):
    scorer = SCORERS[slug]
    is_clf = slug in CLASSIF
    do_single = prep['do_single']
    do_shuffle = prep['do_shuffle']
    shuf_metas = prep['shuf_metas']
    do_rowshuf = prep.get('do_rowshuf', False)
    row_metas = prep.get('row_metas', [None] * len(test))
    agg = {'s_c': 0, 's_t': 0, 's_ex': 0, 'n_c': 0, 'n_t': 0, 'n_ex': 0}
    parse_fail = 0
    (per, invalid) = ([], [])
    (sg, sp, ng, np_) = ([], [], [], [])
    for (idx, (r, g)) in enumerate(zip(test, gens)):
        clean_g = strip_thinking(g)
        pj = extract_json(clean_g)
        if pj is None:
            parse_fail += 1
            invalid.append({'id': r['id'], 'raw': g, 'cleaned_raw': clean_g})
        gj = json.loads(r['target'])
        (sc, st, sex) = scorer(pj, gj, r.get('accept'), n=strict)
        (nc, nt, nex) = scorer(pj, gj, r.get('accept'), n=loose)
        agg['s_c'] += sc
        agg['s_t'] += st
        agg['s_ex'] += sex
        agg['n_c'] += nc
        agg['n_t'] += nt
        agg['n_ex'] += nex
        if is_clf:
            gv = gold_repr(slug, gj)
            pv = pred_repr(slug, pj)
            sg.append(strict(gv))
            sp.append(strict(pv) if pv is not None else INVALID)
            ng.append(loose(gv))
            np_.append(loose(pv) if pv is not None else INVALID)
        rec = {'id': r['id'], 'dataset': r.get('dataset'), 'gold': gj, 'pred': pred_repr(slug, pj), 'parsed_json': pj, 'raw': g, 'cleaned_raw': clean_g, 'strict_correct': sc, 'strict_total': st, 'strict_exact': sex, 'norm_correct': nc, 'norm_exact': nex, 'parse_ok': int(pj is not None)}
        if do_shuffle and shuf_metas[idx] is not None:
            sm = shuf_metas[idx]
            rec['shuffle'] = {'applied': not sm.get('skipped', False), 'reason': sm.get('reason'), 'old_target': sm.get('old_target'), 'new_target': sm.get('new_target'), 'perm': sm.get('perm')}
        if do_rowshuf and row_metas[idx] is not None:
            rm = row_metas[idx]
            rec['shuffle_rows'] = {'applied': not rm.get('skipped', False), 'reason': rm.get('reason'), 'n_rows': rm.get('n_rows'), 'n_cols': rm.get('n_cols'), 'col_perms': rm.get('col_perms')}
        per.append(rec)
    metrics = {'n': len(test), 'single_column': do_single, 'shuffle_columns': do_shuffle, 'shuffle_mode': prep.get('shuffle_mode') if do_shuffle else None, 'n_shuffled': prep['n_shuffled'] if do_shuffle else None, 'shuffle_skipped': prep['shuffle_skipped'] if do_shuffle else None, 'shuffle_rows': do_rowshuf, 'n_row_shuffled': prep.get('n_row_shuffled') if do_rowshuf else None, 'row_shuffle_skipped': prep.get('row_shuffle_skipped') if do_rowshuf else None, 'micro_acc': agg['s_c'] / max(1, agg['s_t']), 'exact_match': agg['s_ex'] / max(1, len(test)), 'norm_micro_acc': agg['n_c'] / max(1, agg['n_t']), 'norm_exact_match': agg['n_ex'] / max(1, len(test)), 'parse_fail': parse_fail}
    if is_clf:
        (macro_f1, weighted_f1) = macro_weighted_f1(sg, sp)
        (n_macro_f1, n_weighted_f1) = macro_weighted_f1(ng, np_)
        metrics.update({'macro_f1': macro_f1, 'weighted_f1': weighted_f1, 'norm_macro_f1': n_macro_f1, 'norm_weighted_f1': n_weighted_f1, 'n_gold_classes': len(set(sg)), 'f1_note': 'strict perfect-label F1, averaged over gold-present classes' + ('; accuracy is okay-tolerant' if slug == 'cta_mmtu' else '')})
    if slug == 'em_mmtu':
        m_prf = binary_prf(sg, sp, strict('match'))
        nm_prf = binary_prf(ng, np_, loose('match'))
        metrics.update({'match_precision': m_prf['precision'], 'match_recall': m_prf['recall'], 'match_f1': m_prf['f1'], 'match_support': m_prf['support'], 'norm_match_f1': nm_prf['f1'], 'positive_class': 'match'})
        by_ds = {}
        for (r, gv, pv) in zip(test, sg, sp):
            d = r.get('dataset', '?')
            by_ds.setdefault(d, {'g': [], 'p': []})
            by_ds[d]['g'].append(gv)
            by_ds[d]['p'].append(pv)
        per_ds = {}
        for (d, gp) in sorted(by_ds.items()):
            prf = binary_prf(gp['g'], gp['p'], strict('match'))
            acc = sum((1 for (a, b) in zip(gp['g'], gp['p']) if a == b)) / max(1, len(gp['g']))
            per_ds[d] = {'n': len(gp['g']), 'acc': acc, 'match_f1': prf['f1'], 'match_precision': prf['precision'], 'match_recall': prf['recall'], 'match_support': prf['support']}
        metrics['per_dataset'] = per_ds
    with open(f'{out_dir}/{base}_{slug}_predictions.jsonl', 'w') as f:
        for p in per:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    if invalid:
        with open(f'{out_dir}/{base}_{slug}_invalid.jsonl', 'w') as f:
            for p in invalid:
                f.write(json.dumps(p, ensure_ascii=False) + '\n')
    f1s = f" | macro_f1={metrics['macro_f1']:.3f} weighted_f1={metrics['weighted_f1']:.3f}" if is_clf else ''
    if slug == 'em_mmtu':
        f1s += f" | match_f1={metrics['match_f1']:.3f} P={metrics['match_precision']:.3f} R={metrics['match_recall']:.3f}"
    print(f"{slug}: micro_acc={metrics['micro_acc']:.3f} exact={metrics['exact_match']:.3f} | norm_micro={metrics['norm_micro_acc']:.3f} norm_exact={metrics['norm_exact_match']:.3f}{f1s} | parse_fail={parse_fail}/{len(test)}")
    if slug == 'em_mmtu':
        for (d, s) in metrics['per_dataset'].items():
            print(f"    [{d}] n={s['n']} acc={s['acc']:.3f} match_f1={s['match_f1']:.3f}")
    return metrics

def is_full_model_dir(path):
    if not path:
        return False
    return os.path.exists(os.path.join(path, 'config.json')) and (not os.path.exists(os.path.join(path, 'adapter_config.json')))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=f'{ROOT}/Qwen3-4B-Instruct-2507')
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--adapter2', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tasks', nargs='+', default=['cta', 'cpa', 'hvm'], choices=['cta', 'cpa', 'hvm', 'cta_mmtu', 'cpa_mmtu', 'cta_mmtu_ctx', 'cpa_mmtu_ctx', 'cea_mmtu', 'cta_viznet', 'em_mmtu', 'tabfact_mmtu', 'cea_mmtu_bin', 'wikiqa_mmtu'])
    ap.add_argument('--attn_impl', default='sdpa', choices=['sdpa', 'flash_attention_2', 'eager'])
    ap.add_argument('--max_new_tokens', type=int, default=400)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--task_prefix', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--drop_empty_target', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--target_col_mask', action='store_true')
    ap.add_argument('--target_first', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--target_sandwich', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--sandwich_mask', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--target_end', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--target_end_tasks', type=str, default=None)
    ap.add_argument('--merge_adapter', type=str, default=None)
    ap.add_argument('--table_first', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--table_first_tasks', type=str, default=None)
    ap.add_argument('--table_last', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--role_gate_mlp', dest='role_gate_mlp', action='store_true', default=None)
    ap.add_argument('--no_role_gate_mlp', dest='role_gate_mlp', action='store_false')
    ap.add_argument('--ffn_role_scale', dest='ffn_role_scale', action='store_true', default=None)
    ap.add_argument('--no_ffn_role_scale', dest='ffn_role_scale', action='store_false')
    ap.add_argument('--soft_role_prompt', dest='soft_role_prompt', action='store_true', default=None)
    ap.add_argument('--no_soft_role_prompt', dest='soft_role_prompt', action='store_false')
    ap.add_argument('--struct_role_bias', dest='struct_role_bias', action='store_true', default=None)
    ap.add_argument('--no_struct_role_bias', dest='struct_role_bias', action='store_false')
    ap.add_argument('--ctx_relay_mask', dest='ctx_relay_mask', action='store_true', default=None)
    ap.add_argument('--no_ctx_relay_mask', dest='ctx_relay_mask', action='store_false')
    ap.add_argument('--lora_role_matrix', dest='lora_role_matrix', action='store_true', default=None)
    ap.add_argument('--no_lora_role_matrix', dest='lora_role_matrix', action='store_false')
    ap.add_argument('--role_matrix_mode', choices=list(RM.EVAL_MODES), default=RM.DEFAULT_MODE)
    ap.add_argument('--lora_rank_gate', dest='lora_rank_gate', action='store_true', default=None)
    ap.add_argument('--lora_recv_gate', dest='lora_recv_gate', action='store_true', default=None)
    ap.add_argument('--no_lora_recv_gate', dest='lora_recv_gate', action='store_false')
    ap.add_argument('--split_gate_proj', dest='split_gate_proj', action='store_true', default=None)
    ap.add_argument('--swiglu_thres', dest='swiglu_thres', action='store_true', default=None)
    ap.add_argument('--no_swiglu_thres', dest='swiglu_thres', action='store_false', default=None)
    ap.add_argument('--swiglu_thres_off', action='store_true')
    ap.add_argument('--split_swap_branches', action='store_true')
    ap.add_argument('--no_split_gate_proj', dest='split_gate_proj', action='store_false', default=None)
    ap.add_argument('--recv_gate_neutral', action='store_true')
    ap.add_argument('--no_lora_rank_gate', dest='lora_rank_gate', action='store_false')
    ap.add_argument('--lora_ablate', choices=list(LTA.ABLATE_CHOICES), default='none')
    ap.add_argument('--lora_scale', type=float, default=1.0)
    ap.add_argument('--decision_margin', action='store_true')
    ap.add_argument('--col_major_tasks', type=str, default=None)
    ap.add_argument('--single_column', action='store_true')
    ap.add_argument('--shuffle_columns', action='store_true')
    ap.add_argument('--shuffle_rows', action='store_true')
    ap.add_argument('--shuffle_mode', default='full', choices=list(_COL_SHUFFLE_MODES))
    ap.add_argument('--shuffle_seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    if args.shuffle_mode != 'full' and (not args.shuffle_columns):
        ap.error('--shuffle_mode only has an effect with --shuffle_columns')
    if args.shuffle_columns and args.single_column:
        ap.error('--shuffle_columns and --single_column cannot be combined (single_column leaves only the target column, leaving nothing to shuffle)')
    (args.task_prefix, args.drop_empty_target) = resolve_manifest_flags(args.adapter, args.adapter2, args.task_prefix, args.drop_empty_target)
    col_major_tasks = resolve_col_major(args.adapter, args.adapter2, args.col_major_tasks)
    table_first = resolve_table_first(args.adapter, args.adapter2, args.table_first)
    table_first_tasks = resolve_layout_tasks(args.adapter, args.adapter2, args.table_first_tasks, 'table_first_tasks', 'table_first')
    table_last = resolve_table_last(args.adapter, args.adapter2, args.table_last)
    if table_first and table_last:
        raise SystemExit('--table_first and --table_last cannot be combined: they name opposite positions for the same block. Pass --no-table_first or --no-table_last to override an adapter manifest that somehow carries both.')
    target_end = resolve_target_end(args.adapter, args.adapter2, args.target_end)
    target_end_tasks = resolve_layout_tasks(args.adapter, args.adapter2, args.target_end_tasks, 'target_end_tasks', 'target_end')
    target_first = resolve_target_first(args.adapter, args.adapter2, args.target_first)
    if target_end and target_first:
        raise SystemExit('--target_end and --target_first cannot both be on: each pins the same column(s) to a different end of the table. No adapter can have been trained with both (train_lora refuses the pair), so this is a mixed-up invocation or a mixed-up adapter dir.')
    target_sandwich = resolve_target_sandwich(args.adapter, args.adapter2, args.target_sandwich)
    sandwich_mask = resolve_sandwich_mask(args.adapter, args.adapter2, args.sandwich_mask) if target_sandwich else False
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    full_ft = is_full_model_dir(args.adapter)
    if full_ft:
        _bad = [n for (n, v) in (('--adapter2', args.adapter2), ('--merge_adapter', args.merge_adapter)) if v]
        if _bad:
            raise SystemExit(f"{args.adapter} is a FULL fine-tuned model, not an adapter, so {', '.join(_bad)} has nothing to compose with it.")
        print(f'[full_finetune] {args.adapter} holds a full model (config.json, no adapter_config.json); loading it as the model. --model {args.model} is used for the tokenizer only, and no adapter is layered on top.')
    model = ML.load_causal_lm(args.adapter if full_ft else args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn_impl, device_map={'': 0})
    merge_adapter_path = resolve_merge_adapter(args.adapter, args.adapter2, args.merge_adapter)
    if merge_adapter_path:
        _n_before = sum((1 for _ in model.parameters()))
        model = PeftModel.from_pretrained(model, merge_adapter_path).merge_and_unload()
        print(f'[merge_adapter] merged {merge_adapter_path} into the base weights ({_n_before} base tensors); the adapter under test is loaded on top of it.')
    if args.adapter and args.adapter2:
        model = PeftModel.from_pretrained(model, args.adapter, adapter_name='stage1')
        model.load_adapter(args.adapter2, adapter_name='stage2')
        model.base_model.set_adapter(['stage1', 'stage2'])
    elif args.adapter and (not full_ft):
        model = PeftModel.from_pretrained(model, args.adapter)
    soft_role = args.soft_role_prompt
    if soft_role is None:
        soft_role = SP.is_soft_role_adapter(args.adapter) or SP.is_soft_role_adapter(args.adapter2)
        print(f'[{SP.MECHANISM}] auto-detected from the adapter directory: {soft_role}')
    else:
        print(f'[{SP.MECHANISM}] explicitly set: {soft_role}')
    soft_role_info = None
    if soft_role:
        bad = [n for (n, v) in (('--lora_ablate', args.lora_ablate != 'none'), ('--decision_margin', bool(args.decision_margin))) if v]
        if bad:
            raise SystemExit(f"[{SP.MECHANISM}] cannot be combined with {', '.join(bad)}: those paths build their own forwards from input_ids and would drop the inserted role positions.")
        _clash = [n for (n, f) in (('role_gate_mlp', RGM.is_role_gate_adapter), ('lora_role_matrix', RM.is_role_matrix_adapter), ('lora_rank_gate', RKG.is_rank_gate_adapter), ('lora_recv_gate', RCV.is_recv_gate_adapter), ('split_gate_proj', SGL.is_split_gate_adapter), ('swiglu_thres', SWT.is_swiglu_thres_adapter), (FRS.MECHANISM, FRS.is_ffn_role_adapter)) if f(args.adapter) or f(args.adapter2)]
        if _clash:
            raise SystemExit(f"[{SP.MECHANISM}] this adapter also carries {', '.join(_clash)}; those mechanisms read per-token ids built against the sequence BEFORE the role positions are inserted, and train_lora refuses the pair. Refusing to serve it.")
        src = args.adapter if SP.is_soft_role_adapter(args.adapter) else args.adapter2
        if not src:
            raise SystemExit(f'--soft_role_prompt was requested but neither adapter carries {SP.STATE_FILE}; there are no role vectors to load.')
        soft_role_info = SP.attach(model, slots=SP.slots_from_adapter(src), param=SP.param_from_adapter(src))
        soft_role_info['loaded'] = SP.load(model, src)
        soft_role_info['loaded_from'] = src
        soft_role_info['config'] = SP.read_config(src)
        print(f"[{SP.MECHANISM}] {soft_role_info['n_slots']} role embedding(s) {list(soft_role_info['slots'])} at d_role={soft_role_info['role_dim']} + the shared d_role -> d_model encoder ({soft_role_info['n_soft_params']:,} parameters) restored from {src}")
        print(f'[{SP.MECHANISM}] loaded vectors: {SP.stats(model)}')
    struct_role = args.struct_role_bias
    if struct_role is None:
        struct_role = SRB.is_struct_role_adapter(args.adapter) or SRB.is_struct_role_adapter(args.adapter2)
        print(f'[{SRB.MECHANISM}] auto-detected from the adapter directory: {struct_role}')
    else:
        print(f'[{SRB.MECHANISM}] explicitly set: {struct_role}')
    struct_role_info = None
    if struct_role:
        if args.attn_impl not in SRB.ALLOWED_BASE:
            raise SystemExit(f'{SRB.MECHANISM} needs --attn_impl in {list(SRB.ALLOWED_BASE)} (got {args.attn_impl!r}): flash_attention_2 cannot take an additive attention bias.')
        _bad = [n for (n, v) in (('--lora_ablate', args.lora_ablate != 'none'), ('--decision_margin', bool(args.decision_margin))) if v]
        if _bad:
            raise SystemExit(f"[{SRB.MECHANISM}] cannot be combined with {', '.join(_bad)}: those paths build their own forwards and would score the adapter under ordinary attention.")
        _clash = [n for (n, f) in (('soft_role_prompt', SP.is_soft_role_adapter), ('ctx_relay_mask', lambda d: bool(CRM.read_marker(d))), ('target_sandwich', lambda d: bool(TSW.read_marker(d)))) if f(args.adapter) or f(args.adapter2)]
        if _clash:
            raise SystemExit(f"[{SRB.MECHANISM}] this adapter also carries {', '.join(_clash)}; those mechanisms install their own attention path or change the token sequence, and train_lora refuses the pair. Refusing to serve it.")
        src = args.adapter if SRB.is_struct_role_adapter(args.adapter) else args.adapter2
        if not SRB.is_struct_role_adapter(src):
            raise SystemExit(f'--struct_role_bias was requested but neither adapter carries {SRB.STATE_FILE}; there is no bias to load.')
        struct_role_info = SRB.attach(model, base_impl=args.attn_impl, relations=SRB.relations_from_adapter(src))
        struct_role_info['loaded'] = SRB.load(model, src)
        struct_role_info['loaded_from'] = src
        struct_role_info['config'] = SRB.read_config(src)
        args.bs = SRB.clamp_gen_batch(args.bs, where='--bs')
        _srs = SRB.stats(model)
        print(f"[{SRB.MECHANISM}] beta{struct_role_info['beta_shape']} ({struct_role_info['n_beta_params']:,} trained scalars) restored from {src}; relation set '{struct_role_info['relation_set']}': {', '.join(struct_role_info['relations_active'])}" + (f" (pinned at 0: {', '.join(struct_role_info['relations_pinned_at_zero'])})" if struct_role_info['relations_pinned_at_zero'] else ''))
        for _rel in struct_role_info['relations_active']:
            print(f"[{SRB.MECHANISM}] {_rel:<12} mean={_srs['per_relation_mean'][_rel]:+.5f}  |mean|={_srs['per_relation_absmean'][_rel]:.5f}  max|beta|={_srs['per_relation_absmax'][_rel]:.5f}")
        struct_role_info['stats'] = _srs
    role_gate = args.role_gate_mlp
    if role_gate is None:
        role_gate = RGM.is_role_gate_adapter(args.adapter) or RGM.is_role_gate_adapter(args.adapter2)
        print(f'[role_gate_mlp] auto-detected from the adapter directory: {role_gate}')
    else:
        print(f'[role_gate_mlp] explicitly set: {role_gate}')
    role_gate_info = None
    if role_gate:
        bad = [n for (n, v) in (('--lora_ablate', args.lora_ablate != 'none'), ('--lora_scale', args.lora_scale != 1.0), ('--decision_margin', bool(args.decision_margin))) if v]
        if bad:
            raise SystemExit(f"role-gate modulation cannot be combined with {', '.join(bad)}: those paths run forwards that carry no per-token role ids.")
        src = args.adapter if RGM.is_role_gate_adapter(args.adapter) else args.adapter2
        if not src:
            raise SystemExit(f'--role_gate_mlp was requested but neither adapter carries {RGM.STATE_FILE}; there are no role vectors to load.')
        role_gate_info = RGM.attach(model)
        loaded = RGM.load(model, src)
        role_gate_info['loaded_from'] = src
        role_gate_info['loaded'] = loaded
        print(f"[role_gate_mlp] {role_gate_info['n_gated_mlps']} gated MLPs, {role_gate_info['n_role_params']:,} role parameters restored from {src}")
        print(f"[role_gate_mlp] placement check: {role_gate_info['placement_check']}")
        print(f'[role_gate_mlp] loaded vector norms: {RGM.stats(model)}')
    ffn_role = args.ffn_role_scale
    if ffn_role is None:
        ffn_role = FRS.is_ffn_role_adapter(args.adapter) or FRS.is_ffn_role_adapter(args.adapter2)
        print(f'[{FRS.MECHANISM}] auto-detected from the adapter directory: {ffn_role}')
    else:
        print(f'[{FRS.MECHANISM}] explicitly set: {ffn_role}')
    ffn_role_info = None
    if ffn_role:
        bad = [n for (n, v) in (('--lora_ablate', args.lora_ablate != 'none'), ('--lora_scale', args.lora_scale != 1.0), ('--decision_margin', bool(args.decision_margin))) if v]
        if bad:
            raise SystemExit(f"[{FRS.MECHANISM}] cannot be combined with {', '.join(bad)}: those paths run forwards that carry no per-token role ids.")
        src = args.adapter if FRS.is_ffn_role_adapter(args.adapter) else args.adapter2
        if not src:
            raise SystemExit(f'--ffn_role_scale was requested but neither adapter carries {FRS.STATE_FILE}; there are no scale vectors to load.')
        ffn_role_info = FRS.attach(model)
        ffn_role_info['loaded_from'] = src
        ffn_role_info['loaded'] = FRS.load(model, src)
        print(f"[{FRS.MECHANISM}] {ffn_role_info['n_gated_mlps']} gated MLPs, {ffn_role_info['n_scale_params']:,} scale parameters restored from {src}")
        print(f"[{FRS.MECHANISM}] placement check: {ffn_role_info['placement_check']}")
        print(FRS.format_stats(model))
    role_matrix = args.lora_role_matrix
    if role_matrix is None:
        role_matrix = RM.is_role_matrix_adapter(args.adapter) or RM.is_role_matrix_adapter(args.adapter2)
        print(f'[lora_role_matrix] auto-detected from the adapter directory: {role_matrix}')
    else:
        print(f'[lora_role_matrix] explicitly set: {role_matrix}')
    if args.role_matrix_mode != RM.DEFAULT_MODE and (not role_matrix):
        ap.error('--role_matrix_mode only has an effect with the K/V role matrices installed')
    rank_gate = args.lora_rank_gate
    if rank_gate is None:
        rank_gate = RKG.is_rank_gate_adapter(args.adapter) or RKG.is_rank_gate_adapter(args.adapter2)
        print(f'[lora_rank_gate] auto-detected from the adapter directory: {rank_gate}')
    else:
        print(f'[lora_rank_gate] explicitly set: {rank_gate}')
    role_matrix_info = None
    if role_matrix:
        bad = [n for (n, v) in (('--lora_ablate', args.lora_ablate != 'none'), ('--lora_scale', args.lora_scale != 1.0), ('--decision_margin', bool(args.decision_margin))) if v]
        if bad:
            raise SystemExit(f"the role matrices cannot be combined with {', '.join(bad)}: those paths rewrite or bypass the LoRA residual the matrices act inside, and build no per-token role scope for them.")
        src = args.adapter if RM.is_role_matrix_adapter(args.adapter) else args.adapter2
        if not RM.is_role_matrix_adapter(src):
            raise SystemExit(f'--lora_role_matrix was requested but neither adapter carries {RM.STATE_FILE}; there are no role matrices to load.')
        role_matrix_info = RM.attach(model, mode=args.role_matrix_mode)
        role_matrix_info['loaded_from'] = src
        role_matrix_info['loaded'] = RM.load(model, src)
        print(f"[lora_role_matrix] {role_matrix_info['n_modules']} patched modules ({role_matrix_info['modules_per_family']}), r={role_matrix_info['lora_r']}, {role_matrix_info['n_role_matrix_params']:,} parameters restored from {src}")
        print(f'[lora_role_matrix] mode={args.role_matrix_mode} ' + {'learned': '(the four matrices as trained)', 'identity': '(all replaced by I -- ordinary LoRA over the same A/B)', 'swap': '(target <-> context exchanged within each task)'}[args.role_matrix_mode])
        _d = RM.get_bank(model).diagnostics(per_module=False)['summary']
        print('[lora_role_matrix] ' + '  '.join((f"||R_{n} - I||_F={_d[f'{n}_minus_I_fro_mean']:.5f}" for n in RM.SLOT_NAMES)))
        role_matrix_info['diagnostics'] = _d
    rank_gate_info = None
    if rank_gate:
        bad = [n for (n, v) in (('--lora_ablate', args.lora_ablate != 'none'), ('--lora_scale', args.lora_scale != 1.0)) if v]
        if bad:
            raise SystemExit(f"the K/V rank gate cannot be combined with {', '.join(bad)}: those paths rewrite the LoRA residual the gate acts inside.")
        if args.decision_margin:
            _nz = [t for t in args.tasks if RKG.task_row(t) != RKG.TASK_NONE]
            if _nz:
                raise SystemExit(f'--decision_margin cannot be combined with the K/V rank gate for {_nz}: those tasks use a learned gate row, and the margin path builds no table-token scope for it.')
        src = args.adapter if RKG.is_rank_gate_adapter(args.adapter) else args.adapter2
        if not src:
            raise SystemExit(f'--lora_rank_gate was requested but neither adapter carries {RKG.STATE_FILE}; there are no gate matrices to load.')
        cfg = RKG.read_config(src) or {}
        rank_gate_info = RKG.attach(model, mode=cfg.get('lora_rank_gate', 'kv'), exclude_target_tasks=cfg.get('exclude_target_tasks', ()))
        loaded = RKG.load(model, src)
        rank_gate_info['loaded_from'] = src
        rank_gate_info['loaded'] = loaded
        print(f"[lora_rank_gate] {rank_gate_info['n_gated_modules']} gated modules ({rank_gate_info['modules_per_family']}), r={rank_gate_info['lora_r']}, {rank_gate_info['n_gate_params']:,} gate parameters restored from {src}")
        print(f"[lora_rank_gate] task rows: {rank_gate_info['task_rows']} (row 0 is a FIXED ZERO matrix -> ordinary LoRA)")
        print(f"[lora_rank_gate] token scope per task (from the adapter): {rank_gate_info['token_scope_by_task']}   (exclude_target_tasks={rank_gate_info['exclude_target_tasks']})")
        print(f"[lora_rank_gate] gated roles: {rank_gate_info['gated_roles']}")
        print(f'[lora_rank_gate] loaded gate norms: {RKG.gate_weight_stats(model)}')
        _sg = bool(loaded.get('trained_with_gate_stop_gradient', False))
        rank_gate_info['gate_stop_gradient'] = _sg
        rank_gate_info['gate_stop_gradient_runtime'] = RKG.GATE_STOP_GRADIENT
        print(f"[lora_rank_gate] trained with gate_stop_gradient={_sg} ({('gradient-isolated' if _sg else 'pre-iso COUPLED-gradient')} controller inputs)")
    recv_gate = args.lora_recv_gate
    if recv_gate is None:
        recv_gate = RCV.is_recv_gate_adapter(args.adapter) or RCV.is_recv_gate_adapter(args.adapter2)
        print(f'[lora_recv_gate] auto-detected from the adapter directory: {recv_gate}')
    else:
        print(f'[lora_recv_gate] explicitly set: {recv_gate}')
    recv_gate_info = None
    if recv_gate:
        if rank_gate:
            raise SystemExit('this adapter carries BOTH a sender-side K/V rank gate and a receiver-side V-LoRA retrieval gate. They are separate experiments and are never combined; pass --no_lora_rank_gate or --no_lora_recv_gate to choose one.')
        bad = [n for (n, v) in (('--lora_ablate', args.lora_ablate != 'none'), ('--lora_scale', args.lora_scale != 1.0)) if v]
        if bad:
            raise SystemExit(f"the receiver gate cannot be combined with {', '.join(bad)}: those paths rewrite the LoRA residual the gate acts on.")
        if args.decision_margin:
            _nz = [t for t in args.tasks if RCV.task_row(t) != RCV.TASK_NONE]
            if _nz:
                raise SystemExit(f'--decision_margin cannot be combined with the receiver gate for {_nz}: those tasks use a learned gate row, and the margin path builds neither sender streams nor receiver positions for it.')
        src = args.adapter if RCV.is_recv_gate_adapter(args.adapter) else args.adapter2
        if not src:
            raise SystemExit(f'--lora_recv_gate was requested but neither adapter carries {RCV.STATE_FILE}; there are no controllers to load.')
        cfg = RCV.read_config(src) or {}
        recv_gate_info = RCV.attach(model, mode=cfg.get('lora_recv_gate', 'v'), exclude_target_tasks=cfg.get('exclude_target_tasks', ()), base_impl=cfg.get('base_attn_impl', args.attn_impl))
        loaded = RCV.load(model, src)
        recv_gate_info['loaded_from'] = src
        recv_gate_info['loaded'] = loaded
        print(f"[lora_recv_gate] {recv_gate_info['n_layers']} layers x {len(RCV.STREAMS)} streams x {len(recv_gate_info['gate_tasks'])} tasks, r={recv_gate_info['lora_r']}, {recv_gate_info['n_gate_params']:,} controller parameters restored from {src}")
        print(f"[lora_recv_gate] task rows: {recv_gate_info['task_rows']} (row 0 is a FIXED ZERO matrix -> ordinary LoRA)")
        print(f"[lora_recv_gate] target available per task (from the adapter): {recv_gate_info['target_available_by_task']}   (exclude_target_tasks={recv_gate_info['exclude_target_tasks']})")
        print(f'[lora_recv_gate] loaded controller norms: {RCV.gate_weight_stats(model)}')
        _sg = bool(loaded.get('trained_with_gate_stop_gradient', False))
        recv_gate_info['gate_stop_gradient'] = _sg
        recv_gate_info['gate_stop_gradient_runtime'] = RCV.GATE_STOP_GRADIENT
        print(f"[lora_recv_gate] trained with gate_stop_gradient={_sg} ({('gradient-isolated' if _sg else 'pre-iso COUPLED-gradient')} controller inputs)")
        if args.recv_gate_neutral:
            RCV.set_apply_gate(False)
            recv_gate_info['neutralised'] = True
            print('[lora_recv_gate] --recv_gate_neutral: gP = gC = 1 forced; the correction is exactly zero. Same weights, same kernel, no gate.')
    elif args.recv_gate_neutral:
        raise SystemExit('--recv_gate_neutral needs the receiver gate on; with no gate loaded there is nothing to neutralise and the run would just be the ordinary adapter.')
    ctx_relay = resolve_ctx_relay(args.adapter, args.adapter2, args.ctx_relay_mask)
    ctx_relay_info = None
    if ctx_relay:
        if args.attn_impl not in CRM.ALLOWED_BASE:
            raise SystemExit(f'{CRM.MECHANISM} needs --attn_impl in {list(CRM.ALLOWED_BASE)} (got {args.attn_impl!r}): flash_attention_2 cannot take a custom 4D attention mask.')
        if not target_end:
            raise SystemExit(f'{CRM.MECHANISM} requires --target_end, and this run resolved target_end=False. Without it the designated column is wherever the corpus put it, so on every prompt whose target comes first the mask would DELETE the context instead of relaying it.')
        if table_first or table_last:
            raise SystemExit(f'{CRM.MECHANISM} cannot be combined with --table_first/--table_last: the receiver set is the POST-TABLE region, and both move the query out of it.')
        bad = [nm for (nm, v) in (('--lora_ablate', args.lora_ablate != 'none'), ('--lora_recv_gate', recv_gate), ('--lora_rank_gate', rank_gate)) if v]
        if bad:
            raise SystemExit(f"{CRM.MECHANISM} cannot be combined with {', '.join(bad)}: those paths run their own forwards / own attention mask and would not carry this one.")
        ctx_relay_info = CRM.attach(model, base_impl=args.attn_impl)
        print(f'[{CRM.MECHANISM}] attached: ' + '; '.join((f'{k.upper()} {v}' for (k, v) in CRM.RELAY.items())))
        print(f"[{CRM.MECHANISM}] cut keys: roles {ctx_relay_info['context_roles']}; kept: roles {ctx_relay_info['relay_roles']} + every non-table token")
        print(f"[{CRM.MECHANISM}] receivers: {ctx_relay_info['receiver_scope']}")
        print(f"[{CRM.MECHANISM}] {ctx_relay_info['mask_kind']} 4D mask on {ctx_relay_info['base_attn_impl']}, pre-hook on {ctx_relay_info['prehook_on']}")
    tgt_col_info = None
    if args.target_col_mask:
        if args.attn_impl not in TCM.ALLOWED_BASE:
            raise SystemExit(f'--target_col_mask needs --attn_impl in {list(TCM.ALLOWED_BASE)} (got {args.attn_impl!r}): flash_attention_2 cannot take a custom 4D mask.')
        _clash = [n for (n, on) in ((CRM.MECHANISM, ctx_relay), (TSW.MECHANISM, target_sandwich and sandwich_mask), (SRB.MECHANISM, struct_role)) if on]
        if _clash:
            raise SystemExit(f"--target_col_mask cannot be combined with {', '.join(_clash)}: both install their own 4D attention mask / kernel and the second one silently wins.")
        _bad = [n for (n, v) in (('--lora_ablate', args.lora_ablate != 'none'), ('--decision_margin', bool(args.decision_margin))) if v]
        if _bad:
            raise SystemExit(f"--target_col_mask cannot be combined with {', '.join(_bad)}: those paths run forwards that carry no per-token column/role context.")
        tgt_col_info = TCM.attach(model, base_impl=args.attn_impl)
        print(f"[{TCM.MECHANISM}] INFERENCE-ONLY mask attached ({tgt_col_info['mask_kind']} 4D, base {tgt_col_info['base_attn_impl']})")
        for (_r, _v) in TCM.RULE.items():
            print(f'[{TCM.MECHANISM}]   {_r:<12} may read {_v}')
        print(f"[{TCM.MECHANISM}] cut keys: {tgt_col_info['cut_keys']}")
        print(f"[{TCM.MECHANISM}] untouched: {tgt_col_info['untouched_keys']}")
    tgt_sandwich_info = None
    if target_sandwich and (not sandwich_mask):
        print(f"[{TSW.MECHANISM}] serialization only: the target column is written at both ends of the table and attention is the model's ordinary causal pattern.")
    if target_sandwich and sandwich_mask:
        if args.attn_impl not in TSW.ALLOWED_BASE:
            raise SystemExit(f'{TSW.MECHANISM} needs --attn_impl in {list(TSW.ALLOWED_BASE)} (got {args.attn_impl!r}): flash_attention_2 cannot take a custom 4D attention mask.')
        if ctx_relay:
            raise SystemExit(f'{TSW.MECHANISM} cannot be combined with {CRM.MECHANISM}: both install their own 4D attention mask through the same argument, and the second one silently wins.')
        if table_first or table_last:
            raise SystemExit(f'{TSW.MECHANISM} cannot be combined with --table_first/--table_last: the cut query set is the POST-TABLE region, and both move the query out of it.')
        bad = [nm for (nm, v) in (('--lora_ablate', args.lora_ablate != 'none'), ('--lora_recv_gate', recv_gate), ('--lora_rank_gate', rank_gate)) if v]
        if bad:
            raise SystemExit(f"{TSW.MECHANISM} cannot be combined with {', '.join(bad)}: those paths run their own forwards / own attention mask and would not carry this one.")
        tgt_sandwich_info = TSW.attach(model, base_impl=args.attn_impl)
        print(f'[{TSW.MECHANISM}] attached: ' + '; '.join((f'{k.upper()} {v}' for (k, v) in TSW.SANDWICH.items())))
        print(f"[{TSW.MECHANISM}] cut keys: {tgt_sandwich_info['cut_keys']}")
        print(f"[{TSW.MECHANISM}] cut queries: {tgt_sandwich_info['cut_queries']}")
        print(f"[{TSW.MECHANISM}] {tgt_sandwich_info['mask_kind']} 4D mask on {tgt_sandwich_info['base_attn_impl']}, pre-hook on {tgt_sandwich_info['prehook_on']}")
    split_gate = args.split_gate_proj
    if split_gate is None:
        split_gate = SGL.is_split_gate_adapter(args.adapter) or SGL.is_split_gate_adapter(args.adapter2)
        print(f'[split_gate_proj] auto-detected from the adapter directory: {split_gate}')
    else:
        print(f'[split_gate_proj] explicitly set: {split_gate}')
    split_gate_info = None
    if split_gate:
        src = args.adapter if SGL.is_split_gate_adapter(args.adapter) else args.adapter2
        if not SGL.is_split_gate_adapter(src):
            raise SystemExit(f'--split_gate_proj was requested but neither adapter carries {SGL.STATE_FILE}; there are no split pairs to load.')
        cfg = SGL.read_config(src) or {}
        if cfg.get('r') is None:
            raise SystemExit(f'{src}/{SGL.CONFIG_FILE} carries no rank; refusing to guess it.')
        split_gate_info = SGL.attach(model, r=cfg['r'], alpha=cfg['lora_alpha'], dropout=0.0, modules=SGL.config_modules(cfg))
        split_gate_info['loaded_from'] = src
        split_gate_info['loaded'] = SGL.load(model, src)
        print(f"[split_gate_proj] {split_gate_info['n_modules']} MLP modules {split_gate_info['n_modules_per_family']} x {list(SGL.BRANCHES)} @ r={split_gate_info['r']} alpha={split_gate_info['lora_alpha']:g} -> {split_gate_info['n_split_params']:,} parameters loaded from {src}")
        print(f"[split_gate_proj] {split_gate_info['formula']}")
        print(f"[split_gate_proj] {split_gate_info['token_scope']}")
        print(f'[split_gate_proj] loaded weight norms: {SGL.weight_stats(model)}')
        split_gate_info['swap_branches'] = bool(args.split_swap_branches)
        if args.split_swap_branches:
            SGL.set_swap(True)
            print('[split_gate_proj] SWAP: table tokens are routed through the NON-TABLE pair and every non-table token (pre-table, post-table, padding and every generated token) through the TABLE pair. Weights, prompts and the region rule are unchanged; only the branch each token reads is.')
    elif args.split_swap_branches:
        raise SystemExit('--split_swap_branches needs a split adapter: with no split pairs attached there are no branches to swap and the run would just be the ordinary adapter.')
        if args.decision_margin:
            raise SystemExit('--decision_margin is not supported for a split gate_proj adapter: the margin pass builds its own padded batch and would have to carry the table / non-table token split with it. Run the margin probe on a non-split adapter.')
    swiglu_thres = args.swiglu_thres
    if swiglu_thres is None:
        swiglu_thres = SWT.is_swiglu_thres_adapter(args.adapter) or SWT.is_swiglu_thres_adapter(args.adapter2)
        print(f'[swiglu_thres] auto-detected from the adapter directory: {swiglu_thres}')
    else:
        print(f'[swiglu_thres] explicitly set: {swiglu_thres}')
    swiglu_info = None
    if swiglu_thres:
        src = args.adapter if SWT.is_swiglu_thres_adapter(args.adapter) else args.adapter2
        if not SWT.is_swiglu_thres_adapter(src):
            raise SystemExit(f'--swiglu_thres was requested but neither adapter carries {SWT.STATE_FILE}; there are no temperatures to load.')
        swiglu_info = SWT.attach(model)
        swiglu_info['loaded_from'] = src
        swiglu_info['loaded'] = SWT.load(model, src)
        print(f"[swiglu_thres] {swiglu_info['n_layers']} gated MLPs x {len(SWT.BRANCHES)} temperatures loaded from {src}")
        print(f"[swiglu_thres] {swiglu_info['formula']}")
        print(f"[swiglu_thres] {swiglu_info['token_scope']}")
        _bs = SWT.beta_summary(model)
        swiglu_info['betas'] = _bs
        for b in SWT.BRANCHES:
            print(f"[swiglu_thres] beta_{b:<9} min={_bs[b]['min']:.4f} mean={_bs[b]['mean']:.4f} max={_bs[b]['max']:.4f}")
        print(f"[swiglu_thres] beta_table - beta_nontable: mean={_bs['table_minus_nontable']['mean']:+.4f} ({_bs['table_minus_nontable']['n_layers_table_sharper']}/{swiglu_info['n_layers']} layers sharper on table tokens)")
        swiglu_info['forced_off'] = bool(args.swiglu_thres_off)
        if args.swiglu_thres_off:
            SWT.set_enabled(False)
            print('[swiglu_thres] OFF: every beta is forced back to exactly 1.0, so the gate activation is the baseline SiLU. The checkpoint is untouched.')
        else:
            SWT.set_enabled(True)
        if args.decision_margin:
            raise SystemExit('--decision_margin is not supported for a swiglu_thres adapter: the margin pass builds its own padded batch and would have to carry the table / non-table token split with it. Run the margin probe on a plain adapter.')
    elif args.swiglu_thres_off:
        raise SystemExit('--swiglu_thres_off needs the temperatures attached; with none loaded there is nothing to force back to 1.0 and the run would just be the ordinary adapter.')
    force_greedy_generation(model, tok)
    model.eval()
    masker = None
    if args.lora_ablate != 'none' or args.lora_scale != 1.0:
        masker = LTA.LoraRegionMasker(model, mode=args.lora_ablate, scale=args.lora_scale)
        print(f"[lora_ablate] mode={args.lora_ablate}: hooked {masker.n_hooked} lora_B modules (generated tokens: LoRA {('ON' if LTA.generated_token_kept(args.lora_ablate) else 'OFF')})")
        if args.lora_scale != 1.0:
            print(f'[lora_scale] LoRA residual multiplied by {args.lora_scale} everywhere')
        if masker.n_hooked == 0:
            print('[lora_ablate] WARNING: no LoRA modules found -- this is a base-model run, so the ablation is a no-op (already LoRA-free everywhere).')
    out_dir = os.path.dirname(args.out) or '.'
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.out))[0]
    results = {'adapter': args.adapter, 'adapter2': args.adapter2, 'model': args.model, 'task_prefix': args.task_prefix, 'single_column': args.single_column, 'shuffle_columns': args.shuffle_columns, 'shuffle_rows': args.shuffle_rows, 'shuffle_mode': args.shuffle_mode if args.shuffle_columns else None, 'shuffle_seed': args.shuffle_seed if args.shuffle_columns or args.shuffle_rows else None, 'drop_empty_target': args.drop_empty_target, 'col_major_tasks': sorted(col_major_tasks), 'merge_adapter': merge_adapter_path, 'table_first': table_first, 'table_first_tasks': sorted(table_first_tasks), 'table_last': table_last, 'target_end': target_end, 'target_end_tasks': sorted(target_end_tasks), 'target_first': target_first, 'target_col_mask': bool(args.target_col_mask), 'target_col_mask_info': tgt_col_info, 'attn_impl': args.attn_impl, 'lora_ablate': args.lora_ablate, 'lora_scale': args.lora_scale, 'role_gate_mlp': bool(role_gate), 'role_gate_info': role_gate_info, 'ffn_role_scale': bool(ffn_role), 'ffn_role_info': ffn_role_info, 'soft_role_prompt': bool(soft_role), 'soft_role_prompt_info': soft_role_info, 'struct_role_bias': bool(struct_role), 'struct_role_relations': None if struct_role_info is None else struct_role_info['relations_active'], 'struct_role_bias_info': struct_role_info, 'split_gate_proj': bool(split_gate), 'split_swap_branches': bool(args.split_swap_branches), 'swiglu_thres': bool(swiglu_thres), 'swiglu_thres_off': bool(args.swiglu_thres_off), 'swiglu_thres_info': swiglu_info, 'split_gate_info': split_gate_info, 'lora_rank_gate': bool(rank_gate), 'rank_gate_info': rank_gate_info, 'lora_role_matrix': bool(role_matrix), 'role_matrix_mode': args.role_matrix_mode if role_matrix else None, 'role_matrix_info': role_matrix_info, 'lora_recv_gate': bool(recv_gate), 'recv_gate_neutral': bool(args.recv_gate_neutral), 'recv_gate_info': recv_gate_info, 'ctx_relay_mask': bool(ctx_relay), 'ctx_relay_info': ctx_relay_info, 'target_sandwich': bool(target_sandwich), 'sandwich_mask': bool(sandwich_mask), 'target_sandwich_info': tgt_sandwich_info, 'decision_margin': bool(args.decision_margin), 'tasks': {}}
    warn_flag_scope(args.tasks, args.single_column, args.shuffle_columns, args.shuffle_rows)
    for slug in args.tasks:
        (test, prompts, prep) = load_and_prepare(slug, limit=args.limit, drop_empty=args.drop_empty_target, single_column=args.single_column, shuffle_columns=args.shuffle_columns, shuffle_mode=args.shuffle_mode, shuffle_rows=args.shuffle_rows, shuffle_seed=args.shuffle_seed, task_prefix=args.task_prefix, col_major_tasks=col_major_tasks, table_first=table_first, table_first_tasks=table_first_tasks, table_last=table_last, target_end=target_end, target_end_tasks=target_end_tasks, target_first=target_first, target_sandwich=target_sandwich)
        if masker is not None and masker.n_hooked and ('ablate_check' not in results):
            chk_texts = [apply_model_chat_template(tok, c, args.model) for c in prompts[:min(2, len(prompts))]]
            chk_enc = tok(chk_texts, return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
            results['ablate_check'] = LTA.check_masker_identities(model, masker, chk_enc)
            print(f"[lora_ablate] identity check: {results['ablate_check']}")
        if ctx_relay:
            CRM.reset_stats()
        if args.target_col_mask:
            TCM.reset_stats()
        if target_sandwich and sandwich_mask:
            TSW.reset_stats()
        gens = generate_batch(model, tok, prompts, args.max_new_tokens, args.bs, args.model, masker=masker, ablate=args.lora_ablate, role_task=slug if role_gate else None, gate_task=slug if rank_gate else None, recv_task=slug if recv_gate else None, role_matrix_task=slug if role_matrix else None, split_gate=split_gate, swiglu_thres=swiglu_thres, ctx_relay_task=slug if ctx_relay else None, sandwich_task=slug if target_sandwich and sandwich_mask else None, soft_task=slug if soft_role else None, struct_task=slug if struct_role else None, ffn_role_task=slug if ffn_role else None, tgt_col_task=slug if args.target_col_mask else None)
        if ctx_relay:
            _st = CRM.stats()
            metrics_relay = dict(_st)
            if _st['forwards_masked'] == 0:
                raise SystemExit(f'[{CRM.MECHANISM}] {slug}: the mask was installed on ZERO forwards, so this task was generated under ordinary causal attention while the adapter was trained under the mask. Refusing to report the score.')
            if _st['forwards_unmasked']:
                raise SystemExit(f"[{CRM.MECHANISM}] {slug}: {_st['forwards_unmasked']} forward(s) ran with no mask installed. Generation must be masked end to end; refusing to report a partly masked score.")
            print(f"[{CRM.MECHANISM}] {slug}: {_st['forwards_masked']:,} masked forwards, 0 unmasked")
        if args.target_col_mask:
            _ts = TCM.stats()
            metrics_tgtcol = dict(_ts)
            if _ts['forwards_masked'] == 0:
                raise SystemExit(f'[{TCM.MECHANISM}] {slug}: the mask was installed on ZERO forwards, so this task was generated under ordinary causal attention. Refusing to report the score as a masked one.')
            if _ts['forwards_unmasked']:
                raise SystemExit(f"[{TCM.MECHANISM}] {slug}: {_ts['forwards_unmasked']} forward(s) ran with no mask installed. Generation must be masked end to end; refusing to report a partly masked score.")
            print(f"[{TCM.MECHANISM}] {slug}: {_ts['forwards_masked']:,} masked forwards, 0 unmasked")
        if target_sandwich and sandwich_mask:
            _st = TSW.stats()
            metrics_sandwich = dict(_st)
            if _st['forwards_masked'] == 0:
                raise SystemExit(f'[{TSW.MECHANISM}] {slug}: the mask was installed on ZERO forwards, so this task was generated under ordinary causal attention while the adapter was trained under the mask. Refusing to report the score.')
            if _st['forwards_unmasked']:
                raise SystemExit(f"[{TSW.MECHANISM}] {slug}: {_st['forwards_unmasked']} forward(s) ran with no mask installed. Generation must be masked end to end; refusing to report a partly masked score.")
            print(f"[{TSW.MECHANISM}] {slug}: {_st['forwards_masked']:,} masked forwards, 0 unmasked")
        metrics = score_and_dump(slug, test, gens, prep, out_dir, base)
        if ctx_relay:
            metrics['ctx_relay_forwards'] = metrics_relay
        if target_sandwich and sandwich_mask:
            metrics['target_sandwich_forwards'] = metrics_sandwich
        if args.decision_margin and slug in MARGIN_TASKS:
            (recs, msum) = decision_margin(model, tok, slug, prompts, test, args.bs, args.model, masker=masker, ablate=args.lora_ablate, gate_task=slug if rank_gate else None, recv_task=slug if recv_gate else None)
            mpath = f'{out_dir}/{base}_{slug}_margin.jsonl'
            with open(mpath, 'w') as f:
                for r in recs:
                    f.write(json.dumps(r) + '\n')
            metrics['decision_margin'] = msum
            fd = msum['first_div']
            print(f"[decision_margin] {slug}: mean margin {msum['mean_margin']:+.4f} | argmax acc {msum['argmax_accuracy']:.4f} | argmax ENTAILED rate {msum['argmax_entailed_rate']:.4f} -> {mpath}")
            print(f"[decision_margin] {slug}: first divergent answer token #{fd['answer_token_index']} ({fd['entailed_token']!r} vs {fd['refuted_token']!r} after {fd['shared_prefix']!r}): mean logit margin {fd['mean_logit_margin']:+.4f} | acc {fd['argmax_accuracy']:.4f} | agrees with full answer {fd['agreement_with_full_margin']:.4f}")
        elif args.decision_margin:
            print(f'[decision_margin] {slug}: skipped (not a TabFact task)')
        if masker is not None:
            metrics['lora_ablate_stats'] = masker.stats_dict()
        results['tasks'][slug] = metrics
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote', args.out)
if __name__ == '__main__':
    main()
