import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import role_low_rank as RLR
ROLE_NONE = RLR.ROLE_NONE
TASK_ROLES = {'cta': {'context': (RLR.ROLE_CTA_CONTEXT,), 'designated': (RLR.ROLE_CTA_TARGET,)}, 'cpa': {'context': (RLR.ROLE_CPA_CONTEXT,), 'designated': (RLR.ROLE_CPA_HEAD, RLR.ROLE_CPA_TAIL)}}
CONTEXT_ROLES = tuple((r for t in TASK_ROLES.values() for r in t['context']))
DESIGNATED_ROLES = tuple((r for t in TASK_ROLES.values() for r in t['designated']))
MODES = ('off', 'kv')
DEFAULT_MODE = 'off'
TARGET_MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
MODE_MODULES = {'off': (), 'kv': ('k_proj', 'v_proj')}
GATE_MODULES = MODE_MODULES['kv']
GATE_TASKS = ('cta', 'cpa')
TASK_NONE = 0
TASK_INDEX = {t: i + 1 for (i, t) in enumerate(GATE_TASKS)}
N_TASK_ROWS = len(GATE_TASKS) + 1
PARAM_DTYPE = torch.float32
STATE_FILE = 'rank_gate.pt'
CONFIG_FILE = 'rank_gate_config.json'
GATE_STOP_GRADIENT = True
STOP_GRADIENT_INPUTS = ('lora_A rank code z (controller input and gate-correction multiplicand)',)
GATE_FORMULA_SG = 'z = A(dropout(x)); zs = sg(z); g = 1 + m * tanh(W_task zs); z_out = z + (g - 1) zs; delta = scaling * B(z_out)'

def canonical_task(slug):
    s = str(slug or '').lower()
    for base in ('cta', 'cpa', 'cea', 'hvm', 'em', 'tabfact'):
        if s == base or s.startswith(base + '_'):
            return base
    return s

def task_row(slug):
    return TASK_INDEX.get(canonical_task(slug), TASK_NONE)

def task_rows(slugs, device=None):
    return torch.tensor([task_row(s) for s in slugs], dtype=torch.long, device=device)

def normalize_exclude_target(tasks):
    if tasks is None:
        return ()
    sel = {canonical_task(t) for t in tasks}
    unknown = sorted(sel - set(GATE_TASKS))
    if unknown:
        raise ValueError(f'--rank_gate_exclude_target got {unknown}, which are not gate tasks {list(GATE_TASKS)}; only a gated task can hold its designated columns out.')
    return tuple((t for t in GATE_TASKS if t in sel))

def gated_role_ids(exclude_target_tasks=()):
    sel = set(normalize_exclude_target(exclude_target_tasks))
    ids = []
    for (task, groups) in TASK_ROLES.items():
        ids.extend(groups['context'])
        if task not in sel:
            ids.extend(groups['designated'])
    return tuple(sorted(ids))

def table_mask_from_roles(role_ids, exclude_target_tasks=()):
    r = torch.as_tensor(role_ids)
    keep = gated_role_ids(exclude_target_tasks)
    if set(keep) == set(CONTEXT_ROLES) | set(DESIGNATED_ROLES):
        return (r != ROLE_NONE).to(torch.bool)
    lut = torch.zeros(RLR.N_ROLES, dtype=torch.bool, device=r.device)
    lut[list(keep)] = True
    return lut[r.long()]

def gate_mask(model, role_ids):
    return table_mask_from_roles(role_ids, exclude_target_tasks=get_bank(model).exclude_target_tasks)

class _Ctx:
    task_id = None
    mask_full = None
    cur_mask = None
    valid_full = None
    cur_valid = None
    rows = ()
CTX = _Ctx()

def set_gate_context(task_id, table_mask, valid_mask=None):
    tid = torch.as_tensor(task_id, dtype=torch.long)
    if tid.dim() != 1:
        raise ValueError(f'task ids must be [B], got {tuple(tid.shape)}')
    rows = ()
    if tid.numel():
        rows = tuple(sorted({int(v) for v in tid.tolist()}))
        if rows[0] < 0 or rows[-1] >= N_TASK_ROWS:
            raise ValueError(f'gate row out of range [0, {N_TASK_ROWS - 1}]: saw {list(rows)}. Rows are 0 = none (ordinary LoRA) and 1..{len(GATE_TASKS)} = {list(GATE_TASKS)}.')
    tm = torch.as_tensor(table_mask)
    if tm.dim() != 2 or int(tm.shape[0]) != int(tid.shape[0]):
        raise ValueError(f'table_mask must be [B, T] with B = {int(tid.shape[0])}, got {tuple(tm.shape)}')
    vm = None
    if valid_mask is not None:
        vm = torch.as_tensor(valid_mask)
        if tuple(vm.shape) != tuple(tm.shape):
            raise ValueError(f'valid_mask {tuple(vm.shape)} must have the same shape as table_mask {tuple(tm.shape)}')
        vm = vm.to(device=tm.device).bool()
    CTX.task_id = tid
    CTX.mask_full = tm.to(torch.bool)
    CTX.valid_full = vm
    CTX.rows = rows
    CTX.cur_mask = None
    CTX.cur_valid = None

def set_gate_context_for_slug(slug, table_mask, valid_mask=None, device=None):
    tm = torch.as_tensor(table_mask)
    tid = torch.full((int(tm.shape[0]),), task_row(slug), dtype=torch.long, device=device)
    set_gate_context(tid, tm, valid_mask=valid_mask)

def clear_gate_context():
    CTX.task_id = None
    CTX.mask_full = None
    CTX.cur_mask = None
    CTX.valid_full = None
    CTX.cur_valid = None
    CTX.rows = ()

def _mask_prehook(module, args, kwargs):
    if CTX.mask_full is None:
        CTX.cur_mask = None
        return
    input_ids = kwargs.get('input_ids', None)
    if input_ids is None and args:
        input_ids = args[0]
    if input_ids is None:
        CTX.cur_mask = CTX.mask_full
        return
    seq_len = input_ids.shape[1]
    dev = CTX.mask_full.device
    cp = kwargs.get('cache_position', None)
    cp = torch.arange(seq_len, device=dev) if cp is None else cp.to(dev)
    stored_T = CTX.mask_full.shape[1]
    idx = cp.clamp(max=stored_T - 1)
    in_prompt = cp < stored_T
    CTX.cur_mask = CTX.mask_full[:, idx] & in_prompt
    if CTX.valid_full is None:
        CTX.cur_valid = None
    else:
        CTX.cur_valid = CTX.valid_full[:, idx] | ~in_prompt
ACC_FIELDS = ('n', 'sum_g', 'sum_abs_dev', 'sum_sq_dev', 'sum_sq_gated_minus_vanilla', 'sum_sq_vanilla', 'n_tokens_gated', 'n_tokens_seen')

def blank_acc():
    d = {k: 0.0 for k in ACC_FIELDS}
    d['max_abs_dev'] = 0.0
    return d

def merge_acc(dst, rec):
    for k in ACC_FIELDS:
        dst[k] += rec[k]
    dst['max_abs_dev'] = max(dst['max_abs_dev'], rec['max_abs_dev'])

class RankGateBank(nn.Module):

    def __init__(self, module_names, r, tasks=GATE_TASKS, exclude_target_tasks=()):
        super().__init__()
        self.module_names = list(module_names)
        self.tasks = list(tasks)
        self.r = int(r)
        self.exclude_target_tasks = normalize_exclude_target(exclude_target_tasks)
        self.W = nn.ParameterList([nn.Parameter(torch.zeros(len(self.tasks), self.r, self.r, dtype=PARAM_DTYPE)) for _ in self.module_names])
        self.enabled = True
        self.collect = False
        self.acc = {}

    def select(self, i, tid, dtype, device):
        w = self.W[i].to(device=device, dtype=dtype)
        sel = F.one_hot(tid, N_TASK_ROWS).to(dtype)[:, 1:]
        return (sel @ w.reshape(w.shape[0], -1)).view(-1, self.r, self.r)

    def n_params(self):
        return sum((p.numel() for p in self.W))

    def scope_by_task(self):
        return {t: 'context_cells' if t in self.exclude_target_tasks else 'all_table_cells' for t in self.tasks}

    def extra_repr(self):
        return f'modules={len(self.W)}, tasks={self.tasks}, per_module=[{len(self.tasks)}, {self.r}, {self.r}], scope={self.scope_by_task()}'

    def reset_stats(self):
        self.acc = {}

    @torch.no_grad()
    def _accumulate(self, i, tid_dev, g, z, m, valid):
        gated = m if valid is None else m & valid
        rows = CTX.rows if CTX.rows else tuple(sorted({int(v) for v in tid_dev.tolist()}))
        width = g.shape[-1]
        for k in rows:
            sel = tid_dev == k
            (gk, zk) = (g[sel], z[sel])
            mg = gated[sel].unsqueeze(-1).to(g.dtype)
            gkm = gk * mg
            dkm = (gk - 1.0) * mg
            zkm = zk * mg
            n_gated = gated[sel].sum()
            n_valid = torch.full((), float(sel.sum().item() * g.shape[1]), device=g.device) if valid is None else valid[sel].sum()
            packed = torch.stack([gkm.sum(), dkm.abs().sum(), (dkm * dkm).sum(), dkm.abs().max(), ((dkm * zkm) ** 2).sum(), (zkm * zkm).sum(), n_gated.to(g.dtype), n_valid.to(g.dtype)]).double().tolist()
            (sum_g, sum_abs, sum_sq, max_abs, sum_sq_gmv, sum_sq_van, n_tok_gated, n_tok_valid) = packed
            rec = self.acc.get((i, int(k)))
            if rec is None:
                rec = self.acc[i, int(k)] = blank_acc()
            rec['n_tokens_gated'] += n_tok_gated
            rec['n_tokens_seen'] += n_tok_valid
            rec['n'] += n_tok_gated * width
            rec['sum_g'] += sum_g
            rec['sum_abs_dev'] += sum_abs
            rec['sum_sq_dev'] += sum_sq
            rec['max_abs_dev'] = max(rec['max_abs_dev'], max_abs)
            rec['sum_sq_gated_minus_vanilla'] += sum_sq_gmv
            rec['sum_sq_vanilla'] += sum_sq_van

def _gate_hook(bank, i):

    def hook(_module, _args, z):
        if not bank.enabled:
            return z
        tid = CTX.task_id
        if tid is None:
            raise RuntimeError('the K/V rank gate is attached but no gate context was set for this forward. Call rank_gate.set_gate_context(task_id, table_mask) around every forward / generate; a silent fallback would quietly turn this run into the ordinary-LoRA baseline.')
        m = CTX.cur_mask
        if m is None:
            raise RuntimeError('the K/V rank gate has task ids but no table-token mask for this forward. The cache_position pre-hook did not run. attach() registers it on BOTH the PEFT wrapper and the inner *ForCausalLM because neither alone covers both the training forward and generate(); if this fires, the forward reached the gated modules by a third path that bypassed both.')
        if z.dim() != 3:
            raise RuntimeError(f'the rank gate needs a [B, T, r] rank code, got {tuple(z.shape)}')
        (B, T, r) = z.shape
        if int(tid.shape[0]) != B:
            raise RuntimeError(f'task ids have {int(tid.shape[0])} entries but the rank code has {B} rows; the per-example gate selection is misaligned.')
        if tuple(m.shape) != (B, T):
            raise RuntimeError(f'table mask {tuple(m.shape)} does not align with the rank code {(B, T)}; the token scope / cache_position slicing is misaligned.')
        if r != bank.r:
            raise RuntimeError(f'rank code width {r} != bank rank {bank.r}')
        tid_dev = tid.to(z.device)
        w = bank.select(i, tid_dev, z.dtype, z.device)
        m_dev = m.to(z.device)
        mk = m_dev.to(z.dtype).unsqueeze(-1)
        z_sg = z.detach() if GATE_STOP_GRADIENT else z
        g = 1.0 + mk * torch.tanh(torch.bmm(z_sg, w.transpose(1, 2)))
        if bank.collect:
            v = CTX.cur_valid
            bank._accumulate(i, tid_dev, g.detach(), z.detach(), m_dev, None if v is None else v.to(z.device))
        if not GATE_STOP_GRADIENT:
            return g * z
        return z + (g - 1.0) * z_sg
    return hook

def _lora_rank(lora_a_dict):
    ranks = {sub.out_features for sub in lora_a_dict.values()}
    if len(ranks) != 1:
        raise RuntimeError(f'a LoRA module reports several ranks {sorted(ranks)}')
    return ranks.pop()

def find_gated_modules(model, modules=GATE_MODULES):
    allowed = set(modules)
    out = []
    for (name, m) in model.named_modules():
        la = getattr(m, 'lora_A', None)
        if not isinstance(la, nn.ModuleDict) or not len(la):
            continue
        if name.split('.')[-1] in allowed:
            out.append((name, m))
    return out

def attach(model, mode='kv', tasks=GATE_TASKS, exclude_target_tasks=()):
    if mode not in MODES:
        raise ValueError(f'lora_rank_gate must be one of {list(MODES)}, got {mode!r}')
    if mode == 'off':
        raise ValueError("attach() called with mode 'off'; the caller must not attach at all")
    if getattr(model, 'rank_gate_bank', None) is not None:
        raise RuntimeError('a rank gate is already attached to this model')
    families = MODE_MODULES[mode]
    found = find_gated_modules(model, families)
    if not found:
        raise RuntimeError(f'--lora_rank_gate {mode} found no PEFT LoRA modules among {list(families)}. Is the model a PEFT LoRA model whose target_modules include them?')
    ranks = {_lora_rank(m.lora_A) for (_n, m) in found}
    if len(ranks) != 1:
        raise RuntimeError(f'gated modules disagree on the LoRA rank: {sorted(ranks)}')
    r = ranks.pop()
    names = [n for (n, _m) in found]
    bank = RankGateBank(names, r, tasks=tasks, exclude_target_tasks=exclude_target_tasks)
    dev = next(model.parameters()).device
    bank.to(dev)
    model.rank_gate_bank = bank
    handles = []
    per_family = {}
    for (i, (name, m)) in enumerate(found):
        fam = name.split('.')[-1]
        per_family[fam] = per_family.get(fam, 0) + 1
        for (_adapter, sub) in m.lora_A.items():
            handles.append(sub.register_forward_hook(_gate_hook(bank, i)))
    inner = model.get_base_model() if hasattr(model, 'get_base_model') else model
    hook_targets = [model] if inner is model else [model, inner]
    for t in hook_targets:
        handles.append(t.register_forward_pre_hook(_mask_prehook, with_kwargs=True))
    model._rank_gate_handles = handles
    if len(set(per_family.values())) != 1:
        raise RuntimeError(f'uneven rank-gate coverage across families: {per_family}; every gated family must be patched in the same layers.')
    return {'lora_rank_gate': mode, 'gated_families': sorted(per_family), 'n_gated_modules': len(found), 'modules_per_family': per_family, 'lora_r': r, 'gate_tasks': list(bank.tasks), 'task_rows': {'none': TASK_NONE, **{t: TASK_INDEX[t] for t in bank.tasks}}, 'n_gate_params': bank.n_params(), 'gate_param_shape': [len(bank.tasks), r, r], 'zero_init': True, 'state_file': STATE_FILE, 'formula': GATE_FORMULA_SG, 'forward_equivalent_formula': 'z = A(dropout(x)); g = 1 + m * tanh(W_task z); delta = scaling * B(g * z)', 'gate_stop_gradient': GATE_STOP_GRADIENT, 'stop_gradient_inputs': list(STOP_GRADIENT_INPUTS), 'exclude_target_tasks': list(bank.exclude_target_tasks), 'token_scope_by_task': bank.scope_by_task(), 'gated_roles': [RLR.ROLE_NAMES[i] for i in gated_role_ids(bank.exclude_target_tasks)], 'ungated_roles': [RLR.ROLE_NAMES[i] for i in sorted({RLR.ROLE_NONE} | (set(CONTEXT_ROLES) | set(DESIGNATED_ROLES)) - set(gated_role_ids(bank.exclude_target_tasks)))], 'token_scope_source': 'role_low_rank role ids, selected per task', 'prehook_on': [type(t).__name__ for t in hook_targets]}

def detach(model):
    for h in getattr(model, '_rank_gate_handles', []) or []:
        h.remove()
    model._rank_gate_handles = []
    if getattr(model, 'rank_gate_bank', None) is not None:
        model.rank_gate_bank = None
    clear_gate_context()

def get_bank(model):
    bank = getattr(model, 'rank_gate_bank', None)
    if bank is None:
        raise RuntimeError('no rank gate is attached to this model')
    return bank

def is_attached(model):
    return getattr(model, 'rank_gate_bank', None) is not None

def minimal_logits_kwargs(model):
    import inspect
    inner = model.get_base_model() if hasattr(model, 'get_base_model') else model
    for candidate in (inner, model):
        fn = getattr(candidate, 'forward', None)
        if fn is None:
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        for name in ('logits_to_keep', 'num_logits_to_keep'):
            if name in params:
                return {name: 1}
    return {}

@torch.no_grad()
def verify_zero_gate(model, input_ids, attention_mask=None, task_id=None, table_mask=None, atol=0.0):
    bank = get_bank(model)
    (B, T) = (int(input_ids.shape[0]), int(input_ids.shape[1]))
    if task_id is None:
        if B < N_TASK_ROWS:
            raise ValueError(f"the zero-gate probe needs a batch of at least {N_TASK_ROWS} rows to exercise every gate row ({['none'] + list(bank.tasks)}), got B = {B}.")
        task_id = torch.arange(B, dtype=torch.long) % N_TASK_ROWS
    if table_mask is None:
        if T < 2:
            raise ValueError('the zero-gate probe needs at least 2 positions so the table mask can carry both a gated and an ungated token')
        table_mask = torch.zeros(B, T, dtype=torch.bool)
        table_mask[:, T // 2:] = True
    was_training = model.training
    model.eval()
    kw = {'input_ids': input_ids}
    if attention_mask is not None:
        kw['attention_mask'] = attention_mask
    try:
        prev = bank.enabled
        bank.enabled = False
        clear_gate_context()
        base = model(**kw).logits.float()
        bank.enabled = True
        bank.reset_stats()
        bank.collect = True
        set_gate_context(task_id, table_mask, valid_mask=attention_mask if attention_mask is not None else None)
        gated = model(**kw).logits.float()
    finally:
        bank.enabled = prev
        bank.collect = False
        clear_gate_context()
        if was_training:
            model.train()
    diff = float((gated - base).abs().max().item())
    stats = gate_activation_stats(model)
    overall = stats.get('overall', {})
    rows = sorted(set((int(t) for t in torch.as_tensor(task_id).tolist())))
    out = {'max_abs_logit_diff': diff, 'exact': diff <= atol, 'mean_g': overall.get('mean_g'), 'max_abs_g_minus_1': overall.get('max_abs_dev'), 'frac_tokens_gated': overall.get('frac_tokens_gated'), 'task_rows_exercised': rows, 'task_rows_expected': list(range(N_TASK_ROWS)), 'all_rows_exercised': rows == list(range(N_TASK_ROWS)), 'gate_collected': bool(stats.get('collected'))}
    if not out['all_rows_exercised']:
        raise RuntimeError(f"the zero-gate probe exercised rows {rows}, not every row {list(range(N_TASK_ROWS))} ({['none'] + list(bank.tasks)}).")
    if not out['gate_collected']:
        raise RuntimeError('the zero-gate probe collected no gate statistics: the gated modules never ran, so the identity it reports is vacuous.')
    if out['mean_g'] is None or out['mean_g'] != 1.0 or out['max_abs_g_minus_1'] != 0.0:
        raise RuntimeError(f"the zero-initialized gate is not exactly 1: mean(g) = {out['mean_g']}, max|g-1| = {out['max_abs_g_minus_1']}.")
    if not out['exact']:
        raise RuntimeError(f'the zero-initialized gate is NOT identical to ordinary LoRA: max|logit diff| = {diff:.3e} (tolerance {atol:g}).')
    return out

def _parse_layer(name):
    parts = name.split('.')
    for (a, b) in zip(parts, parts[1:]):
        if a == 'layers' and b.isdigit():
            return int(b)
    return None

def gate_activation_stats(model, per_module=False):
    bank = get_bank(model)
    row_name = {TASK_NONE: 'none'}
    row_name.update({TASK_INDEX[t]: t for t in bank.tasks})
    (groups, layers, total) = ({}, {}, blank_acc())
    add = merge_acc
    blank = blank_acc

    def finish(rec):
        n = rec['n']
        seen = rec['n_tokens_seen']
        if n == 0:
            return {'n_values': 0, 'n_tokens_gated': 0, 'n_tokens_seen': int(seen), 'frac_tokens_gated': rec['n_tokens_gated'] / seen if seen else None}
        rel_den = rec['sum_sq_vanilla'] ** 0.5
        return {'n_values': int(n), 'n_tokens_gated': int(rec['n_tokens_gated']), 'n_tokens_seen': int(seen), 'frac_tokens_gated': rec['n_tokens_gated'] / seen if seen else None, 'mean_g': rec['sum_g'] / n, 'mean_abs_dev': rec['sum_abs_dev'] / n, 'rms_dev': (rec['sum_sq_dev'] / n) ** 0.5, 'max_abs_dev': rec['max_abs_dev'], 'rank_space_rel_change': rec['sum_sq_gated_minus_vanilla'] ** 0.5 / rel_den if rel_den > 0 else None}
    per_mod = {}
    for ((i, row), rec) in bank.acc.items():
        name = bank.module_names[i]
        fam = name.split('.')[-1]
        key = f'{row_name.get(row, row)}_{fam}'
        add(groups.setdefault(key, blank()), rec)
        lay = _parse_layer(name)
        if lay is not None:
            add(layers.setdefault(key, {}).setdefault(lay, blank()), rec)
        add(total, rec)
        if per_module:
            add(per_mod.setdefault(f'{row_name.get(row, row)}|{name}', blank()), rec)
    out = {'collected': bool(bank.acc), 'by_group': {k: finish(v) for (k, v) in sorted(groups.items())}, 'by_group_layer': {k: {str(l): finish(v) for (l, v) in sorted(d.items())} for (k, d) in sorted(layers.items())}, 'overall': finish(total)}
    if per_module:
        out['by_module'] = {k: finish(v) for (k, v) in sorted(per_mod.items())}
    return out

@torch.no_grad()
def gate_weight_stats(model, init_snapshot=None):
    bank = get_bank(model)
    out = {}
    for (k, task) in enumerate(bank.tasks):
        (norms, gnorms, moves) = ([], [], [])
        for (i, w) in enumerate(bank.W):
            norms.append(float(w.data[k].norm()))
            if w.grad is not None:
                gnorms.append(float(w.grad[k].norm()))
            if init_snapshot is not None:
                moves.append(float((w.data[k].cpu() - init_snapshot[i][k]).norm()))
        rec = {'mean_fro': sum(norms) / len(norms), 'max_fro': max(norms), 'n_exactly_zero_modules': sum((1 for x in norms if x == 0.0))}
        if gnorms:
            rec['mean_grad_fro'] = sum(gnorms) / len(gnorms)
            rec['n_zero_grad_modules'] = sum((1 for g in gnorms if g == 0.0))
        if moves:
            rec['mean_move_from_init'] = sum(moves) / len(moves)
        out[task] = rec
    if len(bank.tasks) == 2:
        diffs = [float((w.data[0] - w.data[1]).norm()) for w in bank.W]
        base = [max(float(w.data[0].norm()), float(w.data[1].norm())) for w in bank.W]
        out['task_gap'] = {'mean_fro_diff': sum(diffs) / len(diffs), 'max_fro_diff': max(diffs), 'mean_rel_diff': sum((d / b for (d, b) in zip(diffs, base) if b > 0)) / max(1, sum((1 for b in base if b > 0)))}
    return out

def snapshot(model):
    return [w.detach().cpu().clone() for w in get_bank(model).W]

def save(model, out_dir, extra=None):
    bank = get_bank(model)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {n: w.detach().to('cpu', torch.float32) for (n, w) in zip(bank.module_names, bank.W)}
    torch.save(state, out / STATE_FILE)
    cfg = {'lora_rank_gate': 'kv', 'formula': GATE_FORMULA_SG, 'gate_stop_gradient': GATE_STOP_GRADIENT, 'stop_gradient_inputs': list(STOP_GRADIENT_INPUTS), 'exclude_target_tasks': list(bank.exclude_target_tasks), 'token_scope_by_task': bank.scope_by_task(), 'gate_tasks': list(bank.tasks), 'task_rows': {'none': TASK_NONE, **{t: TASK_INDEX[t] for t in bank.tasks}}, 'module_names': list(bank.module_names), 'lora_r': int(bank.r), 'n_gate_params': bank.n_params(), 'state_file': STATE_FILE}
    if extra:
        cfg.update(extra)
    with open(out / CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)
    return str(out / STATE_FILE)

def load(model, adapter_dir, strict=True):
    bank = get_bank(model)
    d = Path(adapter_dir)
    (st_path, cfg_path) = (d / STATE_FILE, d / CONFIG_FILE)
    if not st_path.exists():
        raise RuntimeError(f'{st_path} not found; this adapter carries no K/V rank gate.')
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    if cfg:
        _saved = cfg.get('gate_tasks')
        if _saved is not None and (not isinstance(_saved, (list, tuple))):
            raise RuntimeError(f'{adapter_dir}: gate_tasks is {_saved!r} ({type(_saved).__name__}), not a list of task names; this side-car is malformed.')
        saved_tasks = list(_saved) if _saved is not None else list(bank.tasks)
        if saved_tasks != list(bank.tasks):
            raise RuntimeError(f"this checkpoint was trained with the gate task order {saved_tasks}, but this code builds {list(bank.tasks)}; loading it would give every example the other task's gate.")
        saved_r = cfg.get('lora_r')
        if saved_r is not None:
            if isinstance(saved_r, (list, tuple, dict)):
                raise RuntimeError(f'{adapter_dir}: lora_r is {saved_r!r} ({type(saved_r).__name__}), not a number; this side-car is malformed.')
            if int(saved_r) != int(bank.r):
                raise RuntimeError(f"checkpoint LoRA rank {saved_r} != this model's {bank.r}")
        saved_ex = cfg.get('exclude_target_tasks')
        if saved_ex is not None:
            if not isinstance(saved_ex, (list, tuple)):
                raise RuntimeError(f'{adapter_dir}: exclude_target_tasks is {saved_ex!r} ({type(saved_ex).__name__}), not a list of task names; this side-car is malformed.')
            saved_ex = normalize_exclude_target(saved_ex)
            if saved_ex != bank.exclude_target_tasks:
                raise RuntimeError(f"this checkpoint was trained with exclude_target_tasks={list(saved_ex)} (token scope {cfg.get('token_scope_by_task')}), but the attached gate has {list(bank.exclude_target_tasks)}; serving it would apply the gate to a different set of tokens than it was trained on.")
    state = torch.load(st_path, map_location='cpu')
    missing = [n for n in bank.module_names if n not in state]
    unexpected = [n for n in state if n not in set(bank.module_names)]
    if strict and (missing or unexpected):
        raise RuntimeError(f'rank-gate checkpoint does not match this model: {len(missing)} missing (e.g. {missing[:2]}), {len(unexpected)} unexpected (e.g. {unexpected[:2]}).')
    with torch.no_grad():
        for (n, w) in zip(bank.module_names, bank.W):
            if n in state:
                t = state[n]
                if tuple(t.shape) != tuple(w.shape):
                    raise RuntimeError(f'gate {n}: checkpoint {tuple(t.shape)} != model {tuple(w.shape)}')
                w.copy_(t.to(w.device, w.dtype))
    return {'loaded': len(bank.module_names) - len(missing), 'missing': len(missing), 'trained_with_gate_stop_gradient': bool(cfg.get('gate_stop_gradient', False))}

def is_rank_gate_adapter(adapter_dir):
    if not adapter_dir:
        return False
    return (Path(adapter_dir) / STATE_FILE).exists()

def read_config(adapter_dir):
    if not adapter_dir:
        return None
    p = Path(adapter_dir) / CONFIG_FILE
    if not p.exists():
        return None
    return json.loads(p.read_text())
