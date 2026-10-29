import os
import json
import math
import re
from pathlib import Path
import torch
import torch.nn as nn
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
ROLE_MODULES = ['k_proj', 'v_proj']
ADAPTER_DTYPE = torch.float32
ROLE_NONE = 0
ROLE_CTA_CONTEXT = 1
ROLE_CTA_TARGET = 2
ROLE_CPA_CONTEXT = 3
ROLE_CPA_HEAD = 4
ROLE_CPA_TAIL = 5
ROLE_VECTOR_NAMES = ('cta_context', 'cta_target', 'cpa_context', 'cpa_head', 'cpa_tail')
N_ROLE_VECTORS = len(ROLE_VECTOR_NAMES)
N_ROLES = N_ROLE_VECTORS + 1
ROLE_NAMES = {ROLE_NONE: 'none'}
ROLE_NAMES.update({i + 1: n for (i, n) in enumerate(ROLE_VECTOR_NAMES)})
ROLE_IDS = {n: i for (i, n) in ROLE_NAMES.items()}
TASK_ROLES = {'cta': (ROLE_CTA_CONTEXT, ROLE_CTA_TARGET), 'cpa': (ROLE_CPA_CONTEXT, ROLE_CPA_HEAD, ROLE_CPA_TAIL)}
ROLE_TASKS = tuple(TASK_ROLES)
ROLE_RANK = {ROLE_NONE: 0}
ROLE_RANK.update({ROLE_IDS[n]: 1 if n.endswith('_context') else 2 for n in ROLE_VECTOR_NAMES})

def task_key(task):
    t = str(task).lower()
    if t.startswith('cta'):
        return 'cta'
    if t.startswith('cpa'):
        return 'cpa'
    raise ValueError(f'no role vocabulary for slug {task!r} (this pipeline covers CTA/CPA only)')

class _RoleCtx:
    role_full = None
    cur_role = None
CTX = _RoleCtx()

def set_role_ids(role_full):
    if role_full.dim() != 2:
        raise ValueError(f'role ids must be [B, T], got {tuple(role_full.shape)}')
    hi = int(role_full.max()) if role_full.numel() else 0
    lo = int(role_full.min()) if role_full.numel() else 0
    if hi >= N_ROLES or lo < 0:
        raise ValueError(f'role id out of range [0, {N_ROLES - 1}]: saw [{lo}, {hi}]')
    CTX.role_full = role_full
    CTX.cur_role = None

def clear_role_ids():
    CTX.role_full = None
    CTX.cur_role = None

def _role_prehook(module, args, kwargs):
    if CTX.role_full is None:
        CTX.cur_role = None
        return
    input_ids = kwargs.get('input_ids', None)
    if input_ids is None and args:
        input_ids = args[0]
    if input_ids is None:
        CTX.cur_role = CTX.role_full
        return
    seq_len = input_ids.shape[1]
    dev = CTX.role_full.device
    cp = kwargs.get('cache_position', None)
    cp = torch.arange(seq_len, device=dev) if cp is None else cp.to(dev)
    stored_T = CTX.role_full.shape[1]
    idx = cp.clamp(max=stored_T - 1)
    valid = (cp < stored_T).long()
    CTX.cur_role = CTX.role_full[:, idx] * valid

class RoleLowRankLinear(nn.Module):

    def __init__(self, base, r, alpha, dropout, use_role=True):
        super().__init__()
        self.use_role = bool(use_role)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        (in_f, out_f) = (base.in_features, base.out_features)
        self.in_features = in_f
        self.out_features = out_f
        self.r = r
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        dev = base.weight.device
        pdt = ADAPTER_DTYPE
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(dev, pdt)
        self.lora_B.to(dev, pdt)
        if self.use_role:
            self.role_vec = nn.Parameter(torch.zeros(N_ROLE_VECTORS, r, device=dev, dtype=pdt))
        else:
            self.register_parameter('role_vec', None)

    def role_table(self):
        if not self.use_role:
            raise RuntimeError('role_table() on a module patched without role vectors')
        zero = self.role_vec.new_zeros(1, self.role_vec.shape[1])
        return torch.cat([zero, self.role_vec], dim=0)

    def role_vector(self, name):
        return self.role_vec[ROLE_IDS[name] - 1]

    def forward(self, x):
        out = self.base(x)
        out_dtype = out.dtype
        z = self.lora_A(self.lora_dropout(x).to(self.lora_A.weight.dtype))
        role = CTX.cur_role
        if role is not None and self.use_role:
            if z.dim() != 3:
                raise RuntimeError(f'role offsets need a [B, T, r] activation, got {tuple(z.shape)}')
            if role.shape[0] != z.shape[0] or role.shape[1] != z.shape[1]:
                raise RuntimeError(f'role ids {tuple(role.shape)} do not match hidden {tuple(z.shape[:2])}; role context / cache_position misaligned')
            z = z + self.role_table().to(z.dtype)[role]
        return out + (self.lora_B(z) * self.scaling).to(out_dtype)

    def trainable_state_dict(self):
        sd = {'lora_A.weight': self.lora_A.weight.detach().cpu(), 'lora_B.weight': self.lora_B.weight.detach().cpu()}
        if self.use_role:
            sd['role_vec'] = self.role_vec.detach().cpu()
        return sd

    def load_trainable(self, sd):

        def _copy(dst, key):
            w = sd[key]
            if tuple(w.shape) != tuple(dst.shape):
                raise RuntimeError(f'shape mismatch for {key}: checkpoint {tuple(w.shape)} vs module {tuple(dst.shape)}')
            dst.data.copy_(w.to(dst.device, dst.dtype))
        _copy(self.lora_A.weight, 'lora_A.weight')
        _copy(self.lora_B.weight, 'lora_B.weight')
        if self.use_role:
            if 'role_vec' not in sd:
                raise RuntimeError('checkpoint has no role_vec for a module patched WITH role vectors; role_modules in the adapter config does not match this patch')
            _copy(self.role_vec, 'role_vec')
        elif 'role_vec' in sd:
            raise RuntimeError('checkpoint carries role_vec for a module patched WITHOUT role vectors; role_modules in the adapter config does not match this patch')

    def trainable_params(self):
        ps = [self.lora_A.weight, self.lora_B.weight]
        if self.use_role:
            ps.append(self.role_vec)
        return ps

def _set_submodule(root, dotted, new):
    parent = root
    parts = dotted.split('.')
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)
_QKV_MODULES = ('q_proj', 'k_proj', 'v_proj')

def init_from_struct_adapter(model, adapter_dir, r, alpha):
    import hashlib
    cfg_path = os.path.join(adapter_dir, 'adapter_config.json')
    if not os.path.exists(cfg_path):
        raise SystemExit(f'--init_qkv_adapter {adapter_dir} has no adapter_config.json')
    cfg = json.load(open(cfg_path))
    if cfg.get('peft_type') not in (None, 'LORA'):
        raise SystemExit(f"--init_qkv_adapter is peft_type={cfg.get('peft_type')}, not LORA.")
    src_mods = set(cfg.get('target_modules') or ())
    if not src_mods or not src_mods <= set(_QKV_MODULES):
        raise SystemExit(f'--init_qkv_adapter target_modules={sorted(src_mods)}; the structural stage must adapt only {list(_QKV_MODULES)}.')
    for (key, mine) in (('r', r), ('lora_alpha', alpha)):
        theirs = cfg.get(key)
        if theirs != mine:
            raise SystemExit(f'--init_qkv_adapter {key}={theirs} but this run uses {key}={mine}. LoRA applies scaling=lora_alpha/r, so copying A/B across different scaling changes the delta the structural weights encode. Match them or retrain stage 1.')
    st = os.path.join(adapter_dir, 'adapter_model.safetensors')
    if os.path.exists(st):
        from safetensors.torch import load_file
        sd = load_file(st)
    else:
        bin_path = os.path.join(adapter_dir, 'adapter_model.bin')
        if not os.path.exists(bin_path):
            raise SystemExit(f'--init_qkv_adapter {adapter_dir} has no adapter_model.safetensors/.bin')
        sd = torch.load(bin_path, map_location='cpu')
    patched = {n: m for (n, m) in model.named_modules() if isinstance(m, RoleLowRankLinear)}
    (plan, copied, seen_B, nonzero_B) = ([], [], 0, 0)
    for (k, v) in sd.items():
        if 'lora_A' not in k and 'lora_B' not in k:
            raise SystemExit(f'--init_qkv_adapter holds a non-LoRA tensor {k}; refusing.')
        if not any((f'.{m}.' in k for m in _QKV_MODULES)):
            raise SystemExit(f'--init_qkv_adapter holds a non-QKV tensor {k}; refusing.')
        which = 'lora_A' if 'lora_A' in k else 'lora_B'
        path = k.split(f'.{which}.')[0]
        for pre in ('base_model.model.', 'base_model.'):
            if path.startswith(pre):
                path = path[len(pre):]
                break
        mod = patched.get(path)
        if mod is None:
            raise SystemExit(f'--init_qkv_adapter tensor {k} has no counterpart in the downstream model (resolved module path {path!r} is not a patched RoleLowRankLinear).')
        dst = getattr(mod, which).weight
        if tuple(dst.shape) != tuple(v.shape):
            raise SystemExit(f'--init_qkv_adapter {k}: shape {tuple(v.shape)} != downstream {tuple(dst.shape)}.')
        plan.append((dst, v))
        copied.append(f'{path}.{which}.weight')
        if which == 'lora_B':
            seen_B += 1
            nonzero_B += int(bool(v.count_nonzero().item()))
    if seen_B == 0:
        raise SystemExit('--init_qkv_adapter contains no lora_B tensors; refusing.')
    if nonzero_B == 0:
        raise SystemExit("--init_qkv_adapter has ALL-ZERO lora_B: that is PEFT's fresh initialization, i.e. an adapter that never trained. Refusing.")
    want = {f'{n}.{w}.weight' for n in patched for w in ('lora_A', 'lora_B') if any((n.endswith(f'.{m}') for m in src_mods))}
    missing = sorted(want - set(copied))
    if missing:
        raise SystemExit(f'{len(missing)} LoRA tensors of {sorted(src_mods)} were NOT initialized from the structural adapter (e.g. {missing[:3]}); they would still hold the fresh initialization.')
    with torch.no_grad():
        for (dst, v) in plan:
            dst.copy_(v.to(dst.dtype))
    sha = hashlib.sha256()
    for k in sorted(sd):
        sha.update(k.encode())
        sha.update(sd[k].float().cpu().numpy().tobytes())
    info = {'init_qkv_adapter': os.path.abspath(adapter_dir), 'structural_target_modules': sorted(src_mods), 'tensors_copied': len(copied), 'lora_B_tensors': seen_B, 'lora_B_nonzero': nonzero_B, 'structural_weights_sha256': sha.hexdigest(), 'structural_r': cfg.get('r'), 'structural_lora_alpha': cfg.get('lora_alpha'), 'role_vectors_initialized_from_stage1': False}
    print(f'[init_qkv_adapter] copied {len(copied)} tensors from {adapter_dir} ({seen_B} lora_B, {nonzero_B} of them non-zero) into {sorted(src_mods)}; every other module keeps its fresh LoRA init and the five role vectors stay at zero.')
    return info

def patch_model(model, r, alpha, dropout, target_modules=None, role_modules=None):
    mods = list(target_modules or TARGET_MODULES)
    rmods = list(ROLE_MODULES if role_modules is None else role_modules)
    unknown = [m for m in rmods if m not in mods]
    if unknown:
        raise ValueError(f'role_modules {unknown} are not in target_modules {mods}')
    targets = [(name, m) for (name, m) in model.named_modules() if isinstance(m, nn.Linear) and name.split('.')[-1] in mods]
    (patched, role_patched) = ([], [])
    for (name, m) in targets:
        use_role = name.split('.')[-1] in rmods
        _set_submodule(model, name, RoleLowRankLinear(m, r, alpha, dropout, use_role=use_role))
        patched.append(name)
        if use_role:
            role_patched.append(name)
    if not patched:
        raise RuntimeError(f'no modules matched target_modules={mods}')
    if not role_patched:
        raise RuntimeError(f'no modules matched role_modules={rmods}')
    handle = model.register_forward_pre_hook(_role_prehook, with_kwargs=True)
    model._role_low_rank_hook_handle = handle
    model._role_low_rank_role_modules = rmods
    return {'patched': patched, 'target_modules': mods, 'role_patched': role_patched, 'role_modules': rmods, 'role_vector_names': list(ROLE_VECTOR_NAMES), 'n_role_vectors_per_module': N_ROLE_VECTORS}

def trainable_parameters(model):
    params = []
    for m in model.modules():
        if isinstance(m, RoleLowRankLinear):
            params.extend(m.trainable_params())
    return params

def count_trainable(model):
    return sum((p.numel() for p in trainable_parameters(model)))

def param_report(model, r):
    n_mod = 0
    n_role_mod = 0
    lora_n = 0
    role_n = 0
    for m in model.modules():
        if isinstance(m, RoleLowRankLinear):
            n_mod += 1
            lora_n += m.lora_A.weight.numel() + m.lora_B.weight.numel()
            if m.use_role:
                n_role_mod += 1
                role_n += m.role_vec.numel()
    measured = sum((p.numel() for p in model.parameters() if p.requires_grad))
    return {'n_patched_modules': n_mod, 'n_role_modules': n_role_mod, 'role_vector_names': list(ROLE_VECTOR_NAMES), 'n_role_vectors_per_module': N_ROLE_VECTORS, 'role_dim': r, 'lora_params': lora_n, 'role_vector_params': role_n, 'role_params_per_module': N_ROLE_VECTORS * r, 'total_trainable': lora_n + role_n, 'measured_trainable_params': measured, 'role_share_of_trainable': role_n / max(1, lora_n + role_n), 'vanilla_equivalent_lora_params': lora_n}

def architecture_summary(model, r, alpha, dropout):
    rep = param_report(model, r)
    rmods = list(getattr(model, '_role_low_rank_role_modules', None) or ROLE_MODULES)
    lines = [f'[role-low-rank] delta_h = scaling * B ( A x + e_role ),  scaling = alpha/r = {alpha}/{r} = {alpha / r:g}', f'[role-low-rank] roles: none(0)=fixed zero, ' + ', '.join((f'{n}({ROLE_IDS[n]})' for n in ROLE_VECTOR_NAMES)), f'[role-low-rank] {N_ROLE_VECTORS} learned vectors in R^{r} per role module, zero-init (== vanilla LoRA at init)', '[role-low-rank] CTA sees {context, target}; CPA sees {context, head, tail} -- the CPA direction is NOT collapsed', '[role-low-rank] A and B are SHARED across tasks; the role offsets ride the SAME B (no separate up-projection)', f"[role-low-rank] patched {rep['n_patched_modules']} modules {TARGET_MODULES}", f"[role-low-rank] role vectors on {rep['n_role_modules']}/{rep['n_patched_modules']} modules {rmods} ONLY; the rest are plain shared LoRA", f"[role-low-rank] trainable: {rep['total_trainable']:,} (lora {rep['lora_params']:,} + role vectors {rep['role_vector_params']:,}, role share {rep['role_share_of_trainable'] * 100:.3f}%)"]
    return ('\n'.join(lines), rep)

def role_architecture_facts(model):
    role_mods = [(n, m) for (n, m) in model.named_modules() if isinstance(m, RoleLowRankLinear) and m.use_role]
    all_mods = [(n, m) for (n, m) in model.named_modules() if isinstance(m, RoleLowRankLinear)]
    if not role_mods:
        return {'has_roles': False, 'n_patched': len(all_mods)}
    (probe_name, probe) = role_mods[0]
    tbl = probe.role_table()
    if tbl.device.type == 'meta' or probe.lora_B.weight.data_ptr() == 0:
        raise RuntimeError('cannot verify the role architecture: the parameters live on a meta/unallocated device, where every tensor reports the same address. Materialise the model before asking what it is.')
    shared_b = all((m.role_table().shape[1] == m.lora_B.weight.shape[1] for (_, m) in role_mods))
    none_is_zero = all((bool(torch.count_nonzero(m.role_table()[ROLE_NONE]) == 0) for (_, m) in role_mods))
    same_count = {tuple(m.role_vec.shape) for (_, m) in role_mods}
    return {'has_roles': True, 'role_vector_names': list(ROLE_VECTOR_NAMES), 'role_ids': {n: ROLE_IDS[n] for n in ROLE_VECTOR_NAMES}, 'task_roles': {t: [ROLE_NAMES[i] for i in ids] for (t, ids) in TASK_ROLES.items()}, 'n_role_vectors_per_module': probe.role_vec.shape[0], 'role_dim': probe.role_vec.shape[1], 'lora_r': probe.r, 'scaling': probe.scaling, 'n_patched': len(all_mods), 'n_role_modules': len(role_mods), 'role_modules': sorted({n.rsplit('.', 1)[-1] for (n, _) in role_mods}), 'offset_rides_the_lora_b': bool(shared_b), 'none_row_is_exactly_zero': bool(none_is_zero), 'none_row_is_a_parameter': any((m.role_vec.shape[0] == N_ROLES for (_, m) in role_mods)), 'consistent_across_role_modules': len(same_count) == 1, 'probe_module': probe_name}

def format_role_architecture(facts):
    if not facts.get('has_roles'):
        return '[arch] plain shared LoRA on %d modules; no role branch.' % facts.get('n_patched', 0)
    L = ['[arch] ---------- what this model actually is (read off the live tensors) ----------']
    L.append('[arch] update rule:  delta_h = %g * B ( A x  +  e[role] )' % facts['scaling'])
    L.append('[arch]   the role offset is added to A x, so it goes through the SAME B as LoRA;')
    L.append('[arch]   there is no separate role up-projection.')
    L.append('[arch] LoRA rank r = %d;  role vector width = %d (they share the rank space)' % (facts['lora_r'], facts['role_dim']))
    L.append('[arch] LoRA A/B on all %d patched modules; the role branch on %d of them (%s).' % (facts['n_patched'], facts['n_role_modules'], ', '.join(facts['role_modules'])))
    L.append('[arch] the role vocabulary is TASK-SHAPED -- %d learned vectors per role module:' % facts['n_role_vectors_per_module'])
    for (task, names) in facts['task_roles'].items():
        L.append('[arch]     %-3s -> %s' % (task.upper(), ', '.join(('%s (id %d)' % (n, facts['role_ids'][n]) for n in names))))
    L.append('[arch]     none (id 0) -> a FIXED ZERO row, shared by both tasks, NOT a parameter')
    L.append('[arch]   -> CPA keeps head and tail apart; they are NOT collapsed into one target.')
    L.append('[arch]   -> the id names its own task, so no task index is used anywhere.')
    L.append('[arch] checks on %s:' % facts['probe_module'])
    L.append('[arch]   role offset shares the LoRA B          : %s' % ('YES' if facts['offset_rides_the_lora_b'] else 'NO -- WRONG'))
    L.append('[arch]   e_none is exactly zero                  : %s' % ('YES' if facts['none_row_is_exactly_zero'] else 'NO -- WRONG'))
    L.append('[arch]   e_none is a trainable parameter         : %s' % ('YES -- WRONG' if facts['none_row_is_a_parameter'] else 'NO (correct)'))
    L.append('[arch]   all %d role modules agree in shape      : %s' % (facts['n_role_modules'], 'YES' if facts['consistent_across_role_modules'] else 'NO -- WRONG'))
    return '\n'.join(L)

def assert_role_architecture(model, expect_role_modules=None):
    facts = role_architecture_facts(model)
    if not facts.get('has_roles'):
        raise RuntimeError('no role modules were patched, but a role architecture was requested')
    problems = []
    if facts['n_role_vectors_per_module'] != N_ROLE_VECTORS:
        problems.append('expected %d learned role vectors per role module, found %d' % (N_ROLE_VECTORS, facts['n_role_vectors_per_module']))
    if facts['role_dim'] != facts['lora_r']:
        problems.append('role width %d != LoRA rank %d; with the shared B the offset is added to A x and must live in the rank space' % (facts['role_dim'], facts['lora_r']))
    if not facts['offset_rides_the_lora_b']:
        problems.append('the role offset does not ride the LoRA up-projection B')
    if not facts['none_row_is_exactly_zero']:
        problems.append('e_none is not exactly zero')
    if facts['none_row_is_a_parameter']:
        problems.append('e_none is a trainable parameter; it must be a fixed zero row')
    want = list(ROLE_MODULES if expect_role_modules is None else expect_role_modules)
    if sorted(facts['role_modules']) != sorted(want):
        problems.append('role modules are %s, expected %s' % (facts['role_modules'], want))
    if not facts['consistent_across_role_modules']:
        problems.append('role modules do not all carry the same role-vector shape')
    if problems:
        raise RuntimeError('the patched model does NOT match the requested architecture:\n  - ' + '\n  - '.join(problems) + '\n\n' + format_role_architecture(facts))
    return facts

def format_param_arithmetic(model):
    (lora_groups, role_groups) = ({}, {})
    for (name, m) in model.named_modules():
        if not isinstance(m, RoleLowRankLinear):
            continue
        leaf = name.rsplit('.', 1)[-1]
        lora_groups.setdefault((leaf, m.in_features, m.out_features, m.r), [0])[0] += 1
        if m.use_role:
            role_groups.setdefault(tuple(m.role_vec.shape), [0])[0] += 1
    L = ['[params] ---------- where every trainable parameter comes from ----------']
    total = 0
    L.append('[params] LoRA A and B (shared across tasks), r x (in + out) per module:')
    for ((leaf, in_f, out_f, r), (cnt,)) in sorted(lora_groups.items()):
        n = cnt * r * (in_f + out_f)
        total += n
        L.append('[params]   %3d x %-10s  %d x (%d + %d) = %s each   -> %s' % (cnt, leaf, r, in_f, out_f, format(r * (in_f + out_f), ','), format(n, ',')))
    for ((n_vec, d), (cnt,)) in sorted(role_groups.items()):
        n = cnt * n_vec * d
        total += n
        L.append('[params] role vectors, %d learned roles x width %d per role module:' % (n_vec, d))
        L.append('[params]   %3d role modules x %d roles (%s) x %d -> %s' % (cnt, n_vec, ', '.join(ROLE_VECTOR_NAMES), d, format(n, ',')))
        L.append('[params]   (e_none is a fixed zero row and costs nothing)')
    measured = sum((p.numel() for p in model.parameters() if p.requires_grad))
    L.append('[params] ' + '-' * 72)
    L.append('[params] the lines above add up to            %s' % format(total, ','))
    L.append('[params] torch reports trainable              %s' % format(measured, ','))
    L.append('[params] %s' % ('MATCH -- the count is explained, nothing else is training.' if total == measured else 'MISMATCH of %s -- something outside these terms requires grad!' % format(measured - total, ',')))
    if total != measured:
        raise RuntimeError('trainable parameter count %s is not explained by the architecture (accounted %s)' % (format(measured, ','), format(total, ',')))
    return '\n'.join(L)

def _norm_stats(vals):
    if not vals:
        return {}
    return {'count': len(vals), 'mean': sum(vals) / len(vals), 'min': min(vals), 'max': max(vals)}
ROLE_DIFF_PAIRS = [('cta_target', 'cta_context'), ('cpa_head', 'cpa_context'), ('cpa_tail', 'cpa_context'), ('cpa_head', 'cpa_tail'), ('cta_target', 'cpa_head'), ('cta_context', 'cpa_context')]

def role_separation_summary(model, per_module=False):
    norms = {n: [] for n in ROLE_VECTOR_NAMES}
    diffs = {f'{a}|{b}': {'e_diff': [], 'b_e_diff': []} for (a, b) in ROLE_DIFF_PAIRS}
    per = {}
    for (name, m) in model.named_modules():
        if not isinstance(m, RoleLowRankLinear) or not m.use_role:
            continue
        with torch.no_grad():
            v = m.role_vec.detach().float()
            b = m.lora_B.weight.detach().float()
            d = {}
            for (i, n) in enumerate(ROLE_VECTOR_NAMES):
                val = float(v[i].norm())
                norms[n].append(val)
                d[n] = val
            for (a, bb) in ROLE_DIFF_PAIRS:
                dv = v[ROLE_IDS[a] - 1] - v[ROLE_IDS[bb] - 1]
                vals = {'e_diff': float(dv.norm()), 'b_e_diff': float((b @ dv).norm())}
                for (k, val) in vals.items():
                    diffs[f'{a}|{bb}'][k].append(val)
                d[f'{a}|{bb}'] = vals
        if per_module:
            per[name] = d
    out = {'roles': list(ROLE_VECTOR_NAMES), 'role_norms': {n: _norm_stats(v) for (n, v) in norms.items()}, 'role_pairs': {k: {kk: _norm_stats(vv) for (kk, vv) in v.items()} for (k, v) in diffs.items()}}
    if per_module:
        out['per_module'] = per
    return out

def format_role_separation(sep):
    lines = []
    for (n, st) in sep['role_norms'].items():
        if st:
            lines.append(f"[role-sep] ||e_{n:<12}||      mean={st['mean']:.6g} min={st['min']:.6g} max={st['max']:.6g} (n={st['count']})")
    for (pair, d) in sep['role_pairs'].items():
        for k in ('e_diff', 'b_e_diff'):
            st = d[k]
            if st:
                lines.append(f"[role-sep] {pair:<25} {k:<9} mean={st['mean']:.6g} min={st['min']:.6g} max={st['max']:.6g} (n={st['count']})")
    return '\n'.join(lines)

def gather_trainable_state(model):
    state = {}
    for (name, m) in model.named_modules():
        if isinstance(m, RoleLowRankLinear):
            for (k, v) in m.trainable_state_dict().items():
                state[f'{name}.{k}'] = v
    return state

def scatter_trainable_state(model, state):
    by_mod = {}
    for (full_k, v) in state.items():
        if full_k.endswith('.role_vec'):
            (mod_name, leaf) = (full_k[:-len('.role_vec')], 'role_vec')
        else:
            mod_name = full_k.rsplit('.', 2)[0]
            leaf = full_k[len(mod_name) + 1:]
        by_mod.setdefault(mod_name, {})[leaf] = v
    for (name, m) in model.named_modules():
        if isinstance(m, RoleLowRankLinear) and name in by_mod:
            m.load_trainable(by_mod[name])

def save_role_adapter(model, out_dir, config):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(gather_trainable_state(model), out / 'role_low_rank.pt')
    with open(out / 'role_low_rank_config.json', 'w') as f:
        json.dump(config, f, indent=2)

def load_role_adapter(model, adapter_dir):
    adapter_dir = Path(adapter_dir)
    cfg_path = adapter_dir / 'role_low_rank_config.json'
    if not cfg_path.exists():
        raise RuntimeError(f'{cfg_path} not found. Role-low-rank adapters are not vanilla PEFT adapters; the patch config must exist and be applied before loading.')
    cfg = json.loads(cfg_path.read_text())
    names = cfg.get('role_vector_names')
    if names is not None and list(names) != list(ROLE_VECTOR_NAMES):
        raise RuntimeError(f'this adapter was trained with the role vocabulary {list(names)}, but this code builds {list(ROLE_VECTOR_NAMES)}. Loading it would attach the wrong vector to every token; re-train, or check out the revision that wrote it.')
    if names is None:
        raise RuntimeError(f'{cfg_path} records no `role_vector_names`, so it predates the task-shaped role vocabulary (CTA{{context,target}} + CPA{{context,head,tail}}). Its two-role-per-task vectors cannot be mapped onto the five-role table; re-train under this design.')
    patch_model(model, cfg['lora_r'], cfg['lora_alpha'], cfg.get('lora_dropout', 0.0), target_modules=cfg.get('target_modules'), role_modules=cfg.get('role_modules'))
    state = torch.load(adapter_dir / 'role_low_rank.pt', map_location='cpu')
    scatter_trainable_state(model, state)
    return cfg

class RoleSpec:
    __slots__ = ('task', 'name_to_role', 'context_role', 'allowed')

    def __init__(self, task, name_to_role, context_role):
        self.task = task
        self.name_to_role = dict(name_to_role)
        self.context_role = context_role
        self.allowed = frozenset((ROLE_NONE, context_role, *name_to_role.values()))

    def __repr__(self):
        return 'RoleSpec(task=%r, designated=%s, context=%s)' % (self.task, {k: ROLE_NAMES[v] for (k, v) in sorted(self.name_to_role.items())}, ROLE_NAMES[self.context_role])
_CTA_TARGET_RE = re.compile('Target entity column:\\s*(\\S+)')
_CPA_HEAD_RE = re.compile('Head column:\\s*(\\S+)')
_CPA_TAIL_RE = re.compile('Tail column:\\s*(\\S+)')

def role_spec(user_prompt, task):
    t = task_key(task)
    if t == 'cta':
        m = _CTA_TARGET_RE.search(user_prompt)
        names = {m.group(1).strip(): ROLE_CTA_TARGET} if m else {}
        return RoleSpec('cta', names, ROLE_CTA_CONTEXT)
    head = _CPA_HEAD_RE.search(user_prompt)
    tail = _CPA_TAIL_RE.search(user_prompt)
    names = {}
    if head:
        names[head.group(1).strip()] = ROLE_CPA_HEAD
    if tail:
        tn = tail.group(1).strip()
        if tn in names:
            raise ValueError(f'CPA prompt declares column {tn!r} as BOTH head and tail; the direction this experiment tests is undefined for it')
        names[tn] = ROLE_CPA_TAIL
    return RoleSpec('cpa', names, ROLE_CPA_CONTEXT)

def _is_separator_line(line):
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    return len(cells) > 0 and all((c != '' and set(c) <= set('-: ') for c in cells))

def scan_table_columns(text):
    n = len(text)
    col = [0] * n
    content = [False] * n
    lines = text.splitlines(keepends=True)
    (offs, p) = ([], 0)
    for ln in lines:
        offs.append(p)
        p += len(ln)
    is_tbl = [ln.strip().startswith('|') for ln in lines]
    header_cell_text = {}
    content_cols = set()
    i = 0
    while i < len(lines):
        if not is_tbl[i]:
            i += 1
            continue
        j = i
        while j < len(lines) and is_tbl[j]:
            j += 1
        row_ctr = 0
        for k in range(i, j):
            (ln, base) = (lines[k], offs[k])
            sep = _is_separator_line(ln)
            if not sep:
                row_ctr += 1
            is_header_row = not sep and row_ctr == 1
            c_idx = 0
            for (t, ch) in enumerate(ln):
                g = base + t
                if ch == '\n':
                    continue
                if ch == '|':
                    col[g] = max(1, c_idx)
                    c_idx += 1
                    continue
                if sep:
                    col[g] = max(1, c_idx)
                    continue
                col[g] = c_idx
                if is_header_row and c_idx >= 1:
                    header_cell_text.setdefault(c_idx, []).append(ch)
                if c_idx >= 1 and (not ch.isspace()):
                    content[g] = True
                    content_cols.add(c_idx)
        i = j
    headers = {c: ''.join(chars).strip() for (c, chars) in header_cell_text.items()}
    return (col, content, headers, content_cols)

def column_role_map(text, spec):
    (_, _, headers, content_cols) = scan_table_columns(text)
    out = {}
    for c in sorted(content_cols):
        h = headers.get(c, '')
        out[c] = (h, spec.name_to_role.get(h, spec.context_role))
    return out

def format_column_roles(text, spec, max_cols=24):
    cmap = column_role_map(text, spec)
    designated = {v: k for (k, v) in spec.name_to_role.items()}
    L = ['[roles] task=%s  designated: %s' % (spec.task.upper(), ', '.join(('%s -> %s' % (ROLE_NAMES[r], designated[r]) for r in sorted(designated))) or '(none declared)')]
    for (c, (h, rid)) in list(cmap.items())[:max_cols]:
        why = 'declared %s' % ROLE_NAMES[rid].split('_', 1)[1] if h in spec.name_to_role else 'not declared -> context'
        L.append('[roles]   column %-2d  header %-14s -> %-12s (%s)' % (c, h or '(empty)', ROLE_NAMES[rid], why))
    if len(cmap) > max_cols:
        L.append('[roles]   ... %d more columns' % (len(cmap) - max_cols))
    missing = [n for n in spec.name_to_role if n not in {h for (h, _) in cmap.values()}]
    if missing:
        L.append('[roles]   !! declared column(s) %s are not header cells of this table; they carry no role' % missing)
    return '\n'.join(L)

def assert_column_roles(text, spec, expected):
    got = {c: ROLE_NAMES[r] for (c, (_, r)) in column_role_map(text, spec).items()}
    want = {int(c): n for (c, n) in expected.items()}
    if got != want:
        raise AssertionError('column roles differ:\n  expected %s\n  got      %s' % (want, got))
    return got

def char_role_ids(text, spec):
    (col, content, headers, content_cols) = scan_table_columns(text)
    col_role = {}
    for c in content_cols:
        col_role[c] = spec.name_to_role.get(headers.get(c, ''), spec.context_role)
    role = [ROLE_NONE] * len(text)
    for g in range(len(text)):
        if content[g]:
            r = col_role.get(col[g])
            if r is not None:
                role[g] = r
    return role

def _span_role(span):
    (best, best_rank) = (ROLE_NONE, 0)
    for r in span:
        k = ROLE_RANK[r]
        if k > best_rank:
            (best, best_rank) = (r, k)
    return best

def tokens_role_ids(tok, text, spec):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    (ids, offs) = (enc['input_ids'], enc['offset_mapping'])
    cmap = char_role_ids(text, spec)
    roles = [_span_role(cmap[a:b]) if b > a else ROLE_NONE for (a, b) in offs]
    bad = set(roles) - spec.allowed
    if bad:
        raise RuntimeError('role construction produced ids %s that task %r may not emit (allowed %s)' % (sorted((ROLE_NAMES[b] for b in bad)), spec.task, sorted((ROLE_NAMES[a] for a in spec.allowed))))
    return (ids, roles)
_SWAP_ALL = {ROLE_NONE: ROLE_NONE, ROLE_CTA_CONTEXT: ROLE_CTA_TARGET, ROLE_CTA_TARGET: ROLE_CTA_CONTEXT, ROLE_CPA_CONTEXT: ROLE_CPA_HEAD, ROLE_CPA_HEAD: ROLE_CPA_TAIL, ROLE_CPA_TAIL: ROLE_CPA_CONTEXT}
_SWAP_HEAD_TAIL = {r: r for r in ROLE_NAMES}
_SWAP_HEAD_TAIL[ROLE_CPA_HEAD] = ROLE_CPA_TAIL
_SWAP_HEAD_TAIL[ROLE_CPA_TAIL] = ROLE_CPA_HEAD
ROLE_PERMUTATIONS = {'role_swap': _SWAP_ALL, 'head_tail_swap': _SWAP_HEAD_TAIL}

def permute_roles(role_ids, how):
    if how not in ROLE_PERMUTATIONS:
        raise ValueError(f'unknown role permutation {how!r} (expected one of {sorted(ROLE_PERMUTATIONS)})')
    table = ROLE_PERMUTATIONS[how]
    return [table[r] for r in role_ids]

def role_id_counts(role_ids):
    return {name: sum((1 for r in role_ids if r == rid)) for (rid, name) in ROLE_NAMES.items()}
