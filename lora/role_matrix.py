import json
from pathlib import Path
import torch
import torch.nn as nn
import role_low_rank as RLR
ROLE_NONE = RLR.ROLE_NONE
SLOT_NAMES = ('cta_target', 'cta_context', 'cpa_target', 'cpa_context')
N_SLOTS = len(SLOT_NAMES)
SLOT_INDEX = {n: i for (i, n) in enumerate(SLOT_NAMES)}
ROLE_TO_SLOT = {RLR.ROLE_CTA_TARGET: SLOT_INDEX['cta_target'], RLR.ROLE_CTA_CONTEXT: SLOT_INDEX['cta_context'], RLR.ROLE_CPA_HEAD: SLOT_INDEX['cpa_target'], RLR.ROLE_CPA_TAIL: SLOT_INDEX['cpa_target'], RLR.ROLE_CPA_CONTEXT: SLOT_INDEX['cpa_context']}
SLOT_SOURCES = {n: tuple(sorted((rid for (rid, si) in ROLE_TO_SLOT.items() if si == i))) for (i, n) in enumerate(SLOT_NAMES)}
ROW_NONE = 0
ROW_OF_SLOT = {s: s + 1 for s in range(N_SLOTS)}
SLOT_OF_ROW = {s + 1: s for s in range(N_SLOTS)}
SWAP_SLOTS = {SLOT_INDEX['cta_target']: SLOT_INDEX['cta_context'], SLOT_INDEX['cta_context']: SLOT_INDEX['cta_target'], SLOT_INDEX['cpa_target']: SLOT_INDEX['cpa_context'], SLOT_INDEX['cpa_context']: SLOT_INDEX['cpa_target']}
DIAG_PAIRS = (('cta_target_minus_context', 'cta_target', 'cta_context'), ('cpa_target_minus_context', 'cpa_target', 'cpa_context'), ('cta_minus_cpa_target', 'cta_target', 'cpa_target'), ('cta_minus_cpa_context', 'cta_context', 'cpa_context'))
ROLE_MATRIX_MODULES = ('k_proj', 'v_proj', 'gate_proj')
TARGET_MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
PARAM_DTYPE = torch.float32
STATE_FILE = 'role_matrix.pt'
CONFIG_FILE = 'role_matrix_config.json'
EVAL_MODES = ('learned', 'identity', 'swap')
DEFAULT_MODE = 'learned'
FORMULA = "z = A(dropout(x)); z' = R_slot z  (R_cta_target | R_cta_context | R_cpa_target | R_cpa_context | I); delta = scaling * B(z')"

def role_rows(role_ids):
    r = torch.as_tensor(role_ids)
    out = torch.zeros_like(r, dtype=torch.long)
    for (rid, slot) in ROLE_TO_SLOT.items():
        out = torch.where(r == rid, torch.full_like(out, ROW_OF_SLOT[slot]), out)
    return out

class _Ctx:
    row_full = None
    cur_row = None
CTX = _Ctx()

def set_role_context(role_ids):
    rows = role_rows(role_ids)
    if rows.dim() != 2:
        raise ValueError(f'role ids must be [B, T], got {tuple(rows.shape)}')
    CTX.row_full = rows
    CTX.cur_row = None

def clear_role_context():
    CTX.row_full = None
    CTX.cur_row = None

def _row_prehook(module, args, kwargs):
    if CTX.row_full is None:
        CTX.cur_row = None
        return
    input_ids = kwargs.get('input_ids', None)
    if input_ids is None and args:
        input_ids = args[0]
    if input_ids is None:
        CTX.cur_row = CTX.row_full
        return
    seq_len = input_ids.shape[1]
    dev = CTX.row_full.device
    cp = kwargs.get('cache_position', None)
    cp = torch.arange(seq_len, device=dev) if cp is None else cp.to(dev)
    stored_T = CTX.row_full.shape[1]
    idx = cp.clamp(max=stored_T - 1)
    in_prompt = cp < stored_T
    CTX.cur_row = torch.where(in_prompt, CTX.row_full[:, idx], torch.full_like(CTX.row_full[:, idx], ROW_NONE))

class RoleMatrixBank(nn.Module):

    def __init__(self, module_names, r, mode=DEFAULT_MODE):
        super().__init__()
        self.module_names = list(module_names)
        self.r = int(r)
        self.n_modules = len(self.module_names)
        eye = torch.eye(self.r, dtype=PARAM_DTYPE)
        self.R = nn.Parameter(eye.expand(self.n_modules, N_SLOTS, self.r, self.r).clone())
        self.enabled = True
        self.set_mode(mode)

    def set_mode(self, mode):
        if mode not in EVAL_MODES:
            raise ValueError(f'role-matrix mode must be one of {list(EVAL_MODES)}, got {mode!r}')
        self.mode = mode
        return self.mode

    def n_params(self):
        return self.R.numel()

    def matrices(self, i, dtype, device):
        R = self.R[i].to(device=device, dtype=dtype)
        if self.mode == 'identity':
            return torch.eye(self.r, dtype=dtype, device=device).expand(N_SLOTS, self.r, self.r)
        if self.mode == 'swap':
            return R[[SWAP_SLOTS[s] for s in range(N_SLOTS)]]
        return R

    @torch.no_grad()
    def diagnostics(self, per_module=True):
        R = self.R.detach().to('cpu', torch.float64)
        eye = torch.eye(self.r, dtype=torch.float64)
        off = ~torch.eye(self.r, dtype=torch.bool)
        keys = [f'{n}_minus_I_fro' for n in SLOT_NAMES] + [f'{n}_diag_energy_frac' for n in SLOT_NAMES] + [f'{n}_fro' for (n, _a, _b) in DIAG_PAIRS]
        (per, agg) = ({}, {k: [] for k in keys})
        for (i, name) in enumerate(self.module_names):
            rec = {}
            for (si, tag) in enumerate(SLOT_NAMES):
                d = R[i, si] - eye
                diag_e = float((d.diagonal() ** 2).sum().item())
                off_e = float((d[off] ** 2).sum().item())
                tot = diag_e + off_e
                rec[f'{tag}_minus_I_fro'] = float(d.norm().item())
                rec[f'{tag}_diag_energy'] = diag_e
                rec[f'{tag}_offdiag_energy'] = off_e
                rec[f'{tag}_diag_energy_frac'] = diag_e / tot if tot > 0 else 0.0
            for (nm, a, b) in DIAG_PAIRS:
                rec[f'{nm}_fro'] = float((R[i, SLOT_INDEX[a]] - R[i, SLOT_INDEX[b]]).norm())
            for k in agg:
                agg[k].append(rec[k])
            if per_module:
                per[name] = rec
        summary = {f'{k}_mean': sum(v) / len(v) if v else 0.0 for (k, v) in agg.items()}
        summary.update({f'{k}_max': max(v) if v else 0.0 for (k, v) in agg.items()})
        return {'mode': self.mode, 'lora_r': self.r, 'n_modules': self.n_modules, 'n_params': self.n_params(), 'summary': summary, 'per_module': per if per_module else None}

    def summary(self):
        with torch.no_grad():
            eye = torch.eye(self.r, dtype=self.R.dtype, device=self.R.device)
            dev = (self.R.detach() - eye).abs()
            return {'n_modules': self.n_modules, 'lora_r': self.r, 'mode': self.mode, 'n_role_matrix_params': self.n_params(), 'slots': list(SLOT_NAMES), 'R_dev_from_identity_absmean': float(dev.mean().item()), 'R_dev_from_identity_absmax': float(dev.max().item())}

def _role_hook(bank, i):

    def hook(_module, _args, z):
        if not bank.enabled:
            return z
        rows = CTX.cur_row
        if rows is None:
            if CTX.row_full is None:
                raise RuntimeError('the role matrices are attached but no role context was set for this forward. Call role_matrix.set_role_context(role_ids) around every forward / generate; a silent fallback would quietly turn this run into the ordinary-LoRA baseline.')
            raise RuntimeError('the role matrices have role ids but no per-step slice for this forward. The cache_position pre-hook did not run. attach() registers it on BOTH the PEFT wrapper and the inner *ForCausalLM because neither alone covers both the training forward and generate().')
        if z.dim() != 3:
            raise RuntimeError(f'the role matrices need a [B, T, r] rank code, got {tuple(z.shape)}')
        (B, T, r) = z.shape
        if tuple(rows.shape) != (B, T):
            raise RuntimeError(f'role rows {tuple(rows.shape)} do not align with the rank code {(B, T)}; the token scope / cache_position slicing is misaligned.')
        if r != bank.r:
            raise RuntimeError(f'rank code width {r} != bank rank {bank.r}')
        rows = rows.to(z.device)
        R = bank.matrices(i, z.dtype, z.device)
        out = z
        for slot in range(N_SLOTS):
            zs = torch.matmul(z, R[slot].transpose(0, 1))
            out = torch.where((rows == ROW_OF_SLOT[slot]).unsqueeze(-1), zs, out)
        return out
    return hook

def _lora_rank(lora_a_dict):
    ranks = {sub.out_features for sub in lora_a_dict.values()}
    if len(ranks) != 1:
        raise RuntimeError(f'a LoRA module reports several ranks {sorted(ranks)}')
    return ranks.pop()

def find_role_modules(model, modules=ROLE_MATRIX_MODULES):
    allowed = set(modules)
    out = []
    for (name, m) in model.named_modules():
        la = getattr(m, 'lora_A', None)
        if not isinstance(la, nn.ModuleDict) or not len(la):
            continue
        if name.split('.')[-1] in allowed:
            out.append((name, m))
    return out

def attach(model, mode=DEFAULT_MODE):
    if getattr(model, 'role_matrix_bank', None) is not None:
        raise RuntimeError('role matrices are already attached to this model')
    found = find_role_modules(model)
    if not found:
        raise RuntimeError(f'--lora_role_matrix found no PEFT LoRA modules among {list(ROLE_MATRIX_MODULES)}. Is the model a PEFT LoRA model whose target_modules include them?')
    ranks = {_lora_rank(m.lora_A) for (_n, m) in found}
    if len(ranks) != 1:
        raise RuntimeError(f'the patched modules disagree on the LoRA rank: {sorted(ranks)}')
    r = ranks.pop()
    names = [n for (n, _m) in found]
    bank = RoleMatrixBank(names, r, mode=mode)
    bank.to(next(model.parameters()).device)
    model.role_matrix_bank = bank
    (handles, per_family) = ([], {})
    for (i, (name, m)) in enumerate(found):
        fam = name.split('.')[-1]
        per_family[fam] = per_family.get(fam, 0) + 1
        for (_adapter, sub) in m.lora_A.items():
            handles.append(sub.register_forward_hook(_role_hook(bank, i)))
    inner = model.get_base_model() if hasattr(model, 'get_base_model') else model
    hook_targets = [model] if inner is model else [model, inner]
    for t in hook_targets:
        handles.append(t.register_forward_pre_hook(_row_prehook, with_kwargs=True))
    model._role_matrix_handles = handles
    if len(set(per_family.values())) != 1:
        raise RuntimeError(f'uneven role-matrix coverage across families: {per_family}; every patched family must be covered in the same layers.')
    n_layers = max(per_family.values())
    return {'lora_role_matrix': True, 'families': sorted(per_family), 'n_modules': len(found), 'modules_per_family': per_family, 'n_layers': n_layers, 'lora_r': r, 'roles': list(SLOT_NAMES), 'role_sources': {n: [RLR.ROLE_NAMES[i] for i in SLOT_SOURCES[n]] for n in SLOT_NAMES}, 'bypassed_roles': [RLR.ROLE_NAMES[ROLE_NONE]], 'n_role_matrix_params': bank.n_params(), 'param_count_formula': f'n_layers * {len(ROLE_MATRIX_MODULES)} modules * {N_SLOTS} roles * r^2', 'param_shape': [len(found), N_SLOTS, r, r], 'identity_init': True, 'mode': bank.mode, 'eval_modes': list(EVAL_MODES), 'state_file': STATE_FILE, 'formula': FORMULA, 'role_source': 'role_low_rank role ids, collapsed to target/context/none', 'untouched_families': [f for f in TARGET_MODULES if f not in ROLE_MATRIX_MODULES], 'prehook_on': [type(t).__name__ for t in hook_targets]}

def detach(model):
    for h in getattr(model, '_role_matrix_handles', []) or []:
        h.remove()
    model._role_matrix_handles = []
    if getattr(model, 'role_matrix_bank', None) is not None:
        model.role_matrix_bank = None
    clear_role_context()

def get_bank(model):
    bank = getattr(model, 'role_matrix_bank', None)
    if bank is None:
        raise RuntimeError('no role matrices are attached to this model')
    return bank

def is_attached(model):
    return getattr(model, 'role_matrix_bank', None) is not None

def set_mode(model, mode):
    return get_bank(model).set_mode(mode)

def snapshot(model):
    return get_bank(model).R.detach().cpu().clone()

def save(model, out_dir, extra=None):
    bank = get_bank(model)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {name: bank.R[i].detach().to('cpu', torch.float32).clone() for (i, name) in enumerate(bank.module_names)}
    torch.save(state, out / STATE_FILE)
    cfg = {'lora_role_matrix': True, 'formula': FORMULA, 'module_names': list(bank.module_names), 'roles': list(SLOT_NAMES), 'role_sources': {n: [RLR.ROLE_NAMES[i] for i in SLOT_SOURCES[n]] for n in SLOT_NAMES}, 'lora_r': int(bank.r), 'n_role_matrix_params': bank.n_params(), 'identity_init': True, 'state_file': STATE_FILE}
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
        raise RuntimeError(f'{st_path} not found; this adapter carries no role matrices.')
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    saved_r = cfg.get('lora_r')
    if saved_r is not None:
        if isinstance(saved_r, (list, tuple, dict)):
            raise RuntimeError(f'{adapter_dir}: lora_r is {saved_r!r} ({type(saved_r).__name__}), not a number; this side-car is malformed.')
        if int(saved_r) != int(bank.r):
            raise RuntimeError(f"checkpoint LoRA rank {saved_r} != this model's {bank.r}")
    saved_roles = cfg.get('roles')
    if saved_roles is not None:
        if not isinstance(saved_roles, (list, tuple)):
            raise RuntimeError(f'{adapter_dir}: roles is {saved_roles!r} ({type(saved_roles).__name__}), not a list; this side-car is malformed.')
        if list(saved_roles) != list(SLOT_NAMES):
            raise RuntimeError(f"this checkpoint stores the role slots in the order {list(saved_roles)}, but this code builds {list(SLOT_NAMES)}; loading it would give every token the other role's matrix.")
    state = torch.load(st_path, map_location='cpu')
    missing = [n for n in bank.module_names if n not in state]
    unexpected = [n for n in state if n not in set(bank.module_names)]
    if strict and (missing or unexpected):
        raise RuntimeError(f'role-matrix checkpoint does not match this model: {len(missing)} missing (e.g. {missing[:2]}), {len(unexpected)} unexpected (e.g. {unexpected[:2]}).')
    with torch.no_grad():
        for (i, name) in enumerate(bank.module_names):
            if name in state:
                t = state[name]
                if tuple(t.shape) != tuple(bank.R[i].shape):
                    raise RuntimeError(f'role matrix {name}: checkpoint {tuple(t.shape)} != model {tuple(bank.R[i].shape)}')
                bank.R[i].copy_(t.to(bank.R.device, bank.R.dtype))
    return {'loaded': len(bank.module_names) - len(missing), 'missing': len(missing)}

def is_role_matrix_adapter(adapter_dir):
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
