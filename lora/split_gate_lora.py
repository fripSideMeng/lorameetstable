import json
import math
import os
import torch
import torch.nn as nn
import recv_gate as RCV
MECHANISM = 'split_gate_proj'
VERSION = 2
STATE_FILE = 'split_gate_lora.pt'
CONFIG_FILE = 'split_gate_lora.json'
SPLIT_MODULES = ('gate_proj', 'up_proj', 'down_proj')
BRANCHES = ('table', 'nontable')
FORMULA = 'y = W0 x + s * B_table(A_table(drop(x))) * m + s * B_nontable(A_nontable(drop(x))) * (1 - m), s = alpha_split / r_split, m = table-token mask'
TOKEN_SCOPE = 'm = 1 on tokens whose region id is recv_gate.R_TABLE (the Markdown table block found by find_markdown_table_span); m = 0 on pre-table, post-table, padding and every generated token'

class _Ctx:
    active = False
    mask_full = None
    cur = None
    modules = ()
    swap = False
CTX = _Ctx()

def set_table_mask(mask):
    m = torch.as_tensor(mask).bool()
    if m.dim() != 2:
        raise ValueError(f'table mask must be [B,T]; got {tuple(m.shape)}')
    CTX.mask_full = m
    CTX.cur = None
    CTX.active = True

def set_table_mask_from_regions(region_ids):
    r = torch.as_tensor(region_ids)
    set_table_mask(r == RCV.R_TABLE)
    return CTX.mask_full

def clear_table_mask():
    CTX.active = False
    CTX.mask_full = None
    CTX.cur = None

def set_swap(on):
    CTX.swap = bool(on)
    return CTX.swap

def swap_enabled():
    return bool(CTX.swap)

def _slice_prehook(module, args, kwargs):
    if not CTX.active or CTX.mask_full is None:
        return None
    cp = kwargs.get('cache_position', None)
    ids = kwargs.get('input_ids', None)
    if ids is None and args:
        ids = args[0]
    T = CTX.mask_full.shape[1]
    if cp is None:
        n = int(ids.shape[1]) if ids is not None else T
        cp = torch.arange(n, device=CTX.mask_full.device)
    cp = cp.to(CTX.mask_full.device).long()
    idx = cp.clamp(max=T - 1)
    CTX.cur = CTX.mask_full[:, idx] & (cp < T)
    return None

class SplitGateLoRA(nn.Module):

    def __init__(self, in_features, out_features, r, alpha, dropout, dtype=None, device=None):
        super().__init__()
        if r <= 0:
            raise ValueError(f'r must be positive; got {r}')
        (self.in_features, self.out_features) = (int(in_features), int(out_features))
        (self.r, self.alpha) = (int(r), float(alpha))
        self.scaling = float(alpha) / float(r)
        self.split_drop = nn.ModuleDict({b: nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity() for b in BRANCHES})
        self.split_A = nn.ModuleDict({b: nn.Linear(self.in_features, self.r, bias=False) for b in BRANCHES})
        self.split_B = nn.ModuleDict({b: nn.Linear(self.r, self.out_features, bias=False) for b in BRANCHES})
        for b in BRANCHES:
            nn.init.kaiming_uniform_(self.split_A[b].weight, a=math.sqrt(5))
            nn.init.zeros_(self.split_B[b].weight)
        if dtype is not None or device is not None:
            self.to(device=device, dtype=dtype)

    def n_params(self):
        return sum((p.numel() for p in self.parameters()))

    def delta(self, x, table_mask):
        w_dtype = self.split_A[BRANCHES[0]].weight.dtype
        xd = x.to(w_dtype)
        m = table_mask.to(device=x.device).unsqueeze(-1)
        out = None
        for (b, keep) in (('table', m), ('nontable', ~m)):
            d = self.split_B[b](self.split_A[b](self.split_drop[b](xd)))
            d = d * keep.to(d.dtype)
            out = d if out is None else out + d
        return out * self.scaling

def _hook(module, args, kwargs, output):
    sg = getattr(module, 'split_gate', None)
    if sg is None:
        return output
    x = kwargs.get('input', None)
    if x is None:
        x = args[0]
    cur = CTX.cur
    if cur is None:
        raise RuntimeError('--split_gate_proj: no table mask is installed for this forward. The mask is per-batch context (set_table_mask / set_table_mask_from_regions) and the model pre-hook slices it; a forward without it would send every token to the same branch silently. Refusing to guess.')
    if cur.shape[0] != x.shape[0] or cur.shape[1] != x.shape[1]:
        raise RuntimeError(f'--split_gate_proj: table mask {tuple(cur.shape)} does not match the hidden states {tuple(x.shape[:2])} of this forward.')
    if CTX.swap:
        cur = ~cur
    return output + sg.delta(x, cur).to(output.dtype)

def module_family(name):
    for fam in SPLIT_MODULES:
        if name == fam or name.endswith('.' + fam):
            return fam
    return None

def _split_linears(model, families=SPLIT_MODULES):
    families = tuple(families)
    bad = [f for f in families if f not in SPLIT_MODULES]
    if bad:
        raise ValueError(f'{bad} are not MLP projections this mechanism serves {SPLIT_MODULES}')
    found = []
    for (name, mod) in model.named_modules():
        fam = module_family(name)
        if fam is None or fam not in families:
            continue
        if 'lora' in name:
            continue
        if hasattr(mod, 'lora_A') and (not isinstance(mod, SplitGateLoRA)):
            raise RuntimeError(f'{name} is already a PEFT LoRA module. --split_gate_proj serves {list(SPLIT_MODULES)} itself, so every one of them must be removed from --target_modules; leaving both in would stack two adapters on one projection.')
        if isinstance(mod, nn.Linear):
            found.append((name, mod))
    if not found:
        raise RuntimeError(f'no {list(families)} Linear found in this model')
    fams = {module_family(n) for (n, _) in found}
    if fams != set(families):
        raise RuntimeError(f'--split_gate_proj was asked for {list(families)}; this model only exposes {sorted(fams)}')
    return found

def _peft_lora_dtype(model):
    for (name, p) in model.named_parameters():
        if 'lora_A' in name and 'weight' in name:
            return p.dtype
    return None

def attach(model, r, alpha, dropout=0.0, modules=None):
    families = tuple(SPLIT_MODULES if modules is None else modules)
    mods = _split_linears(model, families)
    dtype = _peft_lora_dtype(model)
    handles = []
    built = []
    for (name, lin) in mods:
        dev = lin.weight.device
        sg = SplitGateLoRA(lin.in_features, lin.out_features, r, alpha, dropout, dtype=dtype if dtype is not None else lin.weight.dtype, device=dev)
        lin.split_gate = sg
        handles.append(lin.register_forward_hook(_hook, with_kwargs=True))
        built.append((name, sg))
    inner = model.get_base_model() if hasattr(model, 'get_base_model') else model
    hosts = [model] if inner is model else [model, inner]
    for h in hosts:
        handles.append(h.register_forward_pre_hook(_slice_prehook, with_kwargs=True))
    model._split_gate_handles = handles
    model._split_gate_modules = built
    model._split_gate_families = list(families)
    CTX.modules = tuple((sg for (_, sg) in built))
    n = sum((sg.n_params() for (_, sg) in built))
    per_family = {fam: sum((1 for (n_, _) in built if module_family(n_) == fam)) for fam in families}
    return {'mechanism': MECHANISM, 'version': VERSION, 'modules': list(families), 'branches': list(BRANCHES), 'n_modules': len(built), 'n_modules_per_family': per_family, 'r': int(r), 'lora_alpha': float(alpha), 'lora_dropout': float(dropout), 'scaling': float(alpha) / float(r), 'n_split_params': int(n), 'param_dtype': str(dtype) if dtype is not None else None, 'formula': FORMULA, 'token_scope': TOKEN_SCOPE, 'zero_init': True, 'layer_names': [n_ for (n_, _) in built], 'prehook_on': [type(h).__name__ for h in hosts]}

def detach(model):
    for h in getattr(model, '_split_gate_handles', []):
        h.remove()
    for (name, _) in getattr(model, '_split_gate_modules', []):
        mod = model.get_submodule(name)
        if hasattr(mod, 'split_gate'):
            del mod.split_gate
    model._split_gate_handles = []
    model._split_gate_modules = []
    model._split_gate_families = []
    CTX.modules = ()
    clear_table_mask()

def is_attached(model):
    return bool(getattr(model, '_split_gate_modules', ()))

def split_modules(model):
    mods = getattr(model, '_split_gate_modules', ())
    if not mods:
        raise RuntimeError('--split_gate_proj is not attached to this model')
    return mods

def attached_families(model):
    fams = getattr(model, '_split_gate_families', None)
    if not fams:
        fams = {module_family(n) for (n, _) in split_modules(model)}
    return [f for f in SPLIT_MODULES if f in set(fams)]

def config_modules(cfg):
    if not cfg:
        return list(SPLIT_MODULES)
    if cfg.get('modules'):
        return list(cfg['modules'])
    if cfg.get('module'):
        return [cfg['module']]
    return list(SPLIT_MODULES)

def verify_zero_init(model, input_ids, attention_mask=None):
    mods = split_modules(model)
    for (name, sg) in mods:
        for b in BRANCHES:
            if float(sg.split_B[b].weight.abs().max()) != 0.0:
                raise RuntimeError(f'{name}.{b}: B is not zero; this check is only meaningful at initialisation')
    dev = next(model.parameters()).device
    ids = torch.as_tensor(input_ids).to(dev)
    am = torch.ones_like(ids) if attention_mask is None else torch.as_tensor(attention_mask).to(dev)
    m = torch.zeros(ids.shape, dtype=torch.bool, device=dev)
    m[:, ids.shape[1] // 2:] = True
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            set_table_mask(m)
            on = model(input_ids=ids, attention_mask=am).logits.float()
            clear_table_mask()
            for (_, sg) in mods:
                sg.eval()
            saved = []
            for (name, _) in mods:
                lin = model.get_submodule(name)
                saved.append((lin, lin.split_gate))
                del lin.split_gate
            try:
                off = model(input_ids=ids, attention_mask=am).logits.float()
            finally:
                for (lin, sg) in saved:
                    lin.split_gate = sg
    finally:
        if was_training:
            model.train()
    d = float((on - off).abs().max())
    if d != 0.0:
        raise RuntimeError(f'zero-init identity FAILED: max|logit diff| = {d:.3e}; the mechanism is mis-inserted')
    return {'max_abs_logit_diff': d, 'frac_tokens_table': float(m.float().mean())}

def _branch_stats(mods, b):
    a = torch.stack([sg.split_A[b].weight.detach().float().norm() for (_, sg) in mods])
    bb = torch.stack([sg.split_B[b].weight.detach().float().norm() for (_, sg) in mods])
    return {'mean_norm_A': float(a.mean()), 'mean_norm_B': float(bb.mean()), 'max_norm_B': float(bb.max())}

def weight_stats(model):
    mods = split_modules(model)
    out = {}
    for b in BRANCHES:
        rec = _branch_stats(mods, b)
        rec['by_family'] = {fam: _branch_stats([m for m in mods if module_family(m[0]) == fam], b) for fam in attached_families(model)}
        out[b] = rec
    return out

def state_dict(model):
    sd = {}
    for (name, sg) in split_modules(model):
        for (k, v) in sg.state_dict().items():
            sd[f'{name}.{k}'] = v.cpu()
    return sd

def save(model, out_dir, extra=None):
    mods = split_modules(model)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state_dict(model), os.path.join(out_dir, STATE_FILE))
    sg0 = mods[0][1]
    cfg = {'mechanism': MECHANISM, 'version': VERSION, 'modules': attached_families(model), 'branches': list(BRANCHES), 'formula': FORMULA, 'token_scope': TOKEN_SCOPE, 'zero_init': True, 'n_modules': len(mods), 'r': sg0.r, 'lora_alpha': sg0.alpha, 'scaling': sg0.scaling, 'n_split_params': int(sum((sg.n_params() for (_, sg) in mods))), 'layer_names': [n for (n, _) in mods], 'state_file': STATE_FILE}
    if extra:
        clash = sorted(set(extra) & set(cfg))
        if clash:
            raise ValueError(f"save(extra=...) would overwrite the side-car's own identity fields {clash}; name them differently (e.g. run_lora_alpha).")
        cfg = {**extra, **cfg}
    with open(os.path.join(out_dir, CONFIG_FILE), 'w') as f:
        json.dump(cfg, f, indent=2)
    return cfg

def load(model, adapter_dir, strict=True):
    mods = split_modules(model)
    path = os.path.join(adapter_dir, STATE_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f'no {STATE_FILE} in {adapter_dir}')
    cfg = read_config(adapter_dir)
    sg0 = mods[0][1]
    (declared, mine_fams) = (config_modules(cfg), attached_families(model))
    if declared != mine_fams:
        raise RuntimeError(f"{adapter_dir} (version {cfg.get('version')}) holds split pairs for {declared} but this model is attached on {mine_fams}; attach(modules=...) must match the side-car. Version 1 split gate_proj alone; version {VERSION} splits the whole MLP.")
    for (key, mine) in (('r', sg0.r), ('lora_alpha', sg0.alpha), ('n_modules', len(mods))):
        theirs = cfg.get(key)
        if theirs is not None and theirs != mine:
            raise RuntimeError(f'{adapter_dir} was trained with {key}={theirs!r} but this model is attached with {mine!r}; refusing to serve it under a different configuration.')
    if cfg.get('mechanism', MECHANISM) != MECHANISM:
        raise RuntimeError(f"{adapter_dir} holds mechanism {cfg.get('mechanism')!r}, not {MECHANISM!r}")
    state = torch.load(path, map_location='cpu')
    (missing, unexpected) = ([], list(state.keys()))
    for (name, sg) in mods:
        pre = f'{name}.'
        sub = {k[len(pre):]: v for (k, v) in state.items() if k.startswith(pre)}
        for k in list(sub):
            unexpected.remove(pre + k)
        (m, u) = sg.load_state_dict(sub, strict=False)
        missing += [pre + k for k in m]
        unexpected += [pre + k for k in u]
    if strict and (missing or unexpected):
        raise RuntimeError(f'split-gate state mismatch: missing={missing} unexpected={unexpected}')
    for (_, sg) in mods:
        sg.to(dtype=_peft_lora_dtype(model) or sg.split_A[BRANCHES[0]].weight.dtype)
    return cfg

def is_split_gate_adapter(adapter_dir):
    if not adapter_dir:
        return False
    return os.path.exists(os.path.join(adapter_dir, STATE_FILE))

def read_config(adapter_dir):
    if not adapter_dir:
        return {}
    p = os.path.join(adapter_dir, CONFIG_FILE)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)

def split_rank(lora_r, lora_alpha):
    if int(lora_r) % 2 != 0:
        raise SystemExit(f'--split_gate_proj halves the rank, so --lora_r must be even; got {lora_r}.')
    return (int(lora_r) // 2, float(lora_alpha) / 2.0)
