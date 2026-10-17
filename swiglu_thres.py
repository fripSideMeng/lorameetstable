import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import recv_gate as RCV
MECHANISM = 'swiglu_thres'
VERSION = 1
STATE_FILE = 'swiglu_thres.pt'
CONFIG_FILE = 'swiglu_thres.json'
BRANCHES = ('table', 'nontable')
(I_TABLE, I_NONTABLE) = (0, 1)
FORMULA = 'SwiGLU gate activation a * sigmoid(beta * a), evaluated as silu(beta*a)/beta; beta = exp(tau_table) on table tokens, exp(tau_nontable) elsewhere; one tau pair per layer, tau init 0 so beta init 1.0'
TOKEN_SCOPE = 'beta_table applies to tokens whose region id is recv_gate.R_TABLE (the Markdown table block found by find_markdown_table_span); beta_nontable applies to pre-table, post-table, padding and every generated token'

class _Ctx:
    active = False
    mask_full = None
    cur = None
    modules = ()
    enabled = True
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

def set_enabled(on):
    CTX.enabled = bool(on)
    return CTX.enabled

def is_enabled():
    return bool(CTX.enabled)

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

class _TemperedSiLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, beta2, table_mask):
        beta = torch.where(table_mask.unsqueeze(-1), beta2[I_TABLE], beta2[I_NONTABLE]).to(a.dtype)
        t = F.silu(beta * a)
        t = t.div_(beta)
        ctx.save_for_backward(a, beta2, table_mask)
        return t

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        (a, beta2, table_mask) = ctx.saved_tensors
        m = table_mask.unsqueeze(-1)
        beta = torch.where(m, beta2[I_TABLE], beta2[I_NONTABLE]).to(a.dtype)
        s = torch.sigmoid(beta * a)
        ds = s * (1.0 - s)
        grad_a = grad_out * (s + beta * a * ds) if ctx.needs_input_grad[0] else None
        grad_beta = None
        if ctx.needs_input_grad[1]:
            per_pos = (grad_out * a * a * ds).sum(dim=-1, dtype=torch.float32)
            gt = torch.where(table_mask, per_pos, torch.zeros_like(per_pos)).sum()
            gn = torch.where(table_mask, torch.zeros_like(per_pos), per_pos).sum()
            grad_beta = torch.stack([gt, gn]).to(beta2.dtype)
        return (grad_a, grad_beta, None)

class SwigluTemperature(nn.Module):

    def __init__(self, base, dtype=None, device=None):
        super().__init__()
        self.base = base
        self.tau = nn.Parameter(torch.zeros(len(BRANCHES), dtype=dtype or torch.float32, device=device))

    def n_params(self):
        return int(self.tau.numel())

    def betas(self):
        return torch.exp(self.tau)

    def beta_dict(self):
        b = self.betas().detach().float().tolist()
        return {name: float(b[i]) for (i, name) in enumerate(BRANCHES)}

    def apply_act(self, a, table_mask):
        return _TemperedSiLU.apply(a, self.betas(), table_mask.to(a.device))

    def forward(self, a):
        if not CTX.enabled:
            return self.base(a)
        cur = CTX.cur
        if cur is None:
            raise RuntimeError('--swiglu_thres: no table mask is installed for this forward. The mask is per-batch context (set_table_mask / set_table_mask_from_regions) and the model pre-hook slices it; a forward without it would give every token the same temperature silently. Refusing to guess.')
        if cur.shape[0] != a.shape[0] or cur.shape[1] != a.shape[1]:
            raise RuntimeError(f'--swiglu_thres: table mask {tuple(cur.shape)} does not match the gate pre-activation {tuple(a.shape[:2])} of this forward.')
        return self.apply_act(a, cur)

def _saved_bytes(fn, *inputs):
    (seen, total) = (set(), [0])

    def pack(t):
        if id(t) not in seen:
            seen.add(id(t))
            total[0] += t.numel() * t.element_size()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn(*inputs)
        del out
    return total[0]

def saved_bytes_audit(batch=2, seq=128, d_int=8192, dtype=torch.bfloat16, device='cpu'):
    a = torch.randn(batch, seq, d_int, dtype=dtype, device=device, requires_grad=True)
    b2 = torch.ones(len(BRANCHES), device=device, requires_grad=True)
    msk = torch.zeros(batch, seq, dtype=torch.bool, device=device)
    msk[:, seq // 2:] = True
    base = _saved_bytes(lambda x: F.silu(x), a)
    ours = _saved_bytes(lambda x, b, m: _TemperedSiLU.apply(x, b, m), a, b2, msk)

    def naive(x, b, m):
        beta = torch.where(m.unsqueeze(-1), b[I_TABLE], b[I_NONTABLE]).to(x.dtype)
        return F.silu(beta * x) / beta
    chained = _saved_bytes(naive, a, b2, msk)
    act = a.numel() * a.element_size()
    return {'shape': [batch, seq, d_int], 'dtype': str(dtype), 'activation_bytes': act, 'baseline_silu_saved_bytes': base, 'swiglu_thres_saved_bytes': ours, 'overhead_bytes': ours - base, 'overhead_ratio_of_one_activation': (ours - base) / act, 'naive_op_chain_saved_bytes': chained, 'naive_op_chain_overhead_bytes': chained - base}

def _gated_mlps(model):
    found = []
    for (name, mod) in model.named_modules():
        if not all((hasattr(mod, p) for p in ('gate_proj', 'up_proj', 'down_proj', 'act_fn'))):
            continue
        act = mod.act_fn
        if isinstance(act, SwigluTemperature):
            raise RuntimeError(f'{name}.act_fn is already a {SwigluTemperature.__name__}; --swiglu_thres is already attached to this model. detach() first.')
        if not isinstance(act, nn.Module):
            raise RuntimeError(f'{name}.act_fn is a plain function, not a module, so it cannot carry a per-layer temperature. This architecture needs its own attach path.')
        found.append((name, mod, act))
    if not found:
        raise RuntimeError('no gated MLP (gate_proj + up_proj + down_proj + act_fn) found in this model; --swiglu_thres only applies to a SwiGLU architecture')
    ids = {id(a) for (_, _, a) in found}
    if len(ids) != len(found):
        raise RuntimeError(f'this model shares {len(found) - len(ids) + 1} activation module(s) across layers, so a per-layer temperature cannot be attached to them')
    return found

def _probe_is_silu(act, device, dtype):
    x = torch.linspace(-4.0, 4.0, 33, device=device, dtype=torch.float32).to(dtype)
    with torch.no_grad():
        got = act(x).float()
        want = F.silu(x).float()
    return (bool(torch.allclose(got, want, atol=0.01, rtol=0.01)), float((got - want).abs().max()))

def attach(model, dtype=None):
    mods = _gated_mlps(model)
    dev = next(model.parameters()).device
    probe_dtype = mods[0][1].gate_proj.weight.dtype
    (ok, diff) = _probe_is_silu(mods[0][2], dev, probe_dtype)
    if not ok:
        raise RuntimeError(f"--swiglu_thres redefines SiLU(a) as a * sigmoid(beta * a), but this model's activation is not SiLU (max|act(x) - silu(x)| = {diff:.3e} on a probe). Refusing to redefine an activation the mechanism was not specified for.")
    (handles, built) = ([], [])
    for (name, mlp, act) in mods:
        st = SwigluTemperature(act, dtype=dtype or torch.float32, device=dev)
        mlp.act_fn = st
        built.append((name, st))
    inner = model.get_base_model() if hasattr(model, 'get_base_model') else model
    hosts = [model] if inner is model else [model, inner]
    for h in hosts:
        handles.append(h.register_forward_pre_hook(_slice_prehook, with_kwargs=True))
    model._swiglu_thres_handles = handles
    model._swiglu_thres_modules = built
    CTX.modules = tuple((st for (_, st) in built))
    return {'mechanism': MECHANISM, 'version': VERSION, 'branches': list(BRANCHES), 'n_layers': len(built), 'n_params': int(sum((st.n_params() for (_, st) in built))), 'param_dtype': str(dtype or torch.float32), 'init_beta': 1.0, 'formula': FORMULA, 'token_scope': TOKEN_SCOPE, 'activation_probe_max_diff': diff, 'memory_audit': saved_bytes_audit(d_int=int(mods[0][1].gate_proj.out_features), dtype=probe_dtype, device='cpu'), 'layer_names': [n for (n, _) in built], 'prehook_on': [type(h).__name__ for h in hosts]}

def detach(model):
    for h in getattr(model, '_swiglu_thres_handles', []):
        h.remove()
    for (name, st) in getattr(model, '_swiglu_thres_modules', []):
        mlp = model.get_submodule(name)
        if isinstance(mlp.act_fn, SwigluTemperature):
            mlp.act_fn = st.base
    model._swiglu_thres_handles = []
    model._swiglu_thres_modules = []
    CTX.modules = ()
    clear_table_mask()
    set_enabled(True)

def is_attached(model):
    return bool(getattr(model, '_swiglu_thres_modules', ()))

def thres_modules(model):
    mods = getattr(model, '_swiglu_thres_modules', ())
    if not mods:
        raise RuntimeError('--swiglu_thres is not attached to this model')
    return mods

def verify_identity(model, input_ids, attention_mask=None):
    mods = thres_modules(model)
    for (name, st) in mods:
        if float(st.tau.abs().max()) != 0.0:
            raise RuntimeError(f'{name}: tau is not zero; this check is only meaningful at initialisation')
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
            saved = []
            for (name, st) in mods:
                mlp = model.get_submodule(name)
                saved.append((mlp, st))
                mlp.act_fn = st.base
            try:
                off = model(input_ids=ids, attention_mask=am).logits.float()
            finally:
                for (mlp, st) in saved:
                    mlp.act_fn = st
    finally:
        if was_training:
            model.train()
    d = float((on - off).abs().max())
    if d != 0.0:
        raise RuntimeError(f'init identity FAILED: max|logit diff| = {d:.3e}; the mechanism is mis-inserted')
    return {'max_abs_logit_diff': d, 'frac_tokens_table': float(m.float().mean())}

def betas_by_layer(model):
    out = {b: [] for b in BRANCHES}
    for (_, st) in thres_modules(model):
        d = st.beta_dict()
        for b in BRANCHES:
            out[b].append(d[b])
    return out

def beta_summary(model):
    by = betas_by_layer(model)
    rec = {}
    for b in BRANCHES:
        v = torch.tensor(by[b])
        rec[b] = {'min': float(v.min()), 'mean': float(v.mean()), 'max': float(v.max())}
    gap = torch.tensor(by['table']) - torch.tensor(by['nontable'])
    rec['table_minus_nontable'] = {'min': float(gap.min()), 'mean': float(gap.mean()), 'max': float(gap.max()), 'n_layers_table_sharper': int((gap > 0).sum())}
    rec['by_layer'] = by
    return rec

def state_dict(model):
    return {f'{name}.tau': st.tau.detach().cpu() for (name, st) in thres_modules(model)}

def save(model, out_dir, extra=None):
    mods = thres_modules(model)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state_dict(model), os.path.join(out_dir, STATE_FILE))
    cfg = {'mechanism': MECHANISM, 'version': VERSION, 'branches': list(BRANCHES), 'formula': FORMULA, 'token_scope': TOKEN_SCOPE, 'init_beta': 1.0, 'n_layers': len(mods), 'n_params': int(sum((st.n_params() for (_, st) in mods))), 'betas': betas_by_layer(model), 'beta_summary': {k: v for (k, v) in beta_summary(model).items() if k != 'by_layer'}, 'layer_names': [n for (n, _) in mods], 'state_file': STATE_FILE}
    if extra:
        clash = sorted(set(extra) & set(cfg))
        if clash:
            raise ValueError(f"save(extra=...) would overwrite the side-car's own identity fields {clash}; name them differently (e.g. run_lora_alpha).")
        cfg = {**extra, **cfg}
    with open(os.path.join(out_dir, CONFIG_FILE), 'w') as f:
        json.dump(cfg, f, indent=2)
    return cfg

def load(model, adapter_dir, strict=True):
    mods = thres_modules(model)
    path = os.path.join(adapter_dir, STATE_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f'no {STATE_FILE} in {adapter_dir}')
    cfg = read_config(adapter_dir)
    if cfg.get('mechanism', MECHANISM) != MECHANISM:
        raise RuntimeError(f"{adapter_dir} holds mechanism {cfg.get('mechanism')!r}, not {MECHANISM!r}")
    for (key, mine) in (('n_layers', len(mods)), ('branches', list(BRANCHES))):
        theirs = cfg.get(key)
        if theirs is not None and theirs != mine:
            raise RuntimeError(f'{adapter_dir} was trained with {key}={theirs!r} but this model has {mine!r}; refusing to serve it under a different configuration.')
    state = torch.load(path, map_location='cpu')
    (missing, unexpected) = ([], list(state.keys()))
    for (name, st) in mods:
        key = f'{name}.tau'
        if key in state:
            unexpected.remove(key)
            with torch.no_grad():
                st.tau.copy_(state[key].to(st.tau.device, st.tau.dtype))
        else:
            missing.append(key)
    if strict and (missing or unexpected):
        raise RuntimeError(f'swiglu-thres state mismatch: missing={missing} unexpected={unexpected}')
    return cfg

def is_swiglu_thres_adapter(adapter_dir):
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
