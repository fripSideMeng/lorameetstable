import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import role_low_rank as RLR
import role_gate_mlp as RGM
MECHANISM = 'ffn_role_scale'
VERSION = 1
SCALE_NAMES = ('cta_target', 'cta_context', 'cpa_target', 'cpa_context')
N_SCALE = len(SCALE_NAMES)
N_SLOTS = N_SCALE + 1
SLOT_NONE = 0
SLOT_CTA_TARGET = 1
SLOT_CTA_CONTEXT = 2
SLOT_CPA_TARGET = 3
SLOT_CPA_CONTEXT = 4
SLOT_NAMES = ('none',) + SCALE_NAMES
ROLE_TO_SLOT = {RLR.ROLE_NONE: SLOT_NONE, RLR.ROLE_CTA_CONTEXT: SLOT_CTA_CONTEXT, RLR.ROLE_CTA_TARGET: SLOT_CTA_TARGET, RLR.ROLE_CPA_CONTEXT: SLOT_CPA_CONTEXT, RLR.ROLE_CPA_HEAD: SLOT_CPA_TARGET, RLR.ROLE_CPA_TAIL: SLOT_CPA_TARGET}
_SLOT_LUT = torch.tensor([ROLE_TO_SLOT[i] for i in range(RLR.N_ROLES)], dtype=torch.long)
PARAM_DTYPE = torch.float32
STATE_FILE = 'ffn_role_scale.pt'
CONFIG_FILE = 'ffn_role_scale_config.json'
ROLE_NONE = RGM.ROLE_NONE
ROLE_TASKS = RGM.ROLE_TASKS
build_roles = RGM.build_roles
pad_roles = RGM.pad_roles
roles_from_attention_mask = RGM.roles_from_attention_mask
role_id_counts = RGM.role_id_counts

def slots_from_roles(role_ids):
    return _SLOT_LUT.to(role_ids.device)[role_ids]

def slot_counts(role_ids):
    s = slots_from_roles(torch.as_tensor(role_ids))
    return {SLOT_NAMES[k]: int((s == k).sum()) for k in range(N_SLOTS)}

class _Ctx:
    slot_full = None
    cur_slot = None
CTX = _Ctx()

def set_role_ids(role_full):
    if role_full.dim() != 2:
        raise ValueError(f'role ids must be [B, T], got {tuple(role_full.shape)}')
    if role_full.numel():
        (hi, lo) = (int(role_full.max()), int(role_full.min()))
        if hi >= RLR.N_ROLES or lo < 0:
            raise ValueError(f'role id out of range [0, {RLR.N_ROLES - 1}]: saw [{lo}, {hi}]')
    CTX.slot_full = slots_from_roles(role_full)
    CTX.cur_slot = None

def clear_role_ids():
    CTX.slot_full = None
    CTX.cur_slot = None

def _prehook(module, args, kwargs):
    if CTX.slot_full is None:
        CTX.cur_slot = None
        return
    input_ids = kwargs.get('input_ids', None)
    if input_ids is None and args:
        input_ids = args[0]
    if input_ids is None:
        CTX.cur_slot = CTX.slot_full
        return
    seq_len = input_ids.shape[1]
    dev = CTX.slot_full.device
    cp = kwargs.get('cache_position', None)
    cp = torch.arange(seq_len, device=dev) if cp is None else cp.to(dev)
    stored_T = CTX.slot_full.shape[1]
    idx = cp.clamp(max=stored_T - 1)
    valid = (cp < stored_T).long()
    CTX.cur_slot = CTX.slot_full[:, idx] * valid

class FFNRoleScaleBank(nn.Module):

    def __init__(self, dims, module_names):
        super().__init__()
        if len(dims) != len(module_names):
            raise ValueError('dims and module_names must line up')
        self.module_names = list(module_names)
        self.deltas = nn.ParameterList([nn.Parameter(torch.zeros(N_SCALE, d, dtype=PARAM_DTYPE)) for d in dims])
        self.enabled = True

    def table(self, i, dtype, device):
        v = self.deltas[i]
        zero = v.new_zeros(1, v.shape[1])
        return torch.cat([zero, v], 0).to(device=device, dtype=dtype)

    def extra_repr(self):
        return f'modules={len(self.deltas)}, conditions={N_SCALE}'

def _scale_hook(bank, i):

    def hook(module, args):
        if not bank.enabled:
            return None
        if not args:
            raise RuntimeError('down_proj was called with no positional input')
        z = args[0]
        slot = CTX.cur_slot
        if slot is None:
            raise RuntimeError('FFN role scaling is active but no role ids were set for this forward. Call ffn_role.set_role_ids(...) around every forward/generate; a silent fallback would turn the method run into the baseline.')
        if z.dim() != 3 or slot.shape[0] != z.shape[0] or slot.shape[1] != z.shape[1]:
            raise RuntimeError(f'role slots {tuple(slot.shape)} do not align with the SwiGLU intermediate activation {tuple(z.shape)}')
        tab = bank.table(i, z.dtype, z.device)
        oh = F.one_hot(slot.to(z.device), N_SLOTS).to(tab.dtype)
        return (z * (1.0 + oh @ tab),) + tuple(args[1:])
    return hook
find_gated_mlps = RGM.find_gated_mlps

def _inter_dim(mlp):
    dp = mlp.down_proj
    d = getattr(dp, 'in_features', None)
    if d is None:
        w = getattr(dp, 'weight', None)
        d = None if w is None else w.shape[1]
    if d is None:
        raise RuntimeError('cannot determine the down projection input dimension')
    return int(d)

def attach(model, verify=True):
    if getattr(model, 'ffn_role_bank', None) is not None:
        raise RuntimeError('FFN role scaling is already attached to this model')
    mlps = find_gated_mlps(model)
    if not mlps:
        raise RuntimeError('no gated (gate_proj/up_proj/down_proj/act_fn) MLP found in this architecture, so there is no SwiGLU intermediate activation to scale. This experiment supports gated decoder backbones only (Llama-3.2, Qwen3); refusing to insert the role signal anywhere else.')
    bad_act = [n for (n, m) in mlps if type(m.act_fn).__name__.lower() not in ('silu', 'siluactivation')]
    if bad_act:
        raise RuntimeError(f'{len(bad_act)} MLP(s) use activation {type(dict(mlps)[bad_act[0]].act_fn).__name__} rather than SiLU (e.g. {bad_act[0]}); this experiment is defined on SwiGLU.')
    names = [n for (n, _) in mlps]
    dims = [_inter_dim(m) for (_, m) in mlps]
    dev = next(model.parameters()).device
    bank = FFNRoleScaleBank(dims, names).to(dev)
    for p in bank.parameters():
        p.requires_grad_(True)
    handles = [m.down_proj.register_forward_pre_hook(_scale_hook(bank, i)) for (i, (_, m)) in enumerate(mlps)]
    inner = model.get_base_model() if hasattr(model, 'get_base_model') else model
    hook_targets = [model] if inner is model else [model, inner]
    for t in hook_targets:
        handles.append(t.register_forward_pre_hook(_prehook, with_kwargs=True))
    model.ffn_role_bank = bank
    model._ffn_role_handles = handles
    info = {MECHANISM: True, 'version': VERSION, 'prehook_on': [type(t).__name__ for t in hook_targets], 'n_gated_mlps': len(mlps), 'scaled_modules': [f'{n}.down_proj' for n in names], 'd_intermediate': sorted(set(dims)), 'condition_names': list(SCALE_NAMES), 'n_scale_params': int(sum((p.numel() for p in bank.parameters()))), 'init': 'zeros (scale = 1 + delta = identity)', 'insertion': 'forward PRE-hook on mlp.down_proj input == post-SwiGLU z, before down_proj'}
    if verify:
        info['placement_check'] = verify_scale_path(model)
    return info

def detach(model):
    for h in getattr(model, '_ffn_role_handles', []):
        h.remove()
    model._ffn_role_handles = []
    if getattr(model, 'ffn_role_bank', None) is not None:
        del model.ffn_role_bank
    clear_role_ids()

def get_bank(model):
    bank = getattr(model, 'ffn_role_bank', None)
    if bank is None:
        raise RuntimeError('no FFN role-scale bank attached to this model')
    return bank

@torch.no_grad()
def verify_scale_path(model, index=0, seed=0):
    bank = get_bank(model)
    (name, mlp) = find_gated_mlps(model)[index]
    d_hidden = int(getattr(mlp.gate_proj, 'in_features', getattr(mlp.gate_proj, 'base_layer', mlp.gate_proj).weight.shape[1]))
    dev = bank.deltas[index].device
    gp = mlp.gate_proj
    dtype = getattr(gp, 'base_layer', gp).weight.dtype
    was_training = mlp.training
    mlp.eval()
    g = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(1, 2, d_hidden, generator=g).to(device=dev, dtype=dtype)
    saved = bank.deltas[index].detach().clone()
    probe = torch.randn(bank.deltas[index].shape, generator=g).to(dev) * 0.1
    bank.deltas[index].copy_(probe)
    probe_roles = torch.tensor([[RLR.ROLE_CTA_TARGET, RLR.ROLE_NONE]], device=dev)
    set_role_ids(probe_roles)
    CTX.cur_slot = slots_from_roles(probe_roles)
    slot_of_probe = int(CTX.cur_slot[0, 0])
    try:
        bank.enabled = False
        z = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
        base_out = mlp.down_proj(z)
        s = torch.ones_like(z)
        s[0, 0] = 1.0 + probe[slot_of_probe - 1].to(z.dtype)
        expect = mlp.down_proj(z * s)
        bank.enabled = True
        got = mlp(x)
    finally:
        bank.enabled = True
        bank.deltas[index].copy_(saved)
        clear_role_ids()
        if was_training:
            mlp.train()
    scale = float(expect.abs().max()) + 1e-06
    err = float((got - expect).abs().max()) / scale
    none_err = float((got[0, 1] - base_out[0, 1]).abs().max()) / scale
    moved = float((expect[0, 0] - base_out[0, 0]).abs().max()) / scale
    if err > 0.001:
        raise RuntimeError(f'[{MECHANISM}] placement check FAILED on {name}: the patched MLP does not equal down_proj(act(gate(x))*up(x)*(1+delta)) (rel err {err:.3e}). The scaling is not landing on the SwiGLU intermediate activation in this architecture.')
    if none_err > 1e-06:
        raise RuntimeError(f'[{MECHANISM}] the `none` role changed the output on {name} (rel err {none_err:.3e}); the identity scale must be exactly 1.')
    if moved < 0.0001:
        raise RuntimeError(f'[{MECHANISM}] the probe delta had no measurable effect on {name}; the hook is not reaching the computation.')
    return {'module': f'{name}.down_proj', 'rel_err_vs_expected': err, 'rel_err_none_vs_base': none_err, 'rel_effect_of_probe': moved}

def save(model, out_dir, extra=None):
    bank = get_bank(model)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {n: v.detach().to('cpu', torch.float32) for (n, v) in zip(bank.module_names, bank.deltas)}
    torch.save(state, out / STATE_FILE)
    cfg = {MECHANISM: True, 'version': VERSION, 'condition_names': list(SCALE_NAMES), 'parameterization': 'delta (scale = 1 + delta)', 'module_names': list(bank.module_names), 'd_intermediate': [int(v.shape[1]) for v in bank.deltas], 'state_file': STATE_FILE}
    if extra:
        cfg.update(extra)
    with open(out / CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)
    return str(out / STATE_FILE)

def load(model, adapter_dir, strict=True):
    bank = get_bank(model)
    d = Path(adapter_dir)
    (cfg_path, st_path) = (d / CONFIG_FILE, d / STATE_FILE)
    if not st_path.exists():
        raise RuntimeError(f'{st_path} not found; this adapter carries no FFN role scales.')
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if list(cfg.get('condition_names', SCALE_NAMES)) != list(SCALE_NAMES):
            raise RuntimeError(f"this checkpoint was trained with the conditions {cfg['condition_names']}, but this code builds {list(SCALE_NAMES)}; loading it would attach the wrong vector to every token.")
    state = torch.load(st_path, map_location='cpu')
    missing = [n for n in bank.module_names if n not in state]
    unexpected = [n for n in state if n not in set(bank.module_names)]
    if strict and (missing or unexpected):
        raise RuntimeError(f'FFN role-scale checkpoint does not match this model: {len(missing)} missing (e.g. {missing[:2]}), {len(unexpected)} unexpected (e.g. {unexpected[:2]}).')
    with torch.no_grad():
        for (n, v) in zip(bank.module_names, bank.deltas):
            if n in state:
                t = state[n]
                if tuple(t.shape) != tuple(v.shape):
                    raise RuntimeError(f'FFN role scale {n}: checkpoint {tuple(t.shape)} != model {tuple(v.shape)}')
                v.copy_(t.to(v.device, v.dtype))
    return {'loaded': len(bank.module_names) - len(missing), 'missing': len(missing)}

def is_ffn_role_adapter(adapter_dir):
    if not adapter_dir:
        return False
    return (Path(adapter_dir) / STATE_FILE).exists()

def stats(model):
    bank = get_bank(model)
    out = {}
    for (k, name) in enumerate(SCALE_NAMES):
        rows = [v.data[k].detach().to(torch.float32).cpu() for v in bank.deltas]
        flat = torch.cat(rows)
        rec = {'mean_delta': float(flat.mean()), 'mean_abs_delta': float(flat.abs().mean()), 'max_abs_delta': float(flat.abs().max()), 'n_params': int(flat.numel())}
        gr = [v.grad[k].detach().to(torch.float32).cpu() for v in bank.deltas if v.grad is not None]
        if gr:
            g = torch.cat(gr)
            rec['mean_abs_grad'] = float(g.abs().mean())
            rec['n_zero_grad_layers'] = sum((1 for t in gr if float(t.abs().max()) == 0.0))
        out[name] = rec
    return out

def per_layer_table(model):
    bank = get_bank(model)
    rows = []
    for (i, (n, v)) in enumerate(zip(bank.module_names, bank.deltas)):
        d = v.data.detach().to(torch.float32).cpu()
        rec = {'layer': i, 'module': n}
        rec.update({name: float(d[k].abs().mean()) for (k, name) in enumerate(SCALE_NAMES)})
        rows.append(rec)
    return rows

def format_stats(model):
    st = stats(model)
    lines = [f'[{MECHANISM}] movement from identity (scale = 1 + delta):', f"    {'condition':<12} {'mean(d)':>12} {'mean|d|':>12} {'max|d|':>12} {'n':>10}"]
    for name in SCALE_NAMES:
        r = st[name]
        lines.append(f"    {name:<12} {r['mean_delta']:>12.3e} {r['mean_abs_delta']:>12.3e} {r['max_abs_delta']:>12.3e} {r['n_params']:>10,}")
    lines.append(f'[{MECHANISM}] per-layer mean|delta|:')
    lines.append('    ' + f"{'layer':>5} " + ' '.join((f'{n:>12}' for n in SCALE_NAMES)))
    for row in per_layer_table(model):
        lines.append('    ' + f"{row['layer']:>5} " + ' '.join((f'{row[n]:>12.3e}' for n in SCALE_NAMES)))
    return '\n'.join(lines)

def snapshot(model):
    bank = get_bank(model)
    return [v.detach().cpu().clone() for v in bank.deltas]
