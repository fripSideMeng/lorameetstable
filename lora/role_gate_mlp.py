import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import role_low_rank as RLR
ROLE_NONE = RLR.ROLE_NONE
ROLE_VECTOR_NAMES = RLR.ROLE_VECTOR_NAMES
N_ROLE_VECTORS = RLR.N_ROLE_VECTORS
N_ROLES = RLR.N_ROLES
ROLE_NAMES = RLR.ROLE_NAMES
role_spec = RLR.role_spec
tokens_role_ids = RLR.tokens_role_ids
role_id_counts = RLR.role_id_counts
task_key = RLR.task_key
ROLE_TASKS = ('cta', 'cpa')
PARAM_DTYPE = torch.float32
STATE_FILE = 'role_gate.pt'
CONFIG_FILE = 'role_gate_config.json'

class _Ctx:
    role_full = None
    cur_role = None
CTX = _Ctx()

def set_role_ids(role_full):
    if role_full.dim() != 2:
        raise ValueError(f'role ids must be [B, T], got {tuple(role_full.shape)}')
    if role_full.numel():
        (hi, lo) = (int(role_full.max()), int(role_full.min()))
        if hi >= N_ROLES or lo < 0:
            raise ValueError(f'role id out of range [0, {N_ROLES - 1}]: saw [{lo}, {hi}]')
    CTX.role_full = role_full
    CTX.cur_role = None

def clear_role_ids():
    CTX.role_full = None
    CTX.cur_role = None

def _prehook(module, args, kwargs):
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

class GateRoleBank(nn.Module):

    def __init__(self, dims, module_names):
        super().__init__()
        if len(dims) != len(module_names):
            raise ValueError('dims and module_names must line up')
        self.module_names = list(module_names)
        self.vectors = nn.ParameterList([nn.Parameter(torch.zeros(N_ROLE_VECTORS, d, dtype=PARAM_DTYPE)) for d in dims])
        self.enabled = True

    def table(self, i, dtype, device):
        v = self.vectors[i]
        zero = v.new_zeros(1, v.shape[1])
        return torch.cat([zero, v], 0).to(device=device, dtype=dtype)

    def extra_repr(self):
        return f'modules={len(self.vectors)}, roles={N_ROLE_VECTORS}'

def _gate_hook(bank, i):

    def hook(module, args, output):
        if not bank.enabled:
            return output
        role = CTX.cur_role
        if role is None:
            raise RuntimeError('role-gate modulation is active but no role ids were set for this forward. Call role_gate_mlp.set_role_ids(...) around every forward/generate; a silent fallback would turn the method run into the baseline.')
        if role.shape[0] != output.shape[0] or role.shape[1] != output.shape[1]:
            raise RuntimeError(f'role ids {tuple(role.shape)} do not align with the gate pre-activation {tuple(output.shape[:2])}')
        tab = bank.table(i, output.dtype, output.device)
        oh = F.one_hot(role.to(output.device), N_ROLES).to(tab.dtype)
        return output + oh @ tab
    return hook

def find_gated_mlps(model):
    out = []
    for (name, m) in model.named_modules():
        if all((hasattr(m, a) for a in ('gate_proj', 'up_proj', 'down_proj', 'act_fn'))):
            out.append((name, m))
    return out

def _gate_dim(mlp):
    gp = mlp.gate_proj
    d = getattr(gp, 'out_features', None)
    if d is None:
        w = getattr(gp, 'weight', None)
        d = None if w is None else w.shape[0]
    if d is None:
        raise RuntimeError('cannot determine the gate projection output dimension')
    return int(d)

def attach(model, verify=True):
    if getattr(model, 'gate_role_bank', None) is not None:
        raise RuntimeError('role-gate modulation is already attached to this model')
    mlps = find_gated_mlps(model)
    if not mlps:
        raise RuntimeError('no gated (gate_proj/up_proj/down_proj/act_fn) MLP found in this architecture, so there is no SwiGLU gate pre-activation to modulate. This experiment supports gated decoder backbones only (Llama-3.2, Qwen3); refusing to insert the role signal anywhere else.')
    bad_act = [n for (n, m) in mlps if type(m.act_fn).__name__.lower() not in ('silu', 'siluactivation')]
    if bad_act:
        raise RuntimeError(f'{len(bad_act)} MLP(s) use activation {type(dict(mlps)[bad_act[0]].act_fn).__name__} rather than SiLU (e.g. {bad_act[0]}); this experiment is defined on the SwiGLU gate.')
    names = [n for (n, _) in mlps]
    dims = [_gate_dim(m) for (_, m) in mlps]
    dev = next(model.parameters()).device
    bank = GateRoleBank(dims, names).to(dev)
    for p in bank.parameters():
        p.requires_grad_(True)
    handles = [m.gate_proj.register_forward_hook(_gate_hook(bank, i)) for (i, (_, m)) in enumerate(mlps)]
    inner = model.get_base_model() if hasattr(model, 'get_base_model') else model
    hook_targets = [model] if inner is model else [model, inner]
    for t in hook_targets:
        handles.append(t.register_forward_pre_hook(_prehook, with_kwargs=True))
    model.gate_role_bank = bank
    model._gate_role_handles = handles
    info = {'role_gate_mlp': True, 'prehook_on': [type(t).__name__ for t in hook_targets], 'n_gated_mlps': len(mlps), 'gate_modules': [f'{n}.gate_proj' for n in names], 'd_intermediate': sorted(set(dims)), 'role_vector_names': list(ROLE_VECTOR_NAMES), 'n_role_params': int(sum((p.numel() for p in bank.parameters()))), 'init': 'zeros', 'insertion': 'forward hook on mlp.gate_proj output == gate pre-activation, before act_fn'}
    if verify:
        info['placement_check'] = verify_gate_path(model)
    return info

def detach(model):
    for h in getattr(model, '_gate_role_handles', []):
        h.remove()
    model._gate_role_handles = []
    if getattr(model, 'gate_role_bank', None) is not None:
        del model.gate_role_bank
    clear_role_ids()

def get_bank(model):
    bank = getattr(model, 'gate_role_bank', None)
    if bank is None:
        raise RuntimeError('no role-gate bank attached to this model')
    return bank

@torch.no_grad()
def verify_gate_path(model, index=0, seed=0):
    bank = get_bank(model)
    (name, mlp) = find_gated_mlps(model)[index]
    d_hidden = int(mlp.gate_proj.in_features)
    dev = bank.vectors[index].device
    gp = mlp.gate_proj
    w = getattr(gp, 'base_layer', gp).weight
    dtype = w.dtype
    was_training = mlp.training
    mlp.eval()
    g = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(1, 2, d_hidden, generator=g).to(device=dev, dtype=dtype)
    saved = bank.vectors[index].detach().clone()
    probe = torch.randn(bank.vectors[index].shape, generator=g).to(dev) * 0.1
    bank.vectors[index].copy_(probe)
    probe_roles = torch.tensor([[1, ROLE_NONE]], device=dev)
    set_role_ids(probe_roles)
    CTX.cur_role = probe_roles
    try:
        bank.enabled = False
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        base_out = mlp.down_proj(mlp.act_fn(gate) * up)
        r = torch.zeros_like(gate)
        r[0, 0] = probe[0].to(gate.dtype)
        expect = mlp.down_proj(mlp.act_fn(gate + r) * up)
        bank.enabled = True
        got = mlp(x)
    finally:
        bank.enabled = True
        bank.vectors[index].copy_(saved)
        clear_role_ids()
        if was_training:
            mlp.train()
    scale = float(expect.abs().max()) + 1e-06
    err = float((got - expect).abs().max()) / scale
    none_err = float((got[0, 1] - base_out[0, 1]).abs().max()) / scale
    moved = float((expect[0, 0] - base_out[0, 0]).abs().max()) / scale
    if err > 0.001:
        raise RuntimeError(f'[role_gate_mlp] placement check FAILED on {name}: the patched MLP does not equal down_proj(act(gate(x)+r)*up(x)) (rel err {err:.3e}). The role signal is not landing on the gate pre-activation in this architecture.')
    if none_err > 1e-06:
        raise RuntimeError(f'[role_gate_mlp] the `none` role changed the output on {name} (rel err {none_err:.3e}); r_none must be exactly zero.')
    if moved < 0.0001:
        raise RuntimeError(f'[role_gate_mlp] the probe vector had no measurable effect on {name}; the hook is not reaching the computation.')
    return {'module': f'{name}.gate_proj', 'rel_err_vs_expected': err, 'rel_err_none_vs_base': none_err, 'rel_effect_of_probe': moved}

def build_roles(tok, raw_prompt, rendered, task, expect_ids=None):
    base = str(task).lower() if task else ''
    if any((base == t or base.startswith(t + '_') for t in ROLE_TASKS)):
        spec = role_spec(raw_prompt, task_key(task))
        (ids, roles) = tokens_role_ids(tok, rendered, spec)
    else:
        ids = tok(rendered, add_special_tokens=False)['input_ids']
        roles = [ROLE_NONE] * len(ids)
    if expect_ids is not None and list(ids) != list(expect_ids):
        raise RuntimeError(f"role tokenization ({len(ids)} tokens) does not reproduce the model's own prompt tokenization ({len(expect_ids)} tokens). Roles must be aligned to the ids the model is fed; refusing to train on a misaligned role sequence.")
    return roles

def pad_roles(role_lists, width, side='right'):
    out = []
    for r in role_lists:
        n = width - len(r)
        if n < 0:
            raise ValueError('role sequence longer than the padded width')
        out.append([ROLE_NONE] * n + list(r) if side == 'left' else list(r) + [ROLE_NONE] * n)
    return out

def roles_from_attention_mask(role_lists, attn_mask):
    T = attn_mask.shape[1]
    rows = []
    for (r, m) in zip(role_lists, attn_mask.tolist()):
        n_pad = len(m) - int(sum(m))
        if any((m[k] == 0 for k in range(n_pad, T))) or any((m[k] == 1 for k in range(n_pad))):
            raise RuntimeError("role ids can only be aligned to a LEFT-padded batch; set tok.padding_side='left'")
        if n_pad + len(r) != T:
            raise RuntimeError(f'left-pad width {n_pad} + {len(r)} role ids != padded length {T}')
        rows.append([ROLE_NONE] * n_pad + list(r))
    return torch.tensor(rows, dtype=torch.long)

def save(model, out_dir, extra=None):
    bank = get_bank(model)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {n: v.detach().to('cpu', torch.float32) for (n, v) in zip(bank.module_names, bank.vectors)}
    torch.save(state, out / STATE_FILE)
    cfg = {'role_gate_mlp': True, 'role_vector_names': list(ROLE_VECTOR_NAMES), 'module_names': list(bank.module_names), 'd_intermediate': [int(v.shape[1]) for v in bank.vectors], 'state_file': STATE_FILE}
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
        raise RuntimeError(f'{st_path} not found; this adapter carries no role-gate vectors.')
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if list(cfg.get('role_vector_names', ROLE_VECTOR_NAMES)) != list(ROLE_VECTOR_NAMES):
            raise RuntimeError(f"this checkpoint was trained with the role vocabulary {cfg['role_vector_names']}, but this code builds {list(ROLE_VECTOR_NAMES)}; loading it would attach the wrong vector to every token.")
    state = torch.load(st_path, map_location='cpu')
    missing = [n for n in bank.module_names if n not in state]
    unexpected = [n for n in state if n not in set(bank.module_names)]
    if strict and (missing or unexpected):
        raise RuntimeError(f'role-gate checkpoint does not match this model: {len(missing)} missing (e.g. {missing[:2]}), {len(unexpected)} unexpected (e.g. {unexpected[:2]}).')
    with torch.no_grad():
        for (n, v) in zip(bank.module_names, bank.vectors):
            if n in state:
                t = state[n]
                if tuple(t.shape) != tuple(v.shape):
                    raise RuntimeError(f'role vector {n}: checkpoint {tuple(t.shape)} != model {tuple(v.shape)}')
                v.copy_(t.to(v.device, v.dtype))
    return {'loaded': len(bank.module_names) - len(missing), 'missing': len(missing)}

def is_role_gate_adapter(adapter_dir):
    if not adapter_dir:
        return False
    return (Path(adapter_dir) / STATE_FILE).exists()

def stats(model, init_snapshot=None):
    bank = get_bank(model)
    out = {}
    for (k, name) in enumerate(ROLE_VECTOR_NAMES):
        (norms, gnorms, moves) = ([], [], [])
        for (i, v) in enumerate(bank.vectors):
            norms.append(float(v.data[k].norm()))
            if v.grad is not None:
                gnorms.append(float(v.grad[k].norm()))
            if init_snapshot is not None:
                moves.append(float((v.data[k].cpu() - init_snapshot[i][k]).norm()))
        rec = {'mean_norm': sum(norms) / len(norms), 'max_norm': max(norms)}
        if gnorms:
            rec['mean_grad_norm'] = sum(gnorms) / len(gnorms)
            rec['n_zero_grad_modules'] = sum((1 for g in gnorms if g == 0.0))
        if moves:
            rec['mean_move_from_init'] = sum(moves) / len(moves)
        out[name] = rec
    return out

def snapshot(model):
    bank = get_bank(model)
    return [v.detach().cpu().clone() for v in bank.vectors]
