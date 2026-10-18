import json
import math
import re
from pathlib import Path
import torch
import torch.nn as nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask, eager_mask
_REG_NAME = 'structural_bias'
_ALLOWED_BASE = ('sdpa', 'eager')
_BASE_MASK_FN = {'sdpa': sdpa_mask, 'eager': eager_mask}

class _AttnCtx:
    active = False
    bias_module = None
    lambda_module = None
    base_attn_forward = None
    task_ids = None
    apply_lambda = True
    role_ids = None
    role_eval_mode = 'learned'
    rowcol_module = None
    expert_eval_mode = 'learned'
    view_ids = None
    expert_mode = 'mixture'
    row_id = None
    col_id = None
    is_header = None
    is_content = None
    rel_full = None
    debug_masks = None
    head_chunk = None
    debug_bias = None
CTX = _AttnCtx()

def set_structural_context(task_ids, row_id, col_id, is_header, is_content, apply_lambda=True, role_ids=None, view_ids=None, expert_mode='mixture'):
    if expert_mode not in EXPERT_MODES:
        raise ValueError(f'expert_mode must be one of {EXPERT_MODES} (got {expert_mode!r})')
    CTX.role_ids = role_ids
    CTX.view_ids = view_ids
    CTX.expert_mode = expert_mode
    CTX.task_ids = task_ids
    CTX.row_id = row_id
    CTX.col_id = col_id
    CTX.is_header = is_header
    CTX.is_content = is_content
    CTX.apply_lambda = apply_lambda
    CTX.rel_full = None
    CTX.active = True

def clear_structural_context():
    CTX.active = False
    CTX.task_ids = CTX.row_id = CTX.col_id = CTX.is_header = CTX.is_content = None
    CTX.apply_lambda = True
    CTX.rel_full = None
    CTX.role_ids = None
    CTX.view_ids = None
    CTX.expert_mode = 'mixture'
(_REL_NONE, _REL_ROW, _REL_COL) = (0, 1, 2)

def _relation_from_vectors(r_q, c_q, h_q, cnt_q, r_k, c_k, h_k, cnt_k):
    ri = r_q.unsqueeze(2)
    rj = r_k.unsqueeze(1)
    ci = c_q.unsqueeze(2)
    cj = c_k.unsqueeze(1)
    hi = h_q.unsqueeze(2).bool()
    hj = h_k.unsqueeze(1).bool()
    both = cnt_q.unsqueeze(2).bool() & cnt_k.unsqueeze(1).bool()
    same_row = both & ~hi & ~hj & (ri == rj) & (ci != cj)
    same_col = both & (ci == cj) & (ri != rj)
    rel = same_row.to(torch.uint8)
    rel += same_col.to(torch.uint8) * _REL_COL
    return rel

class _StructuralBiasFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, beta_row_b, beta_col_b, relation, out_dtype):
        ctx.save_for_backward(relation)
        br = beta_row_b.to(out_dtype)[:, None, None, None]
        bc = beta_col_b.to(out_dtype)[:, None, None, None]
        row = (relation == _REL_ROW)[:, None]
        bias = br * row
        col = (relation == _REL_COL)[:, None].to(out_dtype)
        bias.addcmul_(bc, col)
        return bias

    @staticmethod
    def backward(ctx, grad_out):
        (relation,) = ctx.saved_tensors
        (g_row, g_col) = _reduce_beta_grads(grad_out, relation)
        return (g_row, g_col, None, None)
_BWD_TILE_ELEMS = 1 << 21

def _reduce_beta_grads(grad_out, relation):
    (B, Hm, Tq, Tk) = grad_out.shape
    g_row = grad_out.new_zeros((B,), dtype=torch.float32)
    g_col = grad_out.new_zeros((B,), dtype=torch.float32)
    step = max(1, _BWD_TILE_ELEMS // max(1, Hm * Tk))
    for i in range(0, Tq, step):
        go = grad_out[:, :, i:i + step]
        rel = relation[:, None, i:i + step]
        row_prod = go * (rel == _REL_ROW)
        _record_bias_debug('bwd_row_tile', row_prod)
        g_row += row_prod.sum(dim=(1, 2, 3), dtype=torch.float32)
        col_prod = go * (rel == _REL_COL)
        _record_bias_debug('bwd_col_tile', col_prod)
        g_col += col_prod.sum(dim=(1, 2, 3), dtype=torch.float32)
    return (g_row, g_col)

def _current_relation(q_len, k_len, dev, cache_position):
    row_id = CTX.row_id
    col_id = CTX.col_id
    is_hdr = CTX.is_header
    is_cnt = CTX.is_content
    T = row_id.shape[1]
    if q_len == T and k_len == T:
        if CTX.rel_full is None:
            CTX.rel_full = _relation_from_vectors(row_id, col_id, is_hdr, is_cnt, row_id, col_id, is_hdr, is_cnt)
        return CTX.rel_full
    q_pos = torch.arange(q_len, device=dev) if cache_position is None else cache_position.to(dev)
    k_pos = torch.arange(k_len, device=dev)
    qv = (q_pos < T).long()
    kv = (k_pos < T).long()
    qi = q_pos.clamp(max=T - 1)
    ki = k_pos.clamp(max=T - 1)
    return _relation_from_vectors(row_id[:, qi], col_id[:, qi], is_hdr[:, qi], is_cnt[:, qi] * qv[None, :], row_id[:, ki], col_id[:, ki], is_hdr[:, ki], is_cnt[:, ki] * kv[None, :])

def _to_additive_mask(attention_mask, q_len, k_len, dtype, dev, cache_position):
    minv = torch.finfo(dtype).min
    if attention_mask is None:
        if q_len == k_len:
            m = torch.triu(torch.full((q_len, k_len), minv, dtype=dtype, device=dev), diagonal=1)
        else:
            m = torch.zeros((q_len, k_len), dtype=dtype, device=dev)
        return m[None, None]
    am = attention_mask[..., :k_len]
    if am.dtype == torch.bool:
        return torch.where(am, torch.zeros((), dtype=dtype, device=dev), torch.full((), minv, dtype=dtype, device=dev))
    return am.to(dtype)

def structural_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    base = CTX.base_attn_forward
    if CTX.debug_masks is not None and getattr(module, 'layer_idx', None) == 0:
        CTX.debug_masks.append({'active': bool(CTX.active), 'mask_is_none': attention_mask is None, 'mask_shape': None if attention_mask is None else tuple(attention_mask.shape), 'mask_dtype': None if attention_mask is None else str(attention_mask.dtype)})
    if not CTX.active or CTX.bias_module is None:
        return base(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    L = module.layer_idx
    br = CTX.bias_module.beta_row[CTX.task_ids, L]
    bc = CTX.bias_module.beta_col[CTX.task_ids, L]
    if not torch.is_grad_enabled() and (not (bool(torch.any(br != 0)) or bool(torch.any(bc != 0)))):
        return base(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    q_len = query.shape[2]
    k_len = key.shape[2]
    dev = query.device
    dtype = query.dtype
    relation = _current_relation(q_len, k_len, dev, kwargs.get('cache_position', None))
    add_mask = _to_additive_mask(attention_mask, q_len, k_len, dtype, dev, kwargs.get('cache_position', None))
    bias = _StructuralBiasFn.apply(br, bc, relation, dtype)
    _record_bias_debug('struct_bias', bias)
    bias.add_(add_mask)
    _record_bias_debug('new_mask', bias)
    chunk = _resolve_head_chunk(CTX.head_chunk, query.shape[1], key.shape[1])
    if chunk is not None:
        return _chunked_structural_attention(base, module, query, key, value, bias, chunk, scaling, dropout, kwargs)
    return base(module, query, key, value, bias, scaling=scaling, dropout=dropout, **kwargs)

def _resolve_head_chunk(req, n_q_heads, n_kv_heads):
    if not req or int(req) <= 0 or int(req) >= n_q_heads:
        return None
    n_rep = max(1, n_q_heads // max(1, n_kv_heads))
    aligned = max(n_rep, int(req) // n_rep * n_rep)
    return None if aligned >= n_q_heads else aligned

def _chunked_structural_attention(base, module, query, key, value, mask, chunk, scaling, dropout, kwargs):
    n_q_heads = query.shape[1]
    n_rep = max(1, n_q_heads // key.shape[1])
    outs = []
    for hs in range(0, n_q_heads, chunk):
        he = min(hs + chunk, n_q_heads)
        (ks, ke) = (hs // n_rep, he // n_rep)
        o_c = base(module, query[:, hs:he], key[:, ks:ke], value[:, ks:ke], mask, scaling=scaling, dropout=dropout, **kwargs)
        outs.append(o_c[0] if isinstance(o_c, tuple) else o_c)
    return (torch.cat(outs, dim=2), None)

def _record_bias_debug(tag, t):
    if CTX.debug_bias is not None:
        CTX.debug_bias.append({'tag': tag, 'dtype': str(t.dtype), 'shape': tuple(t.shape)})

def structural_bias_report(bias_module, task_ids, row_id, col_id, is_header, is_content, compute_dtype):
    dev = row_id.device
    T = row_id.shape[1]
    L = 0
    br = bias_module.beta_row[task_ids, L]
    bc = bias_module.beta_col[task_ids, L]
    relation = _relation_from_vectors(row_id, col_id, is_header, is_content, row_id, col_id, is_header, is_content)
    add_mask = _to_additive_mask(None, T, T, compute_dtype, dev, None)
    prev = CTX.debug_bias
    CTX.debug_bias = []
    try:
        bias = _StructuralBiasFn.apply(br, bc, relation, compute_dtype)
        _record_bias_debug('struct_bias', bias)
        bias.add_(add_mask)
        _record_bias_debug('new_mask', bias)
        report = list(CTX.debug_bias)
    finally:
        CTX.debug_bias = prev
    peak_bytes = max((torch.empty(0, dtype=getattr(torch, r['dtype'].split('.')[-1])).element_size() * _numel(r['shape']) for r in report), default=0)
    return (report, peak_bytes)

def _numel(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n
ALL_ATTENTION_FUNCTIONS.register(_REG_NAME, structural_attention_forward)
AttentionMaskInterface.register(_REG_NAME, _BASE_MASK_FN['sdpa'])

class StructuralBias(nn.Module):

    def __init__(self, n_tasks, n_layers, task_names=None):
        super().__init__()
        self.n_tasks = int(n_tasks)
        self.n_layers = int(n_layers)
        self.task_names = list(task_names) if task_names is not None else None
        self.beta_row = nn.Parameter(torch.zeros(self.n_tasks, self.n_layers, dtype=torch.float32))
        self.beta_col = nn.Parameter(torch.zeros(self.n_tasks, self.n_layers, dtype=torch.float32))

    def n_params(self):
        return self.beta_row.numel() + self.beta_col.numel()

    def summary(self):
        with torch.no_grad():
            return {'n_tasks': self.n_tasks, 'n_layers': self.n_layers, 'n_beta_params': self.n_params(), 'beta_row_absmax': float(self.beta_row.detach().abs().max().item()), 'beta_col_absmax': float(self.beta_col.detach().abs().max().item()), 'beta_row_absmean': float(self.beta_row.detach().abs().mean().item()), 'beta_col_absmean': float(self.beta_col.detach().abs().mean().item())}

    def per_task_table(self):
        out = {}
        with torch.no_grad():
            for t in range(self.n_tasks):
                name = self.task_names[t] if self.task_names else str(t)
                out[name] = {'beta_row': [float(x) for x in self.beta_row[t].detach().cpu().tolist()], 'beta_col': [float(x) for x in self.beta_col[t].detach().cpu().tolist()]}
        return out
_LAYER_IDX_RE = re.compile('\\blayers\\.(\\d+)\\b')
LAMBDA_MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
N_ROLES = 2
(ROLE_TEXT, ROLE_TABLE) = (0, 1)
ROLE_EVAL_MODES = ('learned', 'tie_text', 'tie_table', 'identity')

class RoleLambda(nn.Module):

    def __init__(self, n_layers, lora_r, task_names=None, module_names=LAMBDA_MODULES, n_roles=N_ROLES):
        super().__init__()
        self.n_layers = int(n_layers)
        self.lora_r = int(lora_r)
        self.n_roles = int(n_roles)
        self.task_names = list(task_names) if task_names is not None else None
        self.module_names = list(module_names)
        self.n_modules = len(self.module_names)
        shape = (self.n_layers, self.n_modules, self.n_roles, self.lora_r, self.lora_r)
        self.R = nn.Parameter(torch.eye(self.lora_r, dtype=torch.float32).expand(shape).clone())

    def n_params(self):
        return self.R.numel()

    def _role_diff(self):
        with torch.no_grad():
            if self.n_roles < 2:
                return (0.0, 0.0)
            g = self.R.detach()
            diffs = []
            for i in range(self.n_roles):
                for j in range(i + 1, self.n_roles):
                    diffs.append((g[:, :, i] - g[:, :, j]).abs())
            d = torch.stack(diffs)
            return (float(d.mean().item()), float(d.max().item()))

    def summary(self):
        with torch.no_grad():
            eye = torch.eye(self.lora_r, dtype=self.R.dtype, device=self.R.device)
            dev = (self.R.detach() - eye).abs()
            (rdiff_mean, rdiff_max) = self._role_diff()
            return {'n_roles': self.n_roles, 'n_layers': self.n_layers, 'n_modules': self.n_modules, 'module_names': list(self.module_names), 'task_names': list(self.task_names) if self.task_names else None, 'lora_r': self.lora_r, 'n_lambda_params': self.n_params(), 'R_min': float(self.R.detach().min().item()), 'R_max': float(self.R.detach().max().item()), 'R_dev_from_identity_absmean': float(dev.mean().item()), 'R_dev_from_identity_absmax': float(dev.max().item()), 'R_role_diff_absmean': rdiff_mean, 'R_role_diff_absmax': rdiff_max}

    def per_role_table(self):
        out = {}
        names = {ROLE_TEXT: 'text', ROLE_TABLE: 'table'}
        with torch.no_grad():
            g = self.R.detach().cpu()
            for k in range(self.n_roles):
                layers = {}
                for l in range(self.n_layers):
                    layers[f'layer_{l}'] = {self.module_names[mi]: [[float(x) for x in row] for row in g[l, mi, k].tolist()] for mi in range(self.n_modules)}
                out[names.get(k, str(k))] = layers
        return out

    @torch.no_grad()
    def diagnostics(self):
        R = self.R.detach().to('cpu', torch.float64)
        eye = torch.eye(self.lora_r, dtype=torch.float64)
        per_layer = {}
        (rel_fro_all, svals_all) = ([], [])
        dev_all = {k: [] for k in range(self.n_roles)}
        n_pinv = 0
        for l in range(self.n_layers):
            mods = {}
            for (mi, mname) in enumerate(self.module_names):
                ent = {}
                for k in range(self.n_roles):
                    f = float(torch.linalg.matrix_norm(R[l, mi, k] - eye, ord='fro').item())
                    ent[f'R_minus_I_fro_role{k}'] = f
                    dev_all[k].append(f)
                if self.n_roles == 2:
                    (r0, r1) = (R[l, mi, ROLE_TEXT], R[l, mi, ROLE_TABLE])
                    try:
                        rel = r1 @ torch.linalg.inv(r0)
                    except Exception:
                        rel = r1 @ torch.linalg.pinv(r0)
                        ent['rel_used_pinv'] = True
                        n_pinv += 1
                    ent['rel_fro'] = float(torch.linalg.matrix_norm(rel - eye, ord='fro').item())
                    sv = torch.linalg.svdvals(rel)
                    ent['rel_svals'] = [float(x) for x in sv.tolist()]
                    rel_fro_all.append(ent['rel_fro'])
                    svals_all.append(sv)
                mods[mname] = ent
            per_layer[f'layer_{l}'] = mods
        glob = {f'mean_R_minus_I_fro_role{k}': float(sum(v) / len(v)) if v else 0.0 for (k, v) in dev_all.items()}
        if rel_fro_all:
            glob['mean_rel_fro'] = float(sum(rel_fro_all) / len(rel_fro_all))
            sv = torch.stack(svals_all)
            glob['mean_rel_svals'] = [float(x) for x in sv.mean(0).tolist()]
            glob['max_rel_sval'] = float(sv.max().item())
            glob['min_rel_sval'] = float(sv.min().item())
        glob['n_rel_pinv_fallback'] = n_pinv
        return {'summary': self.summary(), 'global': glob, 'per_layer': per_layer}

def set_role_eval_mode(mode):
    if mode not in ROLE_EVAL_MODES:
        raise ValueError(f'role_eval_mode must be one of {ROLE_EVAL_MODES}, got {mode!r}')
    CTX.role_eval_mode = mode

def set_expert_eval_mode(mode):
    if mode not in EXPERT_EVAL_MODES:
        raise ValueError(f'expert_eval_mode must be one of {EXPERT_EVAL_MODES}, got {mode!r}')
    CTX.expert_eval_mode = mode

def _role_matrices(lam, layer_idx, module_idx, dtype):
    mode = CTX.role_eval_mode or 'learned'
    R = lam.R[layer_idx, module_idx]
    if mode == 'identity':
        eye = torch.eye(lam.lora_r, dtype=dtype, device=R.device)
        return (eye, eye)
    if mode == 'tie_text':
        m = R[ROLE_TEXT].to(dtype)
        return (m, m)
    if mode == 'tie_table':
        m = R[ROLE_TABLE].to(dtype)
        return (m, m)
    if mode != 'learned':
        raise RuntimeError(f'unknown role_eval_mode {mode!r}; expected one of {ROLE_EVAL_MODES}')
    return (R[ROLE_TEXT].to(dtype), R[ROLE_TABLE].to(dtype))

def _make_lambda_hook(layer_idx, module_idx):

    def hook(_mod, _inp, out):
        lam = CTX.lambda_module
        if not CTX.active or not CTX.apply_lambda or lam is None:
            return out
        role = CTX.role_ids
        if role is None:
            raise RuntimeError('R-LoRA: CTX.role_ids is None during a forward with the role hooks attached. Pass role_ids to attn_lora.set_structural_context() before the forward (or set apply_lambda=False to deliberately run ordinary shared LoRA).')
        (m_text, m_table) = _role_matrices(lam, layer_idx, module_idx, out.dtype)
        t_out = out.shape[-2] if out.dim() >= 2 else 1
        t_role = role.shape[-1]
        if t_out != t_role:
            if t_out <= t_role:
                return out @ m_text.transpose(0, 1)
            raise RuntimeError(f'R-LoRA: rank code has {t_out} positions but role_ids has {t_role}; the role vector must cover the whole padded batch.')
        h0 = out @ m_text.transpose(0, 1)
        h1 = out @ m_table.transpose(0, 1)
        sel = role.bool()
        while sel.dim() < out.dim():
            sel = sel.unsqueeze(-1)
        return torch.where(sel, h1, h0)
    return hook

def attach_task_lambda(model, lambda_module):
    CTX.lambda_module = lambda_module
    handles = []
    counts = {}
    allowed = set(lambda_module.module_names)
    for (name, m) in model.named_modules():
        la = getattr(m, 'lora_A', None)
        if not isinstance(la, nn.ModuleDict):
            continue
        mod_type = name.split('.')[-1]
        if mod_type not in allowed:
            continue
        mt = _LAYER_IDX_RE.search(name)
        if mt is None:
            raise RuntimeError(f'attach_task_lambda: cannot parse layer index from {name!r}')
        layer_idx = int(mt.group(1))
        if layer_idx >= lambda_module.n_layers:
            raise RuntimeError(f'attach_task_lambda: layer {layer_idx} >= n_layers {lambda_module.n_layers} for {name!r}')
        module_idx = lambda_module.module_names.index(mod_type)
        for (_adapter, sub) in la.items():
            r = sub.out_features
            if r != lambda_module.lora_r:
                raise RuntimeError(f'attach_task_lambda: LoRA rank {r} for {name!r} != lambda_module.lora_r {lambda_module.lora_r}')
            handles.append(sub.register_forward_hook(_make_lambda_hook(layer_idx, module_idx)))
        counts[mod_type] = counts.get(mod_type, 0) + 1
    if not handles:
        raise RuntimeError(f'attach_task_lambda: found no LoRA modules to hook for {list(lambda_module.module_names)} -- is the model a PEFT LoRA model whose target modules include the selected lambda modules?')
    for mod_type in lambda_module.module_names:
        got = counts.get(mod_type, 0)
        if got != lambda_module.n_layers:
            raise RuntimeError(f"attach_task_lambda: {mod_type} attached to {got} layers, expected {lambda_module.n_layers} (every decoder layer). Model's LoRA {mod_type} coverage does not match the R table (is {mod_type} a LoRA target module?).")
    CTX.lambda_handles = handles
    return counts

def detach_task_lambda():
    for h in getattr(CTX, 'lambda_handles', []) or []:
        h.remove()
    CTX.lambda_handles = []
    CTX.lambda_module = None
EXPERT_EVAL_MODES = ('learned', 'row_only', 'col_only', 'identity')
EXPERT_MODES = ('anchor', 'mixture')
(VIEW_ROW, VIEW_COL) = (0, 1)
VIEW_NAMES = {VIEW_ROW: 'row_major', VIEW_COL: 'col_major'}

def _bview(x, ndim):
    return x.view(-1, *[1] * (ndim - 1))

class RowColExperts(nn.Module):

    def __init__(self, n_tasks, n_layers, lora_r, task_names=None, module_names=LAMBDA_MODULES, use_lambda=True):
        super().__init__()
        self.n_tasks = int(n_tasks)
        self.n_layers = int(n_layers)
        self.lora_r = int(lora_r)
        self.task_names = list(task_names) if task_names is not None else None
        self.module_names = list(module_names)
        self.n_modules = len(self.module_names)
        self.use_lambda = bool(use_lambda)
        _R = torch.zeros(self.n_tasks, self.n_layers, self.n_modules, self.lora_r, self.lora_r, dtype=torch.float32)
        if self.use_lambda:
            self.R = nn.Parameter(_R)
        else:
            self.register_buffer('R', _R, persistent=True)
        self.g = nn.Parameter(torch.zeros(self.n_tasks, self.n_layers, self.n_modules, 2, dtype=torch.float32))
        self.b_col = nn.ParameterDict()
        self._out_features = {}

    @staticmethod
    def _key(layer_idx, module_idx):
        return f'l{layer_idx}_m{module_idx}'

    def add_b(self, layer_idx, module_idx, out_features, dtype=torch.float32, device=None):
        k = self._key(layer_idx, module_idx)
        if k in self.b_col:
            return self.b_col[k]
        p = nn.Parameter(torch.zeros(int(out_features), self.lora_r, dtype=dtype, device=device))
        self.b_col[k] = p
        self._out_features[k] = int(out_features)
        return p

    def get_b(self, layer_idx, module_idx):
        return self.b_col[self._key(layer_idx, module_idx)]

    def weights(self):
        return torch.softmax(self.g, dim=-1)

    def n_lambda_params(self):
        return self.R.numel() if self.use_lambda else 0

    def n_gate_params(self):
        return self.g.numel()

    def n_b_params(self):
        return sum((p.numel() for p in self.b_col.values()))

    def n_params(self):
        return self.n_lambda_params() + self.n_gate_params() + self.n_b_params()

    def _task_diff(self):
        with torch.no_grad():
            if self.n_tasks < 2:
                return (0.0, 0.0)
            g = self.R.detach()
            diffs = [(g[i] - g[j]).abs() for i in range(self.n_tasks) for j in range(i + 1, self.n_tasks)]
            d = torch.stack(diffs)
            return (float(d.mean().item()), float(d.max().item()))

    def task_name(self, t):
        return self.task_names[t] if self.task_names else str(t)

    @torch.no_grad()
    def gate_means(self):
        w = self.weights().detach().to(torch.float64)
        out = {}
        for t in range(self.n_tasks):
            wr = w[t, :, :, 0]
            out[self.task_name(t)] = {'w_row_mean': float(wr.mean().item()), 'w_col_mean': float(1.0 - wr.mean().item()), 'w_row_min': float(wr.min().item()), 'w_row_max': float(wr.max().item()), 'w_row_std': float(wr.std(unbiased=False).item())}
        return out

    @torch.no_grad()
    def summary(self):
        dev = self.R.detach().abs()
        (tmean, tmax) = self._task_diff()
        bc = [p.detach() for p in self.b_col.values()]
        b_norm = float(torch.stack([p.norm() for p in bc]).mean().item()) if bc else 0.0
        return {'variant': 'rowcol_experts' if self.use_lambda else 'rowcol_experts_no_lambda', 'use_lambda': self.use_lambda, 'n_tasks': self.n_tasks, 'n_layers': self.n_layers, 'n_modules': self.n_modules, 'module_names': list(self.module_names), 'task_names': list(self.task_names) if self.task_names else None, 'lora_r': self.lora_r, 'n_lambda_params': self.n_lambda_params(), 'n_gate_params': self.n_gate_params(), 'n_b_col_params': self.n_b_params(), 'n_params_total': self.n_params(), 'lambda_dev_from_identity_absmean': float(dev.mean().item()), 'lambda_dev_from_identity_absmax': float(dev.max().item()), 'lambda_task_diff_absmean': tmean, 'lambda_task_diff_absmax': tmax, 'b_col_fro_mean': b_norm, 'gate_means': self.gate_means()}

    def per_task_table(self):
        out = {}
        with torch.no_grad():
            lam = (self.R.detach() + torch.eye(self.lora_r, device=self.R.device, dtype=self.R.dtype)).cpu()
            w = self.weights().detach().cpu()
            for t in range(self.n_tasks):
                layers = {}
                for l in range(self.n_layers):
                    layers[f'layer_{l}'] = {self.module_names[mi]: {'lambda': [[float(x) for x in row] for row in lam[t, l, mi].tolist()], 'w_row': float(w[t, l, mi, 0]), 'w_col': float(w[t, l, mi, 1])} for mi in range(self.n_modules)}
                out[self.task_name(t)] = layers
        return out

    @torch.no_grad()
    def grad_norms(self):
        out = {}
        for (name, p) in (('lambda', self.R), ('gate', self.g)):
            if name == 'lambda' and (not self.use_lambda):
                out['lambda_grad_norm'] = None
                out['lambda_grad_norm_by_task'] = None
                continue
            if p.grad is None:
                out[f'{name}_grad_norm'] = None
                out[f'{name}_grad_norm_by_task'] = None
                continue
            gr = p.grad.detach().to(torch.float64)
            out[f'{name}_grad_norm'] = float(gr.norm().item())
            out[f'{name}_grad_norm_by_task'] = {self.task_name(t): float(gr[t].norm().item()) for t in range(self.n_tasks)}
        return out

    @torch.no_grad()
    def diagnostics(self, model=None):
        R = self.R.detach().to('cpu', torch.float64)
        w = self.weights().detach().to('cpu', torch.float64)
        b_row = {}
        if model is not None:
            for (name, m) in model.named_modules():
                lb = getattr(m, 'lora_B', None)
                if not isinstance(lb, nn.ModuleDict):
                    continue
                mod_type = name.split('.')[-1]
                if mod_type not in self.module_names:
                    continue
                mt = _LAYER_IDX_RE.search(name)
                if mt is None:
                    continue
                for (_a, sub) in lb.items():
                    b_row[self._key(int(mt.group(1)), self.module_names.index(mod_type))] = float(sub.weight.detach().to(torch.float64).norm().item())
        per_layer = {}
        (lam_fro, w_row_all) = ([], {t: [] for t in range(self.n_tasks)})
        for l in range(self.n_layers):
            mods = {}
            for (mi, mname) in enumerate(self.module_names):
                ent = {}
                for t in range(self.n_tasks):
                    tn = self.task_name(t)
                    f = float(torch.linalg.matrix_norm(R[t, l, mi], ord='fro').item())
                    ent[f'lambda_minus_I_fro_task_{tn}'] = f
                    ent[f'w_row_task_{tn}'] = float(w[t, l, mi, 0].item())
                    ent[f'w_col_task_{tn}'] = float(w[t, l, mi, 1].item())
                    lam_fro.append(f)
                    w_row_all[t].append(float(w[t, l, mi, 0].item()))
                k = self._key(l, mi)
                if k in self.b_col:
                    ent['b_col_fro'] = float(self.b_col[k].detach().to(torch.float64).norm().item())
                if k in b_row:
                    ent['b_row_fro'] = b_row[k]
                    if ent.get('b_col_fro') is not None and b_row[k] > 0:
                        ent['b_col_over_b_row'] = ent['b_col_fro'] / b_row[k]
                mods[mname] = ent
            per_layer[f'layer_{l}'] = mods
        glob = {'mean_lambda_minus_I_fro': float(sum(lam_fro) / len(lam_fro)) if lam_fro else 0.0, 'mean_w_row_by_task': {self.task_name(t): float(sum(v) / len(v)) if v else 0.5 for (t, v) in w_row_all.items()}}
        if b_row:
            glob['mean_b_row_fro'] = float(sum(b_row.values()) / len(b_row))
        if self.b_col:
            glob['mean_b_col_fro'] = float(sum((float(p.detach().norm().item()) for p in self.b_col.values())) / len(self.b_col))
        return {'summary': self.summary(), 'global': glob, 'per_layer': per_layer}

def _make_rowcol_lambda_hook(layer_idx, module_idx):

    def hook(_mod, _inp, out):
        lm = CTX.rowcol_module
        if not CTX.active or not CTX.apply_lambda or lm is None:
            return out
        if not lm.use_lambda:
            return out
        if (CTX.expert_eval_mode or 'learned') == 'identity':
            return out
        if CTX.task_ids is None:
            raise RuntimeError('RC-LoRA: CTX.task_ids is None; Lambda is task-indexed. Pass task_ids to set_structural_context() before the forward.')
        R = lm.R[CTX.task_ids, layer_idx, module_idx].to(out.dtype)
        if out.dim() == 3:
            return out + torch.bmm(out, R.transpose(1, 2))
        return out + out @ R[0].transpose(0, 1)
    return hook

def _make_rowcol_b_hook(layer_idx, module_idx):

    def hook(_mod, inp, out):
        lm = CTX.rowcol_module
        if not CTX.active or not CTX.apply_lambda or lm is None:
            return out
        mode = CTX.expert_eval_mode or 'learned'
        if mode not in EXPERT_EVAL_MODES:
            raise RuntimeError(f'unknown expert_eval_mode {mode!r}; expected {EXPERT_EVAL_MODES}')
        if mode == 'row_only':
            return out
        z = inp[0]
        b_c = lm.get_b(layer_idx, module_idx).to(out.dtype)
        out_c = torch.nn.functional.linear(z, b_c)
        if mode == 'col_only':
            return out_c
        if CTX.task_ids is None:
            raise RuntimeError('RC-LoRA: CTX.task_ids is None; the expert gate is task-indexed.')
        w = lm.weights()[CTX.task_ids, layer_idx, module_idx].to(out.dtype)
        (w_row, w_col) = (w[:, 0], w[:, 1])
        if out.dim() < 2:
            raise RuntimeError(f'RC-LoRA: unexpected lora_B output rank {out.dim()}')
        if CTX.expert_mode == 'anchor':
            v = CTX.view_ids
            if v is None:
                raise RuntimeError("RC-LoRA: expert_mode='anchor' needs CTX.view_ids ([B] 0=row-major / 1=col-major). Pass view_ids to set_structural_context(), or use expert_mode='mixture' for validation / evaluation.")
            is_row = v == VIEW_ROW
            coef = torch.where(is_row, w_row, w_col)
            base = torch.where(_bview(is_row, out.dim()), out, out_c)
            return _bview(coef, out.dim()) * base
        return _bview(w_row, out.dim()) * out + _bview(w_col, out.dim()) * out_c
    return hook

def attach_rowcol_experts(model, rc_module):
    CTX.rowcol_module = rc_module
    handles = []
    counts = {}
    allowed = set(rc_module.module_names)
    for (name, m) in model.named_modules():
        la = getattr(m, 'lora_A', None)
        lb = getattr(m, 'lora_B', None)
        if not isinstance(la, nn.ModuleDict) or not isinstance(lb, nn.ModuleDict):
            continue
        mod_type = name.split('.')[-1]
        if mod_type not in allowed:
            continue
        mt = _LAYER_IDX_RE.search(name)
        if mt is None:
            raise RuntimeError(f'attach_rowcol_experts: cannot parse layer index from {name!r}')
        layer_idx = int(mt.group(1))
        if layer_idx >= rc_module.n_layers:
            raise RuntimeError(f'attach_rowcol_experts: layer {layer_idx} >= n_layers {rc_module.n_layers} for {name!r}')
        module_idx = rc_module.module_names.index(mod_type)
        for (_adapter, sub) in la.items():
            r = sub.out_features
            if r != rc_module.lora_r:
                raise RuntimeError(f'attach_rowcol_experts: LoRA rank {r} for {name!r} != rc_module.lora_r {rc_module.lora_r}')
            handles.append(sub.register_forward_hook(_make_rowcol_lambda_hook(layer_idx, module_idx)))
        for (_adapter, sub) in lb.items():
            if sub.in_features != rc_module.lora_r:
                raise RuntimeError(f'attach_rowcol_experts: lora_B in_features {sub.in_features} for {name!r} != lora_r {rc_module.lora_r}')
            rc_module.add_b(layer_idx, module_idx, sub.out_features, dtype=torch.float32, device=sub.weight.device)
            handles.append(sub.register_forward_hook(_make_rowcol_b_hook(layer_idx, module_idx)))
        counts[mod_type] = counts.get(mod_type, 0) + 1
    if not handles:
        raise RuntimeError(f'attach_rowcol_experts: found no LoRA modules to hook for {list(rc_module.module_names)} -- is the model a PEFT LoRA model whose target modules include the selected modules?')
    for mod_type in rc_module.module_names:
        got = counts.get(mod_type, 0)
        if got != rc_module.n_layers:
            raise RuntimeError(f"attach_rowcol_experts: {mod_type} attached to {got} layers, expected {rc_module.n_layers} (every decoder layer). Model's LoRA {mod_type} coverage does not match the table.")
    CTX.rowcol_handles = handles
    return counts

def detach_rowcol_experts():
    for h in getattr(CTX, 'rowcol_handles', []) or []:
        h.remove()
    CTX.rowcol_handles = []
    CTX.rowcol_module = None

def save_rowcol_experts(rc_module, out_dir, config):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'R': rc_module.R.detach().cpu(), 'use_lambda': bool(rc_module.use_lambda), 'g': rc_module.g.detach().cpu(), 'b_col': {k: p.detach().cpu() for (k, p) in rc_module.b_col.items()}, 'module_names': list(rc_module.module_names), 'task_names': rc_module.task_names, 'lora_r': int(rc_module.lora_r), 'variant': 'rowcol_experts'}, out / 'rowcol_experts.pt')
    with open(out / 'rowcol_experts_config.json', 'w') as f:
        json.dump(config, f, indent=2)

def load_rowcol_experts(adapter_dir):
    d = Path(adapter_dir)
    p = d / 'rowcol_experts.pt'
    if not p.exists():
        if (d / 'role_b_lora.pt').exists():
            raise RuntimeError(f'{d}: this adapter was trained with the REMOVED RB-LoRA variant (role_b_lora.pt: task Lambda + per-TOKEN-ROLE B). RC-LoRA replaced it -- the two second-B matrices mean different things and are not interchangeable. Re-train, or check out the revision that wrote it.')
        return (None, {})
    state = torch.load(p, map_location='cpu')
    R = state['R']
    if R.dim() != 5:
        raise RuntimeError(f'{p}: expected a 5-D R [n_tasks, n_layers, n_modules, r, r], got {tuple(R.shape)}.')
    (n_tasks, n_layers, n_modules, r, r2) = (int(x) for x in R.shape)
    if r != r2:
        raise RuntimeError(f'{p}: Lambda is not square in the rank dims: {tuple(R.shape)}')
    g = state['g']
    if tuple(g.shape) != (n_tasks, n_layers, n_modules, 2):
        raise RuntimeError(f'{p}: gate logits have shape {tuple(g.shape)}, expected {(n_tasks, n_layers, n_modules, 2)}')
    module_names = state.get('module_names', list(LAMBDA_MODULES))
    if len(module_names) != n_modules:
        raise RuntimeError(f'{p}: {len(module_names)} module names for {n_modules} slots')
    rc = RowColExperts(n_tasks, n_layers, r, task_names=state.get('task_names'), module_names=module_names, use_lambda=bool(state.get('use_lambda', True)))
    rc.R.data.copy_(R)
    rc.g.data.copy_(g)
    for (k, v) in state['b_col'].items():
        rc.b_col[k] = nn.Parameter(v.clone())
        rc._out_features[k] = int(v.shape[0])
    cfg_path = d / 'rowcol_experts_config.json'
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    return (rc, cfg)

def set_head_chunk(head_chunk):
    CTX.head_chunk = int(head_chunk) if head_chunk else None

def attach_structural_attention(model, bias_module, base_impl='sdpa', head_chunk=None):
    if base_impl not in _ALLOWED_BASE:
        raise ValueError(f'base_impl must be one of {_ALLOWED_BASE} (got {base_impl!r}); flash_attention_2 cannot take the additive structural mask.')
    set_head_chunk(head_chunk)
    if base_impl == 'eager':
        import transformers.models.llama.modeling_llama as _llama
        CTX.base_attn_forward = _llama.eager_attention_forward
    else:
        CTX.base_attn_forward = ALL_ATTENTION_FUNCTIONS[base_impl]
    AttentionMaskInterface.register(_REG_NAME, _BASE_MASK_FN[base_impl])
    CTX.bias_module = bias_module
    n_set = 0
    for m in model.modules():
        cfg = getattr(m, 'config', None)
        if cfg is not None and hasattr(cfg, '_attn_implementation') and m.__class__.__name__.endswith('Attention'):
            cfg._attn_implementation = _REG_NAME
            n_set += 1
    for cfg in _iter_configs(model):
        if hasattr(cfg, '_attn_implementation'):
            cfg._attn_implementation = _REG_NAME
    if n_set == 0:
        raise RuntimeError('attach_structural_attention: no *Attention submodules found to patch')
    return n_set

def _iter_configs(model):
    seen = []
    for attr in ('config',):
        c = getattr(model, attr, None)
        if c is not None:
            seen.append(c)
    base = getattr(model, 'base_model', None)
    if base is not None:
        inner = getattr(base, 'model', None)
        for obj in (base, inner):
            c = getattr(obj, 'config', None)
            if c is not None:
                seen.append(c)
    return seen

def count_layers(model):
    n = 0
    for m in model.modules():
        if m.__class__.__name__.endswith('Attention') and hasattr(m, 'layer_idx'):
            n = max(n, int(m.layer_idx) + 1)
    if n == 0:
        raise RuntimeError('count_layers: could not find attention modules with layer_idx')
    return n

def _is_separator_line(line):
    s = line.strip()
    if not s.startswith('|'):
        return False
    cells = [c.strip() for c in _split_cells(s)]
    return len(cells) > 0 and all((c != '' and set(c) <= set('-: ') for c in cells))

def _split_cells(line):
    s = line
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    return s.split('|')

def char_struct_map(text):
    n = len(text)
    row_id = [0] * n
    col_id = [0] * n
    is_hdr = [0] * n
    is_cnt = [0] * n
    lines = text.splitlines(keepends=True)
    offs = []
    p = 0
    for ln in lines:
        offs.append(p)
        p += len(ln)
    is_tbl = [ln.lstrip().startswith('|') for ln in lines]
    row_counter = 0
    col_key = {}
    col_counter = 0
    block_idx = 0
    i = 0
    while i < len(lines):
        if not is_tbl[i]:
            i += 1
            continue
        j = i
        while j < len(lines) and is_tbl[j]:
            j += 1
        header_seen = False
        for k in range(i, j):
            ln = lines[k]
            base = offs[k]
            sep = _is_separator_line(ln)
            if sep:
                continue
            row_counter += 1
            cur_row = row_counter
            header = not header_seen
            header_seen = True
            c_idx = 0
            for (t, ch) in enumerate(ln):
                g = base + t
                if ch == '\n' or ch == '\r':
                    continue
                if ch == '|':
                    c_idx += 1
                    continue
                if c_idx < 1:
                    continue
                key = (block_idx, c_idx)
                if key not in col_key:
                    col_counter += 1
                    col_key[key] = col_counter
                row_id[g] = cur_row
                col_id[g] = col_key[key]
                is_hdr[g] = 1 if header else 0
                if not ch.isspace():
                    is_cnt[g] = 1
        block_idx += 1
        i = j
    return (row_id, col_id, is_hdr, is_cnt)

def structure_from_offsets(text, offs):
    (cr, cc, ch_hdr, ch_cnt) = char_struct_map(text)
    (row_ids, col_ids, is_header, is_content) = ([], [], [], [])
    n_straddle = 0
    for (a, b) in offs:
        if a >= b:
            row_ids.append(0)
            col_ids.append(0)
            is_header.append(0)
            is_content.append(0)
            continue
        first = None
        cells = set()
        any_cnt = 0
        for g in range(a, b):
            if ch_cnt[g]:
                any_cnt = 1
                cells.add((cr[g], cc[g]))
                if first is None:
                    first = g
        if first is None:
            row_ids.append(0)
            col_ids.append(0)
            is_header.append(0)
            is_content.append(0)
        else:
            if len(cells) > 1:
                n_straddle += 1
            row_ids.append(cr[first])
            col_ids.append(cc[first])
            is_header.append(ch_hdr[first])
            is_content.append(any_cnt)
    return (row_ids, col_ids, is_header, is_content, n_straddle)

def char_table_span(text):
    n = len(text)
    span = [0] * n
    p = 0
    for ln in text.splitlines(keepends=True):
        if ln.lstrip().startswith('|'):
            for g in range(p, p + len(ln)):
                span[g] = 1
        p += len(ln)
    return span

def span_from_offsets(text, offs):
    sp = char_table_span(text)
    out = []
    for (a, b) in offs:
        if a >= b:
            out.append(0)
            continue
        out.append(1 if any((sp[g] for g in range(a, b))) else 0)
    return out

def tokens_structure_ext(tok, text, add_special_tokens=False):
    if not getattr(tok, 'is_fast', False):
        raise RuntimeError('attn_lora requires a fast tokenizer (needs return_offsets_mapping)')
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
    offs = enc['offset_mapping']
    (r, c, h, ct, ns) = structure_from_offsets(text, offs)
    return (enc['input_ids'], r, c, h, ct, span_from_offsets(text, offs), ns)

def tokens_structure(tok, text, add_special_tokens=False):
    if not getattr(tok, 'is_fast', False):
        raise RuntimeError('attn_lora requires a fast tokenizer (needs return_offsets_mapping)')
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
    ids = enc['input_ids']
    (r, c, h, ct, ns) = structure_from_offsets(text, enc['offset_mapping'])
    return (ids, r, c, h, ct, ns)

def pairwise_masks(row_id, col_id, is_header, is_content):
    row_id = row_id.long()
    col_id = col_id.long()
    hdr = is_header.bool()
    cnt = is_content.bool()
    ri = row_id.unsqueeze(2)
    rj = row_id.unsqueeze(1)
    ci = col_id.unsqueeze(2)
    cj = col_id.unsqueeze(1)
    ci_c = cnt.unsqueeze(2)
    cj_c = cnt.unsqueeze(1)
    hi = hdr.unsqueeze(2)
    hj = hdr.unsqueeze(1)
    both = ci_c & cj_c
    same_row = both & ~hi & ~hj & (ri == rj) & (ci != cj)
    same_col = both & (ci == cj) & (ri != rj)
    return (same_row, same_col)

def save_structural_bias(bias_module, out_dir, config, lambda_module=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if bias_module is not None:
        payload = {'beta_row': bias_module.beta_row.detach().cpu(), 'beta_col': bias_module.beta_col.detach().cpu(), 'task_names': bias_module.task_names}
    else:
        payload = {'task_names': config.get('task_names')}
    if lambda_module is not None:
        payload['lambda'] = lambda_module.R.detach().cpu()
        payload['lambda_modules'] = list(lambda_module.module_names)
        payload['lambda_r'] = int(lambda_module.lora_r)
        payload['n_roles'] = int(lambda_module.n_roles)
        payload['lambda_kind'] = 'role_full_matrix'
    torch.save(payload, out / 'structural_bias.pt')
    with open(out / 'attn_config.json', 'w') as f:
        json.dump(config, f, indent=2)

def load_structural_bias(adapter_dir):
    adapter_dir = Path(adapter_dir)
    cfg_path = adapter_dir / 'attn_config.json'
    if not cfg_path.exists():
        raise RuntimeError(f'{cfg_path} not found; not a structural-attention adapter.')
    cfg = json.loads(cfg_path.read_text())
    state = torch.load(adapter_dir / 'structural_bias.pt', map_location='cpu')
    if 'beta_row' not in state:
        return (None, cfg)
    task_names = state.get('task_names', cfg.get('task_names'))
    if state['beta_row'].dim() != 2:
        raise RuntimeError(f"{adapter_dir}/structural_bias.pt has a {state['beta_row'].dim()}-D beta_row {tuple(state['beta_row'].shape)}; this pipeline uses a per-task x per-layer beta table (2-D). Retrain the adapter with the current train_attn_lora.py.")
    (n_tasks, n_layers) = (int(x) for x in state['beta_row'].shape)
    bm = StructuralBias(n_tasks, n_layers, task_names=task_names)
    bm.beta_row.data.copy_(state['beta_row'])
    bm.beta_col.data.copy_(state['beta_col'])
    return (bm, cfg)

def load_task_lambda(adapter_dir):
    p = Path(adapter_dir) / 'structural_bias.pt'
    if not p.exists():
        return None
    state = torch.load(p, map_location='cpu')
    if 'lambda' not in state:
        return None
    lam = state['lambda']
    if lam.dim() == 4:
        raise RuntimeError(f'{p} holds a 4-D lambda {tuple(lam.shape)} -- a checkpoint from the superseded TASK-indexed DIAGONAL variant (B Lambda_t A x). This pipeline now trains the ROLE-conditioned full r x r variant (B R_role A x), whose table is 5-D [n_layers, n_modules, n_roles, r, r]. Retrain the adapter with the current train_attn_lora.py.')
    if lam.dim() != 5:
        raise RuntimeError(f'{p}: expected a 5-D role table [n_layers, n_modules, n_roles, r, r], got {tuple(lam.shape)}.')
    (n_layers, n_modules, n_roles, lora_r, r2) = (int(x) for x in lam.shape)
    if lora_r != r2:
        raise RuntimeError(f'{p}: R is not square in the rank dims: {tuple(lam.shape)}')
    module_names = state.get('lambda_modules', list(LAMBDA_MODULES))
    if len(module_names) != n_modules:
        raise RuntimeError(f'{p}: {len(module_names)} module names for {n_modules} R slots')
    tl = RoleLambda(n_layers, lora_r, task_names=state.get('task_names'), module_names=module_names, n_roles=n_roles)
    tl.R.data.copy_(lam)
    return tl

@torch.no_grad()
def sanity_check_zero(model, input_ids, attention_mask, task_ids, row_id, col_id, is_header, is_content, tol=0.001, debug=False):
    assert CTX.bias_module is not None, 'attach_structural_attention must run first'
    absmax = float(CTX.bias_module.beta_row.abs().max() + CTX.bias_module.beta_col.abs().max())
    if absmax != 0.0:
        raise RuntimeError(f'sanity_check_zero requires all beta == 0 (got absmax sum {absmax})')
    was_training = model.training
    model.eval()
    CTX.debug_masks = [] if debug else None
    try:
        clear_structural_context()
        with torch.no_grad():
            base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        set_structural_context(task_ids, row_id, col_id, is_header, is_content)
        try:
            with torch.no_grad():
                zero_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        finally:
            clear_structural_context()
    finally:
        if was_training:
            model.train()
        dbg = CTX.debug_masks
        CTX.debug_masks = None
    if debug and dbg:
        for d in dbg:
            path = 'structural' if d['active'] else 'ordinary  '
            print(f"    [{path}] attention_mask is None? {d['mask_is_none']}  shape={d['mask_shape']}  dtype={d['mask_dtype']}")
    diff = (zero_logits.float() - base_logits.float()).abs().max().item()
    if diff > tol:
        raise RuntimeError(f'beta==0 logits differ from base by {diff} > tol {tol}')
    return diff

def _selftest():
    import torch as _t
    text = 'Annotate the target column.\n| Name | Age | City |\n|---|---:|---|\n| Alice | 30 | Paris |\n| Bob | 40 | Rome |\nReturn JSON.'
    (r, c, h, ct) = char_struct_map(text)
    print('char map ok; distinct rows:', len(set((x for x in r if x))), 'cols:', len(set((x for x in c if x))))

    class _FakeTok:
        is_fast = True

        def __call__(self, s, return_offsets_mapping=False, add_special_tokens=False):
            (toks, offsms, i) = ([], [], 0)
            n = len(s)
            while i < n:
                if s[i].isspace():
                    i += 1
                    continue
                j = i
                while j < n and (not s[j].isspace()):
                    j += 1
                toks.append(s[i:j])
                offsms.append((i, j))
                i = j
            out = {'input_ids': list(range(len(toks)))}
            if return_offsets_mapping:
                out['offset_mapping'] = offsms
            return out
    tok = _FakeTok()
    (ids, ri, ci, hi, cti, nstr) = tokens_structure(tok, text)
    toks = [text[a:b] for (a, b) in tok(text, return_offsets_mapping=True)['offset_mapping']]

    def find(word):
        return [k for (k, w) in enumerate(toks) if w == word]
    iAlice = find('Alice')[0]
    i30 = find('30')[0]
    iParis = find('Paris')[0]
    iBob = find('Bob')[0]
    i40 = find('40')[0]
    iRome = find('Rome')[0]
    iName = find('Name')[0]
    RID = _t.tensor([ri])
    CID = _t.tensor([ci])
    HID = _t.tensor([hi])
    CT = _t.tensor([cti])
    (sr, sc) = pairwise_masks(RID, CID, HID, CT)
    (sr, sc) = (sr[0], sc[0])

    def chk(name, cond):
        print(('OK ' if cond else 'FAIL ') + name)
        assert cond, name
    chk('row Alice-30', sr[iAlice, i30] == 1)
    chk('row Alice-Paris', sr[iAlice, iParis] == 1)
    chk('row Alice-40==0', sr[iAlice, i40] == 0)
    chk('row symmetric', sr[i30, iAlice] == 1)
    chk('col Alice-Bob', sc[iAlice, iBob] == 1)
    chk('col 30-40', sc[i30, i40] == 1)
    chk('col Alice-30==0', sc[iAlice, i30] == 0)
    chk('diag row 0', sr[iAlice, iAlice] == 0)
    chk('diag col 0', sc[iAlice, iAlice] == 0)
    chk('header not in row (Name-Age)', sr[iName, find('Age')[0]] == 0)
    chk('header in col (Name-Alice)', sc[iName, iAlice] == 1)
    chk('masks disjoint', bool((sr.bool() & sc.bool()).sum() == 0))
    rel = _relation_from_vectors(RID, CID, HID, CT, RID, CID, HID, CT)[0]
    iAge = find('Age')[0]
    iCity = find('City')[0]
    chk('rel same-cell = none', int(rel[iAlice, iAlice]) == _REL_NONE)
    chk('rel same data row = ROW', int(rel[iAlice, i30]) == _REL_ROW and int(rel[iAlice, iParis]) == _REL_ROW)
    chk('rel same column = COL', int(rel[iAlice, iBob]) == _REL_COL and int(rel[i30, i40]) == _REL_COL)
    chk('rel unrelated = none', int(rel[iAlice, i40]) == _REL_NONE and int(rel[iAlice, iRome]) == _REL_NONE)
    chk('rel header-data same col = COL', int(rel[iName, iAlice]) == _REL_COL and int(rel[iAge, i30]) == _REL_COL)
    chk('rel header-header = none', int(rel[iName, iAge]) == _REL_NONE and int(rel[iName, iCity]) == _REL_NONE)
    chk('rel header-data cross-col row = none', int(rel[iName, i30]) == _REL_NONE)
    iAnn = find('Annotate')[0]
    chk('rel non-table token = none', int(rel[iAnn, iAlice]) == _REL_NONE and int(rel[iAlice, iAnn]) == _REL_NONE)
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64, attn_implementation='sdpa')
    _t.manual_seed(0)
    m = Qwen3ForCausalLM(cfg).eval()
    T = 7
    ids_t = _t.randint(0, 64, (2, T))
    am = _t.ones(2, T, dtype=_t.long)
    base_logits = m(input_ids=ids_t, attention_mask=am).logits.detach()
    nL = count_layers(m)
    bias = StructuralBias(2, nL, task_names=['cta', 'cpa'])
    m.structural_bias = bias
    attach_structural_attention(m, bias, base_impl='sdpa')
    RIDb = _t.tensor([[0, 0, 5, 5, 6, 0, 0], [0, 0, 5, 5, 6, 0, 0]])
    CIDb = _t.tensor([[0, 0, 1, 2, 1, 0, 0], [0, 0, 1, 2, 1, 0, 0]])
    HIDb = _t.zeros(2, T, dtype=_t.long)
    CTb = _t.tensor([[0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0]])
    tid = _t.tensor([0, 1])
    (srb, scb) = pairwise_masks(RIDb, CIDb, HIDb, CTb)
    chk('crafted row 2-3', bool(srb[0, 2, 3]) and (not bool(srb[0, 2, 4])))
    chk('crafted col 2-4', bool(scb[0, 2, 4]) and (not bool(scb[0, 2, 3])))
    d = sanity_check_zero(m, ids_t, am, tid, RIDb, CIDb, HIDb, CTb)
    chk(f'beta==0 identity (diff={d:.2e})', d < 0.001)
    chk('structural_bias in AttentionMaskInterface', _REG_NAME in AttentionMaskInterface._global_mapping)
    pad = 3
    ids_p = _t.cat([ids_t, _t.zeros(2, pad, dtype=_t.long)], dim=1)
    am_p = _t.cat([am, _t.zeros(2, pad, dtype=_t.long)], dim=1)
    z = _t.zeros(2, pad, dtype=_t.long)
    RIDp = _t.cat([RIDb, z], 1)
    CIDp = _t.cat([CIDb, z], 1)
    HIDp = _t.cat([HIDb, z], 1)
    CTp = _t.cat([CTb, z], 1)
    dp = sanity_check_zero(m, ids_p, am_p, tid, RIDp, CIDp, HIDp, CTp, debug=True)
    chk(f'beta==0 identity, padded batch (diff={dp:.2e})', dp < 0.001)
    CTX.debug_masks = []
    set_structural_context(tid, RIDp, CIDp, HIDp, CTp)
    try:
        m.eval()
        with _t.no_grad():
            m(input_ids=ids_p, attention_mask=am_p)
    finally:
        clear_structural_context()
    seen = list(CTX.debug_masks)
    CTX.debug_masks = None
    chk('padded batch: mask materialized (not None), 4D', len(seen) >= 1 and all((not s['mask_is_none'] and len(s['mask_shape']) == 4 for s in seen)))
    with _t.no_grad():
        bias.beta_row.fill_(0.7)
        bias.beta_col.fill_(-0.5)
    lp = 5
    ids_1 = ids_t[:1]
    set_structural_context(tid[:1], RIDb[:1], CIDb[:1], HIDb[:1], CTb[:1])
    try:
        log_np = m(input_ids=ids_1, attention_mask=am[:1]).logits[0].detach()
    finally:
        clear_structural_context()
    ids_lp = _t.cat([_t.zeros(1, lp, dtype=_t.long), ids_1], 1)
    am_lp = _t.cat([_t.zeros(1, lp, dtype=_t.long), am[:1]], 1)
    zl = _t.zeros(1, lp, dtype=_t.long)
    set_structural_context(tid[:1], _t.cat([zl, RIDb[:1]], 1), _t.cat([zl, CIDb[:1]], 1), _t.cat([zl, HIDb[:1]], 1), _t.cat([zl, CTb[:1]], 1))
    try:
        log_lp = m(input_ids=ids_lp, attention_mask=am_lp).logits[0].detach()
    finally:
        clear_structural_context()
    leak = (log_np.float() - log_lp[lp:].float()).abs().max().item()
    chk(f'non-zero beta: left padding does not leak (max|diff|={leak:.2e})', leak < 0.0001)
    with _t.no_grad():
        bias.beta_row.zero_()
        bias.beta_col.zero_()
    bias.beta_row.grad = None
    bias.beta_col.grad = None
    set_structural_context(tid, RIDb, CIDb, HIDb, CTb)
    try:
        m.train()
        out0 = m(input_ids=ids_t, attention_mask=am).logits
        out0.float().pow(2).mean().backward()
    finally:
        clear_structural_context()
        m.eval()
    chk('beta==0 gets non-zero grad in training (row)', bias.beta_row.grad.abs().sum() > 0)
    chk('beta==0 gets non-zero grad in training (col)', bias.beta_col.grad.abs().sum() > 0)
    bias.beta_row.grad = None
    bias.beta_col.grad = None
    bias.beta_row.data[0, 0] = 2.0
    bias.beta_col.data[1, 1] = -1.5
    set_structural_context(tid, RIDb, CIDb, HIDb, CTb)
    try:
        out = m(input_ids=ids_t, attention_mask=am).logits
        chk('beta!=0 changes logits', (out.detach() - base_logits).abs().max() > 0.001)
        loss = out.float().pow(2).mean()
        loss.backward()
    finally:
        clear_structural_context()
    chk('grad -> beta_row', bias.beta_row.grad.abs().sum() > 0)
    chk('grad -> beta_col', bias.beta_col.grad.abs().sum() > 0)
    chk('beta param count == T*L*2', bias.n_params() == 2 * nL * 2)
    with _t.no_grad():
        bias.beta_row.zero_()
        bias.beta_col.zero_()
    with _t.no_grad():
        bias.beta_row.data[0, 0] = 1.5
        nz = int((bias.beta_row != 0).sum().item())
    set_structural_context(tid[:1], RIDb[:1], CIDb[:1], HIDb[:1], CTb[:1])
    try:
        with _t.no_grad():
            out_l0 = m(input_ids=ids_t[:1], attention_mask=am[:1]).logits[0].detach().float().clone()
    finally:
        clear_structural_context()
    chk('per-layer beta on (task0,layer0) changes logits', (out_l0 - base_logits[0].float()).abs().max() > 0.001)
    chk('single (task,layer) beta touches exactly one entry', nz == 1)
    chk('beta table is [n_tasks, n_layers]', tuple(bias.beta_row.shape) == (2, nL))
    with _t.no_grad():
        bias.beta_row.zero_()
        bias.beta_col.zero_()
    m.generation_config.do_sample = False
    m.config.use_cache = True
    base_gen = m.generate(input_ids=ids_t, attention_mask=am, max_new_tokens=4, do_sample=False, use_cache=True, pad_token_id=0)
    bias.beta_row.data.zero_()
    bias.beta_col.data.zero_()
    set_structural_context(tid, RIDb, CIDb, HIDb, CTb)
    try:
        zero_gen = m.generate(input_ids=ids_t, attention_mask=am, max_new_tokens=4, do_sample=False, use_cache=True, pad_token_id=0)
    finally:
        clear_structural_context()
    chk('beta==0 generate == base (decode path)', _t.equal(zero_gen, base_gen))
    print('straddle tokens:', nstr)
    _selftest_lambda()
    _selftest_rowcol_experts()
    _selftest_mem_opt()
    print('ALL OK')

def _selftest_mem_opt():
    import torch as _t
    import torch.nn.functional as _F

    def chk(name, cond):
        print(('OK ' if cond else 'FAIL ') + name)
        assert cond, name
    print('-- memory optimization (bias dtype + in-place mask + head-chunk) --')
    _t.manual_seed(0)
    (B, Hq, Hkv, T, D) = (2, 4, 2, 6, 8)
    n_rep = Hq // Hkv
    dtype = _t.bfloat16
    q = _t.randn(B, Hq, T, D, dtype=dtype)
    k = _t.randn(B, Hkv, T, D, dtype=dtype)
    v = _t.randn(B, Hkv, T, D, dtype=dtype)
    RID = _t.tensor([[0, 5, 5, 6, 6, 0]] * B)
    CID = _t.tensor([[0, 1, 2, 1, 2, 0]] * B)
    HID = _t.zeros(B, T, dtype=_t.long)
    CT = _t.tensor([[0, 1, 1, 1, 1, 0]] * B)
    tid = _t.tensor([0, 1])
    relation = _relation_from_vectors(RID, CID, HID, CT, RID, CID, HID, CT)
    add_mask = _to_additive_mask(None, T, T, dtype, q.device, None)

    def _gqa_sdpa(mask):
        k2 = k.repeat_interleave(n_rep, dim=1)
        v2 = v.repeat_interleave(n_rep, dim=1)
        return _F.scaled_dot_product_attention(q, k2, v2, attn_mask=mask)
    bn = StructuralBias(2, 1)
    bo = StructuralBias(2, 1)
    with _t.no_grad():
        init_r = _t.randn(2, 1)
        init_c = _t.randn(2, 1)
        bn.beta_row.copy_(init_r)
        bn.beta_col.copy_(init_c)
        bo.beta_row.copy_(init_r)
        bo.beta_col.copy_(init_c)
    br_n = bn.beta_row[tid, 0]
    bc_n = bn.beta_col[tid, 0]
    row_f = (relation == _REL_ROW)[:, None].to(_t.float32)
    col_f = (relation == _REL_COL)[:, None].to(_t.float32)
    bias_naive_fp32 = br_n[:, None, None, None] * row_f + bc_n[:, None, None, None] * col_f
    chk('naive reference really builds a float32 [B,1,T,T]', bias_naive_fp32.dtype == _t.float32 and tuple(bias_naive_fp32.shape) == (B, 1, T, T))
    mask_naive = add_mask + bias_naive_fp32.to(dtype)
    out_naive = _gqa_sdpa(mask_naive)
    out_naive.float().pow(2).mean().backward()
    br_o = bo.beta_row[tid, 0]
    bc_o = bo.beta_col[tid, 0]
    CTX.debug_bias = []
    try:
        bias_opt = _StructuralBiasFn.apply(br_o, bc_o, relation, dtype)
        _record_bias_debug('struct_bias', bias_opt)
        bias_opt.add_(add_mask)
        _record_bias_debug('new_mask', bias_opt)
        out_opt = _gqa_sdpa(bias_opt)
        out_opt.float().pow(2).mean().backward()
        rep = list(CTX.debug_bias)
    finally:
        CTX.debug_bias = None
    fwd_rec = [r for r in rep if r['tag'] in ('struct_bias', 'new_mask')]
    bwd_rec = [r for r in rep if r['tag'].startswith('bwd_')]
    od = (out_opt.float() - out_naive.float()).abs().max().item()
    chk(f'optimized output == naive per-layer formulation (diff={od:.2e})', od < 1e-06)
    dr = (bo.beta_row.grad - bn.beta_row.grad).abs().max().item()
    dc = (bo.beta_col.grad - bn.beta_col.grad).abs().max().item()
    chk(f'beta_row grad == naive (diff={dr:.2e})', dr < 1e-06)
    chk(f'beta_col grad == naive (diff={dc:.2e})', dc < 1e-06)
    chk('beta grads are nonzero (test is live)', bo.beta_row.grad.abs().sum() > 0 and bo.beta_col.grad.abs().sum() > 0)
    chk('forward struct bias built directly in bf16 (no float32 [B,1,T,T])', len(fwd_rec) == 2 and all((r['dtype'] == 'torch.bfloat16' for r in fwd_rec)))
    chk('forward: exactly ONE [B,1,T,T] buffer (in-place fold, head-broadcast, same shape reused)', all((tuple(r['shape']) == (B, 1, T, T) for r in fwd_rec)))
    chk('backward actually ran (reduction temps recorded)', len(bwd_rec) >= 2)
    chk('backward reduction temps are bf16, NEVER float32', all((r['dtype'] == 'torch.bfloat16' for r in bwd_rec)))
    chk('backward: no recorded structural tensor is float32', not any((r['dtype'] == 'torch.float32' for r in rep)))
    (prof, peak) = structural_bias_report(bo, tid, RID, CID, HID, CT, dtype)
    chk('profile helper: bias dtype bf16, peak = 2 bytes/elem', all((p['dtype'] == 'torch.bfloat16' for p in prof)) and peak == 2 * B * 1 * T * T)
    (Bb, Hb, Tb) = (1, 1, 2048)
    g_big = _t.randn(Bb, Hb, Tb, Tb, dtype=dtype)
    rel_big = _t.randint(0, 3, (Bb, Tb, Tb), dtype=_t.uint8)
    CTX.debug_bias = []
    try:
        (gr_b, gc_b) = _reduce_beta_grads(g_big, rel_big)
        tiles = [r for r in CTX.debug_bias if r['tag'] == 'bwd_row_tile']
    finally:
        CTX.debug_bias = None
    gf = g_big.float()
    gr_ref = (gf * (rel_big == _REL_ROW)[:, None]).sum(dim=(1, 2, 3))
    gc_ref = (gf * (rel_big == _REL_COL)[:, None]).sum(dim=(1, 2, 3))
    chk(f'backward tiling: >1 tile for large T (n_tiles={len(tiles)})', len(tiles) > 1)
    chk('backward tiling: every tile temp is bf16 and SMALLER than the full [B,1,T,T]', all((r['dtype'] == 'torch.bfloat16' and r['shape'][2] < Tb for r in tiles)))
    chk(f'backward tiling matches full-fp32 reference (row diff={(gr_b - gr_ref).abs().max():.1e})', _t.allclose(gr_b, gr_ref, atol=0.01) and _t.allclose(gc_b, gc_ref, atol=0.01))
    with _t.no_grad():
        bo.beta_row.zero_()
        bo.beta_col.zero_()
    z_naive = add_mask + (br_n[:, None, None, None] * row_f + bc_n[:, None, None, None] * col_f).zero_().to(dtype)
    z_bias = _StructuralBiasFn.apply(bo.beta_row[tid, 0], bo.beta_col[tid, 0], relation, dtype)
    z_bias.add_(add_mask)
    out_zero = _gqa_sdpa(z_bias)
    out_naive0 = _gqa_sdpa(z_naive)
    chk('beta==0 optimized == naive-zero (dense mask, exact)', (out_zero.float() - out_naive0.float()).abs().max().item() < 1e-06)
    chk('causal/padding mask preserved (upper triangle masked)', bool((z_bias[0, 0].float()[0, 1:] < -1e+30).all()))
    with _t.no_grad():
        bo.beta_row.copy_(init_r)
        bo.beta_col.copy_(init_c)

    def _opt_forward():
        b = _StructuralBiasFn.apply(bo.beta_row[tid, 0], bo.beta_col[tid, 0], relation, dtype)
        b = b + add_mask
        return _gqa_sdpa(b)
    with _t.no_grad():
        o1 = _opt_forward()
        o2 = _opt_forward()
    chk('repeated optimized forward bit-identical', _t.equal(o1, o2))
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64, attn_implementation='sdpa')
    _t.manual_seed(1)
    m = Qwen3ForCausalLM(cfg).eval()
    nL = count_layers(m)
    nH = cfg.num_attention_heads
    chk('chunk-test model is GQA (n_rep=2)', nH == 4 and cfg.num_key_value_heads == 2)
    bias = StructuralBias(2, nL, task_names=['cta', 'cpa'])
    m.structural_bias = bias
    attach_structural_attention(m, bias, base_impl='sdpa')
    with _t.no_grad():
        bias.beta_row.normal_(0, 0.6)
        bias.beta_col.normal_(0, 0.6)
    Tt = 6
    ids = _t.randint(0, 64, (2, Tt))
    am = _t.ones(2, Tt, dtype=_t.long)
    RIDb = _t.tensor([[0, 5, 5, 6, 6, 0]] * 2)
    CIDb = _t.tensor([[0, 1, 2, 1, 2, 0]] * 2)
    HIDb = _t.zeros(2, Tt, dtype=_t.long)
    CTb = _t.tensor([[0, 1, 1, 1, 1, 0]] * 2)
    tidb = _t.tensor([0, 1])

    def _run_and_grad():
        bias.beta_row.grad = None
        bias.beta_col.grad = None
        set_structural_context(tidb, RIDb, CIDb, HIDb, CTb)
        try:
            m.train()
            out = m(input_ids=ids, attention_mask=am).logits
            out.float().pow(2).mean().backward()
        finally:
            clear_structural_context()
            m.eval()
        return (out.detach().clone(), bias.beta_row.grad.clone(), bias.beta_col.grad.clone())
    set_head_chunk(None)
    (logit_single, gr_single, gc_single) = _run_and_grad()
    for hc in (2, 1):
        set_head_chunk(hc)
        chk(f'head_chunk={hc} resolves to a multiple of n_rep', _resolve_head_chunk(hc, nH, cfg.num_key_value_heads) == 2)
        (logit_c, gr_c, gc_c) = _run_and_grad()
        ld = (logit_c.float() - logit_single.float()).abs().max().item()
        grd = (gr_c - gr_single).abs().max().item()
        gcd = (gc_c - gc_single).abs().max().item()
        chk(f'head_chunk={hc}: logits == single call (diff={ld:.2e})', ld < 0.0001)
        chk(f'head_chunk={hc}: beta_row grad == single (diff={grd:.2e})', grd < 1e-05)
        chk(f'head_chunk={hc}: beta_col grad == single (diff={gcd:.2e})', gcd < 1e-05)
    n_rep = nH // cfg.num_key_value_heads
    chunk2 = _resolve_head_chunk(2, nH, cfg.num_key_value_heads)
    kv_slices = [(hs // n_rep, min(hs + chunk2, nH) // n_rep) for hs in range(0, nH, chunk2)]
    chk('GQA chunk->KV mapping correct (heads[0:2)->kv[0:1), [2:4)->kv[1:2))', kv_slices == [(0, 1), (1, 2)])

    def _eval_logits():
        set_structural_context(tidb, RIDb, CIDb, HIDb, CTb)
        try:
            m.eval()
            with _t.no_grad():
                return m(input_ids=ids, attention_mask=am).logits.detach().clone()
        finally:
            clear_structural_context()
    set_head_chunk(None)
    ev_single = _eval_logits()
    set_head_chunk(2)
    ev_chunk = _eval_logits()
    chk(f'head_chunk eval-mode logits == single call (diff={(ev_chunk.float() - ev_single.float()).abs().max():.2e})', (ev_chunk.float() - ev_single.float()).abs().max().item() < 0.0001)
    set_head_chunk(None)
    chk('head_chunk restored to None (default single call)', CTX.head_chunk is None)
    print('mem-opt OK')

def _selftest_lambda():
    import torch as _t
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model

    def chk(name, cond):
        print(('OK ' if cond else 'FAIL ') + name)
        assert cond, name
    print('-- R-LoRA (role-conditioned full r x r rank-space modulation) --')
    R = 4
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128, attn_implementation='sdpa')
    _t.manual_seed(0)
    m = Qwen3ForCausalLM(cfg)
    m.config.use_cache = False
    tgt = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    lora = LoraConfig(r=R, lora_alpha=8, lora_dropout=0.0, bias='none', task_type='CAUSAL_LM', target_modules=tgt)
    m = get_peft_model(m, lora)
    with _t.no_grad():
        for (n, p) in m.named_parameters():
            if 'lora_B' in n:
                p.normal_(0, 0.2)
    m.eval()
    nL = count_layers(m)
    bias = StructuralBias(2, nL, task_names=['cta', 'cpa'])
    m.structural_bias = bias
    attach_structural_attention(m, bias, base_impl='sdpa')
    lam_mod = RoleLambda(nL, R, task_names=['cta', 'cpa'])
    m.task_lambda = lam_mod
    counts = attach_task_lambda(m, lam_mod)
    chk('R hooks on all seven module families', set(counts) == set(LAMBDA_MODULES))
    chk('R hooked every layer', all((counts[k] == nL for k in LAMBDA_MODULES)))
    chk('R param count == L*n_modules*n_roles*R*R', lam_mod.n_params() == nL * len(LAMBDA_MODULES) * N_ROLES * R * R)
    chk('seven module families by default', len(LAMBDA_MODULES) == 7)
    n_loraA = sum((1 for (n, _) in m.named_parameters() if n.endswith('lora_A.default.weight')))
    n_loraB = sum((1 for (n, _) in m.named_parameters() if n.endswith('lora_B.default.weight')))
    chk('one shared LoRA A/B per module (7 modules x nL)', n_loraA == 7 * nL and n_loraB == 7 * nL)
    T = 7
    ids = _t.randint(0, 64, (2, T))
    am = _t.ones(2, T, dtype=_t.long)
    RID = _t.tensor([[0, 0, 5, 5, 6, 0, 0]] * 2)
    CID = _t.tensor([[0, 0, 1, 2, 1, 0, 0]] * 2)
    HID = _t.zeros(2, T, dtype=_t.long)
    CT = _t.tensor([[0, 0, 1, 1, 1, 0, 0]] * 2)
    tid = _t.tensor([0, 1])
    ROLE = _t.tensor([[0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0, 0]], dtype=_t.uint8)
    with _t.no_grad():
        bias.beta_row.fill_(0.3)
        bias.beta_col.fill_(-0.2)
    set_structural_context(tid, RID, CID, HID, CT, apply_lambda=True, role_ids=ROLE)
    try:
        r_on_identity = m(input_ids=ids, attention_mask=am).logits.detach().clone()
    finally:
        clear_structural_context()
    set_structural_context(tid, RID, CID, HID, CT, apply_lambda=False, role_ids=ROLE)
    try:
        r_off = m(input_ids=ids, attention_mask=am).logits.detach().clone()
    finally:
        clear_structural_context()
    d_id = (r_on_identity.float() - r_off.float()).abs().max().item()
    chk(f'R=identity init vs modulation-off, same nonzero beta (diff={d_id:.2e})', d_id < 0.0001)
    set_structural_context(tid, RID, CID, HID, CT, role_ids=ROLE)
    try:
        base_active = m(input_ids=ids, attention_mask=am).logits.detach().clone()
    finally:
        clear_structural_context()
    q_idx = LAMBDA_MODULES.index('q_proj')
    target_layer = 0
    qb = None
    for (n, mod) in m.named_modules():
        la = getattr(mod, 'lora_A', None)
        if isinstance(la, nn.ModuleDict) and n.split('.')[-1] == 'q_proj' and _LAYER_IDX_RE.search(n) and (int(_LAYER_IDX_RE.search(n).group(1)) == target_layer):
            qb = mod.lora_B['default']
            break
    assert qb is not None, 'could not find layer-0 q_proj lora_B'
    ALL_TABLE = _t.ones(2, T, dtype=_t.uint8)
    v = _t.tensor([1.7, 0.0, -0.9, 0.4])[:R]
    saved_b = qb.weight.detach().clone()
    with _t.no_grad():
        qb.weight.mul_(v.view(1, -1))
    set_structural_context(tid, RID, CID, HID, CT, role_ids=ALL_TABLE)
    try:
        ref_folded = m(input_ids=ids, attention_mask=am).logits.detach().clone()
    finally:
        clear_structural_context()
    with _t.no_grad():
        qb.weight.copy_(saved_b)
        lam_mod.R[target_layer, q_idx, ROLE_TABLE].copy_(_t.diag(v))
    set_structural_context(tid, RID, CID, HID, CT, role_ids=ALL_TABLE)
    try:
        test_R = m(input_ids=ids, attention_mask=am).logits.detach().clone()
    finally:
        clear_structural_context()
    d_math = (test_R.float() - ref_folded.float()).abs().max().item()
    chk(f'R computes B R A h between A and B (diff={d_math:.2e})', d_math < 0.0001)
    chk('non-identity R actually changed the output (vs identity)', (test_R.float() - base_active.float()).abs().max().item() > 0.0001)
    with _t.no_grad():
        lam_mod.R.copy_(_t.eye(R).expand_as(lam_mod.R).clone())
    perm = _t.tensor([1, 0, 3, 2])[:R]
    P = _t.eye(R)[perm]
    with _t.no_grad():
        qb.weight.copy_(saved_b[:, perm])
    set_structural_context(tid, RID, CID, HID, CT, role_ids=ALL_TABLE)
    try:
        ref_perm = m(input_ids=ids, attention_mask=am).logits.detach().clone()
    finally:
        clear_structural_context()
    with _t.no_grad():
        qb.weight.copy_(saved_b)
        lam_mod.R[target_layer, q_idx, ROLE_TABLE].copy_(P)
    set_structural_context(tid, RID, CID, HID, CT, role_ids=ALL_TABLE)
    try:
        test_perm = m(input_ids=ids, attention_mask=am).logits.detach().clone()
    finally:
        clear_structural_context()
    d_perm = (test_perm.float() - ref_perm.float()).abs().max().item()
    chk(f'FULL off-diagonal R (permutation) matches folding it into B (diff={d_perm:.2e})', d_perm < 0.0001)
    with _t.no_grad():
        lam_mod.R.copy_(_t.eye(R).expand_as(lam_mod.R).clone())
    with _t.no_grad():
        g = _t.Generator().manual_seed(5)
        lam_mod.R[:, :, ROLE_TEXT] += _t.empty_like(lam_mod.R[:, :, ROLE_TEXT]).normal_(0, 0.3, generator=g)
        lam_mod.R[:, :, ROLE_TABLE] += _t.empty_like(lam_mod.R[:, :, ROLE_TABLE]).normal_(0, 0.3, generator=g)
    single = ids[:1]
    single_am = am[:1]
    k = 3
    pref = _t.zeros(1, T, dtype=_t.uint8)
    pref[0, :k] = 1
    outs = {}
    for (nm, rr) in (('mixed', pref), ('all_table', _t.ones(1, T, dtype=_t.uint8)), ('all_text', _t.zeros(1, T, dtype=_t.uint8))):
        set_structural_context(tid[:1], RID[:1], CID[:1], HID[:1], CT[:1], role_ids=rr)
        try:
            outs[nm] = m(input_ids=single, attention_mask=single_am).logits.detach().clone()
        finally:
            clear_structural_context()
    d_pref = (outs['mixed'][0, :k].float() - outs['all_table'][0, :k].float()).abs().max().item()
    chk(f'per-TOKEN routing: table-prefix positions match the all-table run (diff={d_pref:.2e})', d_pref < 0.0001)
    chk('per-TOKEN routing: text and table runs actually differ', (outs['all_table'].float() - outs['all_text'].float()).abs().max().item() > 0.0001)
    with _t.no_grad():
        bias.beta_row.zero_()
        bias.beta_col.zero_()
    per_task = []
    for one_tid in (0, 1):
        set_structural_context(_t.tensor([one_tid]), RID[:1], CID[:1], HID[:1], CT[:1], role_ids=pref)
        try:
            per_task.append(m(input_ids=single, attention_mask=single_am).logits.detach().clone())
        finally:
            clear_structural_context()
    d_task = (per_task[0].float() - per_task[1].float()).abs().max().item()
    chk(f'R is task-INDEPENDENT (task0 vs task1, diff={d_task:.2e})', d_task == 0.0)
    set_structural_context(tid, RID, CID, HID, CT, role_ids=ROLE)
    ev = {}
    try:
        for mode in ROLE_EVAL_MODES:
            set_role_eval_mode(mode)
            ev[mode] = m(input_ids=ids, attention_mask=am).logits.detach().clone()
    finally:
        set_role_eval_mode('learned')
        clear_structural_context()
    chk('eval mode tie_text differs from learned', (ev['tie_text'].float() - ev['learned'].float()).abs().max().item() > 0.0001)
    chk('eval mode tie_table differs from learned', (ev['tie_table'].float() - ev['learned'].float()).abs().max().item() > 0.0001)
    chk('eval mode tie_text differs from tie_table', (ev['tie_text'].float() - ev['tie_table'].float()).abs().max().item() > 0.0001)
    set_structural_context(tid, RID, CID, HID, CT, apply_lambda=False, role_ids=ROLE)
    try:
        off = m(input_ids=ids, attention_mask=am).logits.detach().clone()
    finally:
        clear_structural_context()
    d_ident = (ev['identity'].float() - off.float()).abs().max().item()
    chk(f'eval mode identity == ordinary LoRA on the trained A/B (diff={d_ident:.2e})', d_ident < 0.0001)
    raised = False
    CTX.active = True
    CTX.apply_lambda = True
    CTX.role_ids = None
    try:
        m(input_ids=ids, attention_mask=am)
    except RuntimeError as e:
        raised = 'role_ids is None' in str(e)
    finally:
        clear_structural_context()
    chk('missing role_ids raises (no silent ordinary-LoRA fallback)', raised)
    m.train()
    for p in m.parameters():
        p.grad = None
    set_structural_context(tid, RID, CID, HID, CT, role_ids=ROLE)
    try:
        loss = m(input_ids=ids, attention_mask=am, labels=ids.masked_fill(am == 0, -100)).loss
        loss.backward()
    finally:
        clear_structural_context()
        m.eval()
    chk('R gets non-zero grad', lam_mod.R.grad is not None and lam_mod.R.grad.abs().sum() > 0)
    per_slot = lam_mod.R.grad.abs().flatten(start_dim=3).sum(-1)
    chk('every (layer, module, role) slot of R gets grad', int((per_slot == 0).sum().item()) == 0)
    all_A = all((any((p.grad is not None and p.grad.abs().sum() > 0 for (n, p) in m.named_parameters() if 'lora_A' in n and f'.{mm}.' in n)) for mm in LAMBDA_MODULES))
    all_B = all((any((p.grad is not None and p.grad.abs().sum() > 0 for (n, p) in m.named_parameters() if 'lora_B' in n and f'.{mm}.' in n)) for mm in LAMBDA_MODULES))
    chk('shared LoRA A and B get grad on all seven families', all_A and all_B)
    base_frozen = all((p.grad is None or p.grad.abs().sum() == 0 for (n, p) in m.named_parameters() if 'lora_' not in n and 'base_layer' in n))
    chk('frozen base weights get no grad', base_frozen)
    import tempfile
    with _t.no_grad():
        gg = _t.Generator().manual_seed(9)
        lam_mod.R.copy_(lam_mod.R + _t.empty_like(lam_mod.R).normal_(0, 0.25, generator=gg))
    set_structural_context(tid, RID, CID, HID, CT, role_ids=ROLE)
    try:
        pre_save = m(input_ids=ids, attention_mask=am).logits.detach().clone()
    finally:
        clear_structural_context()
    with tempfile.TemporaryDirectory() as td:
        save_structural_bias(bias, td, {'n_tasks': 2, 'n_layers': nL, 'task_names': ['cta', 'cpa']}, lambda_module=lam_mod)
        l2 = load_task_lambda(td)
        chk('R save/load round-trip (shape)', l2 is not None and tuple(l2.R.shape) == tuple(lam_mod.R.shape))
        chk('R save/load round-trip (values)', _t.equal(l2.R, lam_mod.R.detach().cpu()))
        chk('R save/load round-trip (module names, n_roles)', list(l2.module_names) == list(lam_mod.module_names) and l2.n_roles == lam_mod.n_roles)
        detach_task_lambda()
        l2.to(m.device)
        m.task_lambda = l2
        attach_task_lambda(m, l2)
        set_structural_context(tid, RID, CID, HID, CT, role_ids=ROLE)
        try:
            post_load = m(input_ids=ids, attention_mask=am).logits.detach().clone()
        finally:
            clear_structural_context()
        d_rt = (pre_save.float() - post_load.float()).abs().max().item()
        chk(f'R save/load reproduces outputs (diff={d_rt:.2e})', d_rt < 1e-05)
        save_structural_bias(bias, td, {'n_tasks': 2, 'n_layers': nL, 'task_names': ['cta', 'cpa']})
        chk('adapter without R -> load_task_lambda None', load_task_lambda(td) is None)
        _t.save({'beta_row': bias.beta_row.detach().cpu(), 'beta_col': bias.beta_col.detach().cpu(), 'task_names': ['cta', 'cpa'], 'lambda': _t.ones(2, nL, 2, R), 'lambda_modules': ['q_proj', 'k_proj'], 'lambda_r': R}, Path(td) / 'structural_bias.pt')
        rejected = False
        try:
            load_task_lambda(td)
        except RuntimeError as e:
            rejected = '4-D lambda' in str(e)
        chk('superseded 4-D diagonal checkpoint is rejected with a clear message', rejected)
        save_structural_bias(bias, td, {'n_tasks': 2, 'n_layers': nL, 'task_names': ['cta', 'cpa']}, lambda_module=lam_mod)
        (b2, cfg2) = load_structural_bias(td)
        chk('beta save/load round-trip ([n_tasks, n_layers], values, task_names)', b2 is not None and tuple(b2.beta_row.shape) == (2, nL) and _t.equal(b2.beta_row, bias.beta_row.detach().cpu()) and _t.equal(b2.beta_col, bias.beta_col.detach().cpu()) and (list(b2.task_names or []) == ['cta', 'cpa']))
    import json as _json
    smry = lam_mod.summary()
    tbl = lam_mod.per_role_table()
    diag = lam_mod.diagnostics()
    chk('per_role_table shape (roles x layers x modules x r x r)', set(tbl) == {'text', 'table'} and len(tbl['table']) == nL and (set(tbl['table']['layer_0']) == set(LAMBDA_MODULES)) and (len(tbl['table']['layer_0'][LAMBDA_MODULES[0]]) == R) and (len(tbl['table']['layer_0'][LAMBDA_MODULES[0]][0]) == R))
    chk('diagnostics carry rel_fro + r singular values per module', 'mean_rel_fro' in diag['global'] and len(diag['per_layer']['layer_0'][LAMBDA_MODULES[0]]['rel_svals']) == R)
    _json.dumps({'lambda_summary': smry, 'lambda_per_role': tbl, 'role_diag': diag})
    chk('summary + per_role_table + diagnostics JSON-serialize (driver save path)', True)
    detach_task_lambda()
    print('R-LoRA OK')

def _selftest_rowcol_experts():
    import torch as _t
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model

    def chk(name, cond):
        print(('OK ' if cond else 'FAIL ') + name)
        assert cond, name
    print('-- RC-LoRA (shared row/column experts + task Lambda and task gate) --')
    R = 4
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128, attn_implementation='sdpa')
    _t.manual_seed(0)
    m = Qwen3ForCausalLM(cfg)
    m.config.use_cache = False
    tgt = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    m = get_peft_model(m, LoraConfig(r=R, lora_alpha=8, lora_dropout=0.0, bias='none', task_type='CAUSAL_LM', target_modules=tgt))
    m.eval()
    nL = count_layers(m)
    T = 7
    ids = _t.randint(0, 64, (2, T))
    am = _t.ones(2, T, dtype=_t.long)
    RID = _t.tensor([[0, 0, 5, 5, 6, 0, 0]] * 2)
    CID = _t.tensor([[0, 0, 1, 2, 1, 0, 0]] * 2)
    HID = _t.zeros(2, T, dtype=_t.long)
    CT = _t.tensor([[0, 0, 1, 1, 1, 0, 0]] * 2)
    tid = _t.tensor([0, 1])
    VIEW_BOTH = _t.tensor([VIEW_ROW, VIEW_COL], dtype=_t.uint8)

    def fwd(task_ids, view_ids=None, expert_mode='mixture', rows=slice(None)):
        set_structural_context(task_ids, RID[rows], CID[rows], HID[rows], CT[rows], view_ids=view_ids, expert_mode=expert_mode)
        try:
            return m(input_ids=ids[rows], attention_mask=am[rows]).logits.detach().clone()
        finally:
            clear_structural_context()
    with _t.no_grad():
        ord_logits = m(input_ids=ids, attention_mask=am).logits.detach().clone()
    rc = RowColExperts(2, nL, R, task_names=['cta', 'tabfact'])
    m.rowcol = rc
    counts = attach_rowcol_experts(m, rc)
    chk('RC hooks on all seven module families', set(counts) == set(LAMBDA_MODULES))
    chk('RC hooked every layer', all((counts[k] == nL for k in LAMBDA_MODULES)))
    chk('Lambda param count == T*L*n_modules*R*R', rc.n_lambda_params() == 2 * nL * len(LAMBDA_MODULES) * R * R)
    chk('gate param count == T*L*n_modules*2', rc.n_gate_params() == 2 * nL * len(LAMBDA_MODULES) * 2)
    n_loraA = sum((1 for (n, _) in m.named_parameters() if n.endswith('lora_A.default.weight')))
    chk('A remains shared (one lora_A per module x layer)', n_loraA == 7 * nL)
    chk('one COLUMN expert per (layer, module)', len(rc.b_col) == nL * len(LAMBDA_MODULES))
    b_row_numel = sum((p.numel() for (n, p) in m.named_parameters() if n.endswith('lora_B.default.weight')))
    chk('column-expert parameter count == the ordinary lora_B set', rc.n_b_params() == b_row_numel)
    w0 = rc.weights().detach()
    chk(f'gate init softmax == 0.5 (max dev {float((w0 - 0.5).abs().max()):.2e})', float((w0 - 0.5).abs().max()) < 1e-07)
    chk('Lambda init == I (R init zero)', float(rc.R.detach().abs().max()) == 0.0)
    d0 = (fwd(tid, expert_mode='mixture').float() - ord_logits.float()).abs().max().item()
    chk(f'init (Lambda=I, B_r=B_c=0) == ordinary LoRA (diff={d0:.2e})', d0 < 1e-05)
    with _t.no_grad():
        g = _t.Generator().manual_seed(4)
        for (n, p) in m.named_parameters():
            if n.endswith('lora_B.default.weight'):
                p.copy_(_t.empty_like(p).normal_(0, 0.2, generator=g))
        for k in rc.b_col:
            rc.b_col[k].copy_(_t.empty_like(rc.b_col[k]).normal_(0, 0.2, generator=g))
        rc.R.copy_(_t.empty_like(rc.R).normal_(0, 0.25, generator=g))
        rc.g.copy_(_t.empty_like(rc.g).normal_(0, 0.8, generator=g))

    def fold_and_run(task_idx, b_of, use_lambda=True, row=0):
        w = rc.weights().detach()
        saved = {}
        with _t.no_grad():
            for (name, mod) in m.named_modules():
                lb = getattr(mod, 'lora_B', None)
                la = getattr(mod, 'lora_A', None)
                if not isinstance(lb, nn.ModuleDict) or not isinstance(la, nn.ModuleDict):
                    continue
                mtp = name.split('.')[-1]
                if mtp not in rc.module_names:
                    continue
                li = int(_LAYER_IDX_RE.search(name).group(1))
                mi = rc.module_names.index(mtp)
                (wb, wa) = (lb['default'].weight, la['default'].weight)
                saved[name] = (wb.detach().clone(), wa.detach().clone())
                b_r = wb.detach().clone()
                b_c = rc.get_b(li, mi).detach().to(wb.dtype)
                wb.copy_(b_of(b_r, b_c, w[task_idx, li, mi]))
                if use_lambda:
                    lam = (_t.eye(R, dtype=wa.dtype) + rc.R[task_idx, li, mi].detach()).to(wa.dtype)
                    wa.copy_(lam @ wa.detach())
        detach_rowcol_experts()
        with _t.no_grad():
            out = m(input_ids=ids[row:row + 1], attention_mask=am[row:row + 1]).logits.detach().clone()
        with _t.no_grad():
            for (name, mod) in m.named_modules():
                lb = getattr(mod, 'lora_B', None)
                la = getattr(mod, 'lora_A', None)
                if isinstance(lb, nn.ModuleDict) and name in saved:
                    lb['default'].weight.copy_(saved[name][0])
                    la['default'].weight.copy_(saved[name][1])
        m.rowcol = rc
        attach_rowcol_experts(m, rc)
        return out
    hook_mix = fwd(tid[:1], expert_mode='mixture', rows=slice(0, 1))
    ref_mix = fold_and_run(0, lambda br, bc, w: w[0] * br + w[1] * bc)
    d_mix = (hook_mix.float() - ref_mix.float()).abs().max().item()
    chk(f'mixture == w_r B_r + w_c B_c with Lambda folded into A (diff={d_mix:.2e})', d_mix < 0.0001)
    hook_row = fwd(tid[:1], view_ids=_t.tensor([VIEW_ROW], dtype=_t.uint8), expert_mode='anchor', rows=slice(0, 1))
    ref_row = fold_and_run(0, lambda br, bc, w: w[0] * br)
    d_row = (hook_row.float() - ref_row.float()).abs().max().item()
    chk(f'anchor/row-major == w_r B_r only (diff={d_row:.2e})', d_row < 0.0001)
    hook_col = fwd(tid[:1], view_ids=_t.tensor([VIEW_COL], dtype=_t.uint8), expert_mode='anchor', rows=slice(0, 1))
    ref_col = fold_and_run(0, lambda br, bc, w: w[1] * bc)
    d_col = (hook_col.float() - ref_col.float()).abs().max().item()
    chk(f'anchor/col-major == w_c B_c only (diff={d_col:.2e})', d_col < 0.0001)
    chk('anchor row-major != anchor col-major (the experts are distinct)', (hook_row.float() - hook_col.float()).abs().max().item() > 0.0001)
    chk('anchor != mixture (training route is not the inference route)', (hook_row.float() - hook_mix.float()).abs().max().item() > 0.0001)
    mixed = fwd(tid, view_ids=VIEW_BOTH, expert_mode='anchor')
    for row in (0, 1):
        solo = fwd(tid[row:row + 1], view_ids=VIEW_BOTH[row:row + 1], expert_mode='anchor', rows=slice(row, row + 1))
        dd = (mixed[row].float() - solo[0].float()).abs().max().item()
        chk(f'per-example task+view routing, row {row} (diff={dd:.2e})', dd < 0.0001)
    per_task = [fwd(_t.tensor([t]), expert_mode='mixture', rows=slice(0, 1)) for t in (0, 1)]
    chk('task0 != task1 on identical input (Lambda and/or gate are task-specific)', (per_task[0].float() - per_task[1].float()).abs().max().item() > 0.0001)
    saved_R = rc.R.detach().clone()
    with _t.no_grad():
        rc.R.copy_(rc.R[0:1].expand_as(rc.R).clone())
    gate_only = [fwd(_t.tensor([t]), expert_mode='mixture', rows=slice(0, 1)) for t in (0, 1)]
    chk('gate alone is task-specific (identical Lambda, different task -> different output)', (gate_only[0].float() - gate_only[1].float()).abs().max().item() > 0.0001)
    with _t.no_grad():
        rc.R.copy_(saved_R)
    raised = False
    try:
        fwd(tid, view_ids=None, expert_mode='anchor')
    except RuntimeError as e:
        raised = 'view_ids' in str(e)
    chk('anchor without view_ids raises (no silent mixture fallback)', raised)
    try:
        set_structural_context(tid, RID, CID, HID, CT, expert_mode='bogus')
        bad_mode = False
    except ValueError:
        bad_mode = True
    finally:
        clear_structural_context()
    chk('an unknown expert_mode is refused', bad_mode)
    ev = {}
    for mode in EXPERT_EVAL_MODES:
        set_expert_eval_mode(mode)
        ev[mode] = fwd(tid[:1], expert_mode='mixture', rows=slice(0, 1))
    set_expert_eval_mode('learned')
    ref_row_only = fold_and_run(0, lambda br, bc, w: br)
    ref_col_only = fold_and_run(0, lambda br, bc, w: bc)
    chk(f"eval row_only == B_r alone (diff={(ev['row_only'].float() - ref_row_only.float()).abs().max().item():.2e})", (ev['row_only'].float() - ref_row_only.float()).abs().max().item() < 0.0001)
    chk(f"eval col_only == B_c alone (diff={(ev['col_only'].float() - ref_col_only.float()).abs().max().item():.2e})", (ev['col_only'].float() - ref_col_only.float()).abs().max().item() < 0.0001)
    ref_ident = fold_and_run(0, lambda br, bc, w: w[0] * br + w[1] * bc, use_lambda=False)
    d_id = (ev['identity'].float() - ref_ident.float()).abs().max().item()
    chk(f'eval identity == Lambda:=I with the gate kept (diff={d_id:.2e})', d_id < 0.0001)
    chk('eval identity differs from learned', (ev['identity'].float() - ev['learned'].float()).abs().max().item() > 0.0001)

    def backward_once(view_ids, expert_mode):
        m.train()
        for p in m.parameters():
            p.grad = None
        rc.R.grad = None
        rc.g.grad = None
        for p in rc.b_col.values():
            p.grad = None
        set_structural_context(tid, RID, CID, HID, CT, view_ids=view_ids, expert_mode=expert_mode)
        try:
            m(input_ids=ids, attention_mask=am, labels=ids.masked_fill(am == 0, -100)).loss.backward()
        finally:
            clear_structural_context()
            m.eval()
    backward_once(VIEW_BOTH, 'anchor')
    chk('Lambda (R) gets non-zero grad', rc.R.grad is not None and rc.R.grad.abs().sum() > 0)
    chk('gate (g) gets non-zero grad', rc.g.grad is not None and rc.g.grad.abs().sum() > 0)
    n_dead = sum((1 for p in rc.b_col.values() if p.grad is None or p.grad.abs().sum().item() == 0))
    chk('every COLUMN expert gets non-zero grad (a col-major example is in the batch)', n_dead == 0)
    all_A = all((any((p.grad is not None and p.grad.abs().sum() > 0 for (n, p) in m.named_parameters() if 'lora_A' in n and f'.{mm}.' in n)) for mm in LAMBDA_MODULES))
    all_B = all((any((p.grad is not None and p.grad.abs().sum() > 0 for (n, p) in m.named_parameters() if 'lora_B' in n and f'.{mm}.' in n)) for mm in LAMBDA_MODULES))
    chk("shared A and the ROW expert (PEFT's lora_B) get grad on all seven families", all_A and all_B)
    chk('frozen base weights get no grad', all((p.grad is None or p.grad.abs().sum() == 0 for (n, p) in m.named_parameters() if 'lora_' not in n and 'base_layer' in n)))
    backward_once(_t.tensor([VIEW_ROW, VIEW_ROW], dtype=_t.uint8), 'anchor')
    chk('all-row-major batch: NO column expert gets gradient', all((p.grad is None or p.grad.abs().sum().item() == 0 for p in rc.b_col.values())))
    chk('all-row-major batch: the row expert still does', any((p.grad is not None and p.grad.abs().sum() > 0 for (n, p) in m.named_parameters() if n.endswith('lora_B.default.weight'))))
    backward_once(_t.tensor([VIEW_COL, VIEW_COL], dtype=_t.uint8), 'anchor')
    chk('all-col-major batch: every column expert gets gradient', all((p.grad is not None and p.grad.abs().sum() > 0 for p in rc.b_col.values())))
    chk('all-col-major batch: NO row expert (lora_B) gets gradient', all((p.grad is None or p.grad.abs().sum().item() == 0 for (n, p) in m.named_parameters() if n.endswith('lora_B.default.weight'))))
    saved_bcol = {k: p.detach().clone() for (k, p) in rc.b_col.items()}
    with _t.no_grad():
        for (name, mod) in m.named_modules():
            lb = getattr(mod, 'lora_B', None)
            if not isinstance(lb, nn.ModuleDict):
                continue
            mtp = name.split('.')[-1]
            if mtp not in rc.module_names:
                continue
            li = int(_LAYER_IDX_RE.search(name).group(1))
            rc.b_col[rc._key(li, rc.module_names.index(mtp))].copy_(lb['default'].weight.detach().to(_t.float32))
    backward_once(VIEW_BOTH, 'mixture')
    g_mix = float(rc.g.grad.abs().max().item())
    backward_once(VIEW_BOTH, 'anchor')
    g_anc = float(rc.g.grad.abs().max().item())
    chk(f'with B_r == B_c the MIXTURE gives the gate no gradient (max|dg|={g_mix:.2e})', g_mix < 1e-08)
    chk(f'the ANCHORED route does give it one (max|dg|={g_anc:.2e})', g_anc > 1e-06)
    with _t.no_grad():
        for (k, v) in saved_bcol.items():
            rc.b_col[k].copy_(v)
    import json as _json
    import tempfile
    pre_save = fwd(tid, expert_mode='mixture')
    with tempfile.TemporaryDirectory() as td:
        save_rowcol_experts(rc, td, {'n_tasks': 2, 'n_layers': nL, 'task_names': ['cta', 'tabfact']})
        (rc2, cfg2) = load_rowcol_experts(td)
        chk('RC save/load round-trip (Lambda R)', rc2 is not None and _t.equal(rc2.R, rc.R.detach().cpu()))
        chk('RC save/load round-trip (gate logits g)', _t.equal(rc2.g, rc.g.detach().cpu()))
        chk('RC save/load round-trip (all column experts)', len(rc2.b_col) == len(rc.b_col) and all((_t.equal(rc2.b_col[k], rc.b_col[k].detach().cpu()) for k in rc.b_col)))
        chk('RC save/load carries the config', cfg2.get('n_layers') == nL)
        detach_rowcol_experts()
        rc2.to(m.device)
        m.rowcol = rc2
        attach_rowcol_experts(m, rc2)
        post_load = fwd(tid, expert_mode='mixture')
        d_rt = (pre_save.float() - post_load.float()).abs().max().item()
        chk(f'RC save/load reproduces outputs (diff={d_rt:.2e})', d_rt < 1e-05)
        chk('adapter without RC -> load_rowcol_experts None', load_rowcol_experts(tempfile.mkdtemp())[0] is None)
        old = tempfile.mkdtemp()
        _t.save({'lam': _t.zeros(1)}, Path(old) / 'role_b_lora.pt')
        refused = False
        try:
            load_rowcol_experts(old)
        except RuntimeError as e:
            refused = 'REMOVED RB-LoRA' in str(e)
        chk('an old RB-LoRA adapter is refused, not silently reinterpreted', refused)
        diag = rc2.diagnostics(model=m)
        chk('RC diagnostics carry the per-task mean gate', 'mean_w_row_by_task' in diag['global'])
        _json.dumps({'summary': rc2.summary(), 'per_task': rc2.per_task_table(), 'diag': diag, 'grads': rc.grad_norms()})
        chk('RC summary + per_task_table + diagnostics + grad_norms JSON-serialize (driver save path)', True)
    detach_rowcol_experts()
    rc_nl = RowColExperts(2, 2, 4, task_names=['cta', 'cpa'], module_names=['q_proj'], use_lambda=False)
    rc_nl.add_b(0, 0, 6)
    rc_nl.add_b(1, 0, 6)
    chk('no_lambda: R is NOT a trainable parameter', not isinstance(rc_nl.R, nn.Parameter) and rc_nl.n_lambda_params() == 0)
    chk('no_lambda: live trainable count == n_params() (gate + column expert only)', sum((p.numel() for p in rc_nl.parameters())) == rc_nl.n_params() == rc_nl.n_gate_params() + rc_nl.n_b_params())
    chk('no_lambda: Lambda is exactly I (R stays zero)', float(rc_nl.R.abs().max().item()) == 0.0)
    chk('no_lambda: gate still starts at exactly 0.5/0.5', abs(float(rc_nl.weights()[0, 0, 0, 0].item()) - 0.5) < 1e-12)
    chk('no_lambda: gate logits still require grad (the one task-specific knob left)', rc_nl.g.requires_grad)
    chk('no_lambda: grad_norms reports lambda as None, gate as a number', rc_nl.grad_norms()['lambda_grad_norm'] is None)
    chk('no_lambda: summary names the variant and serializes', rc_nl.summary()['variant'] == 'rowcol_experts_no_lambda' and json.loads(json.dumps(rc_nl.summary()))['use_lambda'] is False)
    with tempfile.TemporaryDirectory() as _td:
        save_rowcol_experts(rc_nl, _td, {'variant': 'rowcol_experts_no_lambda'})
        (rc_back, _) = load_rowcol_experts(_td)
    chk('no_lambda: save/load round-trip preserves use_lambda and the param count', rc_back.use_lambda is False and rc_back.n_lambda_params() == 0 and (rc_back.n_params() == rc_nl.n_params()))
    print('RC-LoRA (no-Lambda) OK')
    print('RC-LoRA OK')
if __name__ == '__main__':
    _selftest()
