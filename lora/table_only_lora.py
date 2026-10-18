import torch
import torch.nn as nn
import recv_gate as RCV
MECHANISM = 'lora_table_only'
VERSION = 1
FORMULA = 'y = W0 x + s * B(A(drop(x))) * m,  m = 1 on table tokens'
TOKEN_SCOPE = 'm = 1 on tokens whose region id is recv_gate.R_TABLE (the Markdown table block found by find_markdown_table_span); m = 0 on pre-table, post-table, padding and every generated token'

class _Ctx:
    active = False
    mask_full = None
    cur = None
CTX = _Ctx()

def set_table_mask(mask):
    m = torch.as_tensor(mask).bool()
    if m.dim() != 2:
        raise ValueError(f'table mask must be [B,T]; got {tuple(m.shape)}')
    CTX.mask_full = m
    CTX.cur = None
    CTX.active = True
    return CTX.mask_full

def set_table_mask_from_regions(region_ids):
    return set_table_mask(torch.as_tensor(region_ids) == RCV.R_TABLE)

def clear_table_mask():
    CTX.active = False
    CTX.mask_full = None
    CTX.cur = None

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

def _hook(module, args, kwargs, output):
    cur = CTX.cur
    if cur is None:
        raise RuntimeError('--lora_table_only: no table mask is installed for this forward. The mask is per-batch context (set_table_mask / set_table_mask_from_regions) and the model pre-hook slices it; a forward without it would train the adapter on EVERY token silently. Refusing to guess.')
    if cur.shape[0] != output.shape[0] or cur.shape[1] != output.shape[1]:
        raise RuntimeError(f'--lora_table_only: table mask {tuple(cur.shape)} does not match the lora_B output {tuple(output.shape[:2])} of this forward.')
    return output * cur.to(device=output.device, dtype=output.dtype).unsqueeze(-1)

def _lora_b_linears(model):
    found = []
    for (name, mod) in model.named_modules():
        parts = name.split('.')
        if len(parts) >= 2 and parts[-2] == 'lora_B' and isinstance(mod, nn.Linear):
            found.append((name, mod))
    return found

def attach(model):
    mods = _lora_b_linears(model)
    if not mods:
        raise RuntimeError('--lora_table_only found no PEFT lora_B modules on this model; attach() must run AFTER get_peft_model, or there is no adapter to confine.')
    handles = [m.register_forward_hook(_hook, with_kwargs=True) for (_, m) in mods]
    inner = model.get_base_model() if hasattr(model, 'get_base_model') else model
    hosts = [model] if inner is model else [model, inner]
    handles += [h.register_forward_pre_hook(_slice_prehook, with_kwargs=True) for h in hosts]
    model._table_only_handles = handles
    model._table_only_modules = mods
    return {'mechanism': MECHANISM, 'version': VERSION, 'n_lora_B_hooked': len(mods), 'formula': FORMULA, 'token_scope': TOKEN_SCOPE, 'adds_parameters': False, 'prehook_on': [type(h).__name__ for h in hosts], 'layer_names': [n for (n, _) in mods]}

def detach(model):
    for h in getattr(model, '_table_only_handles', []):
        h.remove()
    model._table_only_handles = []
    model._table_only_modules = []
    clear_table_mask()

def is_attached(model):
    return bool(getattr(model, '_table_only_modules', ()))

def verify_masking(model, input_ids, attention_mask=None, seed=0):
    mods = getattr(model, '_table_only_modules', ())
    if not mods:
        raise RuntimeError('--lora_table_only is not attached to this model')
    dev = next(model.parameters()).device
    ids = torch.as_tensor(input_ids).to(dev)
    am = torch.ones_like(ids) if attention_mask is None else torch.as_tensor(attention_mask).to(dev)
    cut = ids.shape[1] // 2
    m = torch.zeros(ids.shape, dtype=torch.bool, device=dev)
    m[:, cut:] = True
    was_training = model.training
    model.eval()
    saved = [(mod, mod.weight.detach().clone()) for (_, mod) in mods]
    g = torch.Generator(device='cpu').manual_seed(seed)
    try:
        with torch.no_grad():
            for (mod, w) in saved:
                mod.weight.copy_((torch.randn(w.shape, generator=g) * 0.02).to(w.device, w.dtype))
            set_table_mask(torch.zeros_like(m))
            base = model(input_ids=ids, attention_mask=am).logits.float()
            set_table_mask(m)
            on = model(input_ids=ids, attention_mask=am).logits.float()
    finally:
        with torch.no_grad():
            for (mod, w) in saved:
                mod.weight.copy_(w)
        clear_table_mask()
        if was_training:
            model.train()
    off_diff = float((on[:, :cut] - base[:, :cut]).abs().max())
    on_diff = float((on[:, cut:] - base[:, cut:]).abs().max())
    if off_diff != 0.0:
        raise RuntimeError(f'--lora_table_only: the LoRA delta LEAKED off the table -- max|logit diff| = {off_diff:.3e} at masked positions that cannot causally see an unmasked one. The mask is not reaching every lora_B.')
    if on_diff == 0.0:
        raise RuntimeError('--lora_table_only: the delta is zero at the UNMASKED positions too, so the hook is eating the whole adapter rather than confining it. Refusing to train.')
    return {'n_lora_B_hooked': len(mods), 'masked_prefix': int(cut), 'max_abs_logit_diff_masked': off_diff, 'max_abs_logit_diff_table': on_diff}
