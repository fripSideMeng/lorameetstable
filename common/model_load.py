from transformers import AutoConfig, AutoModelForCausalLM
_TEXT_ONLY = {'gemma3': 'Gemma3ForCausalLM'}

def resolve_model_class(model_name):
    try:
        cfg = AutoConfig.from_pretrained(model_name)
    except Exception:
        return (AutoModelForCausalLM, 'config unreadable; AutoModelForCausalLM')
    mt = getattr(cfg, 'model_type', None)
    name = _TEXT_ONLY.get(mt)
    if name is not None and hasattr(cfg, 'text_config'):
        import transformers
        return (getattr(transformers, name), f'{mt} is a COMPOSITE checkpoint (has text_config); loading {name} so the vision tower is never materialized and PEFT cannot wrap its q/k/v')
    return (AutoModelForCausalLM, f'model_type={mt}; AutoModelForCausalLM')

def load_causal_lm(model_name, *, torch_dtype, attn_implementation, device_map, **kw):
    (cls, why) = resolve_model_class(model_name)
    print(f'[model_load] {model_name} -> {cls.__name__}   ({why})')
    return cls.from_pretrained(model_name, torch_dtype=torch_dtype, attn_implementation=attn_implementation, device_map=device_map, **kw)
