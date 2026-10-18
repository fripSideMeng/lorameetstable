import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate as E
import role_low_rank as RLR
import train_lora as TL
from train_role_low_rank import role_generate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=f'{E.ROOT}/Qwen3-4B-Instruct-2507')
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tasks', nargs='+', default=['cta', 'cpa'], choices=['cta', 'cpa', 'cta_mmtu', 'cpa_mmtu'])
    ap.add_argument('--max_new_tokens', type=int, default=400)
    ap.add_argument('--gen_bs', type=int, default=16)
    ap.add_argument('--attn_impl', default='sdpa', choices=['sdpa', 'flash_attention_2', 'eager'])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--drop_empty_target', default=None, action=argparse.BooleanOptionalAction)
    ap.add_argument('--table_first', default=None, action=argparse.BooleanOptionalAction)
    ap.add_argument('--eval_role_mode', default='normal', choices=['normal', 'no_role', 'role_swap', 'head_tail_swap'])
    args = ap.parse_args()
    table_first = E.resolve_table_first(args.adapter, None, args.table_first)
    tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn_impl, device_map={'': 0})
    TL.force_greedy_generation(model, tok)
    cfg = {}
    role_sep = None
    if args.adapter:
        cfg = RLR.load_role_adapter(model, args.adapter)
        print(f"loaded role-low-rank adapter regime={cfg.get('regime')} source={cfg.get('cta_cpa_source')} r={cfg.get('lora_r')}/{cfg.get('lora_alpha')} modules={cfg.get('target_modules')} role_modules={cfg.get('role_modules')} role_scheme={cfg.get('role_scheme')} roles={cfg.get('role_vector_names')}")
        RLR.assert_role_architecture(model, expect_role_modules=cfg.get('role_modules'))
        print(RLR.format_role_architecture(RLR.role_architecture_facts(model)))
        sep = RLR.role_separation_summary(model, per_module=True)
        print(RLR.format_role_separation(sep))
        role_sep = sep
    model.eval()
    model.config.use_cache = True
    tok.padding_side = 'left'
    drop_empty = args.drop_empty_target
    if drop_empty is None:
        drop_empty = bool(cfg.get('drop_empty_target', False))
        print(f'[drop_empty_target] auto-detected from adapter config: {drop_empty}')
    else:
        print(f'[drop_empty_target] explicitly set: {drop_empty}')
    out_dir = os.path.dirname(args.out) or '.'
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.out))[0]
    results = {'adapter': args.adapter, 'model': args.model, 'backend': 'hf-role-low-rank', 'role_config': cfg, 'eval_role_mode': args.eval_role_mode, 'eval_role_mode_note': 'no_role, role_swap and head_tail_swap are ABLATIONS of this checkpoint, not baselines; the standard-MTL baseline is the separately trained shared-LoRA model', 'role_separation_summary': role_sep, 'drop_empty_target': drop_empty, 'table_first': bool(table_first), 'tasks': {}}
    for slug in args.tasks:
        (test, prompts, prep) = E.load_and_prepare(slug, limit=args.limit, drop_empty=drop_empty, table_first=table_first)
        print(slug, len(test))
        if prompts:
            print(RLR.format_column_roles(prompts[0], RLR.role_spec(prompts[0], slug)))
        gens = role_generate(model, tok, prompts, args.max_new_tokens, args.gen_bs, args.model, slug, role_mode=args.eval_role_mode)
        results['tasks'][slug] = E.score_and_dump(slug, test, gens, prep, out_dir, base)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote', args.out)
if __name__ == '__main__':
    main()
