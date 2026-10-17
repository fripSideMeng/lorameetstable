# Role-conditioned LoRA for table understanding (CTA / CPA)

Training and evaluation code for LoRA fine-tuning of a causal LM for tabular data related tasks.

Requirements: `torch`, `transformers`, `peft`, `datasets`, `numpy`, `tqdm`.

## Train and evaluate

LoRA fine-tune, then evaluation of the adapter:

```sh
export LORA_GEN_ROOT=.

python -u train_lora.py --model "$BASE_MODEL" --regime multi_ctacpa --seed 42 --budget 2000 --cta_cpa_source t2d --select_mode random --drop_empty_target --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --max_seq_len 1024 --epochs 3 --lr 6e-5 --micro_bs 4 --grad_accum 2 --attn_impl sdpa --val_max 1024 --val_max_new_tokens 64 --val_gen_batch_size 16 --out out/adapter

python -u evaluate.py --model "$BASE_MODEL" --adapter out/adapter --tasks cta_mmtu --max_new_tokens 64 --bs 16 --attn_impl sdpa --out out/cta_mmtu.json

python -u evaluate.py --model "$BASE_MODEL" --adapter out/adapter --tasks cpa_mmtu --max_new_tokens 64 --bs 16 --attn_impl sdpa --out out/cpa_mmtu.json
```

`--micro_bs` / `--grad_accum` are per base model (the values above are for a 3B
model).