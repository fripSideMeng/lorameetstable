# Role-conditioned LoRA for table understanding

Training and evaluation code for LoRA fine-tuning of a causal LM for tabular data related tasks.

Requirements: `torch`, `transformers`, `peft`, `datasets`, `numpy`, `tqdm`.

## Layout

| folder | contents |
|---|---|
| `drivers/` | command-line entry points: standard LoRA training, evaluation, structural training, role-conditioned training and evaluation |
| `common/` | base-model and tokenizer loading |
| `data_prep/` | dataset preparation into `data_prepared/`, plus the synthesized structural-relation and same-row task builders |
| `table/` | table serialization utilities: truncation / budget fitting |
| `lora/` | LoRA variants and role conditioning |
| `attention/` | attention-level mechanisms: structural attention bias, context relay masks, target-column masking, target sandwich, the attention kernels and their training hooks |

## Data

Prepared train/val/test sets are read from `$LORA_GEN_ROOT/data_prepared`
(`LORA_GEN_ROOT` defaults to the current directory) and are built by the scripts
in `data_prep/` from the raw corpora below. Only the training-time table
truncation / row-selection logic is included in this package; the raw corpora
themselves are public:

| corpus | used for | expected path | download |
|---|---|---|---|
| SOTAB (WDC Schema.org Table Annotation Benchmark) | SOTAB CTA/CPA | `./SOTAB/{CTA,CPA}` | https://webdatacommons.org/structureddata/sotab/ |
| MMTU | held-out CTA/CPA test split and the verbatim prompt templates | `./MMTU/data/train-00000-of-00001.parquet` | https://huggingface.co/datasets/MMTU-benchmark/MMTU |

## Train and evaluate

Run from this directory. Multi-task (CTA+CPA) LoRA fine-tune, then MMTU
evaluation of the adapter:

```sh
export LORA_GEN_ROOT=.

export PYTHONPATH="$PWD/drivers:$PWD/common:$PWD/data_prep:$PWD/table:$PWD/lora:$PWD/attention:$PWD/diagnostics${PYTHONPATH:+:$PYTHONPATH}"

python -u drivers/train_lora.py --model "$BASE_MODEL" --regime multi_ctacpa --seed 42 --budget 2000 --cta_cpa_source t2d --select_mode random --drop_empty_target --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --max_seq_len 1024 --epochs 3 --lr 6e-5 --micro_bs 4 --grad_accum 2 --attn_impl sdpa --val_max 1024 --val_max_new_tokens 64 --val_gen_batch_size 16 --out out/adapter

python -u drivers/evaluate.py --model "$BASE_MODEL" --adapter out/adapter --tasks cta_mmtu --max_new_tokens 64 --bs 16 --attn_impl sdpa --out out/cta_mmtu.json

python -u drivers/evaluate.py --model "$BASE_MODEL" --adapter out/adapter --tasks cpa_mmtu --max_new_tokens 64 --bs 16 --attn_impl sdpa --out out/cpa_mmtu.json
```

Role-conditioned variant: run `drivers/train_role_low_rank.py` with the same data
flags, then `drivers/evaluate_role_low_rank.py --model "$BASE_MODEL" --adapter
out/adapter --tasks cta_mmtu cpa_mmtu --eval_role_mode normal --out
out/role_cta_cpa.json`.