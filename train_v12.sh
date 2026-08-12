#!/usr/bin/env bash
# Train v12 on 4× RTX 4090 (DDP). ~17 min for 1 epoch on 96,870 samples.
set -euo pipefail

cd "$(dirname "$0")"

DATA_DIR="/data/storage/zzy/radar_agent_data"
OUT_DIR="output/v9_qa_ddp_v12"

mkdir -p "$OUT_DIR" logs

# 4-GPU DDP
torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  radarlm/vlm/train_v9_qa_ddp.py \
  --jsonl_train "${DATA_DIR}/train_qwen_mt_v12.jsonl" \
  --jsonl_val   "${DATA_DIR}/val_qwen_mt_v12.jsonl" \
  --output_dir  "$OUT_DIR" \
  --num_epochs  1 \
  --lr 2e-5

echo "✔ training done. artifacts in $OUT_DIR"
