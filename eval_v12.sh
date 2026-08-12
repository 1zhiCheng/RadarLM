#!/usr/bin/env bash
# Evaluate v12 on 4× RTX 4090 (DDP). ~25 min for 1500 val + 1500 test.
set -euo pipefail

cd "$(dirname "$0")"

DATA_DIR="/data/storage/zzy/radar_agent_data"
MODEL_DIR="output/v9_qa_ddp_v12"
QWEN_PATH="${MODEL_DIR}/qwen_v12_e1_merged"
PROJ_PATH="${MODEL_DIR}/projector_e1.pt"

if [[ ! -d "$QWEN_PATH" ]]; then
  echo "Merged model not found at $QWEN_PATH — running merge_lora.py first..."
  python radarlm/vlm/merge_lora.py \
    --lora_path "${MODEL_DIR}/lora_e1" \
    --out_path "$QWEN_PATH"
fi

mkdir -p logs

torchrun \
  --nproc_per_node=4 \
  --master_port=29501 \
  radarlm/vlm/eval_v9_v2_runner.py \
  --qwen_path      "$QWEN_PATH" \
  --projector_path "$PROJ_PATH" \
  --jsonl_val  "${DATA_DIR}/val_qwen_mt_v12.jsonl" \
  --jsonl_test "${DATA_DIR}/test_qwen_mt_v12.jsonl" \
  --output_dir "$MODEL_DIR" \
  --max_val  1500 \
  --max_test 1500

echo "✔ eval done. metrics in ${MODEL_DIR}/{val,test}_metrics_v2.json"
