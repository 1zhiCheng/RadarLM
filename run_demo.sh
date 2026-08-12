#!/usr/bin/env bash
# Launch the interactive RadarLM demo on http://localhost:8765
# Requires: merged Qwen2-VL at output/v9_qa_ddp_v12/qwen_v12_e1_merged
#           PKC weights  at radarlm/pkc_backbone/weights/pkcin_silu_gn.pt
set -euo pipefail

cd "$(dirname "$0")"

export QWEN_PATH="${QWEN_PATH:-output/v9_qa_ddp_v12/qwen_v12_e1_merged}"

if [[ ! -d "$QWEN_PATH" ]]; then
  echo "Merged model not found at $QWEN_PATH"
  echo "Run train_v12.sh + eval_v12.sh first, or set QWEN_PATH env."
  exit 1
fi

mkdir -p logs
exec python demo/backend/app.py --host 0.0.0.0 --port 8765
