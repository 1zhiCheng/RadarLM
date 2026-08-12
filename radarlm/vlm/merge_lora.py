"""把 LoRA adapter merge 到 base Qwen2-VL, 保存完整 base model (供评估用).

★ 关键: eval 必须用 merge 后的完整 base model, 否则等于评估纯 base (训练白做).
"""
import argparse
import os
import sys
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
from peft import PeftModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_path", default="/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")
    p.add_argument("--lora_path", required=True, help="LoRA adapter 路径 (peft format)")
    p.add_argument("--out_path", required=True, help="merge 后完整 base 的保存路径")
    args = p.parse_args()

    print(f"[1/4] 加载 base Qwen2-VL from {args.base_path} ...")
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_path, torch_dtype=torch.bfloat16,
    )

    print(f"[2/4] 加载 LoRA adapter from {args.lora_path} ...")
    peft_model = PeftModel.from_pretrained(base, args.lora_path)

    print(f"[3/4] merge_and_unload ...")
    merged = peft_model.merge_and_unload()
    # 现在 merged 是普通 Qwen2VLForConditionalGeneration, LoRA 已合并到 base 权重

    print(f"[4/4] 保存 merged model → {args.out_path}")
    os.makedirs(args.out_path, exist_ok=True)
    merged.save_pretrained(args.out_path, safe_serialization=True)
    # 也复制 tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_path)
    tok.save_pretrained(args.out_path)

    print(f"[Done] merged model 保存到 {args.out_path}")
    print(f"       下一步: 用 {args.out_path} 作为 qwen_path 评估 (无需再传 --lora_path)")


if __name__ == "__main__":
    main()