"""v9 DDP 训练后, 加载 projector, 跑 val + test 全量评估."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/zzy/Myproject/RadarLM")
sys.path.insert(0, "/home/zzy/Myproject/PKC")

from radarlm.pkc_backbone.pkc_silu_wrapper import PKCWithPretrained
from radarlm.vlm.train_v9_qa_ddp import PKCQwenAlign, RadarQADataset, build_qa_prompt
from radarlm.vlm.eval_v9 import eval_qa_pair, aggregate_metrics
from transformers import AutoTokenizer


def evaluate_split(model, ds, split_name, max_samples=None, save_path=None):
    model.eval()
    tok = AutoTokenizer.from_pretrained("/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")
    all_metrics = []
    n = max_samples or len(ds)
    n = min(n, len(ds))
    print(f"[eval {split_name}] {n} samples", flush=True)
    for i in range(n):
        sample = ds[i]
        prompt, gt_answer = build_qa_prompt(sample["conversations"], 512)
        x_rd = sample["x_rd"].cuda()
        x_ra = sample["x_ra"].cuda()
        x_ad = sample["x_ad"].cuda()
        visual_embeds = model.get_visual_embeds(x_rd, x_ra, x_ad)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = enc["input_ids"].cuda()
        attention_mask = enc["attention_mask"].cuda()
        image_grid_thw = torch.tensor([[1, 32, 64]], dtype=torch.long).cuda()
        with torch.no_grad():
            out_ids = model.qwen.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=visual_embeds,
                image_grid_thw=image_grid_thw,
                min_new_tokens=20,
                max_new_tokens=128,
                do_sample=False,
                repetition_penalty=1.1,
            )
        gen = tok.decode(out_ids[0, input_ids.size(1):], skip_special_tokens=True)
        first_q = sample["conversations"][0]["value"].replace("<image>", "").strip()
        m = eval_qa_pair(first_q, gt_answer, gen)
        all_metrics.append(m)
        if i < 2:
            print(f"  [{split_name} {i}] Q: {first_q[:60]}")
            print(f"    TRUE: {gt_answer[:100]}")
            print(f"    GEN:  {gen[:100]}")
        if (i + 1) % 50 == 0:
            print(f"  [{split_name}] {i+1}/{n} done", flush=True)
    agg = aggregate_metrics(all_metrics)
    print(f"[{split_name}] {json.dumps(agg, ensure_ascii=False)}", flush=True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_path", default="/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")
    parser.add_argument("--jsonl_val", default="/data/storage/zzy/radar_agent_data/val_qwen_mt.jsonl")
    parser.add_argument("--jsonl_test", default="/data/storage/zzy/radar_agent_data/test_qwen_mt.jsonl")
    parser.add_argument("--carrada_root", default="/data/storage/zzy/Carrada")
    parser.add_argument("--projector_path", default="/home/zzy/Myproject/RadarLM/output/v9_qa_ddp_full/projector_e3.pt")
    parser.add_argument("--output_dir", default="/home/zzy/Myproject/RadarLM/output/v9_qa_ddp_full")
    args = parser.parse_args()

    print("[Setup] loading data...")
    val_ds = RadarQADataset(args.jsonl_val, args.carrada_root, seed=42)
    test_ds = RadarQADataset(args.jsonl_test, args.carrada_root, seed=42)

    print("[Setup] loading model...")
    model = PKCQwenAlign(args.qwen_path).cuda()
    sd = torch.load(args.projector_path, weights_only=False)
    model.proj_rd.load_state_dict(sd["proj_rd"])
    model.proj_ra.load_state_dict(sd["proj_ra"])
    print(f"[loaded] {args.projector_path}", flush=True)

    # Val (2448 samples)
    val_metrics = evaluate_split(model, val_ds, "val",
                                 save_path=f"{args.output_dir}/val_metrics_2448.json")
    # Test (2130 samples)
    test_metrics = evaluate_split(model, test_ds, "test",
                                  save_path=f"{args.output_dir}/test_metrics_2130.json")
    print("[Done]")


if __name__ == "__main__":
    main()