"""用新的 v2 eval (视图一致性 + 严格数值 + 宽松 bbox) 重新评估 DDP 训好的模型."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/zzy/Myproject/RadarLM")
sys.path.insert(0, "/home/zzy/Myproject/PKC")

from radarlm.pkc_backbone.pkc_silu_wrapper import PKCWithPretrained
from radarlm.vlm.train_v9_qa_ddp import PKCQwenAlign, RadarQADataset, build_qa_prompt
from radarlm.vlm.eval_v9_v2 import eval_qa_pair, aggregate_metrics
from transformers import AutoTokenizer


def build_single_qa_prompt(conversations, qa_idx, num_image_tokens=512):
    """构造单轮 QA prompt: system + image + 之前所有 user/assistant + 当前 user."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")
    image_pad_str = "<|image_pad|>" * num_image_tokens
    system_msg = (
        "你是雷达感知 Agent, 分析 RD/RA/AD 三视图.\n"
        "视图尺寸参考 (必须记忆):\n"
        "  - RD 视图 (距离-多普勒): 256×64 (高×宽)\n"
        "  - RA 视图 (距离-角度): 256×256 (高×宽)\n"
        "  - AD 视图 (角度-多普勒): 256×64 (高×宽)\n"
        "回答要求:\n"
        "  1. 问哪个视图, 必须答哪个视图 (RD/RA/AD)\n"
        "  2. 距离/速度/角度数值要精确 (单位 m 或 m/s 或 °)\n"
        "  3. bbox 格式: 'RD 视图 bbox: [row_min=X, col_min=Y, row_max=Z, col_max=W]'\n"
        "  4. 不要乱猜, 不确定就说'图中无目标'\n"
    )
    parts = [f"<|im_start|>system\n{system_msg}<|im_end|>\n"]
    # 找到第 qa_idx 个 user question
    user_count = 0
    target_q = None
    for j, msg in enumerate(conversations):
        if msg["from"] == "user":
            if user_count == qa_idx:
                target_q = msg["value"].replace("<image>", "").strip()
                # 加 user prompt (含 image_pad)
                if j == 0 or (j > 0 and conversations[j-1]["from"] != "user"):
                    parts.append(f"<|im_start|>user\n{image_pad_str}{target_q}<|im_end|>\n")
                else:
                    parts.append(f"<|im_start|>user\n{target_q}<|im_end|>\n")
                parts.append("<|im_start|>assistant\n")
                break
            user_count += 1
    return "".join(parts)


def evaluate_split(model, ds, split_name, max_samples=None, save_path=None):
    """评估每个 sample 的所有 5 轮 QA (逐轮 generate + eval)."""
    model.eval()
    tok = AutoTokenizer.from_pretrained("/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")
    all_metrics = []
    if max_samples is not None:
        n = min(max_samples, len(ds))
    else:
        n = len(ds)
    # ★ DDP 分片: 按 rank 取 sample 区间
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    start = (n * local_rank) // world_size
    end = (n * (local_rank + 1)) // world_size
    print(f"[eval {split_name} rank{local_rank}/{world_size}] samples [{start}, {end}) of {n}", flush=True)
    for i in range(start, end):
        sample = ds[i]
        qa_pairs = []
        for j in range(len(sample["conversations"]) - 1):
            if sample["conversations"][j]["from"] == "user":
                q = sample["conversations"][j]["value"].replace("<image>", "").strip()
                if j + 1 < len(sample["conversations"]):
                    a = sample["conversations"][j + 1]["value"]
                    qa_pairs.append((q, a))
        x_rd = sample["x_rd"].cuda(local_rank)
        x_ra = sample["x_ra"].cuda(local_rank)
        x_ad = sample["x_ad"].cuda(local_rank)
        visual_embeds = model.get_visual_embeds(x_rd, x_ra, x_ad)
        # v6: RD (16x4=64) + RA (16x16=256) 两个独立 grid
        image_grid_thw = torch.tensor([[1, 16, 4], [1, 16, 16]], dtype=torch.long).cuda(local_rank)
        # 逐轮 evaluate
        for k, (q, gt_a) in enumerate(qa_pairs):
            # 单轮 prompt
            prompt = build_single_qa_prompt(sample["conversations"], k, 320)  # v4: 64+256=320
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = enc["input_ids"].cuda(local_rank)
            attention_mask = enc["attention_mask"].cuda(local_rank)
            with torch.no_grad():
                out_ids = model.qwen.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    pixel_values=visual_embeds, image_grid_thw=image_grid_thw,
                    min_new_tokens=10, max_new_tokens=80, do_sample=False, repetition_penalty=1.1,
                )
            gen_a = tok.decode(out_ids[0, input_ids.size(1):], skip_special_tokens=True).strip()
            m = eval_qa_pair(q, gt_a, gen_a)
            m["sample_idx"] = i
            m["qa_idx"] = k
            m["gen_answer"] = gen_a[:80]
            m["gt_answer"] = gt_a[:80]
            all_metrics.append(m)
        if i < 3:
            print(f"  [{split_name} {i}] {len(qa_pairs)} QA pairs")
            for k in range(min(3, len(qa_pairs))):
                q, gt_a = qa_pairs[k]
                print(f"    Q{k}: {q[:60]}")
                print(f"    TRUE: {gt_a[:60]}")
        if (i + 1) % 100 == 0:
            print(f"  [{split_name}] {i+1}/{n} done", flush=True)
    agg = aggregate_metrics(all_metrics)
    print(f"[{split_name} rank{local_rank}] {json.dumps(agg, ensure_ascii=False)}", flush=True)
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        # ★ DDP: 存 per-QA metrics 列表 (rank 0 再合并)
        # 如果有 world_size > 1, 存全量 per-QA 列表; 否则存聚合
        if world_size > 1:
            with open(save_path, "w") as f:
                json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        else:
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
    parser.add_argument("--lora_path", default="",
                        help="LoRA adapter 路径 (peft format). ★ 已废弃: 现在用 merge_lora.py 先 merge, 直接传 --qwen_path 指向 merged model.")
    parser.add_argument("--output_dir", default="/home/zzy/Myproject/RadarLM/output/v9_qa_ddp_full")
    parser.add_argument("--max_val", type=int, default=2448, help="Max val samples (use 2448 for full)")
    parser.add_argument("--max_test", type=int, default=2130, help="Max test samples (use 2130 for full)")
    args = parser.parse_args()

    # ★ DDP 初始化 (4 卡并行加速 ~4×)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        is_main = rank == 0
    else:
        is_main = True
    print(f"[DDP eval] local_rank={local_rank}, world_size={world_size}, is_main={is_main}", flush=True)

    print("[Setup] loading data...")
    val_ds = RadarQADataset(args.jsonl_val, args.carrada_root, seed=42)
    test_ds = RadarQADataset(args.jsonl_test, args.carrada_root, seed=42)

    print(f"[Setup] loading model on cuda:{local_rank}...")
    model = PKCQwenAlign(args.qwen_path).cuda(local_rank)
    sd = torch.load(args.projector_path, weights_only=False, map_location=f"cuda:{local_rank}")
    # strict=False: 旧 weights 包含 norm (新 model 无), 只 load proj weight/bias
    model.proj_rd.load_state_dict(sd["proj_rd"], strict=False)
    model.proj_ra.load_state_dict(sd["proj_ra"], strict=False)
    print(f"[loaded] {args.projector_path} (strict=False, ignore norm)", flush=True)
    # ★ 关键: qwen_path 必须指向 merge 后的完整 base (见 merge_lora.py)
    # 不要在 eval runner 里做 PeftModel.from_pretrained, 会和 PKCQwenAlign 内 get_peft_model 嵌套冲突
    print(f"[using] qwen_path={args.qwen_path} (应是 merge 后的完整 base)", flush=True)

    # ★ DDP: 每个 rank 写自己的部分 metrics, 然后 rank 0 合并
    val_metrics = evaluate_split(model, val_ds, "val", max_samples=args.max_val,
                                 save_path=f"/tmp/eval_rank{local_rank}_val.json")
    test_metrics = evaluate_split(model, test_ds, "test", max_samples=args.max_test,
                                  save_path=f"/tmp/eval_rank{local_rank}_test.json")

    if world_size > 1:
        dist.barrier()  # 等待所有 rank 完成
        if is_main:
            # rank 0 合并所有 rank 的 metrics
            from radarlm.vlm.eval_v9_v2 import aggregate_metrics
            def _merge(split):
                all_metrics = []
                for r in range(world_size):
                    p = f"/tmp/eval_rank{r}_{split}.json"
                    if os.path.exists(p):
                        with open(p) as f:
                            all_metrics.extend(json.load(f))
                return aggregate_metrics(all_metrics), all_metrics
            for split, sp in [("val", "val_metrics_v2"), ("test", "test_metrics_v2")]:
                agg, am = _merge(split)
                os.makedirs(args.output_dir, exist_ok=True)
                with open(f"{args.output_dir}/{sp}.json", "w") as f:
                    json.dump(agg, f, indent=2, ensure_ascii=False)
                # 写全量 per-QA metrics 用于 badcase 分析
                with open(f"{args.output_dir}/{sp}_full.json", "w") as f:
                    json.dump(am, f, indent=2, ensure_ascii=False)
                print(f"[merged {split}] {sp}.json: {json.dumps(agg, ensure_ascii=False)}", flush=True)
        dist.destroy_process_group()
    print("[Done]")


if __name__ == "__main__":
    main()