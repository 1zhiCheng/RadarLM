"""视觉敏感性测试: 真实/黑图/噪声/错位图 输出差异."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/zzy/Myproject/RadarLM")
sys.path.insert(0, "/home/zzy/Myproject/PKC")

import numpy as np
import torch
from transformers import AutoTokenizer
from radarlm.vlm.train_v9_qa_ddp import PKCQwenAlign, RadarQADataset
from radarlm.vlm.eval_v9_v2_runner import build_single_qa_prompt


def make_black(x_rd, x_ra, x_ad):
    return (torch.zeros_like(x_rd), torch.zeros_like(x_ra), torch.zeros_like(x_ad))


def make_noise(x_rd, x_ra, x_ad, seed=42):
    rng = np.random.RandomState(seed)
    no_rd = torch.from_numpy(rng.rand(*x_rd.shape).astype(np.float32))
    no_ra = torch.from_numpy(rng.rand(*x_ra.shape).astype(np.float32))
    no_ad = torch.from_numpy(rng.rand(*x_ad.shape).astype(np.float32))
    return no_rd.cuda(), no_ra.cuda(), no_ad.cuda()


def make_shuffled(x_rd, x_ra, x_ad):
    """错位图: RD/RA/AD 用错位置的内容."""
    # 保持形状但内容错位 (RD 位置放 RA 内容, 但 RA 形状 256x256 与 RD 256x64 不匹配)
    # 简单做法: 在 RD 范围内随机选 RA 一部分, 但简化不做这个
    return x_ra, x_rd, x_ad  # 会 shape error, 跳过


def test_variant(model, ds, sample_idx, qa_idx, x_rd, x_ra, x_ad, tok, image_grid_thw, label):
    sample = ds[sample_idx]
    visual_embeds = model.get_visual_embeds(x_rd, x_ra, x_ad)
    prompt = build_single_qa_prompt(sample["conversations"], qa_idx, 320)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = enc["input_ids"].cuda()
    attention_mask = enc["attention_mask"].cuda()
    with torch.no_grad():
        out_ids = model.qwen.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=visual_embeds, image_grid_thw=image_grid_thw,
            min_new_tokens=20, max_new_tokens=128, do_sample=False, repetition_penalty=1.1,
        )
    gen = tok.decode(out_ids[0, input_ids.size(1):], skip_special_tokens=True).strip()
    visual_norm = float(visual_embeds.norm())
    visual_mean = float(visual_embeds.mean())
    q = sample["conversations"][qa_idx * 2]["value"].replace("<image>", "").strip()
    gt = sample["conversations"][qa_idx * 2 + 1]["value"]
    print(f"\n=== {label} ===")
    print(f"Q: {q[:80]}")
    print(f"TRUE: {gt[:80]}")
    print(f"GEN:  {gen[:120]}")
    print(f"visual: norm={visual_norm:.4f} mean={visual_mean:.4f}")


def main():
    print("加载 v4 模型...")
    ds = RadarQADataset("/data/storage/zzy/radar_agent_data/test_qwen_mt.jsonl",
                        "/data/storage/zzy/Carrada", seed=42)
    model = PKCQwenAlign("/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct").cuda()
    # 新模型没 norm, 不加载旧权重 (避免 shape mismatch)
    # sd = torch.load("/home/zzy/Myproject/RadarLM/output/v9_qa_ddp_v4/projector_e3.pt", weights_only=False)
    # model.proj_rd.load_state_dict(sd["proj_rd"])
    # model.proj_ra.load_state_dict(sd["proj_ra"])
    model.eval()
    tok = AutoTokenizer.from_pretrained("/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")
    image_grid_thw = torch.tensor([[1, 32, 64]], dtype=torch.long).cuda()

    # 取 5 个有目标样本
    for sample_idx in [0, 1, 2, 3, 4]:
        sample = ds[sample_idx]
        x_rd = sample["x_rd"].cuda()
        x_ra = sample["x_ra"].cuda()
        x_ad = sample["x_ad"].cuda()
        # 第一个 QA (qa_idx=0)
        qa_idx = 0
        print(f"\n========== 样本 {sample_idx} QA {qa_idx} ==========")
        # A. 真实图
        test_variant(model, ds, sample_idx, qa_idx, x_rd, x_ra, x_ad, tok, image_grid_thw, "A. 真实图")
        # B. 黑图
        b_rd, b_ra, b_ad = make_black(x_rd, x_ra, x_ad)
        test_variant(model, ds, sample_idx, qa_idx, b_rd, b_ra, b_ad, tok, image_grid_thw, "B. 黑图")
        # C. 随机噪声
        c_rd, c_ra, c_ad = make_noise(x_rd, x_ra, x_ad)
        test_variant(model, ds, sample_idx, qa_idx, c_rd, c_ra, c_ad, tok, image_grid_thw, "C. 随机噪声")
        # D. 错位图 (RD/RA 互换)
        d_rd, d_ra, d_ad = make_shuffled(x_rd, x_ra, x_ad)
        test_variant(model, ds, sample_idx, qa_idx, d_rd, d_ra, d_ad, tok, image_grid_thw, "D. 错位图 (RD/RA 互换)")


if __name__ == "__main__":
    main()