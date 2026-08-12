"""小样本过拟合测试: 8 样本, 验证修复后模型能否学视觉特征."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/zzy/Myproject/RadarLM")
sys.path.insert(0, "/home/zzy/Myproject/PKC")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
from peft import LoraConfig, get_peft_model

from radarlm.pkc_backbone.pkc_silu_wrapper import PKCWithPretrained
from radarlm.vlm.train_v9_qa_ddp import PKCProjectorX9, RadarQADataset, PKCQwenAlign


def main():
    print("加载 v4 修复版模型 (无 LayerNorm)...")
    ds = RadarQADataset("/data/storage/zzy/radar_agent_data/test_qwen_mt.jsonl",
                        "/data/storage/zzy/Carrada", seed=42)
    model = PKCQwenAlign("/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct").cuda()
    model.eval()
    tok = AutoTokenizer.from_pretrained("/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")

    # 用 5 样本, 重复 100 step 看 loss 能否下降
    n_samples = 5
    n_steps = 100
    sample_indices = list(range(n_samples))

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable: {sum(p.numel() for p in trainable)/1e6:.2f}M params")
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)

    image_grid_thw = torch.tensor([[1, 32, 64]], dtype=torch.long).cuda()
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        idx = sample_indices[step % n_samples]
        sample = ds[idx]
        # 简化: 只用第一个 QA
        qa_pair = sample["conversations"][:2]
        q, gt = qa_pair[0]["value"], qa_pair[1]["value"]
        # 加 image_pad 到 prompt
        from radarlm.vlm.eval_v9_v2_runner import build_single_qa_prompt
        prompt = build_single_qa_prompt(sample["conversations"], 0, 320)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = enc["input_ids"].cuda()
        attention_mask = enc["attention_mask"].cuda()
        visual_embeds = model.get_visual_embeds(
            sample["x_rd"].cuda(), sample["x_ra"].cuda(), sample["x_ad"].cuda())
        # labels: only answer part
        full_text = prompt + gt + "<|im_end|>\n"
        full_enc = tok(full_text, return_tensors="pt", truncation=True, max_length=2048)
        full_ids = full_enc["input_ids"].cuda()
        labels = full_ids.clone()
        prompt_len = input_ids.size(1)
        labels[:, :prompt_len] = -100
        # 同步 attention_mask
        full_attn = full_enc["attention_mask"].cuda()
        # 拼 visual tokens
        out = model.qwen(
            input_ids=full_ids, attention_mask=full_attn,
            pixel_values=visual_embeds, image_grid_thw=image_grid_thw, labels=labels)
        loss = out.loss
        if not torch.isnan(loss):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(loss.item())
        if (step + 1) % 10 == 0:
            avg = sum(losses[-10:]) / min(10, len(losses))
            print(f"[step {step+1}] avg_loss={avg:.4f} (current={loss.item():.4f})")
    elapsed = time.time() - t0
    print(f"\n总: {n_steps} steps, {elapsed:.1f}s, {elapsed/n_steps:.1f}s/step")
    if losses:
        print(f"Loss: start={losses[0]:.4f}, end={losses[-1]:.4f}, drop={losses[0]-losses[-1]:.4f}")
        if losses[0] - losses[-1] > 0.1:
            print("✓ 模型能从视觉学习 (loss 下降)")
        else:
            print("✗ 模型不能从视觉学习 (loss 没显著下降)")
    # 验证: 同样 prompt 不同 visual 应该有不同 logits
    print("\n=== 视觉敏感性 (训练后) ===")
    for label, x_rd, x_ra, x_ad in [
        ("A. 真实图", sample["x_rd"].cuda(), sample["x_ra"].cuda(), sample["x_ad"].cuda()),
        ("B. 黑图", torch.zeros_like(sample["x_rd"].cuda()), torch.zeros_like(sample["x_ra"].cuda()), torch.zeros_like(sample["x_ad"].cuda())),
    ]:
        v_emb = model.get_visual_embeds(x_rd, x_ra, x_ad)
        with torch.no_grad():
            out_ids = model.qwen.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=v_emb, image_grid_thw=image_grid_thw,
                min_new_tokens=20, max_new_tokens=80, do_sample=False, repetition_penalty=1.1)
        gen = tok.decode(out_ids[0, input_ids.size(1):], skip_special_tokens=True).strip()
        print(f"  {label} (v_norm={v_emb.norm().item():.1f}): {gen[:100]}")


if __name__ == "__main__":
    main()