"""PKC + Qwen2-VL 集成 (v9 训练脚本).

包装:
  PKC backbone (frozen) → rd_latent, ra_latent (B, 128, H, W)
  → RadarProjector (1x1 conv + avgpool) → (B, N, 1280) visual tokens
  → Qwen2-VL LLM (LoRA)
  → 4 选 1 multi-choice 预测 (A/B/C/D)
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path("/home/zzy/Myproject/RadarLM/radarlm/pkc_backbone").parent))
sys.path.insert(0, "/home/zzy/Myproject/PKC")

from mvrss.models.pkcin_plus_cvf_aug import PKCIn_plus_cvf_aug
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from radarlm.vlm.pkc_visual_tower import RadarProjector


CLASS_LETTER = ["A", "B", "C", "D"]
CLASS_NAME_CN = {0: "car", 1: "pedestrian", 2: "cyclist", 3: "empty"}


def load_carrada_npy(seq, frame, view, carrada_root):
    name_map = {"rd": "range_doppler_processed",
                "ra": "range_angle_processed",
                "ad": "angle_doppler_processed"}
    p = Path(carrada_root) / seq / name_map[view] / f"{frame}.npy"
    if not p.exists():
        return None
    return np.log1p(np.maximum(np.load(p).astype(np.float32), 0))


class PKCDataset(Dataset):
    """加载 CARRADA 5 帧窗口 + 4 选 1 multi-choice 标签."""
    def __init__(self, ann_path, carrada_root, split='train', T=5, max_samples=None, seed=42):
        self.ann = json.load(open(ann_path))
        self.carrada_root = Path(carrada_root)
        self.T = T
        self.rng = random.Random(seed)
        fo_path = Path(ann_path).parent / "annotations_frame_oriented.json"
        fo = json.load(open(fo_path)) if fo_path.exists() else {}

        # 收集所有 instance
        all_instances = []
        for seq, insts in self.ann.items():
            for iid, frames in insts.items():
                if not frames: continue
                flist = sorted(frames.keys())
                if len(flist) < T: continue
                all_instances.append((seq, iid, flist))

        rng = random.Random(seed)
        rng.shuffle(all_instances)
        n = len(all_instances)
        # 切 split
        n_tr = int(n * 0.7)
        n_va = int(n * 0.85)
        if split == 'train':
            self.instances = all_instances[:n_tr]
        elif split == 'val':
            self.instances = all_instances[n_tr:n_va]
        else:
            self.instances = all_instances[n_va:]

        # 每个 instance 选 1 个 T 窗口, 关联 ground truth
        self.samples = []
        for seq, iid, flist in self.instances:
            start = rng.randint(0, len(flist) - self.T + 1)
            fids = flist[start:start + self.T]
            # 取任意一帧的 label (frames 是 dict, key 格式不固定)
            any_key = list(frames.keys())[0]
            info = frames[any_key]["range_doppler"]
            label = info["label"]  # 0=empty, 1=ped, 2=cyc, 3=car
            if label == 0:
                # 跳过 empty frames (没目标)
                continue
            self.samples.append({
                "seq": seq, "iid": iid, "fids": fids,
                "label": label,  # 1=ped, 2=cyc, 3=car
            })

        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"[PKCDataset {split}] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        seq = s["seq"]
        fids = s["fids"]
        # 加载 T 帧 3 view
        rd_frames = []
        ra_frames = []
        ad_frames = []
        for f in fids:
            rd = load_carrada_npy(seq, f, "rd", self.carrada_root)
            ra = load_carrada_npy(seq, f, "ra", self.carrada_root)
            ad = load_carrada_npy(seq, f, "ad", self.carrada_root)
            if rd is None or ra is None or ad is None:
                # 跳过
                return self.__getitem__((idx + 1) % len(self))
            rd_frames.append(rd)
            ra_frames.append(ra)
            ad_frames.append(ad)
        # 拼成 (1, T, 256, W) 格式 (B, C, T, H, W) 等价 (1, 1, T, H, W)
        x_rd = torch.from_numpy(np.stack(rd_frames, axis=0)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, 256, 64)
        x_ra = torch.from_numpy(np.stack(ra_frames, axis=0)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, 256, 256)
        x_ad = torch.from_numpy(np.stack(ad_frames, axis=0)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, 256, 64)
        return {
            "x_rd": x_rd, "x_ra": x_ra, "x_ad": x_ad,
            "label": torch.tensor(s["label"] - 1, dtype=torch.long),  # 0/1/2
        }


class PKC_Qwen(nn.Module):
    """PKC visual tower + Qwen LLM (frozen visual, train projector + LoRA LLM)."""
    def __init__(self, qwen_path, pkc_n_frames=5, hidden_size=1280, out_h=8, out_w=8, quantize=True):
        super().__init__()
        # 1) PKC visual tower (frozen)
        from radarlm.vlm.pkc_visual_tower import PKCVisualTower
        self.pkc_vt = PKCVisualTower(pkc_n_frames, hidden_size, out_h, out_w).cuda()
        self.pkc_vt.eval()
        for p in self.pkc_vt.parameters():
            p.requires_grad = False
        # 2) Qwen LLM (frozen except LoRA)
        if quantize:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.qwen = Qwen2VLForConditionalGeneration.from_pretrained(
                qwen_path, quantization_config=bnb,
                torch_dtype=torch.bfloat16, device_map="cuda:0",
            )
        else:
            self.qwen = Qwen2VLForConditionalGeneration.from_pretrained(
                qwen_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
            )
        # 替换 visual 为 PKC 输出
        self.qwen.visual = self.pkc_vt.pkc.visual  # 占位
        # 3) 训 LoRA on LLM
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_target_only_in_module="language_model",
        )
        self.qwen = get_peft_model(self.qwen, lora_config)

    def forward(self, x_rd, x_ra, x_ad, input_ids, attention_mask):
        # 1) PKC visual → tokens (frozen, 128 tokens)
        with torch.no_grad():
            visual_tokens = self.pkc_vt(x_rd, x_ra, x_ad)  # (B, 128, 1280)
        # 2) 拼到 input_embeds 中
        embed_layer = self.qwen.base_model.model.model.embed_tokens
        text_embeds = embed_layer(input_ids)  # (B, L, 1280)
        # 找 image_token 位置 (input_ids 中 image_token_id)
        image_token_id = self.qwen.config.image_token_id
        # 简化: 把 visual tokens 拼到末尾
        combined = torch.cat([text_embeds, visual_tokens], dim=1)  # (B, L+128, 1280)
        # 调整 attention_mask
        new_attn = torch.cat([
            attention_mask,
            torch.ones((attention_mask.size(0), 128), device=attention_mask.device, dtype=attention_mask.dtype)
        ], dim=1)
        # 3) 通过 LLM
        out = self.qwen.base_model.model(
            inputs_embeds=combined, attention_mask=new_attn,
        )
        # logits
        logits = self.qwen.lm_head(out.last_hidden_state)
        # 取前 L 个 token (对应文本)
        text_logits = logits[:, :input_ids.size(1)]
        return text_logits


def collate_fn(batch):
    """拼 batch 中的多帧 tensor + tokenize 问题."""
    # 简单: 每条样本只取一个 RD/RA/AD (T=5)
    x_rd = torch.cat([b["x_rd"] for b in batch], dim=0)
    x_ra = torch.cat([b["x_ra"] for b in batch], dim=0)
    x_ad = torch.cat([b["x_ad"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch])
    # 简化: 不做 text input (用纯视觉输入)
    # input_ids, attention_mask 占位
    B = len(batch)
    input_ids = torch.zeros((B, 1), dtype=torch.long)  # 占位
    attention_mask = torch.ones((B, 1), dtype=torch.long)
    return {
        "x_rd": x_rd, "x_ra": x_ra, "x_ad": x_ad,
        "input_ids": input_ids, "attention_mask": attention_mask,
        "labels": labels,
    }


def train_v9():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_path", default="/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")
    parser.add_argument("--ann_path", default="/data/storage/zzy/Carrada/annotations_instance_oriented.json")
    parser.add_argument("--carrada_root", default="/data/storage/zzy/Carrada")
    parser.add_argument("--output_dir", default="/home/zzy/Myproject/RadarLM/output/v9_pkc_qwen")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)

    print("[Setup] loading data...")
    train_ds = PKCDataset(args.ann_path, args.carrada_root, split="train", max_samples=args.max_samples, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0)

    print("[Setup] loading model...")
    model = PKC_Qwen(args.qwen_path).cuda()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    print(f"[Train] {len(train_ds)} samples, {args.num_epochs} epochs")
    for epoch in range(args.num_epochs):
        model.train()
        loss_sum, n = 0, 0
        t0 = time.time()
        for batch in train_loader:
            x_rd = batch["x_rd"].cuda()
            x_ra = batch["x_ra"].cuda()
            x_ad = batch["x_ad"].cuda()
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            logits = model(x_rd, x_ra, x_ad, input_ids, attention_mask)
            # logits shape: (B, 1, vocab_size), labels = (B,)
            loss = F.cross_entropy(logits.squeeze(1), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            n += len(labels)
        t1 = time.time()
        print(f"[Epoch {epoch+1}] loss={loss_sum/n:.3f}  ({t1-t0:.1f}s)")
    print("[Done] loss should decrease if v9 works")


if __name__ == "__main__":
    train_v9()
