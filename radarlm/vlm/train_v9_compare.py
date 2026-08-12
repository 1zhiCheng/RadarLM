"""v9 PKC 三种 latent 对比训练.

对比:
- 'latent': rd_latent, ra_latent (128 通道, LatentSpaceFusionBranch 输出)
- 'x3':     x3_rd, x3_ra (128 通道, ASPP 后)
- 'x4':     x4_rd, x4_ra (256 通道, x3 + rd_latent)

跑 80 epochs, 输出 val_macro 对比.
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
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/zzy/Myproject/RadarLM")
sys.path.insert(0, "/home/zzy/Myproject/PKC")

from radarlm.vlm.pkc_visual_tower import RadarProjector
from radarlm.pkc_backbone.pkc_silu_wrapper import PKCWithPretrained
from radarlm.vlm.train_v9_projector import PKCDataset


class V9Classifier(nn.Module):
    """PKC + projector + head, 支持 3 种 latent mode."""
    def __init__(self, latent_type='latent', pkc_n_frames=5, T=5, hidden_size=128, out_h=4, out_w=4):
        super().__init__()
        self.latent_type = latent_type
        self.pkc_wrapper = PKCWithPretrained(
            n_classes=4, n_frames=pkc_n_frames, device='cuda',
            weights_path="radarlm/pkc_backbone/weights/pkcin_silu_gn.pt",
        ).cuda()
        self.pkc = self.pkc_wrapper.pkc
        for p in self.pkc.parameters():
            p.requires_grad = False
        self.pkc.eval()
        # latent 通道数: latent/x3 = 128, x4 = 256
        if latent_type == 'x4':
            in_ch = 256
        else:
            in_ch = 128
        self.proj_rd = RadarProjector(in_ch=in_ch, out_ch=hidden_size, out_h=out_h, out_w=out_w)
        self.proj_ra = RadarProjector(in_ch=in_ch, out_ch=hidden_size, out_h=out_h, out_w=out_w)
        feat_dim = hidden_size * out_h * out_w * 2
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 3),
        )

    def forward(self, x_rd, x_ra, x_ad):
        with torch.no_grad():
            out = self.pkc(x_rd, x_ra, x_ad, features_only=True, latent_type=self.latent_type)
            rd_latent, ra_latent = out[0], out[1]
        rd_t = self.proj_rd(rd_latent)
        ra_t = self.proj_ra(ra_latent)
        feat = torch.cat([rd_t, ra_t], dim=1).flatten(1)
        return self.head(feat)


def collate_fn(batch):
    x_rd = torch.cat([b["x_rd"] for b in batch], dim=0)
    x_ra = torch.cat([b["x_ra"] for b in batch], dim=0)
    x_ad = torch.cat([b["x_ad"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch])
    return {"x_rd": x_rd, "x_ra": x_ra, "x_ad": x_ad, "labels": labels}


def train_one_mode(latent_type, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda"

    train_ds = PKCDataset(args.ann_path, args.carrada_root, split='train',
                          max_samples=args.max_samples, seed=args.seed)
    val_ds = PKCDataset(args.ann_path, args.carrada_root, split='val', seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=range(len(train_ds)), collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            sampler=range(len(val_ds)), collate_fn=collate_fn, num_workers=0)

    model = V9Classifier(latent_type=latent_type).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-3
    )
    class_weight = torch.tensor([1.0, 1.0, 0.5], device=device)

    def evaluate(loader):
        model.eval()
        correct = Counter()
        total = Counter()
        with torch.no_grad():
            for batch in loader:
                x_rd = batch["x_rd"].to(device)
                x_ra = batch["x_ra"].to(device)
                x_ad = batch["x_ad"].to(device)
                y = batch["labels"].to(device)
                logits = model(x_rd, x_ra, x_ad)
                pred = logits.argmax(dim=1)
                for p, l in zip(pred.cpu().tolist(), y.cpu().tolist()):
                    total[l] += 1
                    if p == l: correct[l] += 1
        per_class = {c: correct[c] / max(1, total[c]) for c in total}
        macro = sum(per_class.values()) / max(1, len(per_class))
        return macro, per_class, total

    best_macro = 0
    history = []
    for epoch in range(args.num_epochs):
        model.train()
        loss_sum, n = 0, 0
        for batch in train_loader:
            x_rd = batch["x_rd"].to(device)
            x_ra = batch["x_ra"].to(device)
            x_ad = batch["x_ad"].to(device)
            y = batch["labels"].to(device)
            logits = model(x_rd, x_ra, x_ad)
            loss = F.cross_entropy(logits, y, weight=class_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            n += len(y)
        if (epoch + 1) % 10 == 0 or epoch == args.num_epochs - 1:
            macro, per_class, total = evaluate(val_loader)
            history.append({"epoch": epoch + 1, "loss": loss_sum / n,
                            "val_macro": macro, "per_class": per_class})
            print(f"[{latent_type} epoch {epoch+1}] loss={loss_sum/n:.3f} | val_macro={macro:.3f} | "
                  f"ped={per_class.get(0,0):.2f} cyc={per_class.get(1,0):.2f} car={per_class.get(2,0):.2f}",
                  flush=True)
            if macro > best_macro:
                best_macro = macro
    return best_macro, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_path", default="/data/storage/zzy/Carrada/annotations_instance_oriented.json")
    parser.add_argument("--carrada_root", default="/data/storage/zzy/Carrada")
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--modes", default="latent,x3,x4")
    args = parser.parse_args()

    results = {}
    for mode in args.modes.split(','):
        mode = mode.strip()
        print(f"\n========== Training with latent_type='{mode}' ==========", flush=True)
        t0 = time.time()
        best_macro, history = train_one_mode(mode, args)
        t1 = time.time()
        results[mode] = {"best_val_macro": best_macro, "history": history, "time_s": t1 - t0}
        print(f"[{mode}] best_val_macro={best_macro:.3f} ({t1-t0:.1f}s)", flush=True)

    print("\n========== Summary ==========")
    for mode, r in results.items():
        print(f"  {mode:8s}: best_val_macro = {r['best_val_macro']:.3f}  ({r['time_s']:.1f}s)")
    out_path = f"/home/zzy/Myproject/RadarLM/output/v9_compare_{int(time.time())}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()