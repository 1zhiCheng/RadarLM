"""v9 PKC 特征 + 简单分类头 (不接 Qwen).

流程:
  PKC backbone (frozen) → 128 维特征
  → Projector (trainable) → 3 类 head
  → 训练: 看 v9 数据 + 模型比 v8 CNN baseline 是否更好
"""
import argparse
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
from torch.utils.data import DataLoader, SubsetRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, "/home/zzy/Myproject/PKC")

from radarlm.vlm.pkc_visual_tower import RadarProjector
# 用真实预训练权重的 PKC (SiLU+GroupNorm 版本, CARRADA SOTA)
from radarlm.pkc_backbone.pkc_silu_wrapper import PKCWithPretrained
from radarlm.data.temporal_dataset import crop_roi, get_split_instance_map


CLASS_NAMES = {0: "background", 1: "pedestrian", 2: "cyclist", 3: "car"}


# PKC 训练时用的 min-max normalize (from /home/zzy/Myproject/PKC/mvrss/config_files/*_stats_all.json)
PKC_NORM_STATS = {
    "rd": (37.59535773996415, 119.08313902425246),  # range_doppler
    "ra": (40.40928894952408, 103.80548746494114),  # range_angle
    "ad": (54.42604354196056, 105.79746676271202),  # angle_doppler
}


def load_carrada_npy(seq, frame, view, carrada_root):
    """按 PKC 训练时的 dB scale 直接加载 + min-max normalize 到 [0, 1]."""
    name_map = {"rd": "range_doppler_processed",
                "ra": "range_angle_processed",
                "ad": "angle_doppler_processed"}
    p = Path(carrada_root) / seq / name_map[view] / f"{frame}.npy"
    if not p.exists():
        return None
    arr = np.load(p).astype(np.float32)
    # PKC normalize: (data - min) / (max - min), clamp [0, 1]
    min_v, max_v = PKC_NORM_STATS[view]
    arr = np.clip((arr - min_v) / (max_v - min_v), 0.0, 1.0)
    return arr


def center_crop(t, h_out, w_out):
    """PKC 期望 RD 256x64, RA 256x256, AD 256x64. 不能太小, 否则 depthwise kernel 报错."""
    if t is None: return None
    H, W = t.shape[:2]
    if H < h_out or W < w_out:
        # PKC 期望尺寸不足, padding (不裁剪, 保 PKC 形状)
        pad_h = max(0, h_out - H)
        pad_w = max(0, w_out - W)
        t = np.pad(t, ((0, pad_h), (0, pad_w)), mode='constant')
        H, W = t.shape[:2]
    h_start = (H - h_out) // 2
    w_start = (W - w_out) // 2
    return t[h_start:h_start + h_out, w_start:w_start + w_out]


class PKCDataset(torch.utils.data.Dataset):
    """加载 PKC 输入: 5 帧 × 3 view (RD/RA/AD)."""
    def __init__(self, ann_path, carrada_root, split='train', T=5, max_samples=None, seed=42, use_hflip=True):
        import json
        self.ann = json.load(open(ann_path))
        self.carrada_root = Path(carrada_root)
        self.T = T
        self.rng = random.Random(seed)
        self.use_hflip = use_hflip
        fo_path = Path(ann_path).parent / "annotations_frame_oriented.json"
        fo = json.load(open(fo_path)) if fo_path.exists() else {}

        # Class-balanced split (按 instance 不重叠)
        all_inst = []
        for seq, insts in self.ann.items():
            for iid, frames in insts.items():
                if not frames: continue
                flist = sorted(frames.keys())
                if len(flist) < T: continue
                any_key = list(frames.keys())[0]
                label = frames[any_key]["range_doppler"]["label"]
                all_inst.append((seq, iid, flist, label))
        rng = random.Random(seed)
        rng.shuffle(all_inst)
        n = len(all_inst)
        n_tr = int(n * 0.7)
        n_va = int(n * 0.85)
        if split == 'train':
            inst_list = all_inst[:n_tr]
        elif split == 'val':
            inst_list = all_inst[n_tr:n_va]
        else:
            inst_list = all_inst[n_va:]

        # 在各 split 内 class-balanced
        from collections import defaultdict
        by_class = defaultdict(list)
        for inst in inst_list:
            by_class[inst[3]].append(inst)
        max_per_class = 200  # 扩到 200/类
        rng2 = random.Random(seed + hash(split))
        # 1 窗口/inst, 但每个 instance 选不同的 start 位置
        windows_per_inst = 1
        selected = []
        for c in [1, 2, 3]:  # ped, cyc, car
            insts_c = by_class.get(c, [])
            rng2.shuffle(insts_c)
            for inst in insts_c[:max_per_class]:
                # 随机选 start 位置
                max_start = max(1, len(inst[2]) - 5 + 1)
                w = rng2.randint(0, max(1, max_start - 1))
                selected.append(inst + (w,))
        self.samples = selected
        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"[PKCDataset {split}] {len(self.samples)} samples, by_class: {Counter(s[3] for s in self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        seq, iid, flist, label, win_idx = s
        max_start = max(1, len(flist) - self.T + 1)
        start = int(win_idx * max_start / 4) % max_start
        fids = flist[start:start + self.T]
        # 加载 5 帧 × 3 view
        rd_frames, ra_frames, ad_frames = [], [], []
        for f in fids:
            rd = center_crop(load_carrada_npy(seq, f, "rd", self.carrada_root), 256, 64)
            ra = center_crop(load_carrada_npy(seq, f, "ra", self.carrada_root), 256, 256)
            ad = center_crop(load_carrada_npy(seq, f, "ad", self.carrada_root), 256, 64)
            if rd is None or ra is None or ad is None:
                rd_frames = [np.zeros((256, 64), dtype=np.float32) for _ in range(self.T)]
                ra_frames = [np.zeros((256, 256), dtype=np.float32) for _ in range(self.T)]
                ad_frames = [np.zeros((256, 64), dtype=np.float32) for _ in range(self.T)]
                break
            rd_frames.append(rd)
            ra_frames.append(ra)
            ad_frames.append(ad)
        # hflip 增强 (PKC 训时也用, 概率 0.5)
        if self.use_hflip and self.rng.random() > 0.5:
            rd_frames = [np.fliplr(f).copy() for f in rd_frames]
            ra_frames = [np.fliplr(f).copy() for f in ra_frames]
            ad_frames = [np.fliplr(f).copy() for f in ad_frames]
        return {
            "x_rd": torch.from_numpy(np.stack(rd_frames)).unsqueeze(0).unsqueeze(0).float(),
            "x_ra": torch.from_numpy(np.stack(ra_frames)).unsqueeze(0).unsqueeze(0).float(),
            "x_ad": torch.from_numpy(np.stack(ad_frames)).unsqueeze(0).unsqueeze(0).float(),
            "label": torch.tensor(label - 1, dtype=torch.long),
        }


class V9Classifier(nn.Module):
    """PKC backbone (真实预训练权重 frozen) → projector (trainable) → 3 类分类."""
    def __init__(self, pkc_n_frames=5, T=5, hidden_size=128, out_h=4, out_w=4,
                 pkc_weights=None):
        super().__init__()
        # 用真实 PKC 预训练权重 (SiLU+GroupNorm, CARRADA SOTA)
        self.pkc_wrapper = PKCWithPretrained(
            n_classes=4, n_frames=pkc_n_frames, device='cuda',
            weights_path=pkc_weights or "radarlm/pkc_backbone/weights/pkcin_silu_gn.pt",
        ).cuda()
        self.pkc = self.pkc_wrapper.pkc  # 兼容 old API
        for p in self.pkc.parameters():
            p.requires_grad = False
        self.pkc.eval()
        # 双视图 projector
        self.proj_rd = RadarProjector(in_ch=128, out_ch=hidden_size, out_h=out_h, out_w=out_w)
        self.proj_ra = RadarProjector(in_ch=128, out_ch=hidden_size, out_h=out_h, out_w=out_w)
        # 3 类 head (容量大一些, 用 LayerNorm 兼容 batch=1)
        feat_dim = hidden_size * out_h * out_w * 2  # rd + ra
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 3),
        )

    def forward(self, x_rd, x_ra, x_ad):
        with torch.no_grad():
            out = self.pkc(x_rd, x_ra, x_ad, features_only=True)
            if isinstance(out, tuple) and len(out) >= 2:
                rd_latent, ra_latent = out[0], out[1]
            else:
                raise ValueError("PKC output unexpected")
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


def train_v9():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_path", default="/data/storage/zzy/Carrada/annotations_instance_oriented.json")
    parser.add_argument("--carrada_root", default="/data/storage/zzy/Carrada")
    parser.add_argument("--output_dir", default="/home/zzy/Myproject/RadarLM/output/v9")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)

    print("[Setup] loading data...")
    train_ds = PKCDataset(args.ann_path, args.carrada_root, split='train',
                          max_samples=args.max_samples, seed=args.seed)
    val_ds = PKCDataset(args.ann_path, args.carrada_root, split='val', seed=args.seed)
    test_ds = PKCDataset(args.ann_path, args.carrada_root, split='test', seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=range(len(train_ds)), collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=range(len(val_ds)), collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, sampler=range(len(test_ds)), collate_fn=collate_fn, num_workers=0)

    print("[Setup] loading model...")
    model = V9Classifier(pkc_n_frames=5, T=5, hidden_size=128, out_h=4, out_w=4).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pkc_total = sum(p.numel() for p in model.pkc.parameters())
    print(f"[Model] trainable (projector + head): {trainable:,} ({trainable/1e6:.2f}M)")
    print(f"[Model] PKC frozen: {pkc_total:,} ({pkc_total/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-3
    )
    # class weight (counter collapse)
    class_weight = torch.tensor([1.0, 1.0, 0.5], device=device)  # car 权小, 抑制预测 car

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
    for epoch in range(args.num_epochs):
        model.train()
        t0 = time.time()
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
        t1 = time.time()
        macro, per_class, total = evaluate(val_loader)
        if macro > best_macro:
            best_macro = macro
            torch.save(model.state_dict(), f"{args.output_dir}/best.pth")
        if (epoch + 1) % 5 == 0:
            print(f"[Epoch {epoch+1}] loss={loss_sum/n:.3f} | val_macro={macro:.3f} | "
                  f"ped={per_class.get(0,0):.2f} cyc={per_class.get(1,0):.2f} car={per_class.get(2,0):.2f} | {t1-t0:.1f}s | best={best_macro:.3f}")

    # Test
    model.load_state_dict(torch.load(f"{args.output_dir}/best.pth"))
    macro, per_class, total = evaluate(test_loader)
    print(f"\n[Test] macro_acc={macro:.3f} | per_class={per_class}")
    print(f"per_class_n: {dict(total)}")

    # save
    with open(f"{args.output_dir}/test_results.json", "w") as f:
        import json
        json.dump({
            "macro_acc": float(macro),
            "per_class_acc": {str(k): float(v) for k, v in per_class.items()},
            "per_class_n": dict(total),
            "n_params": trainable + pkc_total,
            "n_trainable": trainable,
            "config": vars(args),
        }, f, indent=2)


if __name__ == "__main__":
    train_v9()
