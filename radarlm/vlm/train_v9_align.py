"""v9: PKC + Qwen2-VL 视觉-文本对齐训练 (正确用法).

关键: Qwen2-VL.forward 通过 input_ids 中的 image_token_id (151655) 位置,
用 masked_scatter 把 visual_embeds 替换进去.

我们的方案:
  PKC frozen (6M) → x4_rd, x4_ra (256 ch, 64x64)
  → Projector (256→3584) → 256 patches/视图 = 512 total visual tokens
  → prompt 含 512 个 image_token_id → masked_scatter 替换为 PKC features
  → Qwen2-VL LLM (LoRA) 处理 → 训练 4 选 1 multi-choice
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

sys.path.insert(0, "/home/zzy/Myproject/RadarLM")
sys.path.insert(0, "/home/zzy/Myproject/PKC")

from radarlm.pkc_backbone.pkc_silu_wrapper import PKCWithPretrained
from transformers import Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


PKC_NORM_STATS = {
    "rd": (37.59535773996415, 119.08313902425246),
    "ra": (40.40928894952408, 103.80548746494114),
    "ad": (54.42604354196056, 105.79746676271202),
}


def load_carrada_npy(seq, frame, view, carrada_root):
    name_map = {"rd": "range_doppler_processed",
                "ra": "range_angle_processed",
                "ad": "angle_doppler_processed"}
    p = Path(carrada_root) / seq / name_map[view] / f"{frame}.npy"
    if not p.exists(): return None
    arr = np.load(p).astype(np.float32)
    min_v, max_v = PKC_NORM_STATS[view]
    return np.clip((arr - min_v) / (max_v - min_v), 0.0, 1.0)


def center_crop(t, h_out, w_out):
    if t is None: return None
    H, W = t.shape[:2]
    if H < h_out or W < w_out:
        t = np.pad(t, ((0, max(0, h_out - H)), (0, max(0, w_out - W))), mode='constant')
        H, W = t.shape[:2]
    return t[(H - h_out) // 2:(H - h_out) // 2 + h_out,
             (W - w_out) // 2:(W - w_out) // 2 + w_out]


class PKCDatasetMC(Dataset):
    def __init__(self, ann_path, carrada_root, split='train', T=5, max_samples=None, seed=42, use_hflip=True):
        import json
        self.ann = json.load(open(ann_path))
        self.carrada_root = Path(carrada_root)
        self.T = T
        self.rng = random.Random(seed)
        self.use_hflip = use_hflip
        all_inst = []
        for seq, insts in self.ann.items():
            for iid, frames in insts.items():
                if not frames: continue
                flist = sorted(frames.keys())
                if len(flist) < T: continue
                any_key = list(frames.keys())[0]
                label = frames[any_key]["range_doppler"]["label"]
                all_inst.append((seq, iid, flist, label))
        rng = random.Random(seed); rng.shuffle(all_inst)
        n = len(all_inst); n_tr = int(n * 0.7); n_va = int(n * 0.85)
        if split == 'train': inst_list = all_inst[:n_tr]
        elif split == 'val': inst_list = all_inst[n_tr:n_va]
        else: inst_list = all_inst[n_va:]
        from collections import defaultdict
        by_class = defaultdict(list)
        for inst in inst_list: by_class[inst[3]].append(inst)
        max_per_class = 200
        rng2 = random.Random(seed + hash(split))
        selected = []
        for c in [1, 2, 3]:
            insts_c = by_class.get(c, [])
            rng2.shuffle(insts_c)
            for inst in insts_c[:max_per_class]:
                max_start = max(1, len(inst[2]) - self.T + 1)
                w = rng2.randint(0, max(1, max_start - 1))
                selected.append(inst + (w,))
        self.samples = selected
        if max_samples: self.samples = self.samples[:max_samples]
        print(f"[PKCDatasetMC {split}] {len(self.samples)} samples, by_class: {Counter(s[3] for s in self.samples)}")
        self.label_to_letter = {1: "A", 2: "B", 3: "C"}
        # prompt template (不含 A/B/C 字母, 避免 model 靠 prompt 复读)
        self.prompt_template = (
            "Question: This is a 5-frame radar sequence. "
            "Is the target a pedestrian, cyclist, or car?\nAnswer:"
        )

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        seq, iid, flist, label, win_idx = s
        max_start = max(1, len(flist) - self.T + 1)
        start = int(win_idx * max_start / 4) % max_start
        fids = flist[start:start + self.T]
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
            rd_frames.append(rd); ra_frames.append(ra); ad_frames.append(ad)
        if self.use_hflip and self.rng.random() > 0.5:
            rd_frames = [np.fliplr(f).copy() for f in rd_frames]
            ra_frames = [np.fliplr(f).copy() for f in ra_frames]
            ad_frames = [np.fliplr(f).copy() for f in ad_frames]
        return {
            "x_rd": torch.from_numpy(np.stack(rd_frames)).unsqueeze(0).unsqueeze(0).float(),
            "x_ra": torch.from_numpy(np.stack(ra_frames)).unsqueeze(0).unsqueeze(0).float(),
            "x_ad": torch.from_numpy(np.stack(ad_frames)).unsqueeze(0).unsqueeze(0).float(),
            "prompt": self.prompt_template,
            "answer": self.label_to_letter[label],
        }


class PKCProjector(nn.Module):
    """PKC x4_rd/x4_ra (256 ch) → Qwen hidden_size (3584) visual tokens."""
    def __init__(self, in_ch=256, out_ch=3584, num_patches_h=16, num_patches_w=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((num_patches_h, num_patches_w))
        self.proj = nn.Linear(in_ch, out_ch)
        self.norm = nn.LayerNorm(out_ch)
        # 用小初始化, 避免 visual_embeds 数值过大
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # x: (B, 256, 64, 64)
        x = self.pool(x)  # (B, 256, 16, 16)
        x = x.flatten(2).transpose(1, 2)  # (B, 256, 256)
        x = self.proj(x)  # (B, 256, 3584)
        x = self.norm(x)
        # clamp 限制数值范围, 避免 nan in bfloat16
        x = torch.clamp(x, min=-10.0, max=10.0)
        return x  # (B, num_patches, hidden_size)


class PKCQwenAlign(nn.Module):
    def __init__(self, qwen_path, pkc_weights="radarlm/pkc_backbone/weights/pkcin_silu_gn.pt"):
        super().__init__()
        # 1) PKC frozen
        self.pkc_wrapper = PKCWithPretrained(
            n_classes=4, n_frames=5, device='cuda', weights_path=pkc_weights,
        ).cuda()
        self.pkc = self.pkc_wrapper.pkc
        for p in self.pkc.parameters(): p.requires_grad = False
        self.pkc.eval()

        # 2) Qwen2-VL (bf16, 不量化, 更稳)
        self.qwen = Qwen2VLForConditionalGeneration.from_pretrained(
            qwen_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        )
        # 冻结 Qwen ViT
        for p in self.qwen.visual.parameters(): p.requires_grad = False
        # 不调用 prepare_model_for_kbit_training (那是为 4-bit 设计的)

        # Monkey-patch Qwen ViT: 直接返回输入 (我们的 PKC visual_embeds)
        # 这样 Qwen2VL.forward 走原逻辑: masked_scatter 到 image_token_id 位置
        self.num_image_tokens = 512  # 256 rd + 256 ra
        class PKCVisualStub(nn.Module):
            """Stub ViT: 直接返回 input (reshape 成 Qwen 期望的 2D (num_tokens, hidden))."""
            def __init__(self):
                super().__init__()
                self.dtype = torch.bfloat16
            def get_dtype(self):
                return self.dtype
            def forward(self, pixel_values, grid_thw=None):
                # pixel_values 形状: (B, num_image_tokens, hidden_size)
                # Qwen2-VL 期望 (num_image_features_total, hidden_size)
                # 这里 num_features = batch * num_tokens (因为 forward 后续要 scatter 到 image_mask)
                # 实际看 Qwen 代码: n_image_features = image_embeds.shape[0]
                # 然后 flatten + mask 匹配 num_image_tokens
                # 我们让 pixel_values 直接是 (num_image_tokens * B, hidden_size)
                if pixel_values.dim() == 3:
                    B, N, H = pixel_values.shape
                    pixel_values = pixel_values.view(B * N, H)
                return pixel_values.to(dtype=torch.bfloat16)
        self.qwen.visual = PKCVisualStub()

        self.hidden_size = self.qwen.config.hidden_size
        self.image_token_id = self.qwen.config.image_token_id  # 151655

        # 3) Projector: 256 → hidden_size
        self.proj_rd = PKCProjector(in_ch=256, out_ch=self.hidden_size).cuda()
        self.proj_ra = PKCProjector(in_ch=256, out_ch=self.hidden_size).cuda()

        # 4) LoRA on language_model
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            exclude_modules=r"visual.*",
        )
        self.qwen = get_peft_model(self.qwen, lora_cfg)
        print(f"[PKCQwenAlign] hidden_size={self.hidden_size}, image_token_id={self.image_token_id}")
        print(f"[PKCQwenAlign] Qwen LoRA applied to language_model")

    def get_visual_embeds(self, x_rd, x_ra, x_ad):
        """PKC → x4_rd, x4_ra → projector → visual_embeds (B, 512, 3584)."""
        with torch.no_grad():
            x4_rd, x4_ra = self.pkc(x_rd, x_ra, x_ad, features_only=True, latent_type='x4')
        rd_t = self.proj_rd(x4_rd)  # (B, 256, 3584)
        ra_t = self.proj_ra(x4_ra)
        return torch.cat([rd_t, ra_t], dim=1)  # (B, 512, 3584)


def collate_fn(batch, image_token_id, num_image_tokens=512):
    """构造 input_ids 含 image_token_id 占位 + 拼上 prompt/answer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")

    x_rd = torch.cat([b["x_rd"] for b in batch], dim=0)
    x_ra = torch.cat([b["x_ra"] for b in batch], dim=0)
    x_ad = torch.cat([b["x_ad"] for b in batch], dim=0)

    # 构造 input_ids: <image_pad> * num_image_tokens + prompt + " " + answer + <|im_end|>
    # 用 chat template: <|im_start|>user\n<image_pad>...<image_pad>Question...<|im_end|>\n<|im_start|>assistant\n
    # 简化: prompt + answer
    image_pad_str = "<|image_pad|>" * num_image_tokens
    full_texts = []
    prompt_only_texts = []
    for b in batch:
        # prompt: image_pads + question + "\n" (不含 "Answer:" 让模型自己生成)
        prompt_only_texts.append(f"{image_pad_str}{b['prompt']}")
        # full: prompt + " " + answer
        full_texts.append(f"{image_pad_str}{b['prompt']} {b['answer']}")
    enc_full = tok(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    enc_prompt_only = tok(prompt_only_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = enc_full["input_ids"]
    attention_mask = enc_full["attention_mask"]
    labels = input_ids.clone()
    for i in range(len(batch)):
        prompt_len = enc_prompt_only["attention_mask"][i].sum().item()
        labels[i, :prompt_len] = -100
    # image_grid_thw: 对应 num_image_tokens 个 image_pad
    # Qwen2-VL 默认 spatial_merge_size=2, num_tokens = grid_h * grid_w / merge^2
    # 我们直接给 num_image_tokens = grid_h * grid_w / 4, 所以 grid_h*grid_w = 4 * num_image_tokens
    # num_image_tokens = 512 → grid_h*grid_w = 2048, 比如 32x64 (近似 16x16 patches × 2 view × 4 merge)
    grid_h = 32; grid_w = 32  # 32*32 = 1024, but we want 512 image pads
    # 实际上 image_grid_thw = (T, H, W), num_patches = T * H * W
    # 我们 num_image_tokens=512 = T*H*W = 1*32*16? 不行
    # Qwen2-VL 内部: spatial_merge_size=2, num_visual_tokens = T * H * W / 4
    # 所以 num_visual_tokens=512 → T*H*W = 2048. 用 T=1, H=64, W=32? 不 Qwen 期望 square-ish
    # 直接给 (1, 1, num_image_tokens) -> T=1, H=1, W=num_image_tokens 实际太宽
    # Qwen2-VL 用 grid_thw 确定 num_visual_tokens: num_visual_tokens = T*H*W
    # 但 model 还做 merge (spatial_merge_size=2), 所以 actual num = T*H*W/spatial_merge_size^2
    # 等等 spatial_merge_size = 2, 在 merger 里做 2x2 merge, 所以 actual num = T*H*W / 4
    # 我们要 actual num=512, T*H*W=2048. 用 T=1, H=32, W=64 -> 2048 -> 512 after merge
    image_grid_thw = torch.tensor([[1, 32, 64]] * len(batch), dtype=torch.long)
    return {"x_rd": x_rd, "x_ra": x_ra, "x_ad": x_ad,
            "input_ids": input_ids, "attention_mask": attention_mask,
            "labels": labels, "image_grid_thw": image_grid_thw}


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_path", default="/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")
    parser.add_argument("--ann_path", default="/data/storage/zzy/Carrada/annotations_instance_oriented.json")
    parser.add_argument("--carrada_root", default="/data/storage/zzy/Carrada")
    parser.add_argument("--output_dir", default="/home/zzy/Myproject/RadarLM/output/v9_qwen_align")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)  # 降低避免 nan
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_image_tokens", type=int, default=512)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("[Setup] loading data...")
    train_ds = PKCDatasetMC(args.ann_path, args.carrada_root, split='train',
                            max_samples=args.max_samples, seed=args.seed)
    val_ds = PKCDatasetMC(args.ann_path, args.carrada_root, split='val', seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=range(len(train_ds)),
                              collate_fn=lambda b: collate_fn(b, model.qwen.config.image_token_id, args.num_image_tokens),
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            sampler=range(len(val_ds)),
                            collate_fn=lambda b: collate_fn(b, model.qwen.config.image_token_id, args.num_image_tokens),
                            num_workers=0)

    print("[Setup] loading model...")
    model = PKCQwenAlign(args.qwen_path).cuda()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    print(f"[Train] {len(train_ds)} train, {len(val_ds)} val, {args.num_epochs} epochs, "
          f"trainable params: {sum(p.numel() for p in trainable_params)/1e6:.2f}M")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")
    letter_ids = [tok(" A", add_special_tokens=False)["input_ids"][0],
                  tok(" B", add_special_tokens=False)["input_ids"][0],
                  tok(" C", add_special_tokens=False)["input_ids"][0]]

    def evaluate():
        model.eval()
        correct, total = 0, 0
        per_class = Counter(); per_class_total = Counter()
        with torch.no_grad():
            for batch in val_loader:
                x_rd = batch['x_rd'].cuda(); x_ra = batch['x_ra'].cuda(); x_ad = batch['x_ad'].cuda()
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                image_grid_thw = batch['image_grid_thw'].cuda()
                # 拿 PKC visual embeds
                visual_embeds = model.get_visual_embeds(x_rd, x_ra, x_ad)
                # Qwen2-VL forward: 传 input_ids + visual_embeds (代替 ViT)
                # 用 inputs_embeds=None 让 Qwen 自己 scatter
                out = model.qwen(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=visual_embeds,  # 用 visual_embeds 替换 pixel_values
                    image_grid_thw=image_grid_thw,
                )
                last_logits = out.logits[:, -1, :]
                abcd_logits = last_logits[:, letter_ids]
                pred_idx = abcd_logits.argmax(dim=1).item()
                # 真实 answer
                valid = batch['labels'][0][batch['labels'][0] != -100]
                target_id = valid[0].item() if len(valid) > 0 else -1
                target_idx = letter_ids.index(target_id) if target_id in letter_ids else -1
                cls = ["ped", "cyc", "car"][target_idx] if target_idx >= 0 else "unk"
                per_class_total[cls] += 1
                if pred_idx == target_idx:
                    correct += 1
                    per_class[cls] += 1
                total += 1
        per_class_acc = {c: per_class[c] / max(1, per_class_total[c]) for c in per_class_total}
        macro = sum(per_class_acc.values()) / max(1, len(per_class_acc))
        return correct / max(1, total), per_class_acc, per_class_total

    best_macro = 0
    for epoch in range(args.num_epochs):
        model.train()
        loss_sum, n = 0, 0
        t0 = time.time()
        for batch in train_loader:
            x_rd = batch['x_rd'].cuda(); x_ra = batch['x_ra'].cuda(); x_ad = batch['x_ad'].cuda()
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            image_grid_thw = batch['image_grid_thw'].cuda()
            visual_embeds = model.get_visual_embeds(x_rd, x_ra, x_ad)
            out = model.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=visual_embeds,
                image_grid_thw=image_grid_thw,
                labels=labels,
            )
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            # grad clip
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            loss_sum += loss.item() if not torch.isnan(loss).item() else 0.0
            n += 1
        t1 = time.time()
        print(f"[Epoch {epoch+1}] avg_loss={loss_sum/n:.3f} ({t1-t0:.1f}s)", flush=True)
        torch.save({"proj_rd": model.proj_rd.state_dict(),
                    "proj_ra": model.proj_ra.state_dict()}, f"{args.output_dir}/projector_e{epoch+1}.pt")
        if (epoch + 1) % 2 == 0:
            acc, per_cls, total_cls = evaluate()
            macro = sum(per_cls.values()) / max(1, len(per_cls))
            print(f"  [val] acc={acc:.3f} macro={macro:.3f} per_class={per_cls} n={dict(total_cls)}", flush=True)
            if macro > best_macro:
                best_macro = macro
                torch.save({"proj_rd": model.proj_rd.state_dict(),
                            "proj_ra": model.proj_ra.state_dict()}, f"{args.output_dir}/projector_best.pt")

    print(f"[Done] best_macro={best_macro:.3f}")


if __name__ == "__main__":
    train()