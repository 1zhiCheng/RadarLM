"""v9: PKC + Qwen2-VL 多模态感知 QA 训练 (用 train_qwen_mt.jsonl 数据).

数据: 一帧 (单 RD/RA/AD) + 多轮 QA 对话 (5 QA/frame)
训练目标: projector (PKC → Qwen) + Qwen LoRA 让模型能正确回答感知问题
         (类别、bbox、速度、链式推理等)
"""
import argparse
import json
import os
import random
import re
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
from radarlm.vlm.eval_v9 import eval_qa_pair, aggregate_metrics
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
from peft import LoraConfig, get_peft_model


PKC_NORM_STATS = {
    "rd": (37.59535773996415, 119.08313902425246),
    "ra": (40.40928894952408, 103.80548746494114),
    "ad": (54.42604354196056, 105.79746676271202),
}


def parse_image_id(image_id: str):
    """'2019-09-16-12-52-12_000163' -> ('2019-09-16-12-52-12', '000163')."""
    m = re.match(r"^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(\d+)$", image_id)
    if not m:
        return None, None
    return m.group(1), m.group(2).zfill(6)


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


class RadarQADataset(Dataset):
    """加载 train_qwen_mt.jsonl 风格数据: 单帧 + 多轮 QA."""
    def __init__(self, jsonl_path, carrada_root, split='train', max_samples=None, seed=42, use_hflip=True):
        import json
        self.carrada_root = Path(carrada_root)
        self.use_hflip = use_hflip
        self.rng = random.Random(seed)
        # load
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                self.samples.append(item)
        # split 用文件名后缀决定 (train_qwen_mt / val_qwen_mt / test_qwen_mt)
        # 不再在文件内部做 70/15/15 split (那是 bug), 直接用全量
        if max_samples and max_samples < len(self.samples):
            rng2 = random.Random(seed + 1)
            self.samples = rng2.sample(self.samples, max_samples)
        print(f"[RadarQADataset {split}] {len(self.samples)} samples (from {Path(jsonl_path).name})")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        seq, frame = parse_image_id(item["id"])
        rd = center_crop(load_carrada_npy(seq, frame, "rd", self.carrada_root), 256, 64)
        ra = center_crop(load_carrada_npy(seq, frame, "ra", self.carrada_root), 256, 256)
        ad = center_crop(load_carrada_npy(seq, frame, "ad", self.carrada_root), 256, 64)
        if rd is None or ra is None or ad is None:
            rd = np.zeros((256, 64), dtype=np.float32)
            ra = np.zeros((256, 256), dtype=np.float32)
            ad = np.zeros((256, 64), dtype=np.float32)
        if self.use_hflip and self.rng.random() > 0.5:
            rd = np.fliplr(rd).copy()
            ra = np.fliplr(ra).copy()
            ad = np.fliplr(ad).copy()
        # T=5: 同一帧重复 5 次
        rd5 = np.stack([rd] * 5)
        ra5 = np.stack([ra] * 5)
        ad5 = np.stack([ad] * 5)
        return {
            "x_rd": torch.from_numpy(rd5).unsqueeze(0).unsqueeze(0).float(),  # (1,1,5,256,64)
            "x_ra": torch.from_numpy(ra5).unsqueeze(0).unsqueeze(0).float(),  # (1,1,5,256,256)
            "x_ad": torch.from_numpy(ad5).unsqueeze(0).unsqueeze(0).float(),
            "conversations": item["conversations"],
        }


class PKCProjector(nn.Module):
    def __init__(self, in_ch=256, out_ch=3584, num_patches_h=16, num_patches_w=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((num_patches_h, num_patches_w))
        self.proj = nn.Linear(in_ch, out_ch)
        self.norm = nn.LayerNorm(out_ch)
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = self.norm(x)
        x = torch.clamp(x, min=-10.0, max=10.0)
        return x


class PKCQwenAlign(nn.Module):
    def __init__(self, qwen_path, pkc_weights="radarlm/pkc_backbone/weights/pkcin_silu_gn.pt"):
        super().__init__()
        self.pkc_wrapper = PKCWithPretrained(
            n_classes=4, n_frames=5, device='cuda', weights_path=pkc_weights,
        ).cuda()
        self.pkc = self.pkc_wrapper.pkc
        for p in self.pkc.parameters(): p.requires_grad = False
        self.pkc.eval()

        # 4-bit 量化以省显存 (不用 prepare_model_for_kbit_training)
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.qwen = Qwen2VLForConditionalGeneration.from_pretrained(
            qwen_path, quantization_config=bnb,
            torch_dtype=torch.bfloat16, device_map="cuda:0",
        )
        for p in self.qwen.visual.parameters(): p.requires_grad = False
        self.hidden_size = self.qwen.config.hidden_size
        self.image_token_id = self.qwen.config.image_token_id
        self.proj_rd = PKCProjector(in_ch=256, out_ch=self.hidden_size).cuda()
        self.proj_ra = PKCProjector(in_ch=256, out_ch=self.hidden_size).cuda()

        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            exclude_modules=r"visual.*",
        )
        self.qwen = get_peft_model(self.qwen, lora_cfg)
        # Monkey-patch visual (peft 包装了 model, 需要在 base_model 上改)
        class PKCVisualStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.dtype = torch.bfloat16
            def get_dtype(self): return self.dtype
            def forward(self, pixel_values, grid_thw=None):
                if pixel_values.dim() == 3:
                    B, N, H = pixel_values.shape
                    pixel_values = pixel_values.view(B * N, H)
                return pixel_values.to(dtype=torch.bfloat16)
        # peft 包装: qwen.base_model.model 是 Qwen2VLModel
        self.qwen.base_model.model.visual = PKCVisualStub()
        print(f"[PKCQwenAlign] hidden={self.hidden_size}, img_token={self.image_token_id}")

    def get_visual_embeds(self, x_rd, x_ra, x_ad):
        with torch.no_grad():
            x4_rd, x4_ra = self.pkc(x_rd, x_ra, x_ad, features_only=True, latent_type='x4')
        rd_t = self.proj_rd(x4_rd)
        ra_t = self.proj_ra(x4_ra)
        return torch.cat([rd_t, ra_t], dim=1)  # (B, 512, 3584)


def build_qa_prompt(conversations, num_image_tokens=512, include_all=True):
    """构造 multi-turn conversation prompt + answer.

    Qwen2-VL chat template:
      <|im_start|>user\n<|image_pad|>×N + question<|im_end|>\n
      <|im_start|>assistant\nanswer<|im_end|>\n

    我们用 train_qwen_mt.jsonl 的多轮 conversations, 取所有 user+assistant 轮.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")
    image_pad_str = "<|image_pad|>" * num_image_tokens
    parts = []
    # 取前 5 轮 (1 个 user + 1 个 assistant, 重复)
    turns = []
    for msg in conversations:
        if msg["from"] == "user":
            user_text = msg["value"].replace("<image>", "").strip()  # 去掉 <image> 占位
            turns.append(("user", user_text))
        elif msg["from"] == "assistant":
            turns.append(("assistant", msg["value"]))
    # 构造 chat format
    for role, text in turns:
        if role == "user":
            # 第一次 user 拼接 image_pad
            if len([t for t in parts if "user" in t]) == 0:
                parts.append(f"<|im_start|>user\n{image_pad_str}{text}<|im_end|>\n")
            else:
                parts.append(f"<|im_start|>user\n{text}<|im_end|>\n")
        else:
            parts.append(f"<|im_start|>assistant\n{text}<|im_end|>\n")
    full_text = "".join(parts)
    # prompt_only: 最后一个 assistant 内容去掉, 加 <|im_start|>assistant\n
    if parts and "<|im_start|>assistant\n" in parts[-1]:
        # 最后一个是 assistant 回答, prompt 是去掉它 + 加 assistant 开始
        prompt_text = "".join(parts[:-1]) + "<|im_start|>assistant\n"
        answer_text = parts[-1].replace("<|im_start|>assistant\n", "").replace("<|im_end|>\n", "")
    else:
        prompt_text = full_text
        answer_text = ""
    return prompt_text, answer_text


def collate_qa(batch, image_token_id, num_image_tokens=512):
    """构造 training batch: prompt + answer.

    数值加权: answer 中的数字 token (0-9, '.', '-', 'm', '°') 加大 loss 权重,
    让模型更关注数值精度。
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")
    x_rd = torch.cat([b["x_rd"] for b in batch], dim=0)
    x_ra = torch.cat([b["x_ra"] for b in batch], dim=0)
    x_ad = torch.cat([b["x_ad"] for b in batch], dim=0)
    prompts, fulls = [], []
    answers = []
    for b in batch:
        prompt, answer = build_qa_prompt(b["conversations"], num_image_tokens)
        fulls.append(prompt + answer + "<|im_end|>\n")
        prompts.append(prompt)
        answers.append(answer)
    enc_full = tok(fulls, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    enc_prompt = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    input_ids = enc_full["input_ids"]
    attention_mask = enc_full["attention_mask"]
    labels = input_ids.clone()
    # weight: 数值相关 token 权重 5x (数字/小数点/负号/m/°)
    weight = torch.ones_like(labels, dtype=torch.float)
    NUM_TOK_IDS = set()
    for ch in "0123456789.-":
        ids = tok.encode(ch, add_special_tokens=False)
        NUM_TOK_IDS.update(ids)
    for unit in ["m", "°", "个", "%"]:
        ids = tok.encode(unit, add_special_tokens=False)
        NUM_TOK_IDS.update(ids)
    NUM_TOK_MASK = torch.zeros(max(tok.vocab_size, max(input_ids.max().item() + 1, 152064)),
                                dtype=torch.bool)
    for t in NUM_TOK_IDS:
        if t < NUM_TOK_MASK.size(0):
            NUM_TOK_MASK[t] = True
    for i in range(len(batch)):
        prompt_len = enc_prompt["attention_mask"][i].sum().item()
        labels[i, :prompt_len] = -100
        # 在 answer 部分, 数值 token 加大权重
        ans_mask = labels[i] != -100
        is_num = NUM_TOK_MASK[input_ids[i]]
        weight[i] = torch.where(ans_mask & is_num, 5.0, 1.0)
        weight[i, labels[i] == -100] = 0.0
    image_grid_thw = torch.tensor([[1, 32, 64]] * len(batch), dtype=torch.long)
    return {"x_rd": x_rd, "x_ra": x_ra, "x_ad": x_ad,
            "input_ids": input_ids, "attention_mask": attention_mask,
            "labels": labels, "weight": weight, "image_grid_thw": image_grid_thw}


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_path", default="/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")
    parser.add_argument("--jsonl_train", default="/data/storage/zzy/radar_agent_data/train_qwen_mt.jsonl")
    parser.add_argument("--jsonl_val", default="/data/storage/zzy/radar_agent_data/val_qwen_mt.jsonl")
    parser.add_argument("--jsonl_test", default="/data/storage/zzy/radar_agent_data/test_qwen_mt.jsonl")
    parser.add_argument("--carrada_root", default="/data/storage/zzy/Carrada")
    parser.add_argument("--output_dir", default="/home/zzy/Myproject/RadarLM/output/v9_qa")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_samples", type=int, default=99999)  # 全量数据
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_image_tokens", type=int, default=512)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("[Setup] loading data...")
    train_ds = RadarQADataset(args.jsonl_train, args.carrada_root, split='train',
                              max_samples=args.max_samples, seed=args.seed)
    val_ds = RadarQADataset(args.jsonl_val, args.carrada_root, split='val', seed=args.seed)
    test_ds = RadarQADataset(args.jsonl_test, args.carrada_root, split='test', seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=range(len(train_ds)),
                              collate_fn=lambda b: collate_qa(b, None, args.num_image_tokens),
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            sampler=range(len(val_ds)),
                            collate_fn=lambda b: collate_qa(b, None, args.num_image_tokens),
                            num_workers=0)

    print("[Setup] loading model...")
    model = PKCQwenAlign(args.qwen_path).cuda()
    model.image_token_id = model.qwen.config.image_token_id
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"[Train] {len(train_ds)} train, {len(val_ds)} val, {args.num_epochs} epochs, "
          f"trainable: {sum(p.numel() for p in trainable)/1e6:.2f}M")

    def evaluate(num_samples=None):
        """跑 val generation, 多维度评价."""
        model.eval()
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")
        all_metrics = []
        sample_n = num_samples or len(val_ds)
        for i in range(min(sample_n, len(val_ds))):
            sample = val_ds[i]
            prompt, gt_answer = build_qa_prompt(sample["conversations"], args.num_image_tokens)
            x_rd = sample["x_rd"].cuda(); x_ra = sample["x_ra"].cuda(); x_ad = sample["x_ad"].cuda()
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
                    min_new_tokens=20,        # 防止直接 eos
                    max_new_tokens=128,
                    do_sample=False,
                    repetition_penalty=1.1,
                )
            gen = tok.decode(out_ids[0, input_ids.size(1):], skip_special_tokens=True)
            # 提取第一个 user question
            first_q = sample["conversations"][0]["value"].replace("<image>", "").strip()
            m = eval_qa_pair(first_q, gt_answer, gen)
            all_metrics.append(m)
            if i < 3:
                print(f"  [val {i}] Q: {first_q[:60]}")
                print(f"    TRUE: {gt_answer[:100]}")
                print(f"    GEN:  {gen[:100]}")
        return aggregate_metrics(all_metrics)

    for epoch in range(args.num_epochs):
        model.train()
        loss_sum, n = 0, 0
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            x_rd = batch["x_rd"].cuda(); x_ra = batch["x_ra"].cuda(); x_ad = batch["x_ad"].cuda()
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()
            weight = batch["weight"].cuda()
            image_grid_thw = batch["image_grid_thw"].cuda()
            visual_embeds = model.get_visual_embeds(x_rd, x_ra, x_ad)
            out = model.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=visual_embeds,
                image_grid_thw=image_grid_thw,
                labels=labels,
            )
            logits = out.logits  # (B, L, vocab)
            # 标准 CE loss (用 labels=-100 自动 mask, 不用手算 weight)
            # 防止之前 weighted loss 稀释成 0
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # 加 eos 惩罚: 如果第一个非 mask 的 predicted token 是 eos, 额外 loss
            # (防止模型走捷径输出空)
            eos_id = 151645  # Qwen2 EOS token
            valid_mask = shift_labels != -100
            if valid_mask.any():
                first_valid_idx = valid_mask.float().argmax(dim=1)  # (B,)
                B = logits.size(0)
                first_logits = shift_logits[torch.arange(B), first_valid_idx]  # (B, V)
                eos_prob = F.softmax(first_logits, dim=-1)[:, eos_id]
                eos_penalty = eos_prob.mean() * 2.0  # 加重让模型不在第一个 token 出 eos
                loss = loss + eos_penalty
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            if not torch.isnan(loss).item():
                loss_sum += loss.item(); n += 1
            if step % 20 == 0:
                print(f"  [ep{epoch+1} step{step}] loss={loss.item():.3f}", flush=True)
        t1 = time.time()
        print(f"[Epoch {epoch+1}] avg_loss={loss_sum/max(1,n):.3f} ({t1-t0:.1f}s)", flush=True)
        # save
        torch.save({"proj_rd": model.proj_rd.state_dict(),
                    "proj_ra": model.proj_ra.state_dict()}, f"{args.output_dir}/projector_e{epoch+1}.pt")
        # val (用 50 个 sample 快速测, 避免太慢)
        if (epoch + 1) % 1 == 0:
            print(f"  [val] evaluating {min(50, len(val_ds))} samples...", flush=True)
            metrics = evaluate(num_samples=50)
            print(f"  [val] {json.dumps(metrics, ensure_ascii=False)}", flush=True)
            # save metrics
            with open(f"{args.output_dir}/val_metrics_e{epoch+1}.json", "w") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        # 最后 epoch 在 test 上评估 (全量)
        if epoch + 1 == args.num_epochs:
            print(f"  [test] evaluating full {len(test_ds)} samples...", flush=True)
            # 临时把 val_ds 替换成 test_ds
            val_ds_backup = val_ds
            val_ds.__dict__.update(test_ds.__dict__)
            val_ds.samples = test_ds.samples
            test_metrics = evaluate(num_samples=len(test_ds))
            val_ds.samples = val_ds_backup.samples  # 恢复
            print(f"  [test] {json.dumps(test_metrics, ensure_ascii=False)}", flush=True)
            with open(f"{args.output_dir}/test_metrics.json", "w") as f:
                json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print("[Done]")


if __name__ == "__main__":
    train()