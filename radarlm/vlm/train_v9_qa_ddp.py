"""v9 多模态雷达感知 QA 训练 (DDP 数据并行版本).

用法:
    torchrun --nproc_per_node=4 --master_port=29501 radarlm/vlm/train_v9_qa_ddp.py

或 accelerate:
    accelerate launch --num_processes=4 radarlm/vlm/train_v9_qa_ddp.py
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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

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


def parse_image_id(image_id: str):
    import re
    m = re.match(r"^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(\d+)$", image_id)
    if not m: return None, None
    return m.group(1), m.group(2).zfill(6)


class RadarQADataset(Dataset):
    def __init__(self, jsonl_path, carrada_root, max_samples=None, seed=42, use_hflip=True):
        import json
        self.carrada_root = Path(carrada_root)
        self.use_hflip = use_hflip
        self.rng = random.Random(seed)
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip(): continue
                self.samples.append(json.loads(line))
        if max_samples and max_samples < len(self.samples):
            rng2 = random.Random(seed + 1)
            self.samples = rng2.sample(self.samples, max_samples)
        print(f"[RadarQADataset] {len(self.samples)} samples (from {Path(jsonl_path).name})", flush=True)

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
        rd5 = np.stack([rd] * 5)
        ra5 = np.stack([ra] * 5)
        ad5 = np.stack([ad] * 5)
        return {
            "x_rd": torch.from_numpy(rd5).unsqueeze(0).unsqueeze(0).float(),
            "x_ra": torch.from_numpy(ra5).unsqueeze(0).unsqueeze(0).float(),
            "x_ad": torch.from_numpy(ad5).unsqueeze(0).unsqueeze(0).float(),
            "conversations": item["conversations"],
            # v12 curriculum: 三个阶段 prompt 都要传给 collate (collate 随机选一个)
            "stage_1_prompt": item.get("stage_1_prompt", ""),
            "stage_2_prompt": item.get("stage_2_prompt", ""),
            "stage_3_prompt": item.get("stage_3_prompt", ""),
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


class PKCProjectorX9(nn.Module):
    """v4 projector: 输入 4 通道 segmentation logits, 池化到 64/256 patches, 转 3584 dim.

    关键: 不加 LayerNorm! LayerNorm 会把每个 token 标准化到 mean=0 std=1,
    导致真实图/黑图/噪声的输出几乎一样 (视觉信息被抹掉).
    改用 clamp 保留全局信息.
    """
    def __init__(self, in_ch=4, out_ch=3584, num_patches_h=16, num_patches_w=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((num_patches_h, num_patches_w))
        self.proj = nn.Linear(in_ch, out_ch)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        x = self.pool(x)
        B, C, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)
        x = self.proj(x)
        # clamp 保留 4 通道 logits 的真实值 (不大改动)
        x = torch.clamp(x, min=-20.0, max=20.0)
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

        self.qwen = Qwen2VLForConditionalGeneration.from_pretrained(
            qwen_path, torch_dtype=torch.bfloat16, device_map={"": "cuda"},
        )
        for p in self.qwen.visual.parameters(): p.requires_grad = False
        # Gradient checkpointing: 减少 activation 显存 (DDP 多卡训练必需)
        self.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.hidden_size = self.qwen.config.hidden_size
        self.image_token_id = self.qwen.config.image_token_id
        # v4: 用 x9_rd/x9_ra (4 通道 segmentation logits) 作为 visual features
        # x9_rd: (B, 4, 256, 64) -> pool (B, 4, 16, 4) = 64 patches
        # x9_ra: (B, 4, 256, 256) -> pool (B, 4, 16, 16) = 256 patches
        self.proj_rd = PKCProjectorX9(in_ch=4, out_ch=self.hidden_size, num_patches_h=16, num_patches_w=4).cuda()
        self.proj_ra = PKCProjectorX9(in_ch=4, out_ch=self.hidden_size, num_patches_h=16, num_patches_w=16).cuda()

        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            exclude_modules=r"visual.*",
        )
        self.qwen = get_peft_model(self.qwen, lora_cfg)
        # Enable input require grads for gradient checkpointing + LoRA
        if hasattr(self.qwen, "enable_input_require_grads"):
            self.qwen.enable_input_require_grads()
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
        self.qwen.base_model.model.visual = PKCVisualStub()

    def get_visual_embeds(self, x_rd, x_ra, x_ad):
        with torch.no_grad():
            # v4: x9_rd/x9_ra 最后一帧 segmentation logits (4 通道, 4 类)
            x9_rd, x9_ra = self.pkc(x_rd, x_ra, x_ad, features_only=True, latent_type='x9')
        # x9_rd: (B, 4, 256, 64) -> pool (4, 16, 4) -> 64 patches
        # x9_ra: (B, 4, 256, 256) -> pool (4, 16, 16) -> 256 patches
        rd_t = self.proj_rd(x9_rd)  # (B, 64, 3584)
        ra_t = self.proj_ra(x9_ra)  # (B, 256, 3584)
        return torch.cat([rd_t, ra_t], dim=1)  # (B, 320, 3584)


def build_qa_prompt(conversations, num_image_tokens=320):
    """构建多轮 QA prompt + 答案.

    安全关键 prompt 设计 (自动驾驶要求):
    - System message 明确告诉模型 RD/RA/AD 视图大小
    - 强制模型回答时明确指出视图和格式
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")
    image_pad_str = "<|image_pad|>" * num_image_tokens

    # System message: 明确视图大小 + 格式要求 + 平衡漏检/虚警
    system_msg = (
        "你是自动驾驶雷达感知 Agent, 安全关键. 分析 RD/RA/AD 三视图检测目标.\n"
        "★ 漏检和虚警同等重要 — 不能为了防漏检而瞎说有目标.\n"
        "★ 如果 RD/RA/AD 三视图都没有反射簇, 必答'图中无目标'.\n"
        "视图尺寸参考 (必须记忆):\n"
        "  - RD 视图 (距离-多普勒): 256×64 (高×宽), row 0-255, col 0-63\n"
        "  - RA 视图 (距离-角度): 256×256 (高×宽), row 0-255, col 0-255\n"
        "  - AD 视图 (角度-多普勒): 256×64 (高×宽), row 0-255, col 0-63\n"
        "回答要求 (严格):\n"
        "  1. ★ 视图必须 100% 对应: 问哪个视图必须答哪个视图, 串视图算错\n"
        "  2. 距离/速度/角度数值要精确 (单位 m 或 m/s 或 °)\n"
        "  3. bbox 格式: 'X 视图 bbox: [row_min=R0, col_min=C0, row_max=R1, col_max=C1]'\n"
        "     row_min < row_max 且 col_min < col_max, 必须落在视图范围内\n"
        "  4. 无目标时不要输出 bbox, 直接答'无目标'\n"
        "  5. ★ 类别题必答 A/B/C/D 单字母, 不能答'有目标'/'无目标'模糊文本 (如 '图中检测到 1 个目标' 不是 valid 类别答案)\n"
        "★ 谨慎判断流程: 先扫三视图是否有反射簇 (RD row 200-250, RA 角度 ±5° 集中区):\n"
        "  - 有反射簇 → 仔细看 RD 反射簇大小 (≥10 行=汽车, 5-10 行=行人/骑行者, <5 行=杂波)\n"
        "  - 无反射簇 → 必答 D (无目标)\n"
        "Few-shot 例子:\n"
        "  Q: 图中目标是什么类别? → A/B/C (有目标时) / D (无目标时)\n"
        "  Q: RD 视图 bbox? → RD 视图 bbox: [row_min=80, col_min=40, row_max=85, col_max=41] (有目标时)\n"
        "                            → 图中无目标 (无目标时)\n"
        "  Q: RA 视图有反射簇吗? → RA 视图 (256×256): 在行 X-Y, 列 Z-W 处有反射簇 (有目标时)\n"
        "                          → RA 视图 (256×256): 未见明显反射簇 (无目标时)\n"
        "  Q: 距离是多少? → 距离约 33.2 m (有目标时) / 无目标 (无目标时)\n"
    )

    parts = [f"<|im_start|>system\n{system_msg}<|im_end|>\n"]
    turns = []
    for msg in conversations:
        if msg["from"] == "user":
            turns.append(("user", msg["value"].replace("<image>", "").strip()))
        elif msg["from"] == "assistant":
            turns.append(("assistant", msg["value"]))

    for role, text in turns:
        if role == "user":
            if len([t for t in parts if "user" in t]) == 0:
                parts.append(f"<|im_start|>user\n{image_pad_str}{text}<|im_end|>\n")
            else:
                parts.append(f"<|im_start|>user\n{text}<|im_end|>\n")
        else:
            parts.append(f"<|im_start|>assistant\n{text}<|im_end|>\n")
    full_text = "".join(parts)
    if parts and "<|im_start|>assistant\n" in parts[-1]:
        prompt_text = "".join(parts[:-1]) + "<|im_start|>assistant\n"
        answer_text = parts[-1].replace("<|im_start|>assistant\n", "").replace("<|im_end|>\n", "")
    else:
        prompt_text = full_text
        answer_text = ""
    return prompt_text, answer_text


def collate_qa(batch, image_token_id, num_image_tokens=320):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")
    x_rd = torch.cat([b["x_rd"] for b in batch], dim=0)
    x_ra = torch.cat([b["x_ra"] for b in batch], dim=0)
    x_ad = torch.cat([b["x_ad"] for b in batch], dim=0)
    prompts, fulls, sample_weights = [], [], []
    for b in batch:
        # v12 curriculum: 优先用 stage_1/2/3_prompt (含 PKC 解码的 object list)
        # 兼容老格式 (用 conversations)
        # 关键: stage_3 prompt 没有 <image_pad>, 需要注入占位符
        stage_n = np.random.randint(0, 3)  # 0=Stage 1, 1=Stage 2, 2=Stage 3
        stage_key = f'stage_{stage_n+1}_prompt'
        if stage_key in b:
            prompt = b[stage_key]
            # 注入 320 个 <image_pad> 占位符 (确保 token 数匹配 features)
            if "<|image_pad|>" not in prompt:
                # Stage 3 prompt 没图 token, 在 user msg 开头注入
                prompt = prompt.replace(
                    "<|im_start|>user\n",
                    f"<|im_start|>user\n{'<|image_pad|>' * num_image_tokens}\n", 1)
            answer = b['conversations'][1]['value']  # GT 答案
        else:
            prompt, answer = build_qa_prompt(b["conversations"], num_image_tokens)
        fulls.append(prompt + answer + "<|im_end|>\n")
        prompts.append(prompt)
        first_assist = answer.strip()
        if first_assist in ["A", "B", "C"]:
            sample_weights.append(1.2)
        else:
            sample_weights.append(1.0)
    enc_full = tok(fulls, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    enc_prompt = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    input_ids = enc_full["input_ids"]
    attention_mask = enc_full["attention_mask"]
    labels = input_ids.clone()
    for i in range(len(batch)):
        prompt_len = enc_prompt["attention_mask"][i].sum().item()
        labels[i, :prompt_len] = -100
    # v6: RD (16×4=64 patches) + RA (16×16=256 patches) 两个独立 grid, 给 mrope 独立位置编码
    image_grid_thw = torch.tensor([[1, 16, 4], [1, 16, 16]] * len(batch), dtype=torch.long)
    # weight: 与 labels 同样 shape. 训练时 shift 一步 (labels[:, 1:])
    weights = torch.zeros_like(labels, dtype=torch.float)
    for i, w in enumerate(sample_weights):
        ans_mask = labels[i] != -100
        weights[i, ans_mask] = w
    return {"x_rd": x_rd, "x_ra": x_ra, "x_ad": x_ad,
            "input_ids": input_ids, "attention_mask": attention_mask,
            "labels": labels, "weight": weights, "image_grid_thw": image_grid_thw}


def setup_ddp():
    """初始化 DDP 环境."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return local_rank, world_size, rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_path", default="/data/storage/zzy/radar_agent_data/models/Qwen2-VL-7B-Instruct")
    parser.add_argument("--jsonl_train", default="/data/storage/zzy/radar_agent_data/train_qwen_mt.jsonl")
    parser.add_argument("--jsonl_val", default="/data/storage/zzy/radar_agent_data/val_qwen_mt.jsonl")
    parser.add_argument("--carrada_root", default="/data/storage/zzy/Carrada")
    parser.add_argument("--output_dir", default="/home/zzy/Myproject/RadarLM/output/v9_qa_ddp")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_image_tokens", type=int, default=320,
                        help="v4: 64 (RD) + 256 (RA) = 320 visual tokens")
    parser.add_argument("--max_train_samples", type=int, default=8088)
    args = parser.parse_args()

    local_rank, world_size, rank = setup_ddp()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[DDP] world_size={world_size}, local_rank={local_rank}", flush=True)

    # 数据
    train_ds = RadarQADataset(args.jsonl_train, args.carrada_root,
                              max_samples=args.max_train_samples, seed=args.seed)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(train_ds, batch_size=1,
                              sampler=train_sampler,
                              shuffle=(train_sampler is None),
                              collate_fn=lambda b: collate_qa(b, None, args.num_image_tokens),
                              num_workers=0)

    if is_main:
        print(f"[Train] {len(train_ds)} train, world_size={world_size}", flush=True)

    # 模型 (每个 rank 独立加载到自己 GPU)
    if is_main:
        print("[Setup] loading model...", flush=True)
    model = PKCQwenAlign(args.qwen_path).cuda()
    if world_size > 1:
        # DDP 包装: 注意 qwen 是 PeftModel (LoRA), 可以直接 DDP
        # 但 projector 不在 PeftModel 内, 需要手动 wrap
        # 简单做法: 把 projector 和 qwen 一起放进一个 nn.Module
        # 这里用 DDP 默认 wrap (会 wrap 所有 parameters)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)
    if is_main:
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"[Model] trainable: {sum(p.numel() for p in trainable)/1e6:.2f}M", flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        loss_sum, n = 0, 0
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            x_rd = batch["x_rd"].cuda(local_rank)
            x_ra = batch["x_ra"].cuda(local_rank)
            x_ad = batch["x_ad"].cuda(local_rank)
            input_ids = batch["input_ids"].cuda(local_rank)
            attention_mask = batch["attention_mask"].cuda(local_rank)
            labels = batch["labels"].cuda(local_rank)
            image_grid_thw = batch["image_grid_thw"].cuda(local_rank)
            visual_embeds = model.module.get_visual_embeds(x_rd, x_ra, x_ad) if world_size > 1 else model.get_visual_embeds(x_rd, x_ra, x_ad)
            out = model.module.qwen(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=visual_embeds, image_grid_thw=image_grid_thw, labels=labels,
            ) if world_size > 1 else model.qwen(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=visual_embeds, image_grid_thw=image_grid_thw, labels=labels,
            )
            logits = out.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # Weighted CE: sample_weight (有目标 2x, 无目标 0.5x) 防止 model 倾向答"无目标"
            log_probs = F.log_softmax(shift_logits, dim=-1)
            B, L, V = log_probs.shape
            # weight 需要和 shift_labels 同样 shape: labels[:, 1:]
            shift_weight = batch["weight"].cuda(local_rank)[:, 1:].contiguous()
            valid_mask = shift_labels != -100
            lp = log_probs.view(-1, V)[valid_mask.view(-1)]
            tgt = shift_labels.view(-1)[valid_mask.view(-1)]
            w = shift_weight.view(-1)[valid_mask.view(-1)]
            nll = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
            loss = (nll * w).sum() / w.sum().clamp(min=1.0)
            eos_id = 151645
            valid_mask = shift_labels != -100
            if valid_mask.any():
                first_valid_idx = valid_mask.float().argmax(dim=1)
                first_logits = shift_logits[torch.arange(B, device=local_rank), first_valid_idx]
                eos_prob = F.softmax(first_logits, dim=-1)[:, eos_id]
                eos_penalty = eos_prob.mean() * 2.0
                loss = loss + eos_penalty
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            if not torch.isnan(loss).item():
                loss_sum += loss.item(); n += 1
        if is_main:
            print(f"[Epoch {epoch+1}] avg_loss={loss_sum/max(1,n):.4f} ({time.time()-t0:.1f}s)", flush=True)
            # save projector + LoRA adapter
            proj = model.module if world_size > 1 else model
            torch.save({"proj_rd": proj.proj_rd.state_dict(),
                        "proj_ra": proj.proj_ra.state_dict()}, f"{args.output_dir}/projector_e{epoch+1}.pt")
            # ★ 关键: 保存 LoRA adapter, 否则评估时白训
            qwen_save = proj.qwen
            qwen_save.save_pretrained(f"{args.output_dir}/lora_e{epoch+1}")
            print(f"[saved] projector + lora → {args.output_dir}/(projector|lora)_e{epoch+1}", flush=True)

    if world_size > 1:
        dist.destroy_process_group()
    if is_main:
        print("[Done]")


if __name__ == "__main__":
    main()