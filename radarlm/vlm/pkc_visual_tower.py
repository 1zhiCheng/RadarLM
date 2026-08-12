"""PKC 视觉塔 + Projector (v9 集成).

包装 PKC backbone, 取 rd/ra 维度的 latent features (128 通道),
加 projector 映射到 Qwen hidden size (1280), 输出 visual tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.insert(0, str(Path("/home/zzy/Myproject/RadarLM/radarlm/pkc_backbone").parent))
sys.path.insert(0, "/home/zzy/Myproject/PKC")

from .pkc_model import PKCIn_plus_cvf_aug


class RadarProjector(nn.Module):
    """PKC latent features (128ch) → Qwen hidden (1280)."""
    def __init__(self, in_ch=128, out_ch=1280, out_h=8, out_w=8):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((out_h, out_w))
        self.norm = nn.LayerNorm(out_ch)
        self.out_h = out_h
        self.out_w = out_w

    def forward(self, x):
        x = self.proj(x)
        x = self.pool(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PKCVisualTower(nn.Module):
    """PKC backbone + 双视图 projector 输出 Qwen 视觉 token."""
    def __init__(self, pkc_n_frames=5, hidden_size=1280, out_h=8, out_w=8, device='cuda'):
        super().__init__()
        self.pkc = PKCIn_plus_cvf_aug(
            n_classes=4, n_frames=pkc_n_frames, device=device
        )
        for p in self.pkc.parameters():
            p.requires_grad = False
        self.pkc.eval()
        self.proj_rd = RadarProjector(in_ch=128, out_ch=hidden_size, out_h=out_h, out_w=out_w)
        self.proj_ra = RadarProjector(in_ch=128, out_ch=hidden_size, out_h=out_h, out_w=out_w)
        self.hidden_size = hidden_size

    def train(self, mode=True):
        super().train(mode)
        self.pkc.eval()
        return self

    def forward(self, x_rd, x_ra, x_ad=None):
        with torch.no_grad():
            out = self.pkc(x_rd, x_ra, x_ad if x_ad is not None else torch.zeros_like(x_rd), features_only=True)
            if isinstance(out, tuple) and len(out) >= 2:
                rd_latent, ra_latent = out[0], out[1]
            else:
                raise ValueError("PKC output unexpected")
        rd_tokens = self.proj_rd(rd_latent)
        ra_tokens = self.proj_ra(ra_latent)
        visual_tokens = torch.cat([rd_tokens, ra_tokens], dim=1)
        return visual_tokens

    @property
    def num_patches(self):
        return self.proj_rd.out_h * self.proj_rd.out_w * 2
