"""加载真实 PKC 预训练权重 (SiLU+GroupNorm 版本).

不修改 PKC 原项目; 在 radarlm/pkc_backbone/ 内复制模型 + 权重.

用法:
    from radarlm.pkc_backbone.pkc_silu_wrapper import PKCWithPretrained
    pkc = PKCWithPretrained(n_classes=4, n_frames=5, device='cuda')
    rd_latent, ra_latent = pkc(x_rd, x_ra, x_ad, features_only=True)
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# radarlm 内的 SiLU+GroupNorm 版本
sys.path.insert(0, str(Path(__file__).parent))
from pkcin_silu_gn import PKCIn_plus_cvf_aug

# 默认权重路径 (radarlm 内的副本)
DEFAULT_WEIGHTS = str(Path(__file__).parent / "weights" / "pkcin_silu_gn.pt")


class PKCWithPretrained(nn.Module):
    """PKC (SiLU+GroupNorm) + 真实 CARRADA 预训练权重.

    权重来自: /data/storage/zzy/logs/carrada/pkcin_plus_cvf_aug/原版+AXFLLoss+SiLU/results/model.pt
    训练 300 epochs, SOTA, CARRADA 语义分割.
    """
    def __init__(self, n_classes=4, n_frames=5, device='cuda',
                 weights_path=DEFAULT_WEIGHTS):
        super().__init__()
        self.pkc = PKCIn_plus_cvf_aug(n_classes=n_classes, n_frames=n_frames, device=device)
        # 加载真实权重
        state_dict = torch.load(weights_path, map_location='cpu')
        # 兼容 model.pt 可能是 {'state_dict': ...} 或纯 state_dict
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # 去前缀 (有些保存会带 'module.' 或 'model.')
        new_sd = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '').replace('model.', '')
            new_sd[new_k] = v
        missing, unexpected = self.pkc.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"[PKCWithPretrained] missing keys: {len(missing)}, e.g. {missing[:3]}")
        if unexpected:
            print(f"[PKCWithPretrained] unexpected keys: {len(unexpected)}, e.g. {unexpected[:3]}")
        # 冻结所有参数
        for p in self.pkc.parameters():
            p.requires_grad = False
        self.pkc.eval()
        print(f"[PKCWithPretrained] loaded {weights_path}, {sum(p.numel() for p in self.pkc.parameters())/1e6:.2f}M params, all frozen")

    def forward(self, x_rd, x_ra, x_ad, features_only=False):
        return self.pkc(x_rd, x_ra, x_ad, features_only=features_only)