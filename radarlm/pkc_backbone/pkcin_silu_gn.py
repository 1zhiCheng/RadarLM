# MVRSS-Net w/ Vanilla Peak Conv
# The implementation of Peak Conv referred deformable Conv (https://github.com/4uiiurz1/pytorch-deform-conv-v2)
# zlw @20220628
# Modified: Replace CA-CFAR kernel with cross-shaped kernel
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

class DoubleConvBlock(nn.Module):
    """ (2D conv => GroupNorm => SiLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        # 替换LeakyReLU为SiLU，BN为GroupNorm（分组数8，适配通道数）
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.GroupNorm(8, out_ch),  # 分组数8，需保证out_ch能被8整除（32/64/128均满足）
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Double3DConvBlock(nn.Module):
    """ (3D conv => GroupNorm => SiLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        # 替换LeakyReLU为SiLU，BN为GroupNorm（3D版本）
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class ConvBlock(nn.Module):
    """ (2D conv => GroupNorm => SiLU) """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        # 替换LeakyReLU为SiLU，BN为GroupNorm
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class ConvBlock_o(nn.Module):
    """ (2D conv => BN => LeakyReLU) """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class CVFM(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, pad=0, dil=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, k_size, padding=pad, dilation=dil, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.norm2d = nn.BatchNorm2d(out_ch)
        self.activate = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm2d(x)
        x = self.activate(x)
        return x

class TDConvBlock(nn.Module):
    """ (3D conv => BN => LeakyReLU)"""

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class DoubleADA(nn.Module):
    """
    Temporal Attention Integration (TAI) module

    PARAMETERS
    ----------
    in_ch: int
        Number of input channels
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.v_conv_block1_3x1 = ConvBlock(in_ch=in_ch, out_ch=out_ch, k_size=(3, 1), pad=(3, 0), dil=(3, 1))
        self.h_conv_block1_1x3 = ConvBlock(in_ch=out_ch, out_ch=out_ch, k_size=(1, 3), pad=(0, 3), dil=(1, 3))
        self.v_conv_block2_3x1 = ConvBlock(in_ch=out_ch, out_ch=out_ch, k_size=(3, 1), pad=(3, 0), dil=(3, 1))
        self.h_conv_block2_1x3 = ConvBlock(in_ch=out_ch, out_ch=out_ch, k_size=(1, 3), pad=(0, 3), dil=(1, 3))

    def forward(self, x):
        x_1 = self.v_conv_block1_3x1(x)
        x_2 = self.h_conv_block1_1x3(x_1)
        x_3 = self.v_conv_block2_3x1(x_2)
        x_4 = self.h_conv_block2_1x3(x_3)
        return x_4

class ASPPBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling
    Parallel conv blocks with different dilation rate
    """

    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.global_avg_pool = nn.AvgPool2d((64, 64))
        self.conv1_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, dilation=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=6, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=12, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=18, dil=18)

    def forward(self, x):
        x1 = F.interpolate(self.global_avg_pool(x), size=(64, 64), align_corners=False,
                           mode='bilinear')
        x1 = self.conv1_1x1(x1)
        x2 = self.single_conv_block1_1x1(x)
        x3 = self.single_conv_block1_3x3(x)
        x4 = self.single_conv_block2_3x3(x)
        x5 = self.single_conv_block3_3x3(x)
        x_cat = torch.cat((x2, x3, x4, x5, x1), 1)
        return x_cat

class ViewQueryModule(nn.Module):
    """
    View fusion branch for combining radar views

    PARAMETERS
    ----------
    n_views: int
        Number of radar views to be fused.
    """

    def __init__(self, in_ch, out_ch, supply=False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.supply = supply
        self.single_conv_block_RD_1x1x1 = TDConvBlock(in_ch=256, out_ch=128, k_size=1, pad=0, dil=1)
        self.single_conv_block_RA_1x1x1 = TDConvBlock(in_ch=256, out_ch=128, k_size=1, pad=0, dil=1)
        self.latent_space_extraction_block = LatentSpaceFusionBranch(in_ch=384, out_ch=128)

    def forward(self, x_rd, x_ra, x_ad):   
        x_rd_r = torch.unsqueeze(x_rd, 3)
        x_rd_r = torch.repeat_interleave(x_rd_r, repeats=x_ra.shape[3], dim=3)
        x_ra_r = torch.rot90(x_ra, 2, [2, 3])
        x_ra_r = torch.unsqueeze(x_ra_r, -1)
        x_ra_r = torch.repeat_interleave(x_ra_r, repeats=x_rd.shape[3], dim=4)
        x_ad_r = torch.unsqueeze(x_ad, 2)
        x_ad_r = torch.repeat_interleave(x_ad_r, repeats=x_rd.shape[2], dim=2)
        x_rad = torch.cat((x_rd_r, x_ad_r, x_ra_r), 1)
        x_rd_latent, x_ra_latent = self.latent_space_extraction_block(x_rad)
        
        # 改正
        x_ard = torch.permute(x_rad[:, 128:, :, :, :], (0, 1, 3, 2, 4))
        x_dra = torch.rot90(torch.permute(x_rad[:, :256, :, :, :], (0, 1, 4, 2, 3)), 2, [3, 4])
        x_ard_q = self.single_conv_block_RD_1x1x1(x_ard)
        x_dra_q = self.single_conv_block_RA_1x1x1(x_dra)
        
        return x_ard_q, x_dra_q, x_rd_latent, x_ra_latent

        # 降维 之前
        # x_rad_rd = self.single_conv_block_RD_1x1x1(x_rad)
        # x_rad_ra = self.single_conv_block_RA_1x1x1(x_rad)

        # return x_rad_rd, x_rad_ra, x_rd_latent, x_ra_latent

class LatentSpaceFusionBranch(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.max_pool_D2 = nn.MaxPool3d((1, 64, 1), stride=(1, 64, 1))
        self.max_pool_D3 = nn.MaxPool3d((1, 1, 64), stride=(1, 1, 64))
        self.rd_single_conv_block1_1x1 = ConvBlock(in_ch=self.in_ch, out_ch=self.out_ch, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block1_1x1 = ConvBlock(in_ch=self.in_ch, out_ch=self.out_ch, k_size=1, pad=0, dil=1)

    def forward(self, x_rad):
        # 从拼接得到的rad latent矩阵中压缩得到rd视图的隐藏空间信息
        x_rad2rd = self.max_pool_D2(x_rad)
        x_rad2rd = torch.squeeze(x_rad2rd, 3)  # remove a axis

        # 从拼接得到的rad latent矩阵中压缩得到ra视图的隐藏空间信息
        x_rad2ra = self.max_pool_D3(x_rad)
        x_rad2ra = torch.squeeze(x_rad2ra, 4)
        x_rad2ra = torch.rot90(x_rad2ra, 2, [2, 3])  # align ra view
        
        # 降维投影回原始通道数，得到隐空间特征
        x_rd_latent = self.rd_single_conv_block1_1x1(x_rad2rd)
        x_ra_latent = self.ra_single_conv_block1_1x1(x_rad2ra)

        return x_rd_latent, x_ra_latent
    
class ViewAttentionModule(nn.Module):
    def __init__(self, supply=False):
        super().__init__()
        self.supply = supply
        self.alpha = 0.5 # 偏移量，固定0.5即可
        self.rd_single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=64, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=64, k_size=1, pad=0, dil=1)
        self.rd_single_conv_block2_1x1 = ConvBlock_o(in_ch=64, out_ch=4, k_size=3, pad=1, dil=1)
        self.ra_single_conv_block2_1x1 = ConvBlock_o(in_ch=64, out_ch=4, k_size=3, pad=1, dil=1)
        # 训练使用，评估不使用
        self.rd_upconv = nn.ConvTranspose2d(64, 64, (4, 1), stride=(4, 1))
        self.ra_upconv = nn.ConvTranspose2d(64, 64, 4, stride=4)
        self.rd_inter = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1)
        self.ra_inter = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1)
        self.gaussian_blur = GaussianBlur(kernel_size=3, sigma=(0.5, 1.0))

    def forward(self, x_ard_q, x_dra_q, x_rd_k, x_ra_k):
        """supply=False时的forward，接收全部参数"""
        B, C, H, W = x_rd_k.shape
        x_rd_tem1 = self.rd_single_conv_block1_1x1(x_rd_k)
        x_rd_tem = self.rd_single_conv_block2_1x1(x_rd_tem1)
        rd_max_vals, rd_max_indices = torch.max(x_rd_tem, dim=1, keepdim=True)  # (B,1,H,W), (B,1,H,W)
        # 生成二进制掩码：最大值通道不为0则为1，否则为0
        rd_mask = (rd_max_indices != 0).float()  # (B,1,H,W) 独热编码模
        x_ra_tem1 = self.ra_single_conv_block1_1x1(x_ra_k)
        x_ra_tem = self.ra_single_conv_block2_1x1(x_ra_tem1)
        ra_max_vals, ra_max_indices = torch.max(x_ra_tem, dim=1, keepdim=True)  # (B,1,H,W), (B,1,H,W)
        # 生成二进制掩码：最大值通道不为0则为1，否则为0
        ra_mask = (ra_max_indices != 0).float()  # (B,1,H,W) 独热编码模板
        
        # 指数型增强因子，直接替换原有线性公式
        rd_mask = self.gaussian_blur(rd_mask)
        ra_mask = self.gaussian_blur(ra_mask)
        enhance_rd_factor = 1.0 + self.alpha * (rd_mask - 0.5)
        enhance_ra_factor = 1.0 + self.alpha * (ra_mask - 0.5)
        
        x_rd_k = torch.unsqueeze(x_rd_k * enhance_rd_factor, 2)
        x_ra_k = torch.unsqueeze(x_ra_k * enhance_ra_factor, 2)
        # 计算相似度
        x_rd_sim = torch.mul(x_ard_q * enhance_rd_factor.unsqueeze(2), x_rd_k) / (C ** 0.5)
        x_ra_sim = torch.mul(x_dra_q * enhance_ra_factor.unsqueeze(2), x_ra_k) / (C ** 0.5)
        # 维度求和
        x_rd_sim = torch.sum(x_rd_sim, dim=1)
        x_ra_sim = torch.sum(x_ra_sim, dim=1)
        # 计算注意力权重
        x_rd_att = torch.softmax(x_rd_sim.view(B, H, -1), dim=1).view(B, H, W, W)
        x_ra_att = torch.softmax(x_ra_sim.view(B, H, -1), dim=1).view(B, H, W, W)
        
        # 中间指导输出(评估不使用)
        # ===== 核心修改：根据训练/评估状态自动切换输出 =====
        if self.training:  # 训练阶段：计算并输出中间监督项
            x_rd_int = self.rd_upconv(x_rd_tem1)
            x_ra_int = self.ra_upconv(x_ra_tem1)
            x_rd_inter = self.rd_inter(x_rd_int)
            x_ra_inter = self.ra_inter(x_ra_int)
            return x_rd_att, x_ra_att, x_rd_inter, x_ra_inter
        else:  # 评估阶段：仅输出注意力权重
            return x_rd_att, x_ra_att
    
class ViewFusionBranch(nn.Module):
    """
    View fusion branch for combining radar views

    PARAMETERS
    ----------
    n_views: int
        Number of radar views to be fused.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.Viewquery_block_supply = ViewQueryModule(in_ch=self.in_ch, out_ch=self.out_ch, supply=True)
        self.Viewatt_block2 = ViewAttentionModule(supply=True)
        self.rd_single_conv_block1b_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block1b_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)
        self.ad_single_conv_block1b_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)
        self.rd_single_conv_block3_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block3_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)
        
    def forward(self, x_rd, x_ra, x_ad):
        x_rd_p = self.rd_single_conv_block1b_1x1(x_rd)
        x_ra_p = self.ra_single_conv_block1b_1x1(x_ra)
        x_ad_p = self.ad_single_conv_block1b_1x1(x_ad)
        x_ard_s, x_dra_s, rd_latent, ra_latent = self.Viewquery_block_supply(x_rd_p, x_ra_p, x_ad_p)
        if self.training:
            rd_att2, ra_att2, x_rd_inter, x_ra_inter = self.Viewatt_block2(x_ard_s, x_dra_s, x_rd_p, x_ra_p)
        else:
            rd_att2, ra_att2 = self.Viewatt_block2(x_ard_s, x_dra_s, x_rd_p, x_ra_p)
            
        # 补充视图重构增强RD
        rd_att2 = torch.unsqueeze(rd_att2, 1)
        rd_i = torch.sum(rd_att2 * x_ard_s, dim=2)
        x_rd_f = self.rd_single_conv_block3_1x1(rd_i)
        x_rd_f2 = x_rd + x_rd_f  # 残差连接

        # 补充视图重构增强RA
        ra_att2 = torch.unsqueeze(ra_att2, 1)
        ra_i = torch.sum(ra_att2 * x_dra_s, dim=2)
        x_ra_f = self.ra_single_conv_block3_1x1(ra_i)
        x_ra_f2 = x_ra + x_ra_f  # 残差连接
        if self.training:
            return x_rd_f2, x_ra_f2, rd_latent, ra_latent, x_rd_inter, x_ra_inter
        else:
            return x_rd_f2, x_ra_f2, rd_latent, ra_latent

class ViewReconstructionBranch(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv_block_r = CVFM(in_ch=self.in_ch,
                                        out_ch=self.out_ch,
                                        k_size=(1, 128),
                                        pad=0,
                                        dil=1
                                        )
        self.conv_block_a = CVFM(in_ch=self.in_ch,
                                        out_ch=self.out_ch,
                                        k_size=(128, 1),
                                        pad=0,
                                        dil=1
                                        )
        self.conv_block_d = CVFM(in_ch=self.in_ch,
                                        out_ch=self.out_ch,
                                        k_size=(128, 1),
                                        pad=0,
                                        dil=1
                                        )
        self.rd_single_conv_block_1x1 = ConvBlock(in_ch=256, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block_1x1 = ConvBlock(in_ch=256, out_ch=128, k_size=1, pad=0, dil=1)

    def forward(self, x_rd, x_ra, x_ad):
        # 提前公共R轴特征
        x_ra2r = torch.rot90(x_ra, 2, [2, 3])
        x_r = torch.cat((x_rd, x_ra2r), 3)
        r_feature = self.conv_block_r(x_r) # for RD
        r_feature2 = torch.rot90(r_feature, 2, [2, 3]) # for RA

        # 提前公共A轴特征
        x_da = torch.permute(x_ad, (0, 1, 3, 2))
        x_da2a = torch.rot90(x_da, 2, [2, 3])
        x_a = torch.cat((x_ra, x_da2a), 2)
        a_feature = self.conv_block_a(x_a) # for RA

        # 提取公共D轴特征
        x_d = torch.cat((x_ad, x_rd), 2)
        d_feature = self.conv_block_d(x_d) # for RD

        # 增强RD
        x_r1 = torch.repeat_interleave(r_feature, repeats=x_rd.shape[3], dim=3)
        x_d1 = torch.repeat_interleave(d_feature, repeats=x_rd.shape[2], dim=2)
        x_rd_f = torch.cat((x_r1, x_d1), 1)
        x_rd_f = self.rd_single_conv_block_1x1(x_rd_f)
        x_rd_f2 = x_rd + x_rd_f  # 残差连接

        # 增强RA
        x_r2 = torch.repeat_interleave(r_feature2, repeats=x_ra.shape[3], dim=3)
        x_a1 = torch.repeat_interleave(a_feature, repeats=x_ra.shape[2], dim=2)
        x_ra_f = torch.cat((x_r2, x_a1), 1)
        x_ra_f = self.ra_single_conv_block_1x1(x_ra_f)
        x_ra_f2 = x_ra + x_ra_f  # 残差连接

        return x_rd_f2, x_ra_f2

class EncodingBranch(nn.Module):
    """
    Encoding branch for a single radar view

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view.
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
    """

    def __init__(self, signal_type, device):
        super().__init__()
        self.signal_type = signal_type
        self.device = device
        self.double_3dconv_block1 = Double3DConvBlock(in_ch=1, out_ch=128, k_size=3,
                                                      pad=(0, 1, 1), dil=1)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        '''
        self.double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                  pad=1, dil=1)
        '''
        self.double_ada_block = DoubleADA(in_ch=128, out_ch=128)
        # self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1,
        #                                         pad=0, dil=1)

    def forward(self, x):
        x1 = self.double_3dconv_block1(x)
        x1 = torch.squeeze(x1, 2)  # remove temporal dimension

        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            x1_pad = F.pad(x1, (0, 1, 0, 0), "constant", 0)
            x1_down = self.doppler_max_pool(x1_pad)
        else:
            x1_down = self.max_pool(x1)

        # x2 = self.double_conv_block2(x1_down)
        x2 = self.double_ada_block(x1_down)
        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            x2_pad = F.pad(x2, (0, 1, 0, 0), "constant", 0)
            x2_down = self.doppler_max_pool(x2_pad)
        else:
            x2_down = self.max_pool(x2)

        # x3 = self.single_conv_block1_1x1(x2_down)
        # return input of ASPP block + latent features
        return x2_down


class PKCIn_plus_cvf_aug(nn.Module):
    """ 
    Temporal Multi-View with ASPP Network (TMVA-Net)

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames, device):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.device = device

        # Backbone (encoding)
        self.rd_encoding_branch = EncodingBranch('range_doppler', device)
        self.ra_encoding_branch = EncodingBranch('range_angle', device)
        self.ad_encoding_branch = EncodingBranch('angle_doppler', device)
        self.reconstruction_block = ViewReconstructionBranch(in_ch=128, out_ch=128)
        self.view_fusion_conv_block = ViewFusionBranch(in_ch=128, out_ch=128)

        # ASPP Blocks
        self.rd_aspp_block = ASPPBlock(in_ch=256, out_ch=128)
        self.ra_aspp_block = ASPPBlock(in_ch=256, out_ch=128)
        self.rd_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)

        # Pallel range-Doppler (RD) and range-angle (RA) decoding branches
        self.rd_upconv1 = nn.ConvTranspose2d(256, 128, (2, 1), stride=(2, 1))
        self.ra_upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rd_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.rd_upconv2 = nn.ConvTranspose2d(128, 128, (2, 1), stride=(2, 1))
        self.ra_upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rd_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)

        # Final 1D convs
        self.rd_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)
        self.ra_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)

    def forward(self, x_rd, x_ra, x_ad, features_only=False, latent_type='latent'):
        # Backbone
        ra_feature = self.ra_encoding_branch(x_ra)
        rd_feature = self.rd_encoding_branch(x_rd)
        ad_feature = self.ad_encoding_branch(x_ad)
        # zzy 24S152090
        x1_rd_c, x1_ra_c = self.reconstruction_block(rd_feature, ra_feature, ad_feature)
        if self.training:
            x1_rd_s, x1_ra_s, rd_latent, ra_latent, x_rd_inter, x_ra_inter = self.view_fusion_conv_block(rd_feature, ra_feature, ad_feature)
        else:
            x1_rd_s, x1_ra_s, rd_latent, ra_latent = self.view_fusion_conv_block(rd_feature, ra_feature, ad_feature)
        if features_only:
            if latent_type == 'latent':
                # rd_latent, ra_latent: 跨视图融合 latent, 128 通道, 来自 LatentSpaceFusionBranch
                return rd_latent, ra_latent
            # 走 ASPP + decoder 部分
            x1_rd = torch.cat((x1_rd_s, x1_rd_c), 1)
            x1_ra = torch.cat((x1_ra_s, x1_ra_c), 1)
            x2_rd = self.rd_aspp_block(x1_rd)
            x2_ra = self.ra_aspp_block(x1_ra)
            x3_rd = self.rd_single_conv_block1_1x1(x2_rd)
            x3_ra = self.ra_single_conv_block1_1x1(x2_ra)
            if latent_type == 'x3':
                # x3_rd, x3_ra: ASPP 后 1x1 conv, 128 通道
                return x3_rd, x3_ra
            x4_rd = torch.cat((x3_rd, rd_latent), 1)
            x4_ra = torch.cat((x3_ra, ra_latent), 1)
            if latent_type == 'x4':
                # x4_rd, x4_ra: x3 + rd_latent 拼接, 256 通道 (含 latent 跨视图信息)
                return x4_rd, x4_ra
            if latent_type == 'x9':
                # x9 需要 x8_rd, x8_ra (after upconv2)
                # 注意: x8 在 latent_type=='x4' 之后定义, x9 必须放到 x8 之后
                # 这里用 placeholder, 实际 return 在 line 562 之后处理
                pass

        # 融合补充
        x1_rd = torch.cat((x1_rd_s, x1_rd_c), 1)
        x1_ra = torch.cat((x1_ra_s, x1_ra_c), 1)

        # ASPP blocks
        x2_rd = self.rd_aspp_block(x1_rd)
        x2_ra = self.ra_aspp_block(x1_ra)
        x3_rd = self.rd_single_conv_block1_1x1(x2_rd)
        x3_ra = self.ra_single_conv_block1_1x1(x2_ra)

        # zzy 24S152090
        x4_rd = torch.cat((x3_rd, rd_latent), 1)
        x4_ra = torch.cat((x3_ra, ra_latent), 1)

        # Parallel decoding branches with upconvs
        x5_rd = self.rd_upconv1(x4_rd)
        x5_ra = self.ra_upconv1(x4_ra)
        x6_rd = self.rd_double_conv_block1(x5_rd)
        x6_ra = self.ra_double_conv_block1(x5_ra)

        x7_rd = self.rd_upconv2(x6_rd)
        x7_ra = self.ra_upconv2(x6_ra)
        x8_rd = self.rd_double_conv_block2(x7_rd)
        x8_ra = self.ra_double_conv_block2(x7_ra)

        # Final 1D convolutions
        x9_rd = self.rd_final(x8_rd)
        x9_ra = self.ra_final(x8_ra)

        # v4: 提取 x9_rd/x9_ra 作为 visual features
        if features_only and latent_type == 'x9':
            return x9_rd, x9_ra

        if self.training:
            return x9_rd, x9_ra, x_rd_inter, x_ra_inter
        else:
            return x9_rd, x9_ra