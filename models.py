"""
models.py
=========
RegGAN (Registration-constrained GAN) for minimum-deformation domain adaptation.

Architecture follows:
  Kong et al., "Breaking the Dilemma of Medical Image-to-image Translation",
  NeurIPS 2021  (Reg-GAN)

Key idea:
  - CycleGAN backbone (G_AB, G_BA, D_A, D_B)
  - + a lightweight Registration network R that maps (G_AB(x), y_real) -> deformation field
  - Registration loss penalises large deformations  -> minimum-deformation constraint
  - The generator is also optimised through the registration path
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch: int, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
            nn.ReLU(True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class UpsampleConv(nn.Module):
    """Upsample + Conv (avoids checkerboard artifacts vs ConvTranspose2d)."""
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Generator (ResNet-based, 9 blocks default)
# ─────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    ResNet generator.
    in_ch / out_ch = 1 for grayscale MRI slices.
    """

    def __init__(self, in_ch=1, out_ch=1, ngf=64, n_res=9, use_dropout=False):
        super().__init__()

        # Encoder
        enc = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf,     ngf*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.Conv2d(ngf*2,   ngf*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*4), nn.ReLU(True),
        ]

        # Residual blocks
        res = [ResBlock(ngf*4, use_dropout) for _ in range(n_res)]

        # Decoder
        dec = [
            UpsampleConv(ngf*4, ngf*2),
            UpsampleConv(ngf*2, ngf),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, 7),
            nn.Tanh(),
        ]

        self.net = nn.Sequential(*enc, *res, *dec)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator (PatchGAN 70×70)
# ─────────────────────────────────────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=1, ndf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ch = ndf
        for i in range(1, n_layers):
            ch_prev, ch = ch, min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch_prev, ch, 4, stride=2, padding=1),
                nn.InstanceNorm2d(ch),
                nn.LeakyReLU(0.2, True),
            ]
        ch_prev, ch = ch, min(ch * 2, 512)
        layers += [
            nn.Conv2d(ch_prev, ch, 4, padding=1),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, 1, 4, padding=1),   # output: patch map
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Registration network (VoxelMorph-lite, 2-D)
# ─────────────────────────────────────────────────────────────────────────────

class RegistrationNet(nn.Module):
    """
    Lightweight U-Net that takes (moving, fixed) concatenated (2 channels)
    and predicts a 2-D displacement field (Δx, Δy).

    For MINIMUM deformation:
      - We keep the architecture shallow
      - Deformation is regularised with a smoothness + magnitude penalty
    """

    def __init__(self, nf=16):
        super().__init__()

        def conv_block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1), nn.LeakyReLU(0.2, True),
                nn.Conv2d(oc, oc, 3, padding=1), nn.LeakyReLU(0.2, True),
            )

        self.enc1 = conv_block(2, nf)
        self.enc2 = conv_block(nf, nf*2)
        self.enc3 = conv_block(nf*2, nf*4)

        self.pool = nn.AvgPool2d(2)

        self.dec2 = conv_block(nf*4 + nf*2, nf*2)
        self.dec1 = conv_block(nf*2 + nf,   nf)

        # Flow head – initialise near-zero so deformation starts small
        self.flow = nn.Conv2d(nf, 2, 3, padding=1)
        nn.init.normal_(self.flow.weight, 0, 1e-5)
        nn.init.zeros_(self.flow.bias)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor):
        x = torch.cat([moving, fixed], dim=1)   # (B, 2, H, W)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.dec2(torch.cat([F.interpolate(e3, size=e2.shape[2:],
                                                 mode="bilinear", align_corners=False), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[2:],
                                                 mode="bilinear", align_corners=False), e1], 1))

        flow = self.flow(d1)   # (B, 2, H, W)
        return flow

    @staticmethod
    def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Differentiable bilinear warping via grid_sample.
        img:  (B, C, H, W)
        flow: (B, 2, H, W)  – displacement in pixel units
        """
        B, C, H, W = img.shape
        # Build base grid (no expand – use broadcast to avoid inplace issues)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=img.device),
            torch.linspace(-1, 1, W, device=img.device),
            indexing="ij",
        )
        # (1, H, W, 2) – keep as leaf, do NOT expand inplace
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        # Normalise flow to [-1, 1] space and stack as (B, H, W, 2)
        flow_x = flow[:, 0:1] / (W / 2)   # (B, 1, H, W)
        flow_y = flow[:, 1:2] / (H / 2)
        norm_flow = torch.cat([flow_x, flow_y], dim=1)          # (B, 2, H, W)
        norm_flow = norm_flow.permute(0, 2, 3, 1)               # (B, H, W, 2)

        sample_grid = grid + norm_flow   # broadcast (1,H,W,2) + (B,H,W,2)
        return F.grid_sample(img, sample_grid, align_corners=True,
                             mode="bilinear", padding_mode="border")


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

class GANLoss(nn.Module):
    """LSGAN (MSE) – more stable than BCE for medical images."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target_real: bool):
        target = torch.ones_like(pred) if target_real else torch.zeros_like(pred)
        return F.mse_loss(pred, target)


def gradient_smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    """
    Penalise spatial gradients of the deformation field.
    Encourages smooth (and small) deformations.
    """
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    return (dx.pow(2).mean() + dy.pow(2).mean())


def deformation_magnitude_loss(flow: torch.Tensor) -> torch.Tensor:
    """
    Direct L2 penalty on displacement magnitude.
    Key for 'minimum deformation' constraint.
    """
    return flow.pow(2).mean()
