"""
train.py
========
RegGAN training loop: DESS -> PD domain adaptation.

Usage (local):
    python train.py --splits data/preprocessed/splits.json --out_dir runs/reggan_001

Usage (BigRed / SLURM):
    See bigred_submit.sh

Key hyperparameters for MINIMUM DEFORMATION:
  --lambda_reg_smooth   smoothness regularisation weight (default 10.0)
  --lambda_reg_mag      deformation magnitude penalty   (default 5.0)
  --lambda_cycle        cycle-consistency weight         (default 10.0)
"""

import os
import json
import logging
import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dataset import UnpairedSliceDataset
from models  import (Generator, PatchDiscriminator, RegistrationNet,
                     GANLoss, gradient_smoothness_loss, deformation_magnitude_loss)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Image pool (for discriminator stability)
# ─────────────────────────────────────────────────────────────────────────────

class ImagePool:
    def __init__(self, size: int = 50):
        self.size = size
        self.pool = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return images
        out = []
        for img in images.unbind(0):
            if len(self.pool) < self.size:
                self.pool.append(img)
                out.append(img)
            elif np.random.rand() > 0.5:
                idx = np.random.randint(len(self.pool))
                out.append(self.pool[idx].clone())
                self.pool[idx] = img
            else:
                out.append(img)
        return torch.stack(out, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class RegGANTrainer:

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        log.info(f"Using device: {self.device}")

        # ── Models ──────────────────────────────────────────────────────────
        self.G_AB = Generator(in_ch=1, out_ch=1, ngf=args.ngf, n_res=args.n_res).to(self.device)
        self.G_BA = Generator(in_ch=1, out_ch=1, ngf=args.ngf, n_res=args.n_res).to(self.device)
        self.D_A  = PatchDiscriminator(in_ch=1, ndf=args.ndf).to(self.device)
        self.D_B  = PatchDiscriminator(in_ch=1, ndf=args.ndf).to(self.device)
        # Registration network: maps (fake_B, real_B) -> deformation field
        self.R    = RegistrationNet(nf=args.nf_reg).to(self.device)

        # ── Losses ──────────────────────────────────────────────────────────
        self.crit_gan  = GANLoss()
        self.crit_l1   = nn.L1Loss()

        # ── Optimisers ──────────────────────────────────────────────────────
        self.opt_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=args.lr, betas=(0.5, 0.999),
        )
        self.opt_D = torch.optim.Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=args.lr, betas=(0.5, 0.999),
        )
        self.opt_R = torch.optim.Adam(
            self.R.parameters(), lr=args.lr_reg, betas=(0.5, 0.999),
        )

        # LR schedulers: keep LR for first half, linear decay for second half
        def lr_lambda(epoch):
            n_decay = max(args.epochs - args.epochs // 2, 1)
            return max(0.0, 1.0 - max(0, epoch - args.epochs // 2) / n_decay)

        self.sched_G = torch.optim.lr_scheduler.LambdaLR(self.opt_G, lr_lambda)
        self.sched_D = torch.optim.lr_scheduler.LambdaLR(self.opt_D, lr_lambda)
        self.sched_R = torch.optim.lr_scheduler.LambdaLR(self.opt_R, lr_lambda)

        # ── Data ────────────────────────────────────────────────────────────
        train_ds = UnpairedSliceDataset(args.splits, split="train", aug=True)
        val_ds   = UnpairedSliceDataset(args.splits, split="val",   aug=False)

        # pin_memory only works with CUDA, not MPS
        _pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=_pin, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=4, shuffle=False,
            num_workers=args.num_workers, pin_memory=_pin,
        )

        # ── Misc ────────────────────────────────────────────────────────────
        self.pool_fake_A = ImagePool(50)
        self.pool_fake_B = ImagePool(50)

        os.makedirs(args.out_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb"))

        self.start_epoch = 0
        if args.resume:
            self._load_checkpoint(args.resume)

        # Save args
        with open(os.path.join(args.out_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # ── Forward pass (one batch) ─────────────────────────────────────────────

    def _forward(self, real_A, real_B):
        """
        A = DESS domain
        B = PD domain
        G_AB: DESS -> PD (the translation we care about)
        G_BA: PD -> DESS (needed for cycle consistency)
        R:    Registration(fake_B, real_B) -> deformation field
        """
        # Generator outputs
        fake_B  = self.G_AB(real_A)         # DESS -> fake PD
        rec_A   = self.G_BA(fake_B)         # fake PD -> reconstructed DESS
        fake_A  = self.G_BA(real_B)         # real PD -> fake DESS
        rec_B   = self.G_AB(fake_A)         # fake DESS -> reconstructed PD
        idt_A   = self.G_BA(real_A)         # identity: DESS through G_BA
        idt_B   = self.G_AB(real_B)         # identity: PD through G_AB

        # Registration: align fake_B to a real_B sample
        # (key RegGAN step — replaces direct pixel cycle for B domain)
        flow    = self.R(fake_B.detach(), real_B)
        warped  = RegistrationNet.warp(fake_B, flow.detach())

        return dict(
            fake_B=fake_B, rec_A=rec_A,
            fake_A=fake_A, rec_B=rec_B,
            idt_A=idt_A,   idt_B=idt_B,
            flow=flow,     warped=warped,
        )

    # ── Loss computation ─────────────────────────────────────────────────────

    def _loss_G(self, real_A, real_B, out):
        args = self.args

        # GAN losses
        l_gan_AB = self.crit_gan(self.D_B(out["fake_B"]), True)
        l_gan_BA = self.crit_gan(self.D_A(out["fake_A"]), True)

        # Cycle-consistency
        l_cycle_A = self.crit_l1(out["rec_A"], real_A) * args.lambda_cycle
        l_cycle_B = self.crit_l1(out["rec_B"], real_B) * args.lambda_cycle

        # Identity
        l_idt_A = self.crit_l1(out["idt_A"], real_A) * args.lambda_cycle * 0.5
        l_idt_B = self.crit_l1(out["idt_B"], real_B) * args.lambda_cycle * 0.5

        # Registration loss: warped fake_B should match real_B (NCC-like)
        l_reg_sim = self.crit_l1(out["warped"], real_B) * args.lambda_reg_sim

        # MINIMUM DEFORMATION constraints
        l_smooth = gradient_smoothness_loss(out["flow"]) * args.lambda_reg_smooth
        l_mag    = deformation_magnitude_loss(out["flow"]) * args.lambda_reg_mag

        loss = (l_gan_AB + l_gan_BA +
                l_cycle_A + l_cycle_B +
                l_idt_A   + l_idt_B   +
                l_reg_sim + l_smooth  + l_mag)

        return loss, {
            "G/gan_AB": l_gan_AB.item(),
            "G/gan_BA": l_gan_BA.item(),
            "G/cycle_A": l_cycle_A.item(),
            "G/cycle_B": l_cycle_B.item(),
            "G/idt": (l_idt_A + l_idt_B).item(),
            "G/reg_sim": l_reg_sim.item(),
            "G/smooth": l_smooth.item(),
            "G/mag": l_mag.item(),
            "G/total": loss.item(),
        }

    def _loss_D(self, real, fake_pool, disc):
        l_real = self.crit_gan(disc(real), True)
        l_fake = self.crit_gan(disc(fake_pool), False)
        return (l_real + l_fake) * 0.5

    # ── Training step ────────────────────────────────────────────────────────

    def _step(self, batch, global_step: int):
        real_A = batch["dess"].to(self.device)
        real_B = batch["pd"].to(self.device)

        out = self._forward(real_A, real_B)

        # ── Update Generators FIRST (before R.step would mutate R weights) ───
        self.opt_G.zero_grad()
        for p in self.D_A.parameters(): p.requires_grad_(False)
        for p in self.D_B.parameters(): p.requires_grad_(False)

        l_G, g_metrics = self._loss_G(real_A, real_B, out)
        l_G.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()), 5.0
        )
        self.opt_G.step()

        # ── Update Registration network (after G so R weights dont poison G grad) ─
        self.opt_R.zero_grad()
        flow_r   = self.R(out["fake_B"].detach(), real_B)
        warped_r = RegistrationNet.warp(out["fake_B"].detach(), flow_r)
        l_r_sim  = self.crit_l1(warped_r, real_B) * self.args.lambda_reg_sim
        l_r_smo  = gradient_smoothness_loss(flow_r) * self.args.lambda_reg_smooth
        l_r_mag  = deformation_magnitude_loss(flow_r) * self.args.lambda_reg_mag
        l_R = l_r_sim + l_r_smo + l_r_mag
        l_R.backward()
        self.opt_R.step()

        # ── Update Discriminators ────────────────────────────────────────────
        for p in self.D_A.parameters(): p.requires_grad_(True)
        for p in self.D_B.parameters(): p.requires_grad_(True)

        self.opt_D.zero_grad()
        fake_A_pool = self.pool_fake_A.query(out["fake_A"].detach())
        fake_B_pool = self.pool_fake_B.query(out["fake_B"].detach())
        l_DA = self._loss_D(real_A, fake_A_pool, self.D_A)
        l_DB = self._loss_D(real_B, fake_B_pool, self.D_B)
        l_D  = l_DA + l_DB
        l_D.backward()
        self.opt_D.step()

        # ── Log ─────────────────────────────────────────────────────────────
        if global_step % self.args.log_interval == 0:
            for k, v in g_metrics.items():
                self.writer.add_scalar(k, v, global_step)
            self.writer.add_scalar("D/total", l_D.item(), global_step)
            self.writer.add_scalar("R/total", l_R.item(), global_step)
            self.writer.add_scalar(
                "R/mean_magnitude",
                out["flow"].abs().mean().item(), global_step
            )

        return g_metrics

    # ── Validation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, epoch: int):
        self.G_AB.eval(); self.G_BA.eval(); self.R.eval()
        l1s = []
        images = None
        for batch in self.val_loader:
            real_A = batch["dess"].to(self.device)
            real_B = batch["pd"].to(self.device)
            fake_B = self.G_AB(real_A)
            l1s.append(self.crit_l1(fake_B, real_B).item())   # proxy metric
            if images is None:
                images = (real_A, fake_B, real_B)

        mean_l1 = np.mean(l1s)
        self.writer.add_scalar("val/L1_proxy", mean_l1, epoch)

        # Save sample grid (denorm from [-1,1] to [0,1])
        def denorm(t): return (t + 1) / 2
        grid = make_grid(
            torch.cat([denorm(images[0][:4]),
                       denorm(images[1][:4]),
                       denorm(images[2][:4])], 0),
            nrow=4, normalize=False,
        )
        self.writer.add_image("val/DESS_FakePD_RealPD", grid, epoch)

        self.G_AB.train(); self.G_BA.train(); self.R.train()
        return mean_l1

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, tag="latest"):
        ckpt = {
            "epoch": epoch,
            "global_step": getattr(self, "_last_global_step", 0),
            "G_AB": self.G_AB.state_dict(),
            "G_BA": self.G_BA.state_dict(),
            "D_A":  self.D_A.state_dict(),
            "D_B":  self.D_B.state_dict(),
            "R":    self.R.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
            "opt_R": self.opt_R.state_dict(),
        }
        path = os.path.join(self.args.out_dir, f"ckpt_{tag}.pt")
        torch.save(ckpt, path)
        log.info(f"  Checkpoint saved -> {path}")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.G_AB.load_state_dict(ckpt["G_AB"])
        self.G_BA.load_state_dict(ckpt["G_BA"])
        self.D_A.load_state_dict(ckpt["D_A"])
        self.D_B.load_state_dict(ckpt["D_B"])
        self.R.load_state_dict(ckpt["R"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        self.opt_R.load_state_dict(ckpt["opt_R"])
        self.start_epoch = ckpt["epoch"] + 1
        log.info(f"Resumed from epoch {ckpt['epoch']} step {ckpt.get('global_step', 0)}: {path}")

    # ── Main loop ────────────────────────────────────────────────────────────

    def train(self):
        args = self.args
        global_step = 0
        best_val = float("inf")

        for epoch in range(self.start_epoch, args.epochs):
            self.G_AB.train(); self.G_BA.train()
            self.D_A.train();  self.D_B.train()
            self.R.train()

            for i, batch in enumerate(self.train_loader):
                metrics = self._step(batch, global_step)
                global_step += 1
                self._last_global_step = global_step

                if i % args.log_interval == 0:
                    log.info(
                        f"Ep {epoch:03d}  step {i:04d}  "
                        f"G={metrics['G/total']:.3f}  "
                        f"cycA={metrics['G/cycle_A']:.3f}  "
                        f"reg_sim={metrics['G/reg_sim']:.3f}  "
                        f"smooth={metrics['G/smooth']:.4f}  "
                        f"mag={metrics['G/mag']:.4f}"
                    )

                # Save mid-epoch checkpoint every args.save_every steps
                if args.save_every > 0 and global_step % args.save_every == 0:
                    self._save_checkpoint(epoch, "latest")
                    log.info(f"  Mid-epoch checkpoint saved at step {global_step}")

            # LR step
            self.sched_G.step(); self.sched_D.step(); self.sched_R.step()

            # Validate
            val_l1 = self._validate(epoch)
            log.info(f"  [val] L1 proxy = {val_l1:.4f}")

            # Checkpoint
            self._save_checkpoint(epoch, "latest")
            if val_l1 < best_val:
                best_val = val_l1
                self._save_checkpoint(epoch, "best")

        self.writer.close()
        log.info("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    ap = argparse.ArgumentParser(description="RegGAN: DESS -> PD domain adaptation")

    # Data
    ap.add_argument("--splits",      default="data/preprocessed/splits.json")
    ap.add_argument("--out_dir",     default="runs/reggan_001")
    ap.add_argument("--resume",      default=None, help="Path to checkpoint to resume from")

    # Architecture
    ap.add_argument("--ngf",   type=int, default=48,  help="Generator base filters (reduce for MPS: 48 or 32)")
    ap.add_argument("--ndf",   type=int, default=48,  help="Discriminator base filters")
    ap.add_argument("--n_res", type=int, default=6,   help="Residual blocks (reduce for MPS: 6)")
    ap.add_argument("--nf_reg",type=int, default=16,  help="Registration net base filters (keep small)")

    # Training
    ap.add_argument("--epochs",      type=int,   default=200)
    ap.add_argument("--batch_size",  type=int,   default=2,  help="Reduce to 1 if still OOM on MPS")
    ap.add_argument("--lr",          type=float, default=2e-4)
    ap.add_argument("--lr_reg",      type=float, default=1e-4,
                    help="Registration net LR (lower = less aggressive warping)")
    ap.add_argument("--num_workers", type=int,   default=4)
    ap.add_argument("--log_interval",type=int,   default=50)
    ap.add_argument("--save_every",  type=int,   default=500,
                    help="Save mid-epoch checkpoint every N steps (0=disable)")

    # Loss weights  ← KEY for minimum deformation
    ap.add_argument("--lambda_cycle",      type=float, default=10.0)
    ap.add_argument("--lambda_reg_sim",    type=float, default=5.0,
                    help="Similarity weight between warped fake_B and real_B")
    ap.add_argument("--lambda_reg_smooth", type=float, default=10.0,
                    help="↑ = smoother deformation field")
    ap.add_argument("--lambda_reg_mag",    type=float, default=5.0,
                    help="↑ = smaller deformation magnitude (more rigid)")

    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    trainer = RegGANTrainer(args)
    trainer.train()
