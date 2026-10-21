"""
train_cyclegan.py
==================
Baseline (lever 1, docs/reggan_paper_planning.md): the CycleGAN comparison
missing from this project's own results. RegGAN = CycleGAN + a registration
network R with smoothness/magnitude regularizers constraining the DESS->PD
translation to (near-)rigid. This script is train.py with R and every
reg-related loss term removed -- everything else (Generator/Discriminator
architecture, GAN loss, cycle loss, identity loss, data pipeline, optimizer
settings, image pool, LR schedule) is IDENTICAL, so any Dice/deformation
difference downstream is attributable to the registration constraint, not to
an unrelated architecture or hyperparameter change.

Checkpoint format keeps the "G_AB" key train.py uses, so the existing
inference/infer2.py (which only loads G_AB) works unmodified for translating
volumes with a CycleGAN checkpoint.

Does NOT modify train.py or any RegGAN artifact -- separate script, separate
--out_dir, separate checkpoints.

Usage:
    python train_cyclegan.py --splits data/preprocessed/splits.json \
        --out_dir runs/cyclegan_baseline
"""

import os
import json
import logging
import argparse
import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dataset import UnpairedSliceDataset
from models import Generator, PatchDiscriminator, GANLoss

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


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


class CycleGANTrainer:

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        log.info(f"Using device: {self.device}")

        self.G_AB = Generator(in_ch=1, out_ch=1, ngf=args.ngf, n_res=args.n_res).to(self.device)
        self.G_BA = Generator(in_ch=1, out_ch=1, ngf=args.ngf, n_res=args.n_res).to(self.device)
        self.D_A = PatchDiscriminator(in_ch=1, ndf=args.ndf).to(self.device)
        self.D_B = PatchDiscriminator(in_ch=1, ndf=args.ndf).to(self.device)
        # NO RegistrationNet -- this is the entire difference from RegGANTrainer.

        self.crit_gan = GANLoss()
        self.crit_l1 = nn.L1Loss()

        self.opt_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=args.lr, betas=(0.5, 0.999),
        )
        self.opt_D = torch.optim.Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=args.lr, betas=(0.5, 0.999),
        )

        def lr_lambda(epoch):
            n_decay = max(args.epochs - args.epochs // 2, 1)
            return max(0.0, 1.0 - max(0, epoch - args.epochs // 2) / n_decay)

        self.sched_G = torch.optim.lr_scheduler.LambdaLR(self.opt_G, lr_lambda)
        self.sched_D = torch.optim.lr_scheduler.LambdaLR(self.opt_D, lr_lambda)

        train_ds = UnpairedSliceDataset(args.splits, split="train", aug=True)
        val_ds = UnpairedSliceDataset(args.splits, split="val", aug=False)

        _pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=_pin, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=4, shuffle=False,
            num_workers=args.num_workers, pin_memory=_pin,
        )

        self.pool_fake_A = ImagePool(50)
        self.pool_fake_B = ImagePool(50)

        os.makedirs(args.out_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb"))

        self.start_epoch = 0
        if args.resume:
            self._load_checkpoint(args.resume)

        with open(os.path.join(args.out_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    def _forward(self, real_A, real_B):
        fake_B = self.G_AB(real_A)
        rec_A = self.G_BA(fake_B)
        fake_A = self.G_BA(real_B)
        rec_B = self.G_AB(fake_A)
        idt_A = self.G_BA(real_A)
        idt_B = self.G_AB(real_B)
        return dict(fake_B=fake_B, rec_A=rec_A, fake_A=fake_A, rec_B=rec_B,
                    idt_A=idt_A, idt_B=idt_B)

    def _loss_G(self, real_A, real_B, out):
        args = self.args

        l_gan_AB = self.crit_gan(self.D_B(out["fake_B"]), True)
        l_gan_BA = self.crit_gan(self.D_A(out["fake_A"]), True)

        l_cycle_A = self.crit_l1(out["rec_A"], real_A) * args.lambda_cycle
        l_cycle_B = self.crit_l1(out["rec_B"], real_B) * args.lambda_cycle

        l_idt_A = self.crit_l1(out["idt_A"], real_A) * args.lambda_cycle * 0.5
        l_idt_B = self.crit_l1(out["idt_B"], real_B) * args.lambda_cycle * 0.5

        # NO reg_sim / smooth / mag terms -- this is the causal comparison point.
        loss = (l_gan_AB + l_gan_BA + l_cycle_A + l_cycle_B + l_idt_A + l_idt_B)

        return loss, {
            "G/gan_AB": l_gan_AB.item(),
            "G/gan_BA": l_gan_BA.item(),
            "G/cycle_A": l_cycle_A.item(),
            "G/cycle_B": l_cycle_B.item(),
            "G/idt": (l_idt_A + l_idt_B).item(),
            "G/total": loss.item(),
        }

    def _loss_D(self, real, fake_pool, disc):
        l_real = self.crit_gan(disc(real), True)
        l_fake = self.crit_gan(disc(fake_pool), False)
        return (l_real + l_fake) * 0.5

    def _step(self, batch, global_step: int):
        real_A = batch["dess"].to(self.device)
        real_B = batch["pd"].to(self.device)

        out = self._forward(real_A, real_B)

        self.opt_G.zero_grad()
        for p in self.D_A.parameters(): p.requires_grad_(False)
        for p in self.D_B.parameters(): p.requires_grad_(False)

        l_G, g_metrics = self._loss_G(real_A, real_B, out)
        l_G.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()), 5.0
        )
        self.opt_G.step()

        for p in self.D_A.parameters(): p.requires_grad_(True)
        for p in self.D_B.parameters(): p.requires_grad_(True)

        self.opt_D.zero_grad()
        fake_A_pool = self.pool_fake_A.query(out["fake_A"].detach())
        fake_B_pool = self.pool_fake_B.query(out["fake_B"].detach())
        l_DA = self._loss_D(real_A, fake_A_pool, self.D_A)
        l_DB = self._loss_D(real_B, fake_B_pool, self.D_B)
        l_D = l_DA + l_DB
        l_D.backward()
        self.opt_D.step()

        if global_step % self.args.log_interval == 0:
            for k, v in g_metrics.items():
                self.writer.add_scalar(k, v, global_step)
            self.writer.add_scalar("D/total", l_D.item(), global_step)

        return g_metrics

    @torch.no_grad()
    def _validate(self, epoch: int):
        self.G_AB.eval(); self.G_BA.eval()
        l1s = []
        images = None
        for batch in self.val_loader:
            real_A = batch["dess"].to(self.device)
            real_B = batch["pd"].to(self.device)
            fake_B = self.G_AB(real_A)
            l1s.append(self.crit_l1(fake_B, real_B).item())
            if images is None:
                images = (real_A, fake_B, real_B)

        mean_l1 = np.mean(l1s)
        self.writer.add_scalar("val/L1_unpaired_proxy", mean_l1, epoch)

        def denorm(t): return (t + 1) / 2
        grid = make_grid(
            torch.cat([denorm(images[0][:4]), denorm(images[1][:4]), denorm(images[2][:4])], 0),
            nrow=4, normalize=False,
        )
        self.writer.add_image("val/DESS_FakePD_RealPD", grid, epoch)

        self.G_AB.train(); self.G_BA.train()
        return mean_l1

    def _save_checkpoint(self, epoch: int, tag="latest"):
        ckpt = {
            "epoch": epoch,
            "global_step": getattr(self, "_last_global_step", 0),
            "best_val": getattr(self, "_best_val", float("inf")),
            "G_AB": self.G_AB.state_dict(),
            "G_BA": self.G_BA.state_dict(),
            "D_A": self.D_A.state_dict(),
            "D_B": self.D_B.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
            "sched_G": self.sched_G.state_dict(),
            "sched_D": self.sched_D.state_dict(),
            "pool_fake_A": self.pool_fake_A.pool,
            "pool_fake_B": self.pool_fake_B.pool,
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
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        if "pool_fake_A" in ckpt:
            self.pool_fake_A.pool = ckpt["pool_fake_A"]
            self.pool_fake_B.pool = ckpt["pool_fake_B"]
        resumed_epoch = ckpt["epoch"] + 1
        if "sched_G" in ckpt:
            self.sched_G.load_state_dict(ckpt["sched_G"])
            self.sched_D.load_state_dict(ckpt["sched_D"])
        else:
            for _ in range(resumed_epoch):
                self.sched_G.step(); self.sched_D.step()
        self.start_epoch = resumed_epoch
        self._last_global_step = ckpt.get("global_step", 0)
        self._best_val = ckpt.get("best_val", float("inf"))
        log.info(f"Resumed from epoch {ckpt['epoch']} step {ckpt.get('global_step', 0)}: {path}")

    def train(self):
        args = self.args
        global_step = getattr(self, "_last_global_step", 0)
        best_val = getattr(self, "_best_val", float("inf"))

        for epoch in range(self.start_epoch, args.epochs):
            self.G_AB.train(); self.G_BA.train()
            self.D_A.train(); self.D_B.train()

            for i, batch in enumerate(self.train_loader):
                metrics = self._step(batch, global_step)
                global_step += 1
                self._last_global_step = global_step

                if i % args.log_interval == 0:
                    log.info(
                        f"Ep {epoch:03d}  step {i:04d}  "
                        f"G={metrics['G/total']:.3f}  "
                        f"cycA={metrics['G/cycle_A']:.3f}  "
                        f"cycB={metrics['G/cycle_B']:.3f}"
                    )

                if args.save_every > 0 and global_step % args.save_every == 0:
                    self._save_checkpoint(epoch, "latest")
                    log.info(f"  Mid-epoch checkpoint saved at step {global_step}")

            self.sched_G.step(); self.sched_D.step()

            val_l1 = self._validate(epoch)
            log.info(f"  [val] L1 proxy = {val_l1:.4f}")

            is_best = val_l1 < best_val
            if is_best:
                best_val = val_l1
            self._best_val = best_val
            self._save_checkpoint(epoch, "latest")
            if is_best:
                self._save_checkpoint(epoch, "best")
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, f"epoch_{epoch:03d}")

        self.writer.close()
        log.info("Training complete.")


def get_args():
    ap = argparse.ArgumentParser(description="CycleGAN baseline (lever 1): DESS -> PD, no registration constraint")

    ap.add_argument("--splits", default="data/preprocessed/splits.json")
    ap.add_argument("--out_dir", default="runs/cyclegan_baseline")
    ap.add_argument("--resume", default=None)

    # Same architecture defaults as train.py -- required for a fair comparison.
    ap.add_argument("--ngf", type=int, default=48)
    ap.add_argument("--ndf", type=int, default=48)
    ap.add_argument("--n_res", type=int, default=9)

    ap.add_argument("--epochs", type=int, default=50,
                    help="No R to train -> lower per-step compute than RegGAN. "
                         "Not compute-matched to RegGAN's ~4-epoch checkpoint; "
                         "state this in the writeup when comparing.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=500)

    ap.add_argument("--lambda_cycle", type=float, default=10.0)

    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    trainer = CycleGANTrainer(args)
    trainer.train()
