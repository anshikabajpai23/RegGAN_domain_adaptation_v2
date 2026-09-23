"""
finetune_meniscus_v7_replica_e2.py
=====================================
Phase 5, E2 — slice position via FiLM gating at the bottleneck, on the PLAIN
2.5D replica base (independent of crop+sequence). Diffs directly against
run_v7_replica_seed42, exactly like P1/P2/P4 did.

Distinct from the crop+sequence E2: there is no ConvLSTM hidden state to gate
per-timestep in the plain architecture (one slice per forward pass, no
recurrence), so this applies FiLM ONCE to the single bottleneck feature map,
conditioned on the stack's centre-slice position -- see model_2_5d_film.py's
docstring for why this is the natural equivalent rather than a copy-paste port.

BASE: everything from finetune_meniscus_v7_replica.py is copied verbatim
except build_model() (swapped for model_2_5d_film.build_2_5d_film_model, the
zero-init FiLM wrapper -- proven bit-identical to plain smp.Unet at step 0,
max diff 0.000e+00) and the two dataset classes (thin subclasses adding a
`pos` field, not a rewrite).

Position, same formula as every other Phase 5 position experiment:
    pos = 2 * |idx/(n-1) - 0.5|,  0 at the volume centre (notch), 1 at outer edges
ONE position per sample here (the stack's centre slice), not five -- the plain
2.5D architecture has one slice per forward pass, not a 5-slice window.

n_slices comes from volume_lengths.json (scripts/build_volume_lengths.py),
reused UNCHANGED from the crop+sequence E1 prep: it records true per-volume
slice counts (fake-PD file counts understate this due to the prep step's
min_meniscus_pixels filter), and that fact is completely independent of
whether the downstream model crops in-plane or not -- so the same table is
valid here with zero changes.

Refuses to train if the zero-init check fails, same discipline as every other
Phase 5 position experiment.
"""
import argparse
import csv
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_2_5d_v2 import Meniscus2_5DDataset
from dataset_2_5d_realpd import RealPDDataset
from model_2_5d_film import build_2_5d_film_model, verify_zero_init

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES = 3
PRETRAINED_N_CLASSES = 5
PD_SPACING = (3.6, 0.39, 0.39)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info(f"Seed fixed: {seed} (cudnn.deterministic=True)")


def pos_of(idx, n):
    return 2.0 * abs(idx / max(n - 1, 1) - 0.5)


class PosMeniscus2_5DDataset(Meniscus2_5DDataset):
    """Base already returns patient_id/slice_idx directly -- just attach pos."""
    def __init__(self, img_root, mask_root, lengths, augment=False):
        super().__init__(img_root, mask_root, augment=augment)
        self.lengths = lengths
        missing = {pid for pid, _ in self.items} - set(lengths)
        if missing:
            raise SystemExit(f"volume_lengths.json missing {len(missing)} fake patients, "
                             f"e.g. {sorted(missing)[:5]}")

    def __getitem__(self, i):
        item = super().__getitem__(i)
        n = self.lengths[item["patient_id"]]
        item["pos"] = torch.tensor(pos_of(item["slice_idx"], n), dtype=torch.float32)
        return item


class PosRealPDDataset(RealPDDataset):
    def __init__(self, img_root, mask_root, lengths, augment=False):
        super().__init__(img_root, mask_root, augment=augment)
        self.lengths = lengths
        missing = {s.rsplit("_", 1)[0] for s in self.slices} - set(lengths)
        if missing:
            raise SystemExit(f"volume_lengths.json missing real patients: {sorted(missing)}")

    def __getitem__(self, i):
        item = super().__getitem__(i)
        pid, sidx = self.slices[i].rsplit("_", 1)
        n = self.lengths[pid]
        item["pos"] = torch.tensor(pos_of(int(sidx), n), dtype=torch.float32)
        return item


# ── Losses: verbatim from finetune_meniscus_v7_replica.py ──────────────────

class MergedLoss(nn.Module):
    def __init__(self, w_bg=0.1, w_men=1.5, eps=1e-6):
        super().__init__(); self.w_bg, self.w_men, self.eps = w_bg, w_men, eps

    def forward(self, logits, binary_masks):
        probs = torch.softmax(logits, dim=1)
        p_bg = probs[:, 0]; p_men = probs[:, 1] + probs[:, 2]
        gt_men = (binary_masks > 0).float(); gt_bg = (binary_masks == 0).float()
        ce = -(self.w_bg * gt_bg * torch.log(p_bg.clamp(self.eps)) +
               self.w_men * gt_men * torch.log(p_men.clamp(self.eps))).mean()
        tp = (p_men * gt_men).sum()
        dice = 1.0 - (2 * tp + self.eps) / (p_men.sum() + gt_men.sum() + self.eps)
        return ce + dice


class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes=N_CLASSES, eps=1e-6):
        super().__init__(); self.n_classes = n_classes; self.eps = eps

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1); loss = 0.0
        for c in range(1, self.n_classes):
            p = probs[:, c]; t = (targets == c).float()
            loss += 1.0 - (2 * (p * t).sum() + self.eps) / (p.sum() + t.sum() + self.eps)
        return loss / (self.n_classes - 1)


def dice_binary(preds, targets, eps=1e-6):
    pm = (preds > 0).float(); gm = (targets > 0).float()
    tp = (pm * gm).sum()
    return float((2 * tp + eps) / (pm.sum() + gm.sum() + eps))


def run_epoch(model, real_loader, fake_loader, optimizer, merged_loss_fn, fake_loss_fn, device, train=True):
    model.train() if train else model.eval()
    total_loss, dice_sum, dice_fake_sum, n = 0.0, 0.0, 0.0, 0
    real_iter = iter(real_loader)

    with torch.set_grad_enabled(train):
        for fake_batch in fake_loader:
            xf, yf = fake_batch["image"].to(device), fake_batch["mask"].to(device)
            pf = fake_batch["pos"].to(device)
            logits_f = model(xf, pf)
            loss_f = fake_loss_fn(logits_f, yf)

            try:
                real_batch = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader); real_batch = next(real_iter)
            xr, yr = real_batch["image"].to(device), real_batch["mask"].to(device)
            pr = real_batch["pos"].to(device)
            logits_r = model(xr, pr)
            loss_r = merged_loss_fn(logits_r, yr)

            loss = loss_f + loss_r
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            total_loss += loss.item()
            dice_sum += dice_binary(logits_r.argmax(1), yr)
            dice_fake_sum += dice_binary(logits_f.argmax(1), yf)
            n += 1

    return (total_loss / max(n, 1), dice_sum / max(n, 1), dice_fake_sum / max(n, 1))


@torch.no_grad()
def eval_real_per_patient(model, dataset, device, batch_size=8):
    model.eval(); acc = {}
    for start in range(0, len(dataset), batch_size):
        idxs = range(start, min(start + batch_size, len(dataset)))
        items = [dataset[i] for i in idxs]
        stems = [dataset.slices[i] for i in idxs]
        x = torch.stack([it["image"] for it in items]).to(device)
        y = torch.stack([it["mask"] for it in items]).to(device)
        p = torch.stack([it["pos"] for it in items]).to(device)
        pred = model(x, p).argmax(1)
        pm, gm = (pred > 0).float(), (y > 0).float()
        for b, stem in enumerate(stems):
            pid = stem.rsplit("_", 1)[0]
            a = acc.setdefault(pid, [0.0, 0.0, 0.0])
            a[0] += float((pm[b] * gm[b]).sum()); a[1] += float(pm[b].sum()); a[2] += float(gm[b].sum())
    eps = 1e-6
    return {pid: (2 * tp + eps) / (ps + gs + eps) for pid, (tp, ps, gs) in acc.items()}


@torch.no_grad()
def dump_val_predictions(model, dataset, device, out_dir, batch_size=8):
    try:
        import nibabel as nib
    except ImportError:
        log.warning("nibabel not available — skipping val prediction dump"); return
    model.eval(); os.makedirs(out_dir, exist_ok=True); vols = {}
    for start in range(0, len(dataset), batch_size):
        idxs = range(start, min(start + batch_size, len(dataset)))
        items = [dataset[i] for i in idxs]
        stems = [dataset.slices[i] for i in idxs]
        x = torch.stack([it["image"] for it in items]).to(device)
        y = torch.stack([it["mask"] for it in items])
        p = torch.stack([it["pos"] for it in items]).to(device)
        pred = model(x, p).argmax(1).cpu()
        for b, stem in enumerate(stems):
            pid, sidx = stem.rsplit("_", 1)
            vols.setdefault(pid, {})[int(sidx)] = (
                x[b, 1].cpu().numpy().astype(np.float32),
                y[b].numpy().astype(np.int16), pred[b].numpy().astype(np.int16))
    affine = np.diag([PD_SPACING[0], PD_SPACING[1], PD_SPACING[2], 1.0]).astype(np.float32)
    csv_path = os.path.join(out_dir, "per_slice_dice.csv"); eps = 1e-6
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["patient_id", "slice_idx", "dice", "gt_voxels", "pred_voxels", "false_neg", "false_pos"])
        for pid, slices in sorted(vols.items()):
            order = sorted(slices.keys())
            img = np.stack([slices[s][0] for s in order], 0)
            gt = np.stack([slices[s][1] for s in order], 0)
            pr = np.stack([slices[s][2] for s in order], 0)
            nib.save(nib.Nifti1Image(img, affine), os.path.join(out_dir, f"{pid}_image.nii.gz"))
            nib.save(nib.Nifti1Image((gt > 0).astype(np.int16), affine), os.path.join(out_dir, f"{pid}_gt.nii.gz"))
            nib.save(nib.Nifti1Image((pr > 0).astype(np.int16), affine), os.path.join(out_dir, f"{pid}_pred.nii.gz"))
            for k, s in enumerate(order):
                g = (gt[k] > 0); p_ = (pr[k] > 0)
                tp = float((g & p_).sum()); gs = float(g.sum()); ps = float(p_.sum())
                w.writerow([pid, s, f"{(2*tp + eps) / (ps + gs + eps):.4f}",
                            int(gs), int(ps), int(gs - tp), int(ps - tp)])
            log.info(f"  dumped {pid}: {len(order)} slices")
    log.info(f"  per-slice Dice CSV -> {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--fake_data_root", required=True)
    ap.add_argument("--real_data_root", required=True)
    ap.add_argument("--volume_lengths", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--class_weights", type=float, nargs=3, default=[0.1, 1.5, 1.5])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_zero_init_check", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  LR: {args.lr} -> 1e-8  |  E2: FiLM gating, PLAIN 2.5D, uncropped")

    if not args.skip_zero_init_check:
        ok, diff = verify_zero_init(device="cpu")
        if not ok:
            raise SystemExit(f"ZERO-INIT CHECK FAILED (max diff {diff:.3e}). Refusing to train.")

    with open(args.volume_lengths) as fh:
        vl = json.load(fh)
    meta = vl.get("meta", {})
    log.info(f"volume_lengths: {meta.get('n_fake')} fake, {meta.get('n_real')} real  "
             f"(span_coverage_ok={meta.get('span_coverage_ok')})")

    model = build_2_5d_film_model(args.pretrained_ckpt, device,
                                  pretrained_n_classes=PRETRAINED_N_CLASSES, n_classes=N_CLASSES)
    log.info(f"Loaded DESS baseline from {args.pretrained_ckpt}, wrapped with Unet2_5DFiLM")
    log.info(f"  FiLM gate weight norm at init: {model.position_weight_norm():.3e} (must be 0)")

    fake_train = PosMeniscus2_5DDataset(
        os.path.join(args.fake_data_root, "train", "images"),
        os.path.join(args.fake_data_root, "train", "masks"), vl["fake"], augment=True)
    fake_val = PosMeniscus2_5DDataset(
        os.path.join(args.fake_data_root, "val", "images"),
        os.path.join(args.fake_data_root, "val", "masks"), vl["fake"], augment=False)
    real_train = PosRealPDDataset(
        os.path.join(args.real_data_root, "train", "images"),
        os.path.join(args.real_data_root, "train", "masks"), vl["real"], augment=True)
    real_val = PosRealPDDataset(
        os.path.join(args.real_data_root, "val", "images"),
        os.path.join(args.real_data_root, "val", "masks"), vl["real"], augment=False)

    log.info(f"Fake PD — train: {len(fake_train)} slices  |  val: {len(fake_val)} slices")
    log.info(f"Real PD — train: {len(real_train)} slices  |  val: {len(real_val)} slices")

    fake_train_loader = DataLoader(fake_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
    fake_val_loader = DataLoader(fake_val, args.batch_size, shuffle=False, num_workers=args.num_workers)
    real_train_loader = DataLoader(real_train, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    real_val_loader = DataLoader(real_val, args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

    weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    fake_loss_fn = lambda logits, y: (nn.CrossEntropyLoss(weight=weights)(logits, y) + SoftDiceLoss()(logits, y))
    merged_loss_fn = MergedLoss(w_bg=0.1, w_men=1.5)

    val_pids = sorted({s.rsplit("_", 1)[0] for s in real_val.slices})
    hist_fh = open(os.path.join(args.out_dir, "history.csv"), "w", newline="")
    hist = csv.writer(hist_fh)
    hist.writerow(["epoch", "lr", "loss_tr", "loss_va", "dice_real_tr", "dice_real_va", "gap",
                   "dice_fake_tr", "dice_fake_va", "dice_va_perpatient_mean"]
                  + [f"dice_va_{p}" for p in val_pids] + ["pos_wnorm"])

    best_val_dice, no_improve = -1.0, 0
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, td, tdf = run_epoch(model, real_train_loader, fake_train_loader, optimizer,
                                merged_loss_fn, fake_loss_fn, device, train=True)
        vl_, vd, vdf = run_epoch(model, real_val_loader, fake_val_loader, optimizer,
                                 merged_loss_fn, fake_loss_fn, device, train=False)
        scheduler.step()

        pp = eval_real_per_patient(model, real_val, device, args.batch_size)
        pp_mean = float(np.mean(list(pp.values()))) if pp else float("nan")
        pp_str = "  ".join(f"{p}={pp.get(p, float('nan')):.4f}" for p in val_pids)
        wnorm = model.position_weight_norm()

        log.info(f"Epoch {epoch:03d}  lr={lr:.2e}  loss_tr={tl:.4f} loss_va={vl_:.4f}  |  "
                 f"dice_real tr={td:.4f} va={vd:.4f} gap={td - vd:+.4f}  |  "
                 f"dice_fake tr={tdf:.4f} va={vdf:.4f}  |  pos_wnorm={wnorm:.4e}")
        log.info(f"           per-patient val (clean pass): mean={pp_mean:.4f}   {pp_str}")

        hist.writerow([epoch, f"{lr:.3e}", f"{tl:.5f}", f"{vl_:.5f}", f"{td:.5f}", f"{vd:.5f}", f"{td - vd:+.5f}",
                       f"{tdf:.5f}", f"{vdf:.5f}", f"{pp_mean:.5f}"]
                      + [f"{pp.get(p, float('nan')):.5f}" for p in val_pids] + [f"{wnorm:.6e}"])
        hist_fh.flush()

        torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_latest.pth"))
        if vd > best_val_dice:
            best_val_dice, no_improve = vd, 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_best.pth"))
            log.info(f"  New best: {best_val_dice:.4f} (epoch {epoch}) — dumping val predictions")
            dump_val_predictions(model, real_val, device, os.path.join(args.out_dir, "val_predictions"), args.batch_size)
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                log.info(f"Early stopping at epoch {epoch}. Best: {best_val_dice:.4f}")
                break

    hist_fh.close()
    final_wnorm = model.position_weight_norm()
    log.info(f"Done. Best val Dice: {best_val_dice:.4f}")
    log.info(f"Final FiLM gate weight norm: {final_wnorm:.4e} (started at 0)")


if __name__ == "__main__":
    main()
