"""
finetune_meniscus_fakeonly_e2.py
=================================
Run 3, E2 — slice position via FiLM gating at the bottleneck, FAKE PD ONLY.

BASE: finetune_meniscus_fakeonly_base.py, imported directly, so this arm cannot
drift from the baseline it is measured against. The ONLY changes are the
zero-init FiLM wrapper (model_2_5d_film.Unet2_5DFiLM, proven bit-identical to
the plain model at step 0, max diff 0.000e+00) and the `pos` field the datasets
attach. FiLM is applied ONCE to the single bottleneck feature map, conditioned
on the centre slice's own position -- injecting position where the network has
semantic features rather than at the raw input (that is E1).

    pos = 2 * |idx/(n-1) - 0.5|   0 at the volume centre (notch), 1 at the outer edges

WHY THIS IS A BETTER TEST HERE THAN IN RUN 2
--------------------------------------------
In the mixed regime the model had already seen real PD, so slice position was
information it could read off the image directly -- E2's FiLM gate weight norm barely moved off zero. With no real PD in training, appearance is the thing that shifts
across the domain gap while POSITION DOES NOT: it is a geometric property of
the volume, identical in fake and real. A domain-invariant cue is worth more
exactly when the appearance cue is unreliable, which is the case this arm
creates. The FiLM gate weight norm (logged every epoch as pos_wnorm) is the direct
readout of whether the model actually took the cue this time.

volume_lengths.json is reused UNCHANGED (scripts/build_volume_lengths.py). Only
its "fake" block feeds training. Its "real" block is used solely to compute pos
for the DIAGNOSTIC real-val pass -- volume geometry, not a label, and never
touches selection or gradients.
"""
import argparse
import csv
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_2_5d_v2 import Meniscus2_5DDataset
from dataset_2_5d_realpd import RealPDDataset
from model_2_5d_film import build_2_5d_film_model, verify_zero_init
from finetune_meniscus_fakeonly_base import (
    N_CLASSES, PRETRAINED_N_CLASSES, PD_SPACING, set_seed, SoftDiceLoss,
    dice_binary, add_common_args, history_header,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def pos_of(idx, n):
    return 2.0 * abs(idx / max(n - 1, 1) - 0.5)


class PosMeniscus2_5DDataset(Meniscus2_5DDataset):
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
    """DIAGNOSTIC ONLY."""
    def __init__(self, img_root, mask_root, lengths, augment=False):
        super().__init__(img_root, mask_root, augment=augment)
        self.lengths = lengths
        missing = {s.rsplit("_", 1)[0] for s in self.slices} - set(lengths)
        if missing:
            raise SystemExit(f"volume_lengths.json missing real patients: {sorted(missing)}")

    def __getitem__(self, i):
        item = super().__getitem__(i)
        pid, sidx = self.slices[i].rsplit("_", 1)
        item["pos"] = torch.tensor(pos_of(int(sidx), self.lengths[pid]), dtype=torch.float32)
        return item


def run_epoch(model, fake_loader, optimizer, fake_loss_fn, device, train=True):
    model.train() if train else model.eval()
    total_loss, dice_fake_sum, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in fake_loader:
            x, y = batch["image"].to(device), batch["mask"].to(device)
            p = batch["pos"].to(device)
            logits = model(x, p)
            loss = fake_loss_fn(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            dice_fake_sum += dice_binary(logits.argmax(1), y)
            n += 1
    n = max(n, 1)
    return total_loss / n, dice_fake_sum / n


@torch.no_grad()
def eval_real_per_patient_pos(model, dataset, device, batch_size=8):
    """DIAGNOSTIC ONLY. Position-aware variant of the base's real-val eval."""
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
def dump_val_predictions_pos(model, dataset, device, out_dir, batch_size=8):
    try:
        import nibabel as nib
    except ImportError:
        log.warning("nibabel not available -- skipping val prediction dump"); return
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
    log.info(f"  per-slice Dice CSV -> {csv_path}")


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--volume_lengths", required=True)
    ap.add_argument("--skip_zero_init_check", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  RUN 3 E2: FiLM gating  |  FAKE PD ONLY")

    if not args.skip_zero_init_check:
        ok, diff = verify_zero_init(device="cpu")
        if not ok:
            raise SystemExit(f"ZERO-INIT CHECK FAILED (max diff {diff:.3e}). Refusing to train.")

    with open(args.volume_lengths) as fh:
        vl = json.load(fh)
    m = vl.get("meta", {})
    log.info(f"volume_lengths: {m.get('n_fake')} fake, {m.get('n_real')} real")

    model = build_2_5d_film_model(args.pretrained_ckpt, device,
                                 pretrained_n_classes=PRETRAINED_N_CLASSES, n_classes=N_CLASSES)
    log.info(f"  FiLM gate weight norm at init: {model.position_weight_norm():.3e} (must be 0)")

    fake_train = PosMeniscus2_5DDataset(
        os.path.join(args.fake_data_root, "train", "images"),
        os.path.join(args.fake_data_root, "train", "masks"), vl["fake"], augment=True)
    fake_val = PosMeniscus2_5DDataset(
        os.path.join(args.fake_data_root, "val", "images"),
        os.path.join(args.fake_data_root, "val", "masks"), vl["fake"], augment=False)
    real_val = None
    if args.real_data_root:
        real_val = PosRealPDDataset(
            os.path.join(args.real_data_root, "val", "images"),
            os.path.join(args.real_data_root, "val", "masks"), vl["real"], augment=False)
        log.info(f"Real PD - val ONLY (diagnostic, never trained on): {len(real_val)} slices")
    else:
        log.info("Real PD - NOT LOADED. This run never reads a real-PD label.")
    log.info(f"Fake PD - train: {len(fake_train)} slices  |  val: {len(fake_val)} slices")

    fake_train_loader = DataLoader(fake_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
    fake_val_loader = DataLoader(fake_val, args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

    weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    fake_loss_fn = lambda logits, y: (nn.CrossEntropyLoss(weight=weights)(logits, y) + SoftDiceLoss()(logits, y))

    val_pids = sorted({s.rsplit("_", 1)[0] for s in real_val.slices}) if real_val is not None else []
    hist_fh = open(os.path.join(args.out_dir, "history.csv"), "w", newline="")
    hist = csv.writer(hist_fh)
    hist.writerow(history_header(real_val, val_pids, extra_cols=["pos_wnorm"]))

    best_smooth, no_improve, va_hist = -1.0, 0, []
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, tdf = run_epoch(model, fake_train_loader, optimizer, fake_loss_fn, device, train=True)
        vl_, vdf = run_epoch(model, fake_val_loader, optimizer, fake_loss_fn, device, train=False)
        scheduler.step()

        va_hist.append(vdf)
        smooth = float(np.mean(va_hist[-args.select_smooth:]))
        wnorm = model.position_weight_norm()
        row = [epoch, f"{lr:.3e}", f"{tl:.5f}", f"{vl_:.5f}",
               f"{tdf:.5f}", f"{vdf:.5f}", f"{tdf - vdf:+.5f}", f"{smooth:.5f}"]
        msg = (f"Epoch {epoch:03d}  lr={lr:.2e}  loss_tr={tl:.4f} loss_va={vl_:.4f}  |  "
               f"dice_fake tr={tdf:.4f} va={vdf:.4f} smooth={smooth:.4f}  |  pos_wnorm={wnorm:.4e}")

        if real_val is not None:
            pp = eval_real_per_patient_pos(model, real_val, device, args.batch_size)
            pp_mean = float(np.mean(list(pp.values()))) if pp else float("nan")
            row += [f"{pp_mean:.5f}"] + [f"{pp.get(p, float('nan')):.5f}" for p in val_pids]
            msg += f"  |  [diag] real_va={pp_mean:.4f}"
        row += [f"{wnorm:.6e}"]
        log.info(msg)

        hist.writerow(row); hist_fh.flush()

        torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_latest.pth"))
        if smooth > best_smooth:
            best_smooth, no_improve = smooth, 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_best.pth"))
            log.info(f"  New best (smoothed fake val): {best_smooth:.4f} (epoch {epoch})")
            if real_val is not None:
                dump_val_predictions_pos(model, real_val, device,
                                         os.path.join(args.out_dir, "val_predictions"), args.batch_size)
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                log.info(f"Early stopping at epoch {epoch}. Best smoothed: {best_smooth:.4f}")
                break

    hist_fh.close()
    log.info(f"Done. Best smoothed fake val Dice: {best_smooth:.4f}")
    log.info(f"Final FiLM gate weight norm: {model.position_weight_norm():.4e} (started at 0)")


if __name__ == "__main__":
    main()
