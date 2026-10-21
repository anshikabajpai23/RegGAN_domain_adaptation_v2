"""
finetune_meniscus_v14_tear_amodal.py
====================================
"Tear-amodal" fine-tuning on top of the replica recipe
(finetune_meniscus_v7_replica.py, seed 42 reference = 0.7780 on the 17pt test set).

Everything the replica does is kept: same starting checkpoint
(baseline_best_model.pth), same data, same fake/real two-stream loop, same
Adam + cosine schedule, same whole-wedge losses (imported from the replica
script, not copied), same checkpoint selection (`vd`, pooled real-val Dice).

Each v14 component is an independent flag:
  --tear_synth     paint synthetic tears inside the meniscus on TRAINING slices
                   (labels unchanged: the model must segment the whole wedge)   [R1]
  --visible_head   extra head predicting the visible (dark) meniscus           [R2]
  --contour_head   extra head predicting the outer contour band                 [R3]
  --inclusion      penalty when visible > whole (needs --visible_head)          [R4]
  (shape prior, R5: not implemented yet)
--visible_head / --inclusion only make sense with --tear_synth (without painted
tears, visible == whole). --contour_head works with or without it.

Held-out torn patients: --heldout_ids (default: the 5 torn-meniscus patients)
are checked against every training/validation slice at startup; the run aborts
if any of them is found. They are only for inference afterwards.

Extra diagnostic (logged, does NOT affect checkpoint selection): whole-wedge
Dice on the real val patients WITH fixed synthetic tears painted in
("dice_synthtear_va"). If tear-amodal training works, this should rise toward
the clean val Dice; for the plain replica it typically drops.
"""
import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_2_5d_v2 import Meniscus2_5DDataset
from dataset_2_5d_realpd import RealPDDataset
from dataset_2_5d_tear import TearAmodalDataset
from model_tear_amodal import build_from_dess_baseline, build_from_replica
# losses imported verbatim from the replica so the whole-wedge objective is byte-identical
from finetune_meniscus_v7_replica import MergedLoss, SoftDiceLoss, dice_binary, set_seed, PD_SPACING

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TORN_HELDOUT = ["AC13D01F99D88C", "AC116435D25299", "AC1CEC4C14F850",
                "AC152F0CF07CF3", "AC1A200F61EDC1"]


# ─────────────────────────────────────────────────────────────────────────────
# Extra v14 losses
# ─────────────────────────────────────────────────────────────────────────────

# [EXPLAIN] Dice loss for ONE binary mask (the replica's SoftDiceLoss is multi-class; the new heads are binary).
def soft_dice_binary(prob, target, eps=1e-6):
    tp = (prob * target).sum()
    return 1.0 - (2 * tp + eps) / (prob.sum() + target.sum() + eps)


# [EXPLAIN] ALL the NEW losses live here; the replica's own whole-wedge losses are NOT here (they are imported).
# [EXPLAIN] Returns 0 when no extra head is on, so R0 = replica exactly. Each term appears only if its head exists:
# [EXPLAIN]   R1  : nothing here (only tear painting changes the data)
# [EXPLAIN]   R1c : contour term only
# [EXPLAIN]   R2  : visible term
# [EXPLAIN]   R3  : visible + contour
# [EXPLAIN]   R4  : visible + contour + inclusion
def aux_losses(out, batch, args, device):
    """Returns (total_aux, {name: value}) for whichever heads/terms are enabled."""
    terms = {}
    # [EXPLAIN] VISIBLE term (R2, R3, R4): BCE + Dice of the visible head vs 'visible' mask, weight w_visible = 0.5.
    if "visible" in out:
        vis = batch["visible"].to(device).float()
        logit = out["visible"][:, 0]
        terms["visible"] = args.w_visible * (F.binary_cross_entropy_with_logits(logit, vis)
                                             + soft_dice_binary(torch.sigmoid(logit), vis))
    # [EXPLAIN] CONTOUR term (R1c, R3, R4): BCE of the contour head vs the edge band, weight w_contour = 0.5.
    # [EXPLAIN] pos_weight = 5 because the edge band is a tiny fraction of pixels (otherwise it predicts all zeros).
    if "contour" in out:
        con = batch["contour"].to(device).float()
        pos_w = torch.tensor(args.contour_pos_weight, device=device)
        terms["contour"] = args.w_contour * F.binary_cross_entropy_with_logits(
            out["contour"][:, 0], con, pos_weight=pos_w)
    # [EXPLAIN] INCLUSION term (R4 only): penalise pixels where visible-probability > whole-meniscus-probability,
    # [EXPLAIN] i.e. 'the visible part must lie INSIDE the whole wedge'. relu(p_vis - p_whole) is 0 when that holds.
    # [EXPLAIN] This is what ties the two heads together so (whole - visible) is a sensible tear map.
    if args.inclusion and "visible" in out:
        p_whole = torch.softmax(out["whole"], dim=1)
        p_men = p_whole[:, 1] + p_whole[:, 2]
        p_vis = torch.sigmoid(out["visible"][:, 0])
        terms["inclusion"] = args.w_inclusion * F.relu(p_vis - p_men).mean()
    total = sum(terms.values()) if terms else torch.zeros((), device=device)
    return total, terms


# [EXPLAIN] REPLICA run_epoch: model(xf) -> logits; loss = fake_loss + real_loss. Fake stream drives the epoch and the
# [EXPLAIN] real stream cycles. THIS version is the same loop, with two differences: (1) model returns a dict, so the
# [EXPLAIN] whole-wedge losses read out['whole']; (2) aux losses (above) are added to the total.
def run_epoch(model, real_loader, fake_loader, optimizer, merged_loss_fn, fake_loss_fn,
              args, device, train=True):
    """Replica loop (fake stream drives, real stream cycles); whole-wedge losses
    unchanged, plus the enabled aux losses on BOTH streams."""
    model.train() if train else model.eval()
    total_loss, dice_sum, dice_fake_sum, n = 0.0, 0.0, 0.0, 0
    aux_sum = {}
    real_iter = iter(real_loader)

    with torch.set_grad_enabled(train):
        for fake_batch in fake_loader:
            xf, yf = fake_batch["image"].to(device), fake_batch["mask"].to(device)
            # [EXPLAIN] Forward pass on the fake-PD batch (may contain painted tears if --tear_synth).
            out_f = model(xf)
            # [EXPLAIN] UNCHANGED replica loss (weighted CE + SoftDice) on the whole-wedge head, against the ORIGINAL mask.
            loss_f = fake_loss_fn(out_f["whole"], yf)
            # [EXPLAIN] NEW: extra-head losses for the fake batch (zero for R0/R1).
            aux_f, terms_f = aux_losses(out_f, fake_batch, args, device)

            try:
                real_batch = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                real_batch = next(real_iter)
            xr, yr = real_batch["image"].to(device), real_batch["mask"].to(device)
            out_r = model(xr)
            # [EXPLAIN] UNCHANGED replica loss (MergedLoss, binary) for the real-PD batch.
            loss_r = merged_loss_fn(out_r["whole"], yr)
            aux_r, terms_r = aux_losses(out_r, real_batch, args, device)

            # [EXPLAIN] TOTAL = replica loss (fake + real) + new extra-head losses. For R1 the last two are 0, so R1's loss
            # [EXPLAIN] equals the replica's; only the INPUT images differ (painted tears).
            loss = loss_f + loss_r + aux_f + aux_r
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            total_loss += loss.item()
            for k in set(terms_f) | set(terms_r):
                aux_sum[k] = aux_sum.get(k, 0.0) + sum(float(t[k].detach()) for t in (terms_f, terms_r) if k in t)
            dice_sum += dice_binary(out_r["whole"].argmax(1), yr)
            dice_fake_sum += dice_binary(out_f["whole"].argmax(1), yf)
            n += 1

    n = max(n, 1)
    return total_loss / n, dice_sum / n, dice_fake_sum / n, {k: v / n for k, v in aux_sum.items()}


@torch.no_grad()
def eval_real_per_patient(model, dataset, device, batch_size=8):
    """Replica's clean per-patient pass (each val slice once), on the whole head."""
    model.eval()
    acc = {}
    for start in range(0, len(dataset), batch_size):
        idxs = range(start, min(start + batch_size, len(dataset)))
        items = [dataset[i] for i in idxs]
        stems = [dataset.slices[i] for i in idxs]
        x = torch.stack([it["image"] for it in items]).to(device)
        y = torch.stack([it["mask"] for it in items]).to(device)
        pred = model(x)["whole"].argmax(1)
        pm, gm = (pred > 0).float(), (y > 0).float()
        for b, stem in enumerate(stems):
            pid = stem.rsplit("_", 1)[0]
            a = acc.setdefault(pid, [0.0, 0.0, 0.0])
            a[0] += float((pm[b] * gm[b]).sum()); a[1] += float(pm[b].sum()); a[2] += float(gm[b].sum())
    eps = 1e-6
    return {pid: (2 * tp + eps) / (ps + gs + eps) for pid, (tp, ps, gs) in acc.items()}


@torch.no_grad()
def dump_val_predictions(model, dataset, device, out_dir, batch_size=8):
    """Image / GT / whole prediction (+ visible, + tear map if available) per val patient."""
    import nibabel as nib
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    vols = {}
    for start in range(0, len(dataset), batch_size):
        idxs = range(start, min(start + batch_size, len(dataset)))
        items = [dataset[i] for i in idxs]
        stems = [dataset.slices[i] for i in idxs]
        x = torch.stack([it["image"] for it in items]).to(device)
        y = torch.stack([it["mask"] for it in items])
        out = model(x)
        pred = out["whole"].argmax(1).cpu()
        vis = (torch.sigmoid(out["visible"][:, 0]) > 0.5).cpu() if "visible" in out else None
        for b, stem in enumerate(stems):
            pid, sidx = stem.rsplit("_", 1)
            vols.setdefault(pid, {})[int(sidx)] = (
                x[b, 1].cpu().numpy().astype(np.float32), y[b].numpy().astype(np.int16),
                pred[b].numpy().astype(np.int16),
                vis[b].numpy().astype(np.int16) if vis is not None else None)
    affine = np.diag([PD_SPACING[0], PD_SPACING[1], PD_SPACING[2], 1.0]).astype(np.float32)
    for pid, slices in sorted(vols.items()):
        order = sorted(slices)
        img = np.stack([slices[s][0] for s in order]); gt = np.stack([slices[s][1] for s in order])
        pr = np.stack([slices[s][2] for s in order])
        nib.save(nib.Nifti1Image(img, affine), os.path.join(out_dir, f"{pid}_image.nii.gz"))
        nib.save(nib.Nifti1Image((gt > 0).astype(np.int16), affine), os.path.join(out_dir, f"{pid}_gt.nii.gz"))
        nib.save(nib.Nifti1Image((pr > 0).astype(np.int16), affine), os.path.join(out_dir, f"{pid}_pred.nii.gz"))
        if slices[order[0]][3] is not None:
            vi = np.stack([slices[s][3] for s in order])
            nib.save(nib.Nifti1Image(vi, affine), os.path.join(out_dir, f"{pid}_visible.nii.gz"))
    log.info(f"  dumped {len(vols)} val patients -> {out_dir}")


# [EXPLAIN] Safety: abort if any of the 5 torn patients is in the training/validation lists (they are for inference only).
def check_heldout(heldout, fake_train, fake_val, real_train, real_val):
    """Abort if any held-out (torn) patient appears anywhere in train/val."""
    seen = {}
    for name, ds in (("fake_train", fake_train), ("fake_val", fake_val)):
        seen[name] = {pid for pid, _ in ds.items}
    for name, ds in (("real_train", real_train), ("real_val", real_val)):
        seen[name] = {s.rsplit("_", 1)[0] for s in ds.slices}
    leaks = [(h, name) for h in heldout for name, pids in seen.items()
             if any(h in p for p in pids)]
    if leaks:
        raise RuntimeError(f"HELD-OUT LEAK: {leaks} -- held-out patients must never be trained on.")
    log.info(f"Held-out check passed: none of {heldout} appear in fake/real train/val "
             f"({sum(len(v) for v in seen.values())} patient IDs checked).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True,
                    help="baseline_best_model.pth (5-class DESS model), same start as the replica")
    ap.add_argument("--init_from_replica", default=None,
                    help="optional: warm start from a trained replica ckpt instead (faster pilot, "
                         "not a clean one-change comparison)")
    ap.add_argument("--fake_data_root", required=True)
    ap.add_argument("--real_data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--class_weights", type=float, nargs=3, default=[0.1, 1.5, 1.5])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_batches", type=int, default=0,
                    help="debug only: stop each epoch after N fake batches (0 = full epoch)")
    # v14 components (independent flags)
    ap.add_argument("--tear_synth", action="store_true")
    ap.add_argument("--tear_p", type=float, default=0.5, help="probability a meniscus slice gets a tear")
    ap.add_argument("--tear_neighbor_p", type=float, default=0.6)
    ap.add_argument("--visible_head", action="store_true")
    ap.add_argument("--contour_head", action="store_true")
    ap.add_argument("--inclusion", action="store_true")
    ap.add_argument("--w_visible", type=float, default=0.5)
    ap.add_argument("--w_contour", type=float, default=0.5)
    ap.add_argument("--w_inclusion", type=float, default=0.1)
    ap.add_argument("--contour_pos_weight", type=float, default=5.0)
    ap.add_argument("--heldout_ids", nargs="*", default=TORN_HELDOUT)
    args = ap.parse_args()

    if args.inclusion and not args.visible_head:
        ap.error("--inclusion needs --visible_head")
    if args.visible_head and not args.tear_synth:
        log.warning("--visible_head without --tear_synth: visible == whole, the head learns nothing new")

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comps = [n for n, on in (("tear_synth", args.tear_synth), ("visible_head", args.visible_head),
                             ("contour_head", args.contour_head), ("inclusion", args.inclusion)) if on]
    log.info(f"Device: {device} | v14 components: {comps or 'NONE (= replica)'}")

    build = build_from_replica if args.init_from_replica else build_from_dess_baseline
    # [EXPLAIN] Model: replica start (DESS baseline), plus the extra heads chosen by --visible_head / --contour_head.
    model = build(args.init_from_replica or args.pretrained_ckpt, device,
                  visible_head=args.visible_head, contour_head=args.contour_head)
    log.info(f"Model init from {'replica (warm start)' if args.init_from_replica else 'DESS baseline'}")

    fake_train_b = Meniscus2_5DDataset(os.path.join(args.fake_data_root, "train", "images"),
                                       os.path.join(args.fake_data_root, "train", "masks"), augment=True)
    fake_val_b = Meniscus2_5DDataset(os.path.join(args.fake_data_root, "val", "images"),
                                     os.path.join(args.fake_data_root, "val", "masks"), augment=False)
    real_train_b = RealPDDataset(os.path.join(args.real_data_root, "train", "images"),
                                 os.path.join(args.real_data_root, "train", "masks"), augment=True)
    real_val_b = RealPDDataset(os.path.join(args.real_data_root, "val", "images"),
                               os.path.join(args.real_data_root, "val", "masks"), augment=False)
    check_heldout(args.heldout_ids, fake_train_b, fake_val_b, real_train_b, real_val_b)

    # tears only on TRAINING sets; val stays clean so ckpt selection matches the replica
    # [EXPLAIN] TRAIN sets: synth=args.tear_synth. This is the line that turns R1 on: tears are painted on training slices.
    fake_train = TearAmodalDataset(fake_train_b, synth=args.tear_synth, p=args.tear_p, neighbor_p=args.tear_neighbor_p)
    real_train = TearAmodalDataset(real_train_b, synth=args.tear_synth, p=args.tear_p, neighbor_p=args.tear_neighbor_p)
    # [EXPLAIN] VAL sets: never painted, so ckpt selection and 'clean val' match the replica's numbers.
    fake_val = TearAmodalDataset(fake_val_b, synth=False)
    real_val = TearAmodalDataset(real_val_b, synth=False)
    # diagnostic only: real val with a fixed synthetic tear on every meniscus slice
    # [EXPLAIN] DIAGNOSTIC only: real val patients with a FIXED painted tear on every slice -> 'dice_synthtear_va' in
    # [EXPLAIN] history.csv. It never affects which checkpoint is saved.
    real_val_synthtear = TearAmodalDataset(real_val_b, synth=True, p=1.0,
                                           neighbor_p=args.tear_neighbor_p, fixed_seed=12345)
    log.info(f"Fake PD train {len(fake_train)} / val {len(fake_val)} slices | "
             f"Real PD train {len(real_train)} / val {len(real_val)} slices")

    dl = lambda ds, shuffle: DataLoader(ds, args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
    fake_train_loader, fake_val_loader = dl(fake_train, True), dl(fake_val, False)
    real_train_loader, real_val_loader = dl(real_train, True), dl(real_val, False)
    if args.max_batches:
        from itertools import islice

        class _Cap:
            def __init__(self, loader, n): self.loader, self.n = loader, n
            def __iter__(self): return islice(iter(self.loader), self.n)
        fake_train_loader, fake_val_loader = _Cap(fake_train_loader, args.max_batches), _Cap(fake_val_loader, args.max_batches)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)
    weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    fake_loss_fn = lambda logits, y: nn.CrossEntropyLoss(weight=weights)(logits, y) + SoftDiceLoss()(logits, y)
    merged_loss_fn = MergedLoss(w_bg=0.1, w_men=1.5)

    val_pids = sorted({s.rsplit("_", 1)[0] for s in real_val.slices})
    aux_names = [k for k, on in (("visible", args.visible_head), ("contour", args.contour_head),
                                 ("inclusion", args.inclusion)) if on]
    hist_fh = open(os.path.join(args.out_dir, "history.csv"), "w", newline="")
    hist = csv.writer(hist_fh)
    hist.writerow(["epoch", "lr", "loss_tr", "loss_va", "dice_real_tr", "dice_real_va", "gap",
                   "dice_fake_tr", "dice_fake_va", "dice_va_perpatient_mean", "dice_synthtear_va"]
                  + [f"dice_va_{p}" for p in val_pids] + [f"aux_{k}_tr" for k in aux_names])

    best, no_improve = -1.0, 0
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, td, tdf, taux = run_epoch(model, real_train_loader, fake_train_loader, optimizer,
                                      merged_loss_fn, fake_loss_fn, args, device, train=True)
        vl, vd, vdf, _ = run_epoch(model, real_val_loader, fake_val_loader, optimizer,
                                   merged_loss_fn, fake_loss_fn, args, device, train=False)
        scheduler.step()
        pp = eval_real_per_patient(model, real_val, device, args.batch_size)
        pp_mean = float(np.mean(list(pp.values()))) if pp else float("nan")
        st = eval_real_per_patient(model, real_val_synthtear, device, args.batch_size)
        st_mean = float(np.mean(list(st.values()))) if st else float("nan")

        log.info(f"Epoch {epoch:03d} lr={lr:.2e} loss tr={tl:.4f} va={vl:.4f} | dice_real tr={td:.4f} "
                 f"va={vd:.4f} | dice_fake tr={tdf:.4f} va={vdf:.4f} | clean val={pp_mean:.4f} "
                 f"| synth-tear val={st_mean:.4f} | aux {', '.join(f'{k}={v:.4f}' for k, v in taux.items())}")
        hist.writerow([epoch, f"{lr:.3e}", f"{tl:.5f}", f"{vl:.5f}", f"{td:.5f}", f"{vd:.5f}",
                       f"{td - vd:+.5f}", f"{tdf:.5f}", f"{vdf:.5f}", f"{pp_mean:.5f}", f"{st_mean:.5f}"]
                      + [f"{pp.get(p, float('nan')):.5f}" for p in val_pids]
                      + [f"{taux.get(k, float('nan')):.5f}" for k in aux_names])
        hist_fh.flush()

        torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_latest.pth"))
        # [EXPLAIN] Checkpoint selection is the replica's: pooled real-val Dice of the WHOLE head on clean val slices.
        if vd > best:   # same selection criterion as the replica
            best, no_improve = vd, 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_best.pth"))
            log.info(f"  New best: {best:.4f} (epoch {epoch})")
            dump_val_predictions(model, real_val, device, os.path.join(args.out_dir, "val_predictions"),
                                 args.batch_size)
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                log.info(f"Early stopping at epoch {epoch}. Best: {best:.4f}")
                break

    hist_fh.close()
    log.info(f"Done. Best val Dice: {best:.4f} -> {args.out_dir}")


if __name__ == "__main__":
    main()
