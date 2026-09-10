"""
diagnose_checkpoint.py
======================
Read-only diagnostic for an already-trained meniscus checkpoint.
Trains nothing, changes nothing, saves no weights.

Answers three questions that global Dice hides:

  1. Overfitting or underfitting?
       Compare dice_train vs dice_val on the SAME domain (real PD).
         train ~= val, both low   -> underfitting  (capacity/optimization limited)
         train >> val             -> overfitting   (memorizing the few real patients)
         train ~= val, both high  -> at the ceiling (label noise / task ambiguity)

  2. Is the gap domain gap or overfitting?
       Compare fake-PD Dice vs real-PD Dice for the same model.
       A model that scores well on fake PD and poorly on real PD is telling you
       the TRANSLATION (RegGAN) is the weak link, not the segmenter.

  3. Is a 2-patient validation set able to rank checkpoints at all?
       Bootstrap: repeatedly draw 2 patients at random from the real-PD patients
       and average their Dice. The spread of those means is the noise floor of
       your model-selection signal. If it is wider than the differences between
       your experiments, those experiments were never distinguishable.

Also reports FP vs FN volume, because Dice adds them together and shows one number.

Usage (BigRed):
  python segmentation/diagnose_checkpoint.py \
      --ckpt            segmentation_runs/run_v7_mixed/ckpt_best.pth \
      --fake_data_root  segmentation_data_v2 \
      --real_data_root  real_pd_seg_data_v7
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_2_5d_v2     import Meniscus2_5DDataset
from dataset_2_5d_realpd import RealPDDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES = 3


def load_finetuned(ckpt_path, device):
    """Fine-tuned checkpoints are plain 3-class state_dicts (head already swapped)."""
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=N_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    log.info(f"Loaded fine-tuned checkpoint: {ckpt_path}")
    return model.to(device).eval()


@torch.no_grad()
def accumulate(model, loader, device, tag):
    """
    Per-patient TP / pred / gt counts. Meniscus = (label > 0), matching
    dice_binary() in the training script so numbers are comparable.
    """
    per_patient = {}   # pid -> [tp, pred_sum, gt_sum]
    unknown = 0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        pred = model(x).argmax(1)

        pred_men = (pred > 0).float()
        gt_men   = (y    > 0).float()

        pids = batch.get("patient_id")
        if pids is None:
            pids = ["<unknown>"] * x.shape[0]
            unknown += x.shape[0]

        # per-sample so slices attribute to the right patient
        for b in range(x.shape[0]):
            pid = pids[b]
            tp = float((pred_men[b] * gt_men[b]).sum())
            ps = float(pred_men[b].sum())
            gs = float(gt_men[b].sum())
            acc = per_patient.setdefault(pid, [0.0, 0.0, 0.0])
            acc[0] += tp; acc[1] += ps; acc[2] += gs

    if unknown:
        log.warning(f"[{tag}] {unknown} slices had no patient_id — grouped as <unknown>")
    return per_patient


def summarize(per_patient, tag, eps=1e-6):
    tp  = sum(v[0] for v in per_patient.values())
    ps  = sum(v[1] for v in per_patient.values())
    gs  = sum(v[2] for v in per_patient.values())

    micro = (2 * tp + eps) / (ps + gs + eps)

    dices = {}
    for pid, (t, p, g) in per_patient.items():
        if g == 0 and p == 0:
            continue                      # no meniscus present or predicted
        dices[pid] = (2 * t + eps) / (p + g + eps)

    vals = np.array(list(dices.values()), dtype=np.float64)
    fn = gs - tp      # missed meniscus voxels
    fp = ps - tp      # hallucinated voxels

    print(f"\n--- {tag} ---")
    print(f"  patients                 : {len(dices)}")
    print(f"  Dice (micro, pooled)     : {micro:.4f}   <- matches training-loop convention")
    if len(vals):
        print(f"  Dice (mean per-patient)  : {vals.mean():.4f} +/- {vals.std():.4f}")
        print(f"  Dice (min / median / max): {vals.min():.4f} / {np.median(vals):.4f} / {vals.max():.4f}")
    print(f"  False NEG voxels (missed): {fn:12,.0f}   ({fn / (gs + eps) * 100:5.1f}% of GT)")
    print(f"  False POS voxels (extra) : {fp:12,.0f}   ({fp / (gs + eps) * 100:5.1f}% of GT)")

    if len(vals):
        print(f"  worst 5 patients:")
        for pid, d in sorted(dices.items(), key=lambda kv: kv[1])[:5]:
            print(f"      {pid:<24} {d:.4f}")
    return dices, micro


def bootstrap_small_val(dices, n_draw=2, n_iter=5000, seed=0):
    """
    How much does a val set of `n_draw` patients wobble? This is the noise floor
    of checkpoint selection — differences smaller than this were never real.
    """
    pids = list(dices.keys())
    if len(pids) < n_draw + 1:
        log.warning("Not enough patients to bootstrap.")
        return
    rng = np.random.default_rng(seed)
    vals = np.array([dices[p] for p in pids])
    means = np.array([rng.choice(vals, size=n_draw, replace=False).mean()
                      for _ in range(n_iter)])

    lo, hi = np.percentile(means, [2.5, 97.5])
    print(f"\n=== Bootstrap: what a {n_draw}-patient validation set looks like ===")
    print(f"  drawn from {len(pids)} real-PD patients, {n_iter} resamples")
    print(f"  mean of means : {means.mean():.4f}")
    print(f"  95% range     : {lo:.4f} .. {hi:.4f}   (width {hi - lo:.4f})")
    print(f"  std           : {means.std():.4f}")
    print(f"\n  -> Any experiment-to-experiment difference SMALLER than ~{means.std():.3f}")
    print(f"     is indistinguishable from which patients landed in your val set.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",           required=True)
    ap.add_argument("--fake_data_root", required=True)
    ap.add_argument("--real_data_root", required=True)
    ap.add_argument("--batch_size",     type=int, default=8)
    ap.add_argument("--num_workers",    type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_finetuned(args.ckpt, device)
    log.info(f"Device: {device}  |  model.eval(), no_grad, nothing is trained")

    splits = {
        "FAKE PD  train": Meniscus2_5DDataset(
            os.path.join(args.fake_data_root, "train", "images"),
            os.path.join(args.fake_data_root, "train", "masks"), augment=False),
        "FAKE PD  val":   Meniscus2_5DDataset(
            os.path.join(args.fake_data_root, "val", "images"),
            os.path.join(args.fake_data_root, "val", "masks"), augment=False),
        "REAL PD  train": RealPDDataset(
            os.path.join(args.real_data_root, "train", "images"),
            os.path.join(args.real_data_root, "train", "masks"), augment=False),
        "REAL PD  val":   RealPDDataset(
            os.path.join(args.real_data_root, "val", "images"),
            os.path.join(args.real_data_root, "val", "masks"), augment=False),
    }

    results = {}
    real_dices = {}
    for tag, ds in splits.items():
        loader = DataLoader(ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        per_patient = accumulate(model, loader, device, tag)
        dices, micro = summarize(per_patient, f"{tag}  ({len(ds)} slices)")
        results[tag] = micro
        if tag.startswith("REAL"):
            real_dices.update(dices)

    # ---- the three readings -------------------------------------------------
    print("\n" + "=" * 68)
    print("DIAGNOSIS")
    print("=" * 68)

    rt, rv = results["REAL PD  train"], results["REAL PD  val"]
    ft, fv = results["FAKE PD  train"], results["FAKE PD  val"]

    print(f"\n1. OVERFIT / UNDERFIT  (same domain: real PD)")
    print(f"     train {rt:.4f}   val {rv:.4f}   gap {rt - rv:+.4f}")
    if rt - rv > 0.15:
        print("     -> OVERFITTING. Model memorizes the few real training patients.")
        print("        More real data / stronger augmentation. NOT more capacity.")
    elif rt < 0.80:
        print("     -> UNDERFITTING or at ceiling. Model cannot fit even its training")
        print("        data. Regularizers (restricted CE, boundary, clDice) can only")
        print("        hurt here. Look at capacity, resolution, LR, or label quality.")
    else:
        print("     -> Fits training data well, small gap: near the achievable ceiling")
        print("        for this label quality. Chase labels/resolution, not losses.")

    print(f"\n2. DOMAIN GAP  (same model, two domains)")
    print(f"     fake-PD val {fv:.4f}   real-PD val {rv:.4f}   drop {fv - rv:+.4f}")
    print("     A large drop means RegGAN translation is the weak link, not the")
    print("     segmenter. NOTE: fake-PD masks are aligned by construction and")
    print("     noise-free; real-PD masks are human tracings. Part of this drop is")
    print("     annotation noise, not domain gap — do not attribute it all to RegGAN.")

    print(f"\n3. CAN A 2-PATIENT VAL SET RANK CHECKPOINTS?")
    bootstrap_small_val(real_dices, n_draw=2)

    print("\nReminder: 'micro' Dice pools all voxels (training-loop convention).")
    print("Your reported 0.78 is mean per-patient volume Dice. They are different")
    print("numbers and can move in different directions.\n")


if __name__ == "__main__":
    main()
