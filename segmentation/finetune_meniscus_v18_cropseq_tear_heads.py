"""
finetune_meniscus_v18_cropseq_tear_heads.py
===========================================
v18 = v16 (crop + sequence model, v17 pattern set incl. Fig. 10a, native-aspect geometry) plus two
extra outputs supervised ONLY by the synthetic tears (no real tear labels exist):
  * tear map  : where the tear is (BCE + soft Dice, weight --w_tear)
  * pattern   : which pattern / variant, incl. "none" (cross-entropy, weight --w_cls); classes =
                dataset_2_5d_tear_patterns.PATTERN_CLASSES. p(torn) = 1 - p(none).
Healthy slices (the 1 - pattern_p share, and slices without meniscus) are the negatives: empty tear
map, class "none".

Segmentation part is exactly v16: model trunk, data, losses, schedule, seed, and checkpoint
selection on CLEAN real-val segmentation Dice (so v18's meniscus output is comparable with v16).
The extra heads never influence checkpoint selection.

history.csv diagnostics (synthetic val = the 2 real val patients with a fixed pattern of each
mode / in-place variant painted in; clean val = the same slices untouched):
  dice_synth_<set>_va    meniscus Dice on that synthetic set
  tear_dice_<set>_va     tear-map Dice (pooled) on that set
  acc_<set>_va           pattern accuracy at MODE level (inplace variants count as "inplace")
  clean_fp_rate_va       share of clean val meniscus slices called torn (p(torn) > 0.5)
  torn_auc_va            AUC of p(torn): all synthetic val slices vs clean val meniscus slices
Caveat: these measure learning of OUR synthetic tears. Real-tear detection can only be judged on
real scans (torn 5 vs healthy 17, knee level), which infer_sequence_v18_tear.py writes out.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_sequence import SequenceMeniscusDataset, SequenceRealPDDataset
from dataset_sequence_tear_patterns import SequenceTearPatternDataset
from dataset_2_5d_tear_patterns import PATTERN_CLASSES
from sequence_model_tear import build_sequence_tear_model, SegOnly
from tear_patterns import MODES, MODE_P
from tear_synth import PATTERNS_V17, PATTERN_P_V17
from synthetic_log_utils import process_epoch
from finetune_meniscus_v7_sequence import (MergedLoss, SoftDiceLoss, set_seed, dice_binary,
                                           dump_val_predictions)
from finetune_meniscus_v14_tear_amodal import check_heldout, TORN_HELDOUT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MODE_OF_CLASS = np.array([c.split("_")[0] if c.startswith("inplace_") else c for c in PATTERN_CLASSES])


def aux_loss(out, batch, args, device):
    tear = batch["tear"].to(device).float()[:, None]
    logit = out["tear"]
    bce = F.binary_cross_entropy_with_logits(logit, tear, pos_weight=torch.tensor(args.tear_pos_weight, device=device))
    p = torch.sigmoid(logit)
    sdice = 1 - (2 * (p * tear).sum() + 1) / (p.sum() + tear.sum() + 1)
    ce = F.cross_entropy(out["cls"], batch["pattern_id"].to(device).long())
    return args.w_tear * (bce + sdice) + args.w_cls * ce


def run_epoch(model, real_loader, fake_loader, optimizer, merged_loss_fn, fake_loss_fn, args, device, train):
    model.train() if train else model.eval()
    tot, dsum, dfsum, n = 0.0, 0.0, 0.0, 0
    real_iter = iter(real_loader)
    with torch.set_grad_enabled(train):
        for fb in fake_loader:
            xf, yf = fb["image"].to(device), fb["mask"].to(device)
            of = model(xf)
            loss_f = fake_loss_fn(of["seg"], yf) + aux_loss(of, fb, args, device)
            try:
                rb = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                rb = next(real_iter)
            xr, yr = rb["image"].to(device), rb["mask"].to(device)
            orr = model(xr)
            loss_r = merged_loss_fn(orr["seg"], yr) + aux_loss(orr, rb, args, device)
            loss = loss_f + loss_r
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot += loss.item()
            dsum += dice_binary(orr["seg"].argmax(1), yr)
            dfsum += dice_binary(of["seg"].argmax(1), yf)
            n += 1
    return tot / max(n, 1), dsum / max(n, 1), dfsum / max(n, 1)


@torch.no_grad()
def evaluate_set(model, ds, device, bs=8):
    """One pass: per-patient meniscus Dice, pooled tear Dice, mode-level accuracy, p(torn) per slice."""
    model.eval()
    acc, tp_t, ps_t, gs_t, correct, total, ptorn, has_men = {}, 0.0, 0.0, 0.0, 0, 0, [], []
    for start in range(0, len(ds), bs):
        idx = range(start, min(start + bs, len(ds)))
        items = [ds[i] for i in idx]
        x = torch.stack([it["image"] for it in items]).to(device)
        y = torch.stack([it["mask"] for it in items]).to(device)
        out = model(x)
        pm, gm = (out["seg"].argmax(1) > 0).float(), (y > 0).float()
        for b, i in enumerate(idx):
            a = acc.setdefault(ds.slices[i].rsplit("_", 1)[0], [0.0, 0.0, 0.0])
            a[0] += float((pm[b] * gm[b]).sum()); a[1] += float(pm[b].sum()); a[2] += float(gm[b].sum())
        if "tear" in items[0]:
            t = torch.stack([it["tear"] for it in items]).to(device).float()
            pt = (torch.sigmoid(out["tear"][:, 0]) > 0.5).float()
            tp_t += float((pt * t).sum()); ps_t += float(pt.sum()); gs_t += float(t.sum())
            prob = torch.softmax(out["cls"], 1).cpu().numpy()
            pred_mode = MODE_OF_CLASS[prob.argmax(1)]
            true_mode = MODE_OF_CLASS[[int(it["pattern_id"]) for it in items]]
            correct += int((pred_mode == true_mode).sum()); total += len(items)
            ptorn += list(1 - prob[:, 0]); has_men += [bool(it["mask"].any()) for it in items]
    dice = float(np.mean([(2 * a[0] + 1e-6) / (a[1] + a[2] + 1e-6) for a in acc.values()])) if acc else float("nan")
    return {"dice": dice, "tear_dice": (2 * tp_t + 1e-6) / (ps_t + gs_t + 1e-6) if gs_t else float("nan"),
            "acc": correct / total if total else float("nan"), "ptorn": np.array(ptorn), "has_men": np.array(has_men)}


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg]); r = s.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--init_from", default=None, help="optional: warm-start the trunk from a v16 ckpt_best.pth")
    ap.add_argument("--fake_data_root", required=True)
    ap.add_argument("--real_data_root", required=True)
    ap.add_argument("--box_json", required=True)
    ap.add_argument("--no_native_fix", action="store_true")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--class_weights", type=float, nargs=3, default=[0.1, 1.5, 1.5])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--pattern_p", type=float, default=0.5)
    ap.add_argument("--w_tear", type=float, default=0.5)
    ap.add_argument("--w_cls", type=float, default=0.2)
    ap.add_argument("--tear_pos_weight", type=float, default=5.0)
    ap.add_argument("--heldout_ids", nargs="*", default=TORN_HELDOUT)
    ap.add_argument("--log_epochs", type=int, nargs="*", default=[0])
    ap.add_argument("--log_samples_per_mode", type=int, default=100)
    args = ap.parse_args()
    modes = list(MODES)

    box = json.load(open(args.box_json))
    native_hw = None if args.no_native_fix else (box["y1"] - box["y0"], box["x1"] - box["x0"])
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump({"pattern_classes": list(PATTERN_CLASSES), "native_hw": native_hw, **vars(args)},
              open(os.path.join(args.out_dir, "v18_config.json"), "w"), indent=2)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | v18 crop+seq + tear map + pattern class ({len(PATTERN_CLASSES)} classes) "
             f"| w_tear={args.w_tear} w_cls={args.w_cls} | native-aspect: {native_hw}")

    model = build_sequence_tear_model(args.pretrained_ckpt, device, len(PATTERN_CLASSES), args.init_from)
    j = lambda root, split, kind: os.path.join(root, split, kind)
    fake_train_b = SequenceMeniscusDataset(j(args.fake_data_root, "train", "images"), j(args.fake_data_root, "train", "masks"), augment=True)
    fake_val_b = SequenceMeniscusDataset(j(args.fake_data_root, "val", "images"), j(args.fake_data_root, "val", "masks"), augment=False)
    real_train_b = SequenceRealPDDataset(j(args.real_data_root, "train", "images"), j(args.real_data_root, "train", "masks"), augment=True)
    real_val_b = SequenceRealPDDataset(j(args.real_data_root, "val", "images"), j(args.real_data_root, "val", "masks"), augment=False)
    check_heldout(args.heldout_ids, fake_train_b, fake_val_b, real_train_b, real_val_b)

    log_dir = os.path.join(args.out_dir, "synthetic_log") if args.log_epochs else None
    v17 = dict(inplace_variants=PATTERNS_V17, inplace_p=PATTERN_P_V17, return_tear=True)
    wrap = lambda b, synth, **kw: SequenceTearPatternDataset(b, native_hw=native_hw, synth=synth, p=args.pattern_p,
                                                             modes=modes, mode_p=MODE_P, **v17, **kw)
    fake_train = wrap(fake_train_b, True, source="fake", log_dir=log_dir, save_per_mode=args.log_samples_per_mode, log_variants=True)
    real_train = wrap(real_train_b, True, source="real", log_dir=log_dir, save_per_mode=args.log_samples_per_mode, log_variants=True)
    fake_val, real_val = wrap(fake_val_b, False), wrap(real_val_b, False)
    sv = lambda **kw: SequenceTearPatternDataset(real_val_b, native_hw=native_hw, synth=True, p=1.0, fixed_seed=12345,
                                                 return_tear=True, **kw)
    synth_val = {m: sv(modes=(m,), mode_p=(1.0,), inplace_variants=PATTERNS_V17, inplace_p=PATTERN_P_V17) for m in modes}
    clean_val_t = SequenceTearPatternDataset(real_val_b, native_hw=native_hw, synth=False, return_tear=True)
    for v in PATTERNS_V17:
        synth_val[f"inplace_{v}"] = sv(modes=("inplace",), mode_p=(1.0,), inplace_variants=(v,), inplace_p=(1.0,))
    log.info(f"Fake PD train {len(fake_train)} / val {len(fake_val)} | Real PD train {len(real_train)} / val {len(real_val)}")

    dl = lambda ds, sh: DataLoader(ds, args.batch_size, shuffle=sh, num_workers=args.num_workers)
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
    hist_fh = open(os.path.join(args.out_dir, "history.csv"), "w", newline="")
    hist = csv.writer(hist_fh)
    hist.writerow(["epoch", "lr", "loss_tr", "loss_va", "dice_real_tr", "dice_real_va", "gap", "dice_fake_tr",
                   "dice_fake_va", "dice_va_perpatient_mean", "clean_fp_rate_va", "torn_auc_va"]
                  + [f"{k}_{m}_va" for m in synth_val for k in ("dice_synth", "tear_dice", "acc")])

    best, no_improve = -1.0, 0
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tag = epoch if epoch in args.log_epochs else None
        fake_train.log_tag = real_train.log_tag = tag
        tl, td, tdf = run_epoch(model, real_train_loader, fake_train_loader, optimizer, merged_loss_fn, fake_loss_fn, args, device, True)
        vl, vd, vdf = run_epoch(model, real_val_loader, fake_val_loader, optimizer, merged_loss_fn, fake_loss_fn, args, device, False)
        scheduler.step()
        fake_train.log_tag = real_train.log_tag = None
        if tag is not None:
            try:
                process_epoch(os.path.join(log_dir, f"epoch{tag:03d}"))
            except Exception as e:
                log.warning(f"synthetic-log summary failed for epoch {tag}: {e}")

        clean = evaluate_set(model, clean_val_t, device, args.batch_size)
        neg = clean["ptorn"][clean["has_men"]]
        res = {m: evaluate_set(model, ds, device, args.batch_size) for m, ds in synth_val.items()}
        pos = np.concatenate([res[m]["ptorn"][res[m]["has_men"]] for m in modes])
        fp_rate, t_auc = float((neg > 0.5).mean()) if len(neg) else float("nan"), auc(pos, neg)
        log.info(f"Epoch {epoch:03d} lr={lr:.2e} loss tr={tl:.4f} va={vl:.4f} | dice_real tr={td:.4f} va={vd:.4f} | "
                 f"clean val={clean['dice']:.4f} | torn AUC={t_auc:.3f} clean FP={fp_rate:.3f} | "
                 + " ".join(f"{m}: d={res[m]['dice']:.3f} t={res[m]['tear_dice']:.3f} a={res[m]['acc']:.2f}" for m in modes))
        hist.writerow([epoch, f"{lr:.3e}", f"{tl:.5f}", f"{vl:.5f}", f"{td:.5f}", f"{vd:.5f}", f"{td - vd:+.5f}",
                       f"{tdf:.5f}", f"{vdf:.5f}", f"{clean['dice']:.5f}", f"{fp_rate:.5f}", f"{t_auc:.5f}"]
                      + [f"{res[m][k]:.5f}" for m in synth_val for k in ("dice", "tear_dice", "acc")])
        hist_fh.flush()

        torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_latest.pth"))
        if vd > best:
            best, no_improve = vd, 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_best.pth"))
            log.info(f"  New best: {best:.4f} (epoch {epoch})")
            dump_val_predictions(SegOnly(model), real_val, device, os.path.join(args.out_dir, "val_predictions"), args.batch_size)
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                log.info(f"Early stopping at epoch {epoch}. Best: {best:.4f}")
                break
    hist_fh.close()
    log.info(f"Done. Best val Dice: {best:.4f} -> {args.out_dir}")


if __name__ == "__main__":
    main()
