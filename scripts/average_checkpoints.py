"""
average_checkpoints.py
=======================
Phase 4 step 10 — weight averaging (free, inference-only, no retrain).

Per the doc's own fallback: no run in this project saves per-epoch
checkpoints (only ckpt_best.pth and ckpt_latest.pth, the latter overwritten
every epoch) -- so this averages ckpt_best + ckpt_latest, exactly the
documented fallback, and says so explicitly rather than silently pretending
it did the ideal "last 5 epochs before best" version.

Averaging is a plain element-wise mean of each parameter tensor across the
given checkpoints -- requires identical architecture/state_dict keys, which
holds here since every Phase 4 run uses the same smp.Unet(resnet34, 3->3).

Usage:
  python scripts/average_checkpoints.py \
      --ckpts segmentation_runs/run_v7_replica_seed42/ckpt_best.pth \
              segmentation_runs/run_v7_replica_seed42/ckpt_latest.pth \
      --out   segmentation_runs/run_v7_replica_seed42/ckpt_avg.pth
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="2+ checkpoint paths to average, equal weight each")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.ckpts) < 2:
        raise SystemExit("Need at least 2 checkpoints to average.")

    states = [torch.load(p, map_location="cpu") for p in args.ckpts]
    keys = set(states[0].keys())
    for p, s in zip(args.ckpts[1:], states[1:]):
        if set(s.keys()) != keys:
            raise SystemExit(f"Key mismatch between {args.ckpts[0]} and {p} — "
                             f"are these the same architecture?")

    avg = {}
    for k in keys:
        stacked = torch.stack([s[k].float() for s in states], dim=0)
        avg[k] = stacked.mean(dim=0).to(states[0][k].dtype)

    torch.save(avg, args.out)
    print(f"Averaged {len(args.ckpts)} checkpoints -> {args.out}")
    for p in args.ckpts:
        print(f"  + {p}")
    if len(args.ckpts) == 2 and any("best" in p for p in args.ckpts) and any("latest" in p for p in args.ckpts):
        print("(ckpt_best + ckpt_latest fallback — no per-epoch checkpoints exist in this project)")


if __name__ == "__main__":
    main()
