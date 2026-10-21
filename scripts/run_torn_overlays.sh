#!/bin/bash
# Overlay montages for every patient that has predictions from BOTH models.
# Usage: bash scripts/run_torn_overlays.sh <dir containing raw PD .nii.gz, searched recursively>
# e.g.   bash scripts/run_torn_overlays.sh "$HOME/Desktop/delete/nikhil-segmentations 2"
PD_DIR="$1"; R=results/torn_predictions_replica_seed42; C=results/torn_predictions_crop_sequence
mkdir -p figures/torn
for r in $R/*_meniscus_pred.nii.gz; do
    stem=$(basename "$r" _meniscus_pred.nii.gz); pid=${stem%%_*}
    raw=$(find "$PD_DIR" -name "$stem.nii.gz" | head -1)
    if [ -z "$raw" ]; then echo "SKIP $pid: raw scan $stem.nii.gz not found under $PD_DIR"; continue; fi
    venv/bin/python scripts/overlay_torn_preds.py --pd "$raw" --pred "$r" --pred2 "$C/${stem}_meniscus_pred.nii.gz" \
        --names replica crop_seq --out "figures/torn/$pid.png" 2>&1 | grep -v INFO
done
