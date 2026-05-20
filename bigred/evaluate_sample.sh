#!/bin/bash
#SBATCH --job-name=reggan_inf
#SBATCH --output=reggan_inf_%j.out
#SBATCH --error=reggan_inf_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-interactive
#SBATCH --gres=gpu:1
#SBATCH -A      # ← your IU email

module purge
module load python/3.11 cudatoolkit/12.2
 
source /N/project/prostate_cancer_ai/anshika/regGAN/regGAN/venv/bin/activate
 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
 
mkdir -p /N/project/prostate_cancer_ai/anshika/regGAN/evaluation
 
cd /N/project/prostate_cancer_ai/anshika/regGAN/regGAN
 
echo "=== Starting evaluation ==="
python evaluate.py \
    --fake_pd_dir    /N/project/prostate_cancer_ai/anshika/regGAN/results2/translated_pd \
    --real_pd_dir    /N/project/prostate_cancer_ai/anshika/regGAN/data/iu-dataset/pd-files \
    --dess_slice_dir /N/project/prostate_cancer_ai/anshika/regGAN/preprocessed/slices/dess \
    --mask_dir       /N/project/prostate_cancer_ai/anshika/regGAN/preprocessed/masks \
    --ckpt           /N/project/prostate_cancer_ai/anshika/regGAN/runs/run_002/ckpt_latest.pt \
    --out_dir        /N/project/prostate_cancer_ai/anshika/regGAN/evaluation \
    --meniscus_label 5 6 \
    # --max_slices     1000
 
echo "=== Evaluation done ==="