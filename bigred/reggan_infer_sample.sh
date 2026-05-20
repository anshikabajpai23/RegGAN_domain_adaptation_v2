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
 
mkdir -p /N/project/prostate_cancer_ai/anshika/regGAN/results/translated_pd
 
cd /N/project/prostate_cancer_ai/anshika/regGAN/regGAN
 
echo "=== Starting inference ==="
python infer2.py \
    --ckpt      /N/project/prostate_cancer_ai/anshika/regGAN/runs/run_002/ckpt_latest.pt \
    --dess_root /N/project/prostate_cancer_ai/anshika/regGAN/data/skm-tea-dataset/dess-files \
    --out_dir   /N/project/prostate_cancer_ai/anshika/regGAN/results2/translated_pd \
    --ngf       48 \
    --n_res     9
 
echo "=== Inference done ==="