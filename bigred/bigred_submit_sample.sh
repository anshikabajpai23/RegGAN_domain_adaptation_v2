#!/bin/bash
#SBATCH --job-name=reggan_cl
#SBATCH --output=reggan_cl_%j.out
#SBATCH --error=reggan_cl_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-interactive
#SBATCH --gres=gpu:1
#SBATCH -A      # ← your IU email

module purge
module load python/3.11 cudatoolkit/12.2
 
source /N/project/prostate_cancer_ai/anshika/regGAN/regGAN/venv/bin/activate
 
mkdir -p /N/project/prostate_cancer_ai/anshika/regGAN/regGAN/logs
mkdir -p /N/project/prostate_cancer_ai/anshika/regGAN/preprocessed
mkdir -p /N/project/prostate_cancer_ai/anshika/regGAN/runs/run_003
 
cd /N/project/prostate_cancer_ai/anshika/regGAN/regGAN
 
if [ ! -f "/N/project/prostate_cancer_ai/anshika/regGAN/preprocessed/splits.json" ]; then
    echo "=== Running preprocessing ==="
    python preprocess.py \
        --dess_root  /N/project/prostate_cancer_ai/anshika/regGAN/data/skm-tea-dataset/dess-files \
        --pd_root    /N/project/prostate_cancer_ai/anshika/regGAN/data/iu-dataset/pd-files \
        --out_root   /N/project/prostate_cancer_ai/anshika/regGAN/preprocessed \
        --val_ratio  0.10 \
        --test_ratio 0.10
fi
 
echo "=== Starting RegGAN training ==="
python train.py \
    --splits            /N/project/prostate_cancer_ai/anshika/regGAN/preprocessed/splits.json \
    --out_dir           /N/project/prostate_cancer_ai/anshika/regGAN/runs/run_003 \
    --epochs            200 \
    --batch_size        8 \
    --lr                2e-4 \
    --lr_reg            1e-4 \
    --num_workers       4 \
    --ngf               48 \
    --ndf               48 \
    --n_res             9 \
    --nf_reg            16 \
    --lambda_cycle      10.0 \
    --lambda_reg_sim     5.0 \
    --lambda_reg_smooth 10.0 \
    --lambda_reg_mag     5.0 \
    | tee /N/project/prostate_cancer_ai/anshika/regGAN/runs/run_003/train.log
 
echo "=== Training done ==="
 