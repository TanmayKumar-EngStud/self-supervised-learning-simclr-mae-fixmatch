#!/bin/bash
# SLURM job for the bonus task. Adjust the account/partition/GPU lines to your KTH cluster.
#SBATCH --job-name=dd2412-bonus
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4          # all GPUs of one node; the script uses however many it gets
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=bonus_%j.log
##SBATCH --account=<your-project>
##SBATCH --partition=<gpu-partition>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# module load / conda activate here if your cluster needs it
export HF_HOME=${HF_HOME:-$PWD/hf_cache}       # where the pretrained diffusion model is cached
export OMP_NUM_THREADS=4

# --nproc_per_node=gpu starts one process per visible GPU
torchrun --standalone --nproc_per_node=gpu bonus_synthetic_fmnist.py \
    --stage all --preset standard --data_dir ./data --out_dir ./bonus_out
