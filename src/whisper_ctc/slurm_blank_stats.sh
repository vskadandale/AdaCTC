#!/bin/bash
#SBATCH --job-name=ctc_blank_stats
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=boost_usr_prod
#SBATCH --output=exp/blank_stats/slurm_%j.log

module purge
module load cuda/12.1
source "${CONDA_PATH:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-adactc}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1

ROOT="${ADACTC_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
cd $ROOT/src/whisper_ctc
for tag in frozen_backbone partial_unfreeze; do
  python blank_stats.py --ckpt $ROOT/models/whisper_ctc_${tag}.pth --tag $tag \
    --test_jsonl $DATA_DIR/librispeech_jsonl/librispeech_test_clean.jsonl \
    --out_dir $ROOT/exp/blank_stats --num_workers 8
done
