#!/bin/bash
#SBATCH --job-name=vox_blank_stats
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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
DATA=$DATA_DIR/voxpopuli_jsonl/test.jsonl
OUT=$ROOT/exp/blank_stats/voxpopuli_en
for tag in frozen_backbone partial_unfreeze; do
  python blank_stats.py --ckpt $ROOT/models/whisper_ctc_${tag}.pth --tag $tag \
    --test_jsonl $DATA --out_dir $OUT --num_workers 8
done
python blank_stats.py --model_type wav2vec2 --ckpt $ROOT/models/wav2vec2 --tag wav2vec2 \
  --test_jsonl $DATA --out_dir $OUT --num_workers 8
