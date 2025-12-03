#!/usr/bin/env bash
#SBATCH -A IscrC_CardioAI
#SBATCH -p boost_usr_prod
#SBATCH --time=01:59:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=data-etl
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --mail-user=your@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# 0) Paths
PROJ_DIR="$WORK/data-etl"
ENV_DIR="$CINECA_SCRATCH/data-etl"

mkdir -p "$WORK/logs"

cd "$PROJ_DIR" || { echo "[ERR] Cannot access $PROJ_DIR"; exit 1; }
echo "[INFO] Running from: $(pwd)"

# 1) Activate venv
if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
    echo "[ERR] Virtual environment not found at: $ENV_DIR"
    exit 2
fi
source "$ENV_DIR/bin/activate"
echo "[OK] Python = $(which python)"
# 2) Env file
if [[ -f "$PROJ_DIR/.env.leonardo" ]]; then
  set -a
  source "$PROJ_DIR/.env.leonardo"
  set +a
else
  echo "[WARN] .env.leonardo not found in $PROJ_DIR"
fi
echo "[INFO] Using FAISS vectorstore (Qdrant disabled)"

# 3) HF cache (uses the folder where you downloaded the model)
export HF_HOME="$WORK/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"
echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
# 4) Offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 5) CPU/GPU
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 6) Python path
export PYTHONPATH="$PROJ_DIR:${PYTHONPATH:-}"
# 7) Run ETL
python -m src.main