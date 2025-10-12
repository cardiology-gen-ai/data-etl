#!/usr/bin/env bash
set -euo pipefail

module purge
module load python/3.11.7

PROJ_DIR="$PWD"
ENV_PATH="$CINECA_SCRATCH/data-etl"

cd "$PROJ_DIR"

if [ -d "$ENV_PATH" ] && [ -f "$ENV_PATH/bin/activate" ]; then
  echo "[env] existing venv"
  source "$ENV_PATH/bin/activate"
else
  echo "[env] creating venv"
  python3 -m venv "$ENV_PATH"
  source "$ENV_PATH/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel build
pip install -e .
# python -m pip install --no-cache-dir --force-reinstall --upgrade "cardiologygenai-coordo @ git+https://github.com/cardiology-gen-ai/cardiology-gen-ai.git@main"


export XDG_CACHE_HOME=$WORK/.cache
export FASTEMBED_CACHE_PATH=$XDG_CACHE_HOME/fastembed
mkdir -p "$FASTEMBED_CACHE_PATH"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
python - <<'PY'
from fastembed.sparse import SparseTextEmbedding
m = SparseTextEmbedding(model_name="Qdrant/bm25")
PY

echo "[OK] installation completed"
