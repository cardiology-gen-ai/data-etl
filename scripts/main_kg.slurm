#!/usr/bin/env bash
#SBATCH -A IscrC_CardioAI
#SBATCH -p boost_usr_prod
#SBATCH --time=01:59:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=main_kg
#SBATCH --output=logs_kg/%x-%j.out
#SBATCH --error=logs_kg/%x-%j.err

set -euo pipefail

export KG_PIPELINE_PHASE="${KG_PIPELINE_PHASE:-preprocess}"

PROJECT_DIR="${PROJECT_DIR:-$WORK/data-etl}"
VENV_DIR="${VENV_DIR:-${CINECA_SCRATCH:-$WORK}/data-etl}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/.env.leonardo}"
CONTAINER_IMG="${CONTAINER_IMG:-$WORK/containers/neo4j_5.18.0.sif}"

cd "$PROJECT_DIR"
mkdir -p logs_kg

echo "[INFO] ===== JOB START ====="
echo "[INFO] Date: $(date)"
echo "[INFO] Host: $(hostname)"
echo "[INFO] PWD: $(pwd)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] KG_PIPELINE_PHASE: $KG_PIPELINE_PHASE"
echo "[INFO] PROJECT_DIR: $PROJECT_DIR"
echo "[INFO] VENV_DIR: $VENV_DIR"
echo "[INFO] ENV_FILE: $ENV_FILE"
echo "[INFO] CONTAINER_IMG: $CONTAINER_IMG"

if [[ -f "$ENV_FILE" ]]; then
  echo "[INFO] Loading env file..."
  set -a
  source "$ENV_FILE"
  set +a
else
  echo "[ERR] Env file not found: $ENV_FILE"
  exit 2
fi

echo "[INFO] CONFIG_PATH=${CONFIG_PATH:-<missing>}"
echo "[INFO] APP_CONFIG_PATH=${APP_CONFIG_PATH:-<missing>}"
echo "[INFO] NEO4J_URI=${NEO4J_URI:-<unset>}"

module load python/3.11
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[INFO] Python executable: $(which python)"
python --version

export HF_HOME="${HF_HOME:-$WORK/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "[INFO] HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "[INFO] HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "[INFO] TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"

export KG_CHAT_MODEL="${KG_CHAT_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
export KG_EMBEDDING_MODEL="${KG_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B}"
export KG_LOCAL_FILES_ONLY="${KG_LOCAL_FILES_ONLY:-true}"
export KG_CHAT_MAX_NEW_TOKENS="${KG_CHAT_MAX_NEW_TOKENS:-512}"

echo "[INFO] KG_CHAT_MODEL=$KG_CHAT_MODEL"
echo "[INFO] KG_EMBEDDING_MODEL=$KG_EMBEDDING_MODEL"
echo "[INFO] KG_LOCAL_FILES_ONLY=$KG_LOCAL_FILES_ONLY"
echo "[INFO] KG_CHAT_MAX_NEW_TOKENS=$KG_CHAT_MAX_NEW_TOKENS"

NEEDS_NEO4J=false
case "$KG_PIPELINE_PHASE" in
  preprocess)
    NEEDS_NEO4J=false
    ;;
  graph|entities|embeddings|full)
    NEEDS_NEO4J=true
    ;;
  *)
    echo "[ERR] Unsupported KG_PIPELINE_PHASE='$KG_PIPELINE_PHASE'"
    exit 3
    ;;
esac

echo "[INFO] NEEDS_NEO4J=$NEEDS_NEO4J"

NEO4J_STARTED=0
NEO4J_DATA=""
NEO4J_LOGS=""
NEO4J_RUN=""
NEO4J_TMP=""
NEO4J_CONF=""

cleanup() {
  if [[ "$NEO4J_STARTED" -eq 1 ]]; then
    echo "[INFO] Stopping Neo4j..."
    singularity exec \
      --bind "$NEO4J_DATA:/data" \
      --bind "$NEO4J_LOGS:/logs" \
      --bind "$NEO4J_RUN:/var/lib/neo4j/run" \
      --bind "$NEO4J_TMP:/tmp" \
      --bind "$NEO4J_CONF/neo4j.conf:/var/lib/neo4j/conf/neo4j.conf" \
      "$CONTAINER_IMG" \
      neo4j stop || true
  fi
}
trap cleanup EXIT

if [[ "$NEEDS_NEO4J" == "true" ]]; then
  NEO4J_DATA="${NEO4J_DATA:-$WORK/neo4j_data_test}"
  NEO4J_LOGS="${NEO4J_LOGS:-$WORK/neo4j_logs_test}"
  NEO4J_RUN="${NEO4J_RUN:-$WORK/neo4j_run_test}"
  NEO4J_TMP="${NEO4J_TMP:-$WORK/neo4j_tmp_test}"
  NEO4J_CONF="${NEO4J_CONF:-$WORK/neo4j_conf}"

  mkdir -p "$NEO4J_DATA" "$NEO4J_LOGS" "$NEO4J_RUN" "$NEO4J_TMP" "$NEO4J_CONF"

  echo "[INFO] Neo4j paths:"
  echo "       DATA=$NEO4J_DATA"
  echo "       LOGS=$NEO4J_LOGS"
  echo "       RUN=$NEO4J_RUN"
  echo "       TMP=$NEO4J_TMP"
  echo "       CONF=$NEO4J_CONF"

  if [[ ! -f "$NEO4J_CONF/neo4j.conf" ]]; then
    echo "[ERR] Neo4j config not found: $NEO4J_CONF/neo4j.conf"
    exit 4
  fi

  echo "[INFO] Starting Neo4j..."
  singularity exec \
    --bind "$NEO4J_DATA:/data" \
    --bind "$NEO4J_LOGS:/logs" \
    --bind "$NEO4J_RUN:/var/lib/neo4j/run" \
    --bind "$NEO4J_TMP:/tmp" \
    --bind "$NEO4J_CONF/neo4j.conf:/var/lib/neo4j/conf/neo4j.conf" \
    "$CONTAINER_IMG" \
    neo4j start

  echo "[INFO] Waiting 30s for Neo4j startup..."
  sleep 30

  export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
  export NEO4J_USERNAME="${NEO4J_USERNAME:-neo4j}"
  export NEO4J_PASSWORD="${NEO4J_PASSWORD:-}"

  echo "[INFO] Neo4j runtime connection:"
  echo "       NEO4J_URI=$NEO4J_URI"
  echo "       NEO4J_USERNAME=$NEO4J_USERNAME"
  echo "       NEO4J_PASSWORD=<${#NEO4J_PASSWORD} chars>"

  NEO4J_STARTED=1
fi

if [[ "$KG_PIPELINE_PHASE" == "entities" || "$KG_PIPELINE_PHASE" == "embeddings" || "$KG_PIPELINE_PHASE" == "full" ]]; then
  echo "[INFO] Checking GPU availability..."
  python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY
fi

echo "[INFO] Checking Python imports..."
python - <<'PY'
import src.main_graph
print("Imported src.main_graph successfully")
PY

echo "[INFO] Launching python -m src.main_graph ..."
if python -m src.main_graph; then
  RC=0
else
  RC=$?
fi

echo "[INFO] Python exit code: $RC"
echo "[INFO] ===== JOB END ====="
exit $RC