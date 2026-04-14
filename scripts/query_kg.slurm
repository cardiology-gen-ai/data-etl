#!/usr/bin/env bash
#SBATCH -A IscrC_CardioAI
#SBATCH -p boost_usr_prod
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=query_kg
#SBATCH --output=logs_kg/%x-%j.out
#SBATCH --error=logs_kg/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$WORK/data-etl}"
VENV_DIR="${VENV_DIR:-${CINECA_SCRATCH:-$WORK}/data-etl}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/.env.leonardo}"
CONTAINER_IMG="${CONTAINER_IMG:-$WORK/containers/neo4j_5.18.0.sif}"

cd "$PROJECT_DIR"
mkdir -p logs_kg

# Load environment
if [[ -f "$ENV_FILE" ]]; then
  echo "[INFO] Loading env file: $ENV_FILE"
  set -a
  source "$ENV_FILE"
  set +a
else
  echo "[ERR] Env file not found: $ENV_FILE"
  exit 1
fi

# Python environment
module load python/3.11
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR:${PYTHONPATH:-}"

NEO4J_STARTED=0
NEO4J_DATA="${NEO4J_DATA:-$WORK/neo4j_data_marta}"
NEO4J_LOGS="${NEO4J_LOGS:-$WORK/neo4j_logs_marta}"
NEO4J_RUN="${NEO4J_RUN:-$WORK/neo4j_run_marta}"
NEO4J_TMP="${NEO4J_TMP:-$WORK/neo4j_tmp_marta}"
NEO4J_CONF="${NEO4J_CONF:-$WORK/neo4j_conf}"

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

mkdir -p "$NEO4J_DATA" "$NEO4J_LOGS" "$NEO4J_RUN" "$NEO4J_TMP"

if [[ ! -f "$NEO4J_CONF/neo4j.conf" ]]; then
  echo "[ERR] Neo4j config not found: $NEO4J_CONF/neo4j.conf"
  exit 2
fi

echo "[INFO] Starting Neo4j for queries..."
singularity exec \
  --bind "$NEO4J_DATA:/data" \
  --bind "$NEO4J_LOGS:/logs" \
  --bind "$NEO4J_RUN:/var/lib/neo4j/run" \
  --bind "$NEO4J_TMP:/tmp" \
  --bind "$NEO4J_CONF/neo4j.conf:/var/lib/neo4j/conf/neo4j.conf" \
  "$CONTAINER_IMG" \
  neo4j start

sleep 30
NEO4J_STARTED=1

export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USERNAME="${NEO4J_USERNAME:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-}"

echo "[INFO] Running graph queries..."
if python -m knowledge_graph.query_graph; then
  RC=0
else
  RC=$?
fi

echo "[INFO] Job completed with exit code: $RC"
exit $RC