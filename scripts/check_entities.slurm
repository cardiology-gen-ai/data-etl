#!/usr/bin/env bash
#SBATCH -A IscrC_CardioAI
#SBATCH -p boost_usr_prod
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=check_entities
#SBATCH --output=logs_kg/%x-%j.out
#SBATCH --error=logs_kg/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/leonardo_work/IscrC_CardioAI/data-etl}"
VENV_DIR="${VENV_DIR:-/leonardo_scratch/large/userexternal/mlucas00/data-etl}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/.env.leonardo}"

NEO4J_DATA="${NEO4J_DATA:-/leonardo_work/IscrC_CardioAI/neo4j_data_marta}"
NEO4J_LOGS="${NEO4J_LOGS:-/leonardo_work/IscrC_CardioAI/neo4j_logs_marta}"
NEO4J_RUN="${NEO4J_RUN:-/leonardo_work/IscrC_CardioAI/neo4j_run_marta}"
NEO4J_TMP="${NEO4J_TMP:-/leonardo_work/IscrC_CardioAI/neo4j_tmp_marta}"
NEO4J_CONF="${NEO4J_CONF:-/leonardo_work/IscrC_CardioAI/neo4j_conf}"
CONTAINER_IMG="${CONTAINER_IMG:-/leonardo_work/IscrC_CardioAI/containers/neo4j_5.18.0.sif}"

cd "$PROJECT_DIR"
mkdir -p logs_kg "$NEO4J_DATA" "$NEO4J_LOGS" "$NEO4J_RUN" "$NEO4J_TMP"

module load python/3.11
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR:${PYTHONPATH:-}"

if [[ -f "$ENV_FILE" ]]; then
  echo "[INFO] Loading env file: $ENV_FILE"
  set -a
  source "$ENV_FILE"
  set +a
else
  echo "[ERR] Env file not found: $ENV_FILE"
  exit 1
fi

if [[ ! -f "$NEO4J_CONF/neo4j.conf" ]]; then
  echo "[ERR] Neo4j config not found: $NEO4J_CONF/neo4j.conf"
  exit 2
fi

NEO4J_STARTED=0

cleanup() {
  if [[ "$NEO4J_STARTED" -eq 1 ]]; then
    echo "[INFO] Stopping Neo4j..."
    singularity exec \
      --bind "$NEO4J_DATA:/data" \
      --bind "$NEO4J_LOGS:/logs" \
      --bind "$NEO4J_LOGS:/var/lib/neo4j/logs" \
      --bind "$NEO4J_RUN:/run" \
      --bind "$NEO4J_RUN:/var/lib/neo4j/run" \
      --bind "$NEO4J_TMP:/tmp" \
      --bind "$NEO4J_CONF/neo4j.conf:/var/lib/neo4j/conf/neo4j.conf" \
      "$CONTAINER_IMG" \
      neo4j stop || true
  fi
}
trap cleanup EXIT

echo "[INFO] Starting Neo4j..."
singularity exec \
  --bind "$NEO4J_DATA:/data" \
  --bind "$NEO4J_LOGS:/logs" \
  --bind "$NEO4J_LOGS:/var/lib/neo4j/logs" \
  --bind "$NEO4J_RUN:/run" \
  --bind "$NEO4J_RUN:/var/lib/neo4j/run" \
  --bind "$NEO4J_TMP:/tmp" \
  --bind "$NEO4J_CONF/neo4j.conf:/var/lib/neo4j/conf/neo4j.conf" \
  "$CONTAINER_IMG" \
  neo4j start

echo "[INFO] Waiting 30s for Neo4j startup..."
sleep 30
NEO4J_STARTED=1

export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USERNAME="${NEO4J_USERNAME:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-}"

echo "[INFO] Running entity analysis queries..."
if python -m knowledge_graph.visualize_entities; then
  echo "[INFO] Analysis completed successfully."
  RC=0
else
  RC=$?
  echo "[ERROR] Python script failed with exit code $RC. Check the error log."
fi

echo "[INFO] Job finished with exit code: $RC"
exit $RC