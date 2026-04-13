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

cd "$PROJECT_DIR"
mkdir -p logs_kg

# Load environment
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

# Python environment
module load python/3.11
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR:${PYTHONPATH:-}"

NEO4J_STARTED=0
cleanup() {
  if [[ "$NEO4J_STARTED" -eq 1 ]]; then
    echo "[INFO] Stopping Neo4j..."
    singularity exec \
      --bind "$NEO4J_DATA:/data" \
      --bind "$NEO4J_RUN:/var/lib/neo4j/run" \
      --bind "$NEO4J_CONF/neo4j.conf:/var/lib/neo4j/conf/neo4j.conf" \
      "$WORK/containers/neo4j_5.18.0.sif" \
      neo4j stop || true
  fi
}
trap cleanup EXIT

NEO4J_DATA="${NEO4J_DATA:-$WORK/neo4j_data_marta}"
NEO4J_LOGS="${NEO4J_LOGS:-$WORK/neo4j_logs_marta}"
NEO4J_RUN="${NEO4J_RUN:-$WORK/neo4j_run_marta}"
NEO4J_TMP="${NEO4J_TMP:-$WORK/neo4j_tmp_marta}"
NEO4J_CONF="${NEO4J_CONF:-$WORK/neo4j_conf}"

echo "[INFO] Starting Neo4j for Queries..."
singularity exec \
  --bind "$NEO4J_DATA:/data" \
  --bind "$NEO4J_LOGS:/logs" \
  --bind "$NEO4J_RUN:/var/lib/neo4j/run" \
  --bind "$NEO4J_TMP:/tmp" \
  --bind "$NEO4J_CONF/neo4j.conf:/var/lib/neo4j/conf/neo4j.conf" \
  "$WORK/containers/neo4j_5.18.0.sif" \
  neo4j start

sleep 30
NEO4J_STARTED=1

export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USERNAME="${NEO4J_USERNAME:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-}"

echo "[INFO] Eseguo le query sul grafo..."
python -m knowledge_graph.query_graph
RC=$?

echo "[INFO] Job completato con codice: $RC"
exit $RC