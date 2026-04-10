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

PROJECT_DIR="/leonardo_work/IscrC_CardioAI/data-etl"
VENV_DIR="/leonardo_scratch/large/userexternal/mlucas00/data-etl"
ENV_FILE="$PROJECT_DIR/.env.leonardo"

# Neo4j paths (form now with that folder, because of permission issues)
NEO4J_DATA="/leonardo_work/IscrC_CardioAI/neo4j_data_marta"
NEO4J_LOGS="/leonardo_work/IscrC_CardioAI/neo4j_logs_marta"
NEO4J_RUN="/leonardo_work/IscrC_CardioAI/neo4j_run_marta"
NEO4J_CONF="/leonardo_work/IscrC_CardioAI/neo4j_conf"
CONTAINER_IMG="/leonardo_work/IscrC_CardioAI/containers/neo4j_5.18.0.sif"

cd "$PROJECT_DIR"
mkdir -p logs_kg "$NEO4J_LOGS" "$NEO4J_RUN"

# Load env and variables
module load python/3.11
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR:${PYTHONPATH:-}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

echo "[INFO] Starting Neo4j in Background..."

singularity exec \
  --bind "$NEO4J_DATA:/data" \
  --bind "$NEO4J_LOGS:/logs" \
  --bind "$NEO4J_LOGS:/var/lib/neo4j/logs" \
  --bind "$NEO4J_RUN:/run" \
  --bind "$NEO4J_RUN:/var/lib/neo4j/run" \
  --bind "$NEO4J_CONF/neo4j.conf:/var/lib/neo4j/conf/neo4j.conf" \
  "$CONTAINER_IMG" \
  neo4j start

echo "[INFO] Waiting 30s for Neo4j startup..."
sleep 30

echo "[INFO] Running Entity Analysis Queries..."
if python src/visualize_entities.py; then
    echo "[INFO] Analysis completed successfully."
else
    echo "[ERROR] Python script failed. Check the error log."
fi

# Cleanup
echo "[INFO] Stopping Neo4j..."
singularity exec \
  --bind "$NEO4J_DATA:/data" \
  --bind "$NEO4J_RUN:/var/lib/neo4j/run" \
  --bind "$NEO4J_LOGS:/var/lib/neo4j/logs" \
  --bind "$NEO4J_CONF/neo4j.conf:/var/lib/neo4j/conf/neo4j.conf" \
  "$CONTAINER_IMG" \
  neo4j stop || true

echo "[INFO] Job finished."
exit 0