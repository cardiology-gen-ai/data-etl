#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="$CINECA_SCRATCH/hf_env"

module purge
module load python/3.11.7

if [ -d "$ENV_PATH" ] && [ -f "$ENV_PATH/bin/activate" ]; then
  echo "[env] existing venv"
  source "$ENV_PATH/bin/activate"
else
  echo "[env] creating venv"
  python3 -m venv "$ENV_PATH"
  source "$ENV_PATH/bin/activate"
  python -m pip install --upgrade pip
  pip install transformers huggingface_hub accelerate jq
fi

# Path to env file on Leonardo cluster:
ENV_FILE="$WORK/data-etl/.env.leonardo"

# Optional: config_path override via -c
CONFIG_FILE=""
while getopts "c:h" opt; do
  case $opt in
    c) CONFIG_FILE="$OPTARG" ;;
    h|*) echo "Uso: $0 [-c /path/config.json]"; exit 0 ;;
  esac
done

# Cache HF
: "${HF_BASE_CACHE:=${WORK:-$CINECA_WORK}/hf}"
export HF_HOME="$HF_BASE_CACHE"
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE"

command -v huggingface-cli >/dev/null || { echo "Error: no huggingface-cli (transformers)"; exit 1; }
command -v jq >/dev/null || { echo "Error: missing 'jq'"; exit 1; }

# 1) Load .env.leonardo (HF_TOKEN + APP_CONFIG_PATH)
[[ -f "$ENV_FILE" ]] || { echo "Error: .env.leonardo non found: $ENV_FILE"; exit 2; }
set -a

source "$ENV_FILE"
set +a

# 2) HF_TOKEN
: "${HF_TOKEN:?Errore: HF_TOKEN not present in .env.leonardo}"

# 3) Find config.json
if [[ -z "$CONFIG_FILE" ]]; then
  : "${APP_CONFIG_PATH:?Error: APP_CONFIG_PATH not found on .env.leonardo e no -c given}"
  CONFIG_FILE="$APP_CONFIG_PATH"
fi
[[ -f "$CONFIG_FILE" ]] || { echo "Error: config not found: $CONFIG_FILE"; exit 3; }

echo "Using ENV_FILE: $ENV_FILE"
echo "Using CONFIG_FILE: $CONFIG_FILE"
echo "Using HF_HOME: $HF_HOME"

# 4) Login Hugging Face
echo ">>> Login HF…"
huggingface-cli login --token "$HF_TOKEN" >/dev/null

# 5) Get model name from cardiology_protocols.embeddings.deployment
#    Both string and array supported.
mapfile -t MODELS < <(jq -r '
  .cardiology_protocols
  | .embeddings
  | .deployment
  | if type=="array" then .[] else . end
' "$CONFIG_FILE" | grep -E '.+/.+|^[a-zA-Z0-9._-]+$' | sort -u)

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "Error: no model found in cardiology_protocols.embeddings.deployment"; exit 4
fi

echo ">>> Found Models:"
printf ' - %s\n' "${MODELS[@]}"

# 6) Download tokenizer + weights
python - <<'PY' "${MODELS[@]}"
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

models = sys.argv[1:]
for mid in models:
    try:
        snapshot_download(mid, local_files_only=True)
        print(f"[CACHE] {mid} already present.")
        continue
    except LocalEntryNotFoundError:
        pass
    print(f"[DL] Downloading {mid}…", flush=True)
    snapshot_download(mid)
print("\n=== DONE ===")
PY

deactivate

echo ">>> Done. Use OFFLINE in jobs:"
echo "export HF_HOME=$HF_HOME"
echo "export HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub"
echo "export TRANSFORMERS_OFFLINE=1"
echo "export HF_HUB_OFFLINE=1"
