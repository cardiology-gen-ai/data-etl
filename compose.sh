#!/usr/bin/env bash

ENV_FILE="$(pwd)/.env"
[[ -f "$ENV_FILE" ]] || { echo "Error: .env file not found: $ENV_FILE"; exit 2; }
set -a
source "$ENV_FILE"
set +a

if [ -z "$CONFIG_PATH" ]; then
    echo "Error: CONFIG_PATH not defined in .env"
    exit 3
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: config.json not found in $CONFIG_PATH"
    exit 4
fi

# INDEXING_TYPE=$(jq -r '.cardiology_protocols.indexing.type' "$CONFIG_PATH")
# EMBEDDING_TYPE=$(jq -r '.cardiology_protocols.embeddings.ollama' "$CONFIG_PATH")
INDEXING_TYPE=$(jq -r '.hepatology_protocols.indexing.type' "$CONFIG_PATH")
EMBEDDING_TYPE=$(jq -r '.hepatology_protocols.embeddings.ollama' "$CONFIG_PATH")

PROFILES=()

if [ "$INDEXING_TYPE" = "qdrant" ]; then
  PROFILES+=("qdrant_vectorstore")
fi

if [ "$EMBEDDING_TYPE" = "true" ]; then
    PROFILES+=("ollama_models")
fi

if [[ ${#PROFILES[@]} -eq 0 ]]; then
    echo "No active profile"
    docker compose --profile "default" up -d
else
    echo "Active profiles: ${PROFILES[*]}"
    CMD=(docker compose -f docker-compose.yaml)
    CMD_GENERAL=(docker compose -f ../cardiology-gen-ai/docker-compose.yaml)
    for p in "${PROFILES[@]}"; do
      CMD+=(--profile "$p")
      CMD_GENERAL+=(--profile "$p")
    done
    CMD+=(up -d)
    CMD_GENERAL+=(up -d)
    echo ${CMD[@]}
    echo ${CMD_GENERAL[@]}
    "${CMD_GENERAL[@]}"
    "${CMD[@]}"
fi