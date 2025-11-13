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

INDEXING_TYPE=$(jq -r '.cardiology_protocols.indexing.type' "$CONFIG_PATH")

if [ "$INDEXING_TYPE" = "qdrant" ]; then
  echo "INDEXING_TYPE is qdrant → activating qdrant_vectorstore profile"
  docker compose --profile qdrant_vectorstore up -d
else
  echo "INDEXING_TYPE is not qdrant → starting default services only"
  docker compose up -d
fi