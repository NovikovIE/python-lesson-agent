#!/bin/bash

STORAGE_DIR="$(pwd)/qdrant_storage"

if [ ! -d "$STORAGE_DIR" ]; then
    echo "Storage directory not found. Creating..."
    mkdir -p "$STORAGE_DIR"
else
    echo "Found existing storage at $STORAGE_DIR"
    chmod -R 777 "$STORAGE_DIR"
fi


if [ $(docker ps -a -f name=qdrant-dev | grep -w qdrant-dev | wc -l) -eq 1 ]; then
    echo "'qdrant-dev' exists. Restarting..."
    docker restart qdrant-dev
else
    echo "Creating and starting new container 'qdrant-dev'..."
    mkdir -p ./qdrant_storage
    docker run -d \
      -p 6333:6333 \
      -p 6334:6334 \
      -v "$(pwd)/qdrant_storage:/qdrant/storage" \
      --name qdrant-dev \
      qdrant/qdrant
fi

echo "Qdrant on http://localhost:6333"


echo -e "\nChecking Ollama..."
if ! pgrep -x "ollama" > /dev/null
then
    echo "[warning] Ollama not found. Installing..."
else
    echo "Ollama running."
fi
