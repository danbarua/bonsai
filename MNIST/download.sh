#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"

FILES=(
  "train-images-idx3-ubyte.gz"
  "train-labels-idx1-ubyte.gz"
  "t10k-images-idx3-ubyte.gz"
  "t10k-labels-idx1-ubyte.gz"
)

for file_name in "${FILES[@]}"; do
  curl -sL -o "$file_name" "$BASE_URL/$file_name"
done

ls -la ./*.gz

python3 mnist_loader.py