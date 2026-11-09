#!/bin/zsh
# Wait for a model's weights to finish downloading, then run P0 and P1 on it.
set -u
REPO_DIR=~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B
cd "$(dirname "$0")/.."
until [[ -d $REPO_DIR/snapshots ]] && ls $REPO_DIR/snapshots/*/model*.safetensors >/dev/null 2>&1 \
      && [[ -z "$(ls $REPO_DIR/blobs/*.incomplete 2>/dev/null)" ]]; do sleep 60; done
echo "download complete $(date)"
uv run python scripts/p0_benchmark.py --model qwen3-1.7b --n 64
uv run python scripts/p1_behavior.py --models qwen3-1.7b
