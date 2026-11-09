#!/bin/zsh
# P1 screen: each model in ascending size, one results entry per model, then the comparison entry.
cd "$(dirname "$0")/.."
MODELS=(qwen3-1.7b llama3.2-3b qwen3-4b gemma3-4b qwen3-8b)
for m in $MODELS; do
  uv run python scripts/p1_behavior.py --models $m --n-scenes 150 --near-n 100 --abl-n 100 --cot-n 60 || echo "FAILED $m"
done
uv run python scripts/p1_behavior.py --models $MODELS --reuse
