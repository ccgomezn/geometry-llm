#!/bin/bash  # exact commands of the overnight run
CUDA_VISIBLE_DEVICES=0 /home/cris_anemona_ai/.local/bin/uv run python scripts/p1_behavior.py --models qwen3-1.7b --n-scenes 150 --near-n 0 --abl-n 0 --cot-n 0 --batch-size 32
CUDA_VISIBLE_DEVICES=1 /home/cris_anemona_ai/.local/bin/uv run python scripts/p1_behavior.py --models llama3.2-3b --n-scenes 150 --near-n 0 --abl-n 0 --cot-n 0 --batch-size 32
CUDA_VISIBLE_DEVICES=2 /home/cris_anemona_ai/.local/bin/uv run python scripts/p1_behavior.py --models qwen3-4b --n-scenes 150 --near-n 0 --abl-n 0 --cot-n 0 --batch-size 32
CUDA_VISIBLE_DEVICES=3 /home/cris_anemona_ai/.local/bin/uv run python scripts/p1_behavior.py --models gemma3-4b --n-scenes 150 --near-n 0 --abl-n 0 --cot-n 0 --batch-size 32
CUDA_VISIBLE_DEVICES=0,1 /home/cris_anemona_ai/.local/bin/uv run python scripts/p1_behavior.py --models gemma3-12b --n-scenes 500 --near-n 200 --abl-n 200 --cot-n 0 --batch-size 32
CUDA_VISIBLE_DEVICES=2,3 /home/cris_anemona_ai/.local/bin/uv run python scripts/p1_behavior.py --models qwen3-14b --n-scenes 500 --near-n 200 --abl-n 200 --cot-n 0 --batch-size 32
CUDA_VISIBLE_DEVICES=0 /home/cris_anemona_ai/.local/bin/uv run python scripts/p1_behavior.py --models qwen3-8b --n-scenes 150 --near-n 0 --abl-n 0 --cot-n 0 --batch-size 32
CUDA_VISIBLE_DEVICES=1,2,3 /home/cris_anemona_ai/.local/bin/uv run python scripts/p1_behavior.py --models gemma3-27b --n-scenes 500 --near-n 200 --abl-n 200 --cot-n 0 --batch-size 16
/home/cris_anemona_ai/.local/bin/uv run python scripts/choose_model.py
/home/cris_anemona_ai/.local/bin/uv run python scripts/build_cache.py --model gemma3-12b --dataset main --n 3000 --n-objects 3 --seed 2 --batch-size 16
/home/cris_anemona_ai/.local/bin/uv run python scripts/build_cache.py --model gemma3-12b --dataset eol --n 1000 --n-objects 3 --seed 8 --batch-size 16
/home/cris_anemona_ai/.local/bin/uv run python scripts/build_cache.py --model gemma3-12b --dataset main --n 3000 --n-objects 3 --seed 2 --batch-size 16 --random-init
/home/cris_anemona_ai/.local/bin/uv run python scripts/p2_probes.py --model gemma3-12b --eol eol3
/home/cris_anemona_ai/.local/bin/uv run python scripts/p3_patching.py --model gemma3-12b --n-objects 3 --exclude closer_than --layer-step 2 --batch-size 16
/home/cris_anemona_ai/.local/bin/uv run python scripts/p4_das.py --model gemma3-12b --n-objects 3 --exclude closer_than --n-train 600 --n-test 400 --epochs 3
/home/cris_anemona_ai/.local/bin/uv run python scripts/p4_slider.py --model gemma3-12b --n-objects 3 --exclude closer_than
/home/cris_anemona_ai/.local/bin/uv run python scripts/p4_das.py --model gemma3-12b --n-objects 3 --exclude closer_than --n-train 600 --n-test 400 --epochs 3
