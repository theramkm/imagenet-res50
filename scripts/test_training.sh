#!/bin/bash

# Test 1: Quick test (1 batch)
echo "Running quick test..."
python src/train.py \
    --data-dir /path/to/mini-imagenet \
    --fast-dev-run \
    --num-classes 100 \
    --batch-size 32 \
    --gpus 1

# Test 2: Single epoch test
echo "Running single epoch test..."
python src/train.py \
    --data-dir /path/to/mini-imagenet \
    --num-classes 100 \
    --batch-size 32 \
    --epochs 1 \
    --gpus 1 \
    --limit-train-batches 0.1 \
    --limit-val-batches 0.1

# Test 3: Full mini-ImageNet test
echo "Running full mini-ImageNet test..."
python src/train.py \
    --data-dir /path/to/mini-imagenet \
    --num-classes 100 \
    --batch-size 32 \
    --epochs 5 \
    --gpus 1 