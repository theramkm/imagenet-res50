#!/bin/bash

# Set up environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU only
DATA_DIR="/path/to/kaggle/imagenet"

# Test 1: Quick validation (few batches)
echo "Running quick validation..."
python src/train.py \
    --data-dir $DATA_DIR \
    --fast-dev-run \
    --batch-size 32 \
    --num-workers 4 \
    --precision 32 \  # Use FP32 for initial testing
    --gpus 1

# Test 2: Single epoch with small subset
echo "Running single epoch test..."
python src/train.py \
    --data-dir $DATA_DIR \
    --batch-size 32 \
    --epochs 1 \
    --limit-train-batches 0.01 \
    --limit-val-batches 0.01 \
    --num-workers 4 \
    --gpus 1 \
    --precision 32

# Test 3: Full training test (small number of epochs)
echo "Running full training test..."
python src/train.py \
    --data-dir $DATA_DIR \
    --batch-size 64 \
    --epochs 3 \
    --num-workers 4 \
    --gpus 1 \
    --precision 16-mixed \
    --wandb-project imagenet1k-test 