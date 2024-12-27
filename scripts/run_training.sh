#!/bin/bash

# Function to detect available GPU memory
get_gpu_memory() {
    if command -v nvidia-smi &> /dev/null; then
        memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
        echo $memory
    else
        echo 0
    fi
}

# Get available GPU memory
GPU_MEM=$(get_gpu_memory)

# Base configuration
BASE_CMD="python src/train.py --data-dir $DATA_DIR"

if [ $GPU_MEM -eq 0 ]; then
    # CPU-only configuration
    echo "Running on CPU..."
    $BASE_CMD \
        --gpus 0 \
        --batch-size 32 \
        --num-workers 4 \
        --precision 32

elif [ $GPU_MEM -lt 16000 ]; then
    # Low-memory GPU configuration (e.g., 8GB GPU)
    echo "Running on low-memory GPU..."
    $BASE_CMD \
        --gpus 1 \
        --batch-size 64 \
        --precision 16-mixed \
        --grad-accum 4 \
        --num-workers 4

elif [ $GPU_MEM -lt 32000 ]; then
    # Medium-memory GPU configuration (e.g., 16GB GPU)
    echo "Running on medium-memory GPU..."
    $BASE_CMD \
        --gpus 1 \
        --batch-size 128 \
        --precision 16-mixed \
        --grad-accum 2 \
        --num-workers 8

else
    # High-memory GPU or multi-GPU configuration
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    echo "Running on $NUM_GPUS high-memory GPUs..."
    $BASE_CMD \
        --gpus -1 \
        --batch-size 256 \
        --precision 16-mixed \
        --num-workers 8 \
        --scale-lr
fi 