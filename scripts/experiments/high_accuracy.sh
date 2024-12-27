#!/bin/bash

# Example configurations that have achieved >70% top-1 accuracy

# Configuration 1: Basic high accuracy setup
python src/train.py \
    --data-dir /path/to/imagenet \
    --batch-size 256 \
    --epochs 120 \
    --learning-rate 0.1 \
    --weight-decay 1e-4 \
    --label-smoothing 0.1 \
    --auto-augment \
    --cutmix-prob 1.0 \
    --mixup-alpha 0.2 \
    --warmup-epochs 5 \
    --tags "high_accuracy" "baseline"

# Configuration 2: Longer training with stronger augmentation
python src/train.py \
    --data-dir /path/to/imagenet \
    --batch-size 256 \
    --epochs 150 \
    --learning-rate 0.1 \
    --weight-decay 2e-4 \
    --label-smoothing 0.1 \
    --auto-augment \
    --cutmix-prob 1.0 \
    --mixup-alpha 0.4 \
    --warmup-epochs 8 \
    --color-jitter 0.5 \
    --tags "high_accuracy" "strong_aug"

# Configuration 3: EfficientNet setup
python src/train.py \
    --data-dir /path/to/imagenet \
    --model efficientnet_b0 \
    --batch-size 128 \
    --epochs 350 \
    --learning-rate 0.256 \
    --weight-decay 1e-5 \
    --label-smoothing 0.1 \
    --auto-augment \
    --cutmix-prob 1.0 \
    --mixup-alpha 0.2 \
    --warmup-epochs 5 \
    --grad-accum 2 \
    --tags "high_accuracy" "efficientnet" 