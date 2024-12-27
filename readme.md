# ImageNet Training with PyTorch Lightning

This repository contains a PyTorch Lightning implementation for training models on the ImageNet dataset. The implementation includes support for distributed training, mixed precision, and AWS spot instance handling.

## Features

- Distributed training support with DDP
- Mixed precision training (FP16)
- AWS spot instance interruption handling
- Automatic checkpointing and training resumption
- WandB integration for experiment tracking
- Cosine learning rate scheduling
- Efficient data loading with proper augmentations

## Requirements

- Python 3.8+
- CUDA-capable GPU
- ImageNet dataset

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/imagenet-training.git
cd imagenet-training
```

## Dataset Preparation

The ImageNet dataset should be organized in the following structure:

```
/path/to/imagenet/
├── train/
│ ├── n01440764/
│ │ ├── n01440764_10026.JPEG
│ │ └── ...
│ └── ...
└── val/
├── n01440764/
│ ├── ILSVRC2012_val_00000293.JPEG
│ └── ...
└── ...
```

## Training

Basic training command:
```bash
python src/train.py --data-dir /path/to/imagenet
```

For distributed training:

```bash
python src/train.py --data-dir /path/to/imagenet --distributed
```

### Command Line Arguments

- `--batch-size`: Batch size for training (default: 256)
- `--epochs`: Number of epochs to train (default: 100)
- `--learning-rate`: Initial learning rate (default: 0.1)
- `--weight-decay`: Weight decay (default: 1e-4)
- `--data-dir`: Path to ImageNet dataset (required)
- `--num-workers`: Number of data loading workers (default: 8)
- `--gpus`: Number of GPUs to use (-1 for all available)
- `--wandb-project`: Weights & Biases project name (default: imagenet-training)
- `--checkpoint-dir`: Directory to save checkpoints (default: checkpoints)
- `--resume`: Path to checkpoint to resume training from

### Examples

Train with specific batch size and learning rate:
```bash
python src/train.py --data-dir /path/to/imagenet --batch-size 128 --learning-rate 0.05
```

Resume training from checkpoint:
```bash
python src/train.py --data-dir /path/to/imagenet --resume checkpoints/last.ckpt
```

## AWS Spot Instance Usage

When running on AWS spot instances, the training script automatically handles interruptions by:
1. Detecting the SIGTERM signal
2. Saving a checkpoint
3. Gracefully shutting down

To resume training after a spot instance interruption:
```bash
python src/train.py --data-dir /path/to/imagenet --resume checkpoints/interrupted.ckpt
```

## Monitoring

Training progress can be monitored through Weights & Biases. The following metrics are tracked:
- Training loss and accuracy
- Validation loss and accuracy
- Learning rate
- GPU memory usage
- Training speed (images/second)

## Project Structure

```
project/
├── src/
│   ├── models/
│   │   └── imagenet_module.py
│   ├── data/
│   │   └── imagenet_datamodule.py
│   └── train.py
├── checkpoints/
├── requirements.txt
└── README.md
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

```bash
python src/train.py \
    --data-dir /path/to/imagenet \
    --batch-size 128 \
    --learning-rate 0.05 \
    --epochs 90 \
    --num-workers 4 \
    --wandb-project my-imagenet-training
```
