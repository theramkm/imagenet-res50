# Testing with Kaggle's ImageNet-1K

## Setup

1. Install Kaggle API and set up credentials:
```bash
pip install kaggle
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

2. Download and prepare the dataset:
```bash
python scripts/download_imagenet1k_kaggle.py --output-dir /path/to/kaggle/imagenet
```

## Testing Pipeline

1. Quick validation (5-10 minutes):
```bash
bash scripts/test_kaggle_imagenet.sh
```

2. Memory usage test:
```bash
python src/train.py \
    --data-dir /path/to/kaggle/imagenet \
    --batch-size 32 \
    --epochs 1 \
    --limit-train-batches 0.01 \
    --gpus 1 \
    --precision 16-mixed
```

3. Full training test (few hours):
```bash
python src/train.py \
    --data-dir /path/to/kaggle/imagenet \
    --batch-size 64 \
    --epochs 3 \
    --gpus 1 \
    --precision 16-mixed \
    --wandb-project imagenet1k-test
```

## Common Issues and Solutions

1. Memory Issues:
- Reduce batch size
- Use mixed precision training
- Reduce number of workers
- Enable gradient accumulation

2. Performance Issues:
- Check disk I/O with `iostat -x 1`
- Monitor GPU usage with `nvidia-smi -l 1`
- Adjust num_workers based on CPU cores

3. Dataset Issues:
- Verify dataset structure with `tree -L 2 /path/to/kaggle/imagenet`
- Check image counts: `find /path/to/kaggle/imagenet -name "*.JPEG" | wc -l`

## Resource Requirements

Minimum:
- 16GB GPU memory
- 32GB RAM
- 150GB disk space

Recommended:
- 24GB+ GPU memory
- 64GB RAM
- 250GB SSD/NVMe storage 