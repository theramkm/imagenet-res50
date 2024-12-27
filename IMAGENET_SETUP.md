# ImageNet Setup Guide

## 1. Official Access
1. Go to [ImageNet's official website](https://image-net.org/download-images.php)
2. Create an account and sign in
3. Request access to ImageNet-1K (ILSVRC2012)
4. Once approved, you'll receive download links

## 2. Download Files
You'll need to download:
- ILSVRC2012_img_train.tar (138GB)
- ILSVRC2012_img_val.tar (6.3GB)
- meta.mat (metadata file)

## 3. Preprocessing Script
Create this script to prepare the dataset:

```python:scripts/prepare_imagenet.py
import os
import shutil
import tarfile
from pathlib import Path

def extract_tar(tar_path, extract_path):
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_path)

def prepare_val_dir(val_dir):
    """
    Reorganize validation directory to match PyTorch ImageFolder structure
    """
    # Download valprep.sh from:
    # https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
    os.system(f"bash valprep.sh {val_dir}")

def main():
    # Set paths
    root_dir = Path("/path/to/imagenet")
    train_tar = root_dir / "ILSVRC2012_img_train.tar"
    val_tar = root_dir / "ILSVRC2012_img_val.tar"
    
    train_dir = root_dir / "train"
    val_dir = root_dir / "val"
    
    # Create directories
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Process training data
    print("Extracting training data...")
    extract_tar(train_tar, train_dir)
    
    # Extract individual class archives
    for tar_file in train_dir.glob("*.tar"):
        class_name = tar_file.stem
        class_dir = train_dir / class_name
        class_dir.mkdir(exist_ok=True)
        extract_tar(tar_file, class_dir)
        tar_file.unlink()  # Remove tar file after extraction
    
    # Process validation data
    print("Extracting validation data...")
    extract_tar(val_tar, val_dir)
    prepare_val_dir(val_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()
```

## 4. Setup Steps

1. Create directory structure:
```bash
mkdir -p /path/to/imagenet/{train,val}
```

2. Download required files:
```bash
# Download valprep.sh
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
chmod +x valprep.sh

# Download and run preprocessing script
python scripts/prepare_imagenet.py
```

3. Verify directory structure:
```bash
tree -L 2 /path/to/imagenet
```

Expected output:
```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

## 5. AWS-specific Setup (if using AWS)

1. Create an EBS volume:
```bash
aws ec2 create-volume \
    --volume-type gp3 \
    --size 200 \
    --availability-zone your-az \
    --iops 16000 \
    --throughput 1000
```

2. Mount the volume:
```bash
sudo mkfs -t ext4 /dev/xvdf
sudo mount /dev/xvdf /path/to/imagenet
```

3. Set up fast data loading:
```bash
# Add to /etc/fstab
/dev/xvdf /path/to/imagenet ext4 defaults,nofail,noatime 0 0
```

## 6. Validation

Test your setup with:
```bash
python src/train.py \
    --data-dir /path/to/imagenet \
    --batch-size 32 \
    --epochs 1 \
    --gpus 1
```

## Performance Tips

1. Use multiple workers for data loading:
```bash
--num-workers $(nproc)  # Use number of CPU cores
```

2. Enable memory pinning:
```bash
export CUDA_LAUNCH_BLOCKING=1
```

3. Monitor I/O performance:
```bash
iostat -x 1
```

4. If using AWS:
- Use instances with local NVMe storage (e.g., p4d.24xlarge)
- Consider using FSx for Lustre for better I/O performance
- Use placement groups for multi-node training 