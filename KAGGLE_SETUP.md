# Download ImageNet from Kaggle

## 1. Setup Kaggle

1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to "Account" → "Create API Token"
3. Download `kaggle.json` and place it in `~/.kaggle/`
4. Set proper permissions:
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## 2. Download Script

Create and run this script:

```python:scripts/download_imagenet_kaggle.py
import os
import subprocess
from pathlib import Path

def setup_kaggle():
    """Verify Kaggle credentials"""
    try:
        import kaggle
    except ImportError:
        subprocess.run(['pip', 'install', 'kaggle'])
        import kaggle
    
    if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
        raise FileNotFoundError("Please setup Kaggle API credentials first!")

def download_imagenet(output_dir: str):
    """Download ImageNet from Kaggle"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    datasets = [
        'imagenet-object-localization-challenge',  # Main dataset
        'imagenet-object-localization-metadata'    # Metadata
    ]
    
    for dataset in datasets:
        print(f"Downloading {dataset}...")
        subprocess.run([
            'kaggle', 'competitions', 'download',
            '-c', dataset,
            '-p', str(output_dir)
        ])
        
        # Extract archives
        archives = list(output_dir.glob('*.zip'))
        for archive in archives:
            print(f"Extracting {archive}...")
            subprocess.run(['unzip', '-q', str(archive), '-d', str(output_dir)])
            archive.unlink()  # Remove zip after extraction

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to download ImageNet to')
    args = parser.parse_args()
    
    setup_kaggle()
    download_imagenet(args.output_dir)
```

## 3. Usage

```bash
# Install requirements
pip install kaggle

# Download and extract ImageNet
python scripts/download_imagenet_kaggle.py --output-dir /path/to/imagenet
``` 