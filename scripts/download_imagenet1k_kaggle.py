import os
import subprocess
from pathlib import Path
import kaggle
import zipfile
import shutil

def setup_kaggle():
    """Verify Kaggle credentials"""
    if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
        raise FileNotFoundError("Please setup Kaggle API credentials first!")

def download_imagenet1k(output_dir: str):
    """
    Download ImageNet-1K from Kaggle
    Dataset: https://www.kaggle.com/competitions/imagenet-object-localization-challenge
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading ImageNet-1K from Kaggle...")
    kaggle.api.competition_download_files(
        'imagenet-object-localization-challenge',
        path=output_dir
    )
    
    # Extract the dataset
    print("Extracting files...")
    zip_path = output_dir / 'imagenet-object-localization-challenge.zip'
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    # Organize files into train/val structure
    print("Organizing dataset...")
    organize_dataset(output_dir)
    
    # Cleanup
    print("Cleaning up...")
    zip_path.unlink()
    
    print("Done! Dataset is ready at:", output_dir)

def organize_dataset(data_dir: Path):
    """Organize the Kaggle dataset into PyTorch ImageFolder structure"""
    # Create train and val directories
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Move validation images to correct structure
    val_annotations = data_dir / 'LOC_val_solution.csv'
    if val_annotations.exists():
        with open(val_annotations, 'r') as f:
            next(f)  # Skip header
            for line in f:
                img_name, class_id = line.strip().split(',')[0:2]
                class_name = class_id.split()[0]  # Get first class for multi-label
                
                # Create class directory in val
                class_dir = val_dir / class_name
                class_dir.mkdir(exist_ok=True)
                
                # Move image to class directory
                src = data_dir / 'ILSVRC/Data/CLS-LOC/val' / f'{img_name}.JPEG'
                if src.exists():
                    shutil.copy2(src, class_dir / f'{img_name}.JPEG')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download ImageNet-1K from Kaggle')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save the dataset')
    args = parser.parse_args()
    
    setup_kaggle()
    download_imagenet1k(args.output_dir) 