import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def create_mini_imagenet(source_dir: str, target_dir: str, num_classes: int = 100, images_per_class: int = 100):
    """
    Create a mini version of ImageNet for testing.
    
    Args:
        source_dir: Path to full ImageNet
        target_dir: Where to save mini-ImageNet
        num_classes: Number of classes to include
        images_per_class: Number of images per class
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Create target directories
    for split in ['train', 'val']:
        (target_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Get list of all classes
    classes = sorted([d for d in (source_dir / 'train').iterdir() if d.is_dir()])
    selected_classes = random.sample(classes, num_classes)
    
    print(f"Creating mini-ImageNet with {num_classes} classes...")
    for class_dir in tqdm(selected_classes):
        class_name = class_dir.name
        
        # Process train split
        src_train = source_dir / 'train' / class_name
        dst_train = target_dir / 'train' / class_name
        dst_train.mkdir(parents=True, exist_ok=True)
        
        # Copy random subset of training images
        train_images = list(src_train.glob('*.JPEG'))
        selected_train = random.sample(train_images, min(images_per_class, len(train_images)))
        for img in selected_train:
            shutil.copy2(img, dst_train / img.name)
        
        # Process validation split
        src_val = source_dir / 'val' / class_name
        dst_val = target_dir / 'val' / class_name
        dst_val.mkdir(parents=True, exist_ok=True)
        
        # Copy all validation images for selected classes
        for img in src_val.glob('*.JPEG'):
            shutil.copy2(img, dst_val / img.name)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create mini-ImageNet dataset')
    parser.add_argument('--source-dir', type=str, required=True,
                       help='Path to full ImageNet dataset')
    parser.add_argument('--target-dir', type=str, required=True,
                       help='Where to save mini-ImageNet')
    parser.add_argument('--num-classes', type=int, default=100,
                       help='Number of classes to include (default: 100)')
    parser.add_argument('--images-per-class', type=int, default=100,
                       help='Number of training images per class (default: 100)')
    
    args = parser.parse_args()
    create_mini_imagenet(**vars(args)) 