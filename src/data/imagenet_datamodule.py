import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet
import os
import torch
import psutil

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 256,
        num_workers: int = 8,
        pin_memory: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # Automatically determine optimal number of workers
        if num_workers == -1:
            self.num_workers = os.cpu_count()
        else:
            self.num_workers = num_workers
            
        # Enable pin_memory only if CUDA is available
        self.pin_memory = pin_memory and torch.cuda.is_available()
        
        # Adjust prefetch factor based on available memory
        total_ram = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
        self.prefetch_factor = 2 if total_ram >= 32 else 1
        
        # Standard ImageNet transforms
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageNet(
                self.data_dir,
                split='train',
                transform=self.train_transforms
            )
            self.val_dataset = ImageNet(
                self.data_dir,
                split='val',
                transform=self.val_transforms
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True
        ) 