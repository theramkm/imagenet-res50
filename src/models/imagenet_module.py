import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR

class ImageNetModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 1000,
        learning_rate: float = 0.1,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 256
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize ResNet50 model
        self.model = models.resnet50(weights=None, num_classes=num_classes)
        
        # Initialize weights using He initialization
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)
        
        # Track training speed
        self.batch_start_time = None
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        self.batch_start_time = torch.cuda.Event(enable_timing=True)
        self.batch_end_time = torch.cuda.Event(enable_timing=True)
        
        self.batch_start_time.record()
        
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        
        self.batch_end_time.record()
        torch.cuda.synchronize()
        batch_time = self.batch_start_time.elapsed_time(self.batch_end_time)
        
        # Calculate images per second
        images_per_second = self.hparams.batch_size / (batch_time / 1000.0)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc(logits, y), on_step=True, on_epoch=True)
        self.log('train_images_per_second', images_per_second, on_step=True)
        self.log('gpu_memory_usage', torch.cuda.memory_allocated() / 1024**2, on_step=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        self.log('val_acc', self.val_acc(logits, y), on_epoch=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs,
            eta_min=0
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': 'val_loss'
            }
        } 