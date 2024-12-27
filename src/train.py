import os
import signal
import sys
import argparse
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from models.imagenet_module import ImageNetModule
from data.imagenet_datamodule import ImageNetDataModule

class SpotInstanceHandler:
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGTERM, self.handle_sigterm)
    
    def handle_sigterm(self, signum, frame):
        print("Received SIGTERM signal. Preparing for shutdown...")
        self.interrupted = True

def parse_args():
    parser = argparse.ArgumentParser(description='ImageNet Training with PyTorch Lightning')
    
    # Training hyperparameters
    training = parser.add_argument_group('Training Hyperparameters')
    training.add_argument('--batch-size', type=int, default=256,
                         help='batch size per GPU (default: 256)')
    training.add_argument('--epochs', type=int, default=100,
                         help='number of epochs (default: 100)')
    training.add_argument('--learning-rate', type=float, default=0.1,
                         help='initial learning rate (default: 0.1)')
    training.add_argument('--weight-decay', type=float, default=1e-4,
                         help='weight decay (default: 1e-4)')
    training.add_argument('--momentum', type=float, default=0.9,
                         help='SGD momentum (default: 0.9)')
    training.add_argument('--label-smoothing', type=float, default=0.1,
                         help='label smoothing factor (default: 0.1)')
    
    # Learning rate schedule
    lr = parser.add_argument_group('Learning Rate Schedule')
    lr.add_argument('--lr-scheduler', type=str, default='cosine',
                    choices=['cosine', 'step', 'linear'],
                    help='learning rate scheduler (default: cosine)')
    lr.add_argument('--warmup-epochs', type=int, default=5,
                    help='number of warmup epochs (default: 5)')
    lr.add_argument('--min-lr', type=float, default=1e-5,
                    help='minimum learning rate (default: 1e-5)')
    
    # Data augmentation
    augment = parser.add_argument_group('Data Augmentation')
    augment.add_argument('--color-jitter', type=float, default=0.4,
                        help='color jitter factor (default: 0.4)')
    augment.add_argument('--auto-augment', action='store_true',
                        help='use AutoAugment policy')
    augment.add_argument('--cutmix-prob', type=float, default=0.0,
                        help='cutmix probability (default: 0.0)')
    augment.add_argument('--mixup-alpha', type=float, default=0.0,
                        help='mixup alpha (default: 0.0)')
    
    # Model configuration
    model = parser.add_argument_group('Model Configuration')
    model.add_argument('--model', type=str, default='resnet50',
                      choices=['resnet50', 'resnet101', 'efficientnet_b0'],
                      help='model architecture (default: resnet50)')
    model.add_argument('--dropout', type=float, default=0.0,
                      help='dropout rate (default: 0.0)')
    
    # Performance optimization
    perf = parser.add_argument_group('Performance Optimization')
    perf.add_argument('--precision', type=str, default='16-mixed',
                     choices=['32', '16-mixed', 'bf16-mixed'],
                     help='training precision (default: 16-mixed)')
    perf.add_argument('--grad-accum', type=int, default=1,
                     help='gradient accumulation steps (default: 1)')
    perf.add_argument('--grad-clip', type=float, default=1.0,
                     help='gradient clipping value (default: 1.0)')
    
    # Hardware/System
    hw = parser.add_argument_group('Hardware/System')
    hw.add_argument('--data-dir', type=str, required=True,
                   help='path to ImageNet dataset')
    hw.add_argument('--num-workers', type=int, default=8,
                   help='number of data loading workers per GPU (default: 8)')
    hw.add_argument('--gpus', type=int, default=-1,
                   help='number of GPUs (-1 for all available)')
    
    # Logging and checkpointing
    log = parser.add_argument_group('Logging and Checkpointing')
    log.add_argument('--wandb-project', type=str, default='imagenet-training',
                    help='wandb project name (default: imagenet-training)')
    log.add_argument('--wandb-entity', type=str, default=None,
                    help='wandb entity name (default: None)')
    log.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                    help='path to save checkpoints (default: checkpoints)')
    log.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
    log.add_argument('--tags', type=str, nargs='+', default=[],
                    help='tags for wandb run')
    
    # Testing/Debug options
    debug = parser.add_argument_group('Testing/Debug Options')
    debug.add_argument('--fast-dev-run', action='store_true',
                      help='Run only 1 train, val batch and program (for testing)')
    debug.add_argument('--limit-train-batches', type=float, default=1.0,
                      help='Limit train batches (1.0 = full dataset)')
    debug.add_argument('--limit-val-batches', type=float, default=1.0,
                      help='Limit validation batches (1.0 = full dataset)')
    debug.add_argument('--num-classes', type=int, default=1000,
                      help='Number of classes (use 100 for mini-ImageNet)')
    debug.add_argument('--dry-run', action='store_true',
                      help='Run full training loop with minimal data')
    
    # Scaling options
    scaling = parser.add_argument_group('Scaling Options')
    scaling.add_argument('--auto-scale-batch-size', action='store_true',
                        help='Automatically find the largest batch size that fits in memory')
    scaling.add_argument('--auto-lr-find', action='store_true',
                        help='Automatically find optimal learning rate')
    scaling.add_argument('--scale-lr', action='store_true',
                        help='Scale learning rate with batch size')
    scaling.add_argument('--base-batch-size', type=int, default=256,
                        help='Base batch size for scaling calculations')
    scaling.add_argument('--min-batch-size', type=int, default=32,
                        help='Minimum batch size when auto scaling')
    scaling.add_argument('--max-batch-size', type=int, default=512,
                        help='Maximum batch size when auto scaling')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize spot instance handler
    spot_handler = SpotInstanceHandler()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize wandb logger with multi-GPU support
    logger = WandbLogger(project=args.wandb_project, log_model=True)
    logger.log_hyperparams(vars(args))
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='imagenet-{epoch:02d}-{val_acc:.3f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Determine accelerator and devices based on system
    if torch.cuda.is_available():
        accelerator = "gpu"
        if args.gpus == -1:
            devices = torch.cuda.device_count()
        else:
            devices = min(args.gpus, torch.cuda.device_count())
    else:
        accelerator = "cpu"
        devices = None
        if args.gpus != 0:
            print("Warning: No GPU available, falling back to CPU training")
    
    # Determine strategy based on hardware
    if devices and devices > 1:
        strategy = 'ddp_find_unused_parameters_false'
        # Enable sync_batchnorm for multi-GPU
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        strategy = None  # Use default strategy for single GPU/CPU
    
    # Adjust batch size and learning rate for different setups
    effective_batch_size = args.batch_size
    if devices and devices > 1:
        effective_batch_size *= devices
    
    # Scale learning rate with batch size
    scaled_lr = args.learning_rate * (effective_batch_size / 256)
    
    # Initialize trainer with hardware-aware settings
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=args.precision if accelerator == "gpu" else "32",
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=50,
        sync_batchnorm=True if devices and devices > 1 else False,
        use_distributed_sampler=True if devices and devices > 1 else False,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.grad_accum,
        deterministic=False,
        benchmark=True if accelerator == "gpu" else False,
        # Testing options
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )
    
    # Initialize data module and model
    data_module = ImageNetDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    model = ImageNetModule(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Resume training from checkpoint if specified
    ckpt_path = args.resume
    if not ckpt_path and os.path.exists(os.path.join(args.checkpoint_dir, 'last.ckpt')):
        ckpt_path = os.path.join(args.checkpoint_dir, 'last.ckpt')
    
    # Start training
    try:
        trainer.fit(model, data_module, ckpt_path=ckpt_path)
    except Exception as e:
        if spot_handler.interrupted:
            print("Spot instance interruption detected. Saving checkpoint...")
            # Save checkpoint
            trainer.save_checkpoint(os.path.join(args.checkpoint_dir, 'interrupted.ckpt'))
            sys.exit(0)
        else:
            raise e

if __name__ == '__main__':
    main() 