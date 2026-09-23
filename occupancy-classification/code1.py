"""
Training script for binary occupancy classification in chess piece detection.
Handles overlapping pieces, camera angle variations, and edge cases.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import numpy as np
import cv2
from PIL import Image
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import logging
import json
from typing import Tuple, List, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChessOccupancyDataset(Dataset):
    """
    Dataset for chess occupancy classification with special handling for:
    - Overlapping pieces
    - Camera angle variations
    - Lighting conditions
    - Partial occlusions
    """
    
    def __init__(self, 
                 root_dir: str,
                 transform=None,
                 augment_overlaps: bool = True,
                 handle_camera_angles: bool = True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.augment_overlaps = augment_overlaps
        self.handle_camera_angles = handle_camera_angles
        
        # Load samples
        self.samples = self._load_samples()
        
        # Class weights for handling imbalanced data
        self.class_counts = self._count_classes()
        
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load image paths and labels"""
        samples = []
        
        # Load occupied squares (label = 1)
        occupied_dirs = ['black_piece', 'white_piece']
        for piece_dir in occupied_dirs:
            piece_path = self.root_dir / piece_dir
            if piece_path.exists():
                for img_file in piece_path.glob('*.png'):
                    samples.append((str(img_file), 1))
                for img_file in piece_path.glob('*.jpg'):
                    samples.append((str(img_file), 1))
                    
        # Load empty squares (label = 0)
        empty_path = self.root_dir / 'empty'
        if empty_path.exists():
            for img_file in empty_path.glob('*.png'):
                samples.append((str(img_file), 0))
            for img_file in empty_path.glob('*.jpg'):
                samples.append((str(img_file), 0))
                
        logger.info(f"Loaded {len(samples)} samples")
        return samples
    
    def _count_classes(self) -> Dict[int, int]:
        """Count samples per class for weighted sampling"""
        counts = {0: 0, 1: 0}
        for _, label in self.samples:
            counts[label] += 1
        logger.info(f"Class distribution: Empty={counts[0]}, Occupied={counts[1]}")
        return counts
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for loss function"""
        total = sum(self.class_counts.values())
        weights = [total / (2 * count) for count in self.class_counts.values()]
        return torch.FloatTensor(weights)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Handle special cases
        if label == 1 and self.augment_overlaps:
            image = self._augment_for_overlaps(image)
            
        if self.handle_camera_angles:
            image = self._handle_camera_angles(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label
    
    def _augment_for_overlaps(self, image: np.ndarray) -> np.ndarray:
        """Augment images to simulate overlapping pieces"""
        if random.random() < 0.3:  # 30% chance of overlap augmentation
            h, w = image.shape[:2]
            
            # Create shadow/occlusion effect
            if random.random() < 0.5:
                # Add shadow from top
                shadow_height = random.randint(h//8, h//4)
                shadow = np.ones_like(image) * 0.6
                image[:shadow_height] = (image[:shadow_height] * shadow[:shadow_height]).astype(np.uint8)
            
            # Simulate partial occlusion
            if random.random() < 0.4:
                # Create random mask for partial occlusion
                mask_size = random.randint(10, min(w, h) // 3)
                x1 = random.randint(0, w - mask_size)
                y1 = random.randint(0, h - mask_size)
                
                # Darken occluded region
                occlusion_factor = random.uniform(0.3, 0.7)
                image[y1:y1+mask_size, x1:x1+mask_size] = \
                    (image[y1:y1+mask_size, x1:x1+mask_size] * occlusion_factor).astype(np.uint8)
        
        return image
    
    def _handle_camera_angles(self, image: np.ndarray) -> np.ndarray:
        """Augment for camera angle variations"""
        if random.random() < 0.4:  # 40% chance of perspective augmentation
            h, w = image.shape[:2]
            
            # Random perspective transformation
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            
            # Add random offset to corners to simulate camera angle
            offset = random.randint(5, 15)
            pts2 = np.float32([
                [random.randint(0, offset), random.randint(0, offset)],
                [w - random.randint(0, offset), random.randint(0, offset)],
                [random.randint(0, offset), h - random.randint(0, offset)],
                [w - random.randint(0, offset), h - random.randint(0, offset)]
            ])
            
            M = cv2.getPerspectiveTransform(pts1, pts2)
            image = cv2.warpPerspective(image, M, (w, h))
        
        return image


def get_transforms(input_size: Tuple[int, int], is_training: bool = True):
    """Get data transforms with robust augmentation"""
    
    if is_training:
        # Heavy augmentation for training
        transform = A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.2),
            A.GridDistortion(p=0.2),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        # Simple transforms for validation/test
        transform = A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    return transform


def create_weighted_sampler(dataset: ChessOccupancyDataset) -> WeightedRandomSampler:
    """Create weighted sampler for balanced training"""
    class_counts = dataset.class_counts
    total_samples = len(dataset)
    
    # Calculate weights for each sample
    sample_weights = []
    for _, label in dataset.samples:
        weight = total_samples / (2 * class_counts[label])
        sample_weights.append(weight)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=True
    )
    
    return sampler


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1, gamma: float = 2, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                device: torch.device) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with tqdm(dataloader, desc="Training") as pbar:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_epoch(model: nn.Module, 
                  dataloader: DataLoader, 
                  criterion: nn.Module, 
                  device: torch.device) -> Tuple[float, float, List, List]:
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        with tqdm(dataloader, desc="Validation") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_predictions)
    
    return epoch_loss, epoch_acc, all_labels, all_predictions


def plot_confusion_matrix(y_true: List, y_pred: List, save_path: str = None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Empty', 'Occupied'],
                yticklabels=['Empty', 'Occupied'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    # Configuration
    config = {
        'data_dir': './dataset',  # Set to your dataset folder
        'model_name': 'ResNet18',
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'patience': 15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './models/',
        'use_focal_loss': True,
        'use_weighted_sampler': True,
        'save_every_n_epochs': 10,
    }
    
    # Create save directory
    Path(config['save_dir']).mkdir(exist_ok=True)
    
    # Model definition
    import torchvision.models as models
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    input_size = (224, 224)
    
    device = torch.device(config['device'])
    model.to(device)
    
    # Datasets and dataloaders
    train_transform = get_transforms(input_size, is_training=True)
    val_transform = get_transforms(input_size, is_training=False)
    
    # Assuming train/val split directories exist
    train_dataset = ChessOccupancyDataset(
        root_dir=os.path.join(config['data_dir'], 'train'),
        transform=train_transform,
        augment_overlaps=True,
        handle_camera_angles=True
    )
    
    val_dataset = ChessOccupancyDataset(
        root_dir=os.path.join(config['data_dir'], 'valid'),
        transform=val_transform,
        augment_overlaps=False,
        handle_camera_angles=False
    )
    
    # Create samplers and dataloaders
    if config['use_weighted_sampler']:
        train_sampler = create_weighted_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, 
                                batch_size=config['batch_size'],
                                sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, 
                                batch_size=config['batch_size'],
                                shuffle=True)
    
    val_loader = DataLoader(val_dataset, 
                          batch_size=config['batch_size'],
                          shuffle=False)
    
    # Loss function and optimizer
    if config['use_focal_loss']:
        criterion = FocalLoss(alpha=1, gamma=2)
    else:
        class_weights = train_dataset.get_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['learning_rate'],
                           weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                   mode='min',
                                                   factor=0.5,
                                                   patience=5,
                                                   verbose=True)
    
    early_stopping = EarlyStopping(patience=config['patience'])
    
    # Training loop
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    logger.info("Starting training...")
    
    for epoch in range(config['num_epochs']):
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc, val_labels, val_predictions = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_predictions, average='weighted'
        )
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logger.info(f"Val Precision: {precision:.4f}, Val Recall: {recall:.4f}, Val F1: {f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            logger.info(f"New best model saved with validation accuracy: {val_acc:.4f}")
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Plot final confusion matrix
    plot_confusion_matrix(val_labels, val_predictions, 
                         os.path.join(config['save_dir'], 'confusion_matrix.png'))
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy Curves')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot([scheduler.get_last_lr()[0] for _ in range(len(train_losses))])
    plt.title('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'training_curves.png'))
    plt.show()
    
    logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()