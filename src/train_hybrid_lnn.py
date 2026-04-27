"""
Hybrid LNN Training - ResNet18 + LTC for 95-98% Accuracy
Combines transfer learning with Liquid Neural Networks
Optimized with ChatGPT suggestions for 95%+ accuracy
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
from pathlib import Path
import json

from model_resnet_ltc import ResNetLTC, FocalLoss


# Configuration
class Config:
    # Paths
    DATA_DIR = 'data/organized'
    MODEL_DIR = 'models'
    LOG_DIR = 'logs'
    
    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0003  # Slightly higher for better learning
    WEIGHT_DECAY = 0.0001
    
    # Model
    NUM_CLASSES = 2
    HIDDEN_UNITS = 128
    PRETRAINED = True  # Use ImageNet pretrained weights
    
    # Image
    IMAGE_SIZE = 224  # ResNet standard size
    
    # Training strategy
    EARLY_STOPPING_PATIENCE = 15
    LR_SCHEDULER_PATIENCE = 5
    LR_SCHEDULER_FACTOR = 0.5
    
    # Two-stage training
    FREEZE_EPOCHS = 10  # Train with frozen backbone first
    
    # Data split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    RANDOM_SEED = 42


config = Config()

# Set random seeds
torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.RANDOM_SEED)


class LungCancerDataset(Dataset):
    """Dataset for lung cancer images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image as RGB (ResNet expects 3 channels)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(is_training=True):
    """Get data augmentation transforms"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.9, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def load_data():
    """Load and split data"""
    print("Loading data...")
    
    data_dir = Path(config.DATA_DIR)
    
    # Load cancer images
    cancer_dir = data_dir / 'cancer'
    cancer_images = list(cancer_dir.glob('*.png')) + list(cancer_dir.glob('*.jpg'))
    cancer_labels = [1] * len(cancer_images)
    
    # Load normal images
    normal_dir = data_dir / 'normal'
    normal_images = list(normal_dir.glob('*.png')) + list(normal_dir.glob('*.jpg'))
    normal_labels = [0] * len(normal_images)
    
    # Combine
    all_images = cancer_images + normal_images
    all_labels = cancer_labels + normal_labels
    
    print(f"Total images: {len(all_images)}")
    print(f"Cancer: {len(cancer_images)} ({100*len(cancer_images)/len(all_images):.1f}%)")
    print(f"Normal: {len(normal_images)} ({100*len(normal_images)/len(all_images):.1f}%)")
    
    # Shuffle
    indices = np.random.permutation(len(all_images))
    all_images = [all_images[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    # Split
    n_train = int(len(all_images) * config.TRAIN_SPLIT)
    n_val = int(len(all_images) * config.VAL_SPLIT)
    
    train_images = all_images[:n_train]
    train_labels = all_labels[:n_train]
    
    val_images = all_images[n_train:n_train+n_val]
    val_labels = all_labels[n_train:n_train+n_val]
    
    test_images = all_images[n_train+n_val:]
    test_labels = all_labels[n_train+n_val:]
    
    print(f"\nSplit: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


def get_class_weights(labels):
    """Calculate class weights for imbalanced dataset"""
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler=None):
    """Train for one epoch with mixed precision"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels, all_probs


def plot_training_history(history, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss - Hybrid LNN', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy - Hybrid LNN', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training history to {save_path}")


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Cancer'],
                yticklabels=['Normal', 'Cancer'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Hybrid LNN (ResNet18 + LTC)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix to {save_path}")


def main():
    """Main training function"""
    print("="*70)
    print("HYBRID LNN TRAINING - ResNet18 + LTC")
    print("Target: 95-98% Accuracy with Transfer Learning")
    print("="*70)
    
    # Create directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Load data
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_data()
    
    # Calculate class weights
    class_weights = get_class_weights(train_labels)
    print(f"\nClass weights: Normal={class_weights[0]:.3f}, Cancer={class_weights[1]:.3f}")
    
    # Create datasets
    train_dataset = LungCancerDataset(train_images, train_labels, transform=get_transforms(True))
    val_dataset = LungCancerDataset(val_images, val_labels, transform=get_transforms(False))
    test_dataset = LungCancerDataset(test_images, test_labels, transform=get_transforms(False))
    
    # Create dataloaders (num_workers=0 for Windows)
    # Use shuffle=True for balanced training
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create model
    print("\nCreating Hybrid LNN Model...")
    print("✓ ResNet18 backbone (ImageNet pretrained)")
    print("✓ LTC classifier (Liquid Time-Constant)")
    print("✓ Two-stage training (freeze then fine-tune)")
    
    model = ResNetLTC(
        num_classes=config.NUM_CLASSES,
        hidden_units=config.HIDDEN_UNITS,
        pretrained=config.PRETRAINED
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # CrossEntropyLoss with class weights (more reliable than Focal Loss)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=config.LR_SCHEDULER_FACTOR, 
        patience=config.LR_SCHEDULER_PATIENCE
    )
    
    # Mixed precision scaler for faster training
    scaler = GradScaler() if torch.cuda.is_available() else None
    if scaler:
        print("✓ Using mixed precision training (2-3x faster)")
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("\nStarting training...")
    print(f"Stage 1 (Epochs 1-{config.FREEZE_EPOCHS}): Frozen backbone")
    print(f"Stage 2 (Epochs {config.FREEZE_EPOCHS+1}+): Fine-tuning all layers")
    print("-" * 70)
    
    # Stage 1: Train with frozen backbone
    model.freeze_backbone()
    
    for epoch in range(config.NUM_EPOCHS):
        # Stage 2: Unfreeze backbone after FREEZE_EPOCHS
        if epoch == config.FREEZE_EPOCHS:
            print("\n" + "="*70)
            print(f"STAGE 2: Unfreezing backbone for fine-tuning")
            print("="*70)
            model.unfreeze_backbone()
            # Reduce learning rate for fine-tuning (0.1x)
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.LEARNING_RATE * 0.1
            print(f"Learning rate reduced to: {config.LEARNING_RATE * 0.1:.6f}")
        
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler)
        
        # Validate
        val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': {
                    'hidden_units': config.HIDDEN_UNITS,
                    'num_classes': config.NUM_CLASSES,
                    'pretrained': config.PRETRAINED
                }
            }, os.path.join(config.MODEL_DIR, 'best_hybrid_lnn.pth'))
            print(f"✓ Saved best model with val_acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Plot training history
    plot_training_history(history, os.path.join(config.MODEL_DIR, 'training_history_hybrid_lnn.png'))
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load(os.path.join(config.MODEL_DIR, 'best_hybrid_lnn.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_preds, test_labels, test_probs = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Normal', 'Cancer']))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, os.path.join(config.MODEL_DIR, 'confusion_matrix_hybrid_lnn.png'))
    
    # AUC-ROC
    if len(np.unique(test_labels)) == 2:
        auc = roc_auc_score(test_labels, test_probs)
        print(f"\nAUC-ROC Score: {auc:.4f}")
    
    # Save final results
    results = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'auc_roc': auc if len(np.unique(test_labels)) == 2 else None,
        'total_epochs': len(history['train_acc']),
        'model_type': 'Hybrid LNN (ResNet18 + LTC)',
        'pretrained': config.PRETRAINED
    }
    
    with open(os.path.join(config.MODEL_DIR, 'hybrid_lnn_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*70)
    print(f"TRAINING COMPLETE - HYBRID LNN")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Model Type: Hybrid LNN (ResNet18 + LTC)")
    print(f"Transfer Learning: {'Yes' if config.PRETRAINED else 'No'}")
    print("="*70)


if __name__ == '__main__':
    main()
