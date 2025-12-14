# Model training script
# This script defines the model architecture and runs the training loop.

import json
import os
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchinfo import summary

from utils import setup_logger
from models import create_model
from config import (
    DATA_DIR, PROCESSED_DATA_DIR, MODEL_SAVE_PATH,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, EARLY_STOPPING_PATIENCE,
    IMAGE_SIZE, NUM_CLASSES,
    AUG_RESIZE, AUG_BRIGHTNESS, AUG_CONTRAST, AUG_SATURATION, AUG_ROTATION,
    IMAGE_MEAN, IMAGE_STD,
    LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE
)

logger = setup_logger(__name__)


def get_baseline_accuracy(train_loader):
    """
    Compute baseline accuracy using majority class prediction.
    Returns the accuracy if we always predict the most common class.
    """
    logger.info("\n" + "=" * 50)
    logger.info("BASELINE MODEL (Majority Class Classifier)")
    logger.info("=" * 50)
    
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())
    
    counter = Counter(all_labels)
    most_common_class, most_common_count = counter.most_common(1)[0]
    
    total = len(all_labels)
    baseline_acc = 100.0 * most_common_count / total
    
    logger.info(f"Training set class distribution: {dict(counter)}")
    logger.info(f"Most common class: {most_common_class}")
    logger.info(f"Baseline accuracy (always predict class {most_common_class}): {baseline_acc:.2f}%")
    logger.info("=" * 50 + "\n")
    
    return baseline_acc


def get_data_loaders():
    """Create train and validation data loaders with appropriate transforms."""
    processed_dir = Path(PROCESSED_DATA_DIR)
    
    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_dir}. "
            "Please run 01-data-preprocessing.py first."
        )
    
    train_transforms = transforms.Compose([
        transforms.Resize(AUG_RESIZE),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=AUG_BRIGHTNESS, contrast=AUG_CONTRAST, saturation=AUG_SATURATION),
        transforms.RandomRotation(AUG_ROTATION),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(AUG_RESIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])
    
    train_dataset = datasets.ImageFolder(
        root=processed_dir / 'train',
        transform=train_transforms
    )
    
    val_dataset = datasets.ImageFolder(
        root=processed_dir / 'val',
        transform=val_transforms
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} images")
    logger.info(f"Validation dataset: {len(val_dataset)} images")
    logger.info(f"Class names: {train_dataset.classes}")
    
    return train_loader, val_loader, train_dataset.classes


def create_model(num_classes):
    """
    Create a custom CNN model for ankle alignment classification.
    
    Args:
        num_classes: Number of output classes
    """
    logger.info("\n" + "=" * 50)
    logger.info("MODEL ARCHITECTURE")
    logger.info("=" * 50)
    
    from models import create_model as create_custom_model
    model = create_custom_model('anklenet', num_classes)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: AnkleNet (Custom CNN)")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info("=" * 50 + "\n")
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    
    return val_loss, val_acc


def train():
    """Main training function."""
    logger.info("=" * 50)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Image size: {IMAGE_SIZE}")
    logger.info(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    logger.info(f"Number of classes: {NUM_CLASSES}")
    logger.info("=" * 50 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    logger.info("Loading data...")
    train_loader, val_loader, class_names = get_data_loaders()
    
    baseline_acc = get_baseline_accuracy(train_loader)
    
    model = create_model(NUM_CLASSES)
    model = model.to(device)
    
    logger.info("\nModel Architecture Summary:")
    logger.info("\n" + "=" * 95)
    summary(model, input_size=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE), device=device)
    logger.info("=" * 95 + "\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )
    
    logger.info("=" * 50)
    logger.info("TRAINING PROGRESS")
    logger.info("=" * 50)
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
            }, MODEL_SAVE_PATH)
            
            logger.info(f"New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            logger.info(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    total_time = time.time() - start_time
    logger.info("=" * 50)
    logger.info(f"Training completed in {total_time / 60:.2f} minutes")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Baseline accuracy: {baseline_acc:.2f}%")
    logger.info(f"Improvement over baseline: {best_val_acc - baseline_acc:.2f}%")
    logger.info(f"Model saved to: {MODEL_SAVE_PATH}")
    logger.info("=" * 50)


if __name__ == "__main__":
    train()
