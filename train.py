"""
Main training script for malaria detection using transfer learning.
Supports MobileNetV2 and EfficientNetB0 pretrained models with frozen backbones.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

from data_preprocessing import get_data_loaders
from models import get_model
from utils import save_checkpoint, load_checkpoint, calculate_metrics, print_metrics, \
                 plot_training_history, ensure_dir


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.float().to(device)  # Convert to float for BCE loss
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)  # Binary output (sigmoid)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        preds = (outputs > 0.5).float()  # Binary prediction threshold
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(train_loader)
    metrics = calculate_metrics(all_labels, all_preds)
    
    return epoch_loss, metrics['accuracy']


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probas = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            labels = labels.float().to(device)  # Convert to float for BCE loss
            
            # Forward pass
            outputs = model(images)  # Binary output (sigmoid)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            preds = (outputs > 0.5).float()  # Binary prediction threshold
            probas = outputs  # Already sigmoid probabilities
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    metrics = calculate_metrics(all_labels, all_preds, all_probas)
    
    return epoch_loss, metrics


def train(model, train_loader, val_loader, num_epochs, device, 
          learning_rate=0.001, weight_decay=1e-4, patience=10, 
          model_name='model', save_dir='models'):
    """Main training loop."""
    
    # Setup - Binary classification with BCE loss
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Ensure save directory exists
    ensure_dir(save_dir)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {train_loader.batch_size}\n")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_metrics['accuracy'])
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print_metrics(val_metrics, 'Validation')
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_val_loss = val_loss
            patience_counter = 0
            
            best_model_path = os.path.join(save_dir, f'{model_name}_best.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, best_val_acc, best_model_path)
            print(f"\n✓ New best model saved! (Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_metrics['accuracy'], checkpoint_path)
    
    # Plot training history
    history_path = os.path.join(save_dir, f'{model_name}_training_history.png')
    plot_training_history(train_losses, val_losses, train_accs, val_accs, history_path)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return model, train_losses, val_losses, train_accs, val_accs


def main():
    parser = argparse.ArgumentParser(description='Train malaria detection model using transfer learning')
    parser.add_argument('--data_dir', type=str, default='data/cell_images',
                       help='Path to cell_images directory')
    parser.add_argument('--model', type=str, default='mobilenetv2',
                       choices=['mobilenetv2', 'efficientnetb0', 'efficientnet'],
                       help='Pretrained model architecture (default: mobilenetv2)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size for input (default: 224)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--no_pretrained', action='store_true',
                       help='Do not use pretrained weights (default: use pretrained)')
    parser.add_argument('--unfreeze_backbone', action='store_true',
                       help='Unfreeze backbone for fine-tuning (default: frozen)')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models (default: models)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    # Model
    print(f"\nCreating {args.model} model...")
    model = get_model(
        model_name=args.model,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.unfreeze_backbone
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    if not args.unfreeze_backbone:
        print("  (Backbone is frozen - only classifier head will be trained)")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _, _ = load_checkpoint(args.resume, model)
    
    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        model_name=args.model,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()

