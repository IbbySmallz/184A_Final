"""
Example usage script demonstrating how to use the malaria detection system.
This script shows a complete workflow from training to evaluation.
"""

import os
import torch
from train import train, train_epoch, validate
from evaluate import evaluate_model
from data_preprocessing import get_data_loaders
from models import get_model
from utils import ensure_dir

def main():
    # Configuration
    # Using minimal dataset included in repository for demo
    data_dir = 'data_minimal/cell_images'  # Minimal dataset (400 images) included in repo
    model_name = 'mobilenetv2'  # Options: 'mobilenetv2', 'efficientnetb0'
    batch_size = 32
    num_epochs = 20  # Fast training with frozen backbone
    image_size = 224
    learning_rate = 0.001
    
    # Note: The minimal dataset is included in the repository at data_minimal/cell_images/
    # For full dataset, download from Kaggle and change data_dir to 'data/cell_images'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size
    )
    
    # Create model
    print(f"\nCreating {model_name} model...")
    model = get_model(
        model_name=model_name,
        pretrained=True,  # Use pretrained weights
        freeze_backbone=True  # Freeze backbone (only train classifier head)
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("  (Backbone is frozen - only classifier head will be trained)\n")
    
    # Create directories
    ensure_dir('models')
    ensure_dir('results')
    
    # Train model
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device=device,
        learning_rate=learning_rate,
        model_name=model_name,
        save_dir='models'
    )
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # Load best model
    best_model_path = f'models/{model_name}_best.pth'
    if os.path.exists(best_model_path):
        from utils import load_checkpoint
        load_checkpoint(best_model_path, model)
        
        # Evaluate
        metrics, labels, preds, probas, images = evaluate_model(
            model, test_loader, device, save_dir='results'
        )
        
        print("\nEvaluation complete! Results saved to 'results/' directory")
    else:
        print(f"Best model not found at {best_model_path}")

if __name__ == '__main__':
    main()

