"""
Evaluation script for malaria detection models.
Computes comprehensive metrics and generates visualizations.
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from data_preprocessing import get_data_loaders
from models import get_model
from utils import load_checkpoint, calculate_metrics, print_metrics, \
                 create_confusion_matrix, ensure_dir
from visualize import plot_roc_curve


def evaluate_model(model, test_loader, device, save_dir=None):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probas = []
    all_images = []
    
    criterion = nn.BCELoss()
    running_loss = 0.0
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels = labels.float().to(device)  # Convert to float for BCE loss
            
            # Forward pass
            outputs = model(images)  # Binary output (sigmoid)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Get predictions
            preds = (outputs > 0.5).float()  # Binary prediction threshold
            probas = outputs  # Already sigmoid probabilities
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
            all_images.extend(images.cpu())
    
    # Calculate metrics
    test_loss = running_loss / len(test_loader)
    metrics = calculate_metrics(all_labels, all_preds, all_probas)
    
    # Print results
    print(f"\nTest Loss: {test_loss:.4f}")
    print_metrics(metrics, 'Test')
    
    # Save visualizations
    if save_dir:
        ensure_dir(save_dir)
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        create_confusion_matrix(all_labels, all_preds, save_path=cm_path)
        
        roc_path = os.path.join(save_dir, 'roc_curve.png')
        plot_roc_curve(all_labels, all_probas, save_path=roc_path)
    
    return metrics, all_labels, all_preds, all_probas, all_images


def find_misclassified_samples(model, test_loader, device, num_samples=10, save_dir=None):
    """Find and visualize misclassified samples."""
    model.eval()
    misclassified = []
    
    print("\nFinding misclassified samples...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Analyzing'):
            images = images.to(device)
            labels = labels.float().to(device)
            
            outputs = model(images)  # Binary output (sigmoid)
            preds = (outputs > 0.5).float()  # Binary prediction threshold
            
            # Find misclassified
            incorrect = (preds != labels).cpu().numpy()
            for i in range(len(images)):
                if incorrect[i] and len(misclassified) < num_samples:
                    misclassified.append({
                        'image': images[i].cpu(),
                        'true_label': int(labels[i].cpu().item()),
                        'pred_label': int(preds[i].cpu().item()),
                        'proba': outputs[i].cpu().item()  # Already sigmoid probability
                    })
            
            if len(misclassified) >= num_samples:
                break
    
    print(f"\nFound {len(misclassified)} misclassified samples")
    
    if save_dir and len(misclassified) > 0:
        from visualize import visualize_misclassified
        visualize_misclassified(misclassified, save_path=os.path.join(save_dir, 'misclassified_samples.png'))
    
    return misclassified


def main():
    parser = argparse.ArgumentParser(description='Evaluate malaria detection model')
    parser.add_argument('--data_dir', type=str, default='data/cell_images',
                       help='Path to cell_images directory')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='mobilenetv2',
                       choices=['mobilenetv2', 'efficientnetb0', 'efficientnet'],
                       help='Model architecture type')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size for input')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save evaluation results')
    parser.add_argument('--num_misclassified', type=int, default=10,
                       help='Number of misclassified samples to visualize')
    
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
    print(f"\nLoading {args.model_type} model...")
    model = get_model(
        model_name=args.model_type,
        pretrained=True,
        freeze_backbone=True
    )
    model = model.to(device)
    
    # Load checkpoint
    load_checkpoint(args.model_path, model)
    
    # Evaluate
    ensure_dir(args.save_dir)
    metrics, labels, preds, probas, images = evaluate_model(
        model, test_loader, device, save_dir=args.save_dir
    )
    
    # Find misclassified samples
    misclassified = find_misclassified_samples(
        model, test_loader, device, 
        num_samples=args.num_misclassified,
        save_dir=args.save_dir
    )
    
    # Save metrics to file
    metrics_file = os.path.join(args.save_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Evaluation Metrics\n")
        f.write("=" * 50 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"\nMetrics saved to {metrics_file}")


if __name__ == '__main__':
    main()

