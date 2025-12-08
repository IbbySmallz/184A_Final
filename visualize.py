"""
Visualization tools for malaria detection results.
Includes functions for visualizing misclassified samples and model predictions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import seaborn as sns


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize a tensor image for visualization."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def visualize_misclassified(misclassified_samples, num_samples=None, save_path=None):
    """
    Visualize misclassified samples.
    
    Args:
        misclassified_samples: List of dictionaries with 'image', 'true_label', 
                              'pred_label', and 'proba' keys
        num_samples: Number of samples to visualize (None for all)
        save_path: Path to save the visualization
    """
    if num_samples is None:
        num_samples = len(misclassified_samples)
    else:
        num_samples = min(num_samples, len(misclassified_samples))
    
    class_names = ['Uninfected', 'Parasitized']
    
    # Calculate grid size
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        sample = misclassified_samples[idx]
        image = denormalize(sample['image'])
        image = torch.clamp(image, 0, 1)
        image = image.permute(1, 2, 0).numpy()
        
        true_label = sample['true_label']
        pred_label = sample['pred_label']
        proba = sample['proba']
        
        ax.imshow(image)
        ax.set_title(f'True: {class_names[true_label]}\n'
                    f'Pred: {class_names[pred_label]} ({proba:.2f})',
                    fontsize=10, color='red' if true_label != pred_label else 'green')
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle('Misclassified Samples', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Misclassified samples visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_predictions(model, data_loader, device, num_samples=16, save_path=None):
    """
    Visualize model predictions on random samples.
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    model.eval()
    class_names = ['Uninfected', 'Parasitized']
    
    # Get random samples
    all_images = []
    all_labels = []
    all_preds = []
    all_probas = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            probas = torch.softmax(outputs, dim=1)
            
            for i in range(len(images)):
                if len(all_images) < num_samples:
                    all_images.append(images[i].cpu())
                    all_labels.append(labels[i].cpu().item())
                    all_preds.append(preds[i].cpu().item())
                    all_probas.append(probas[i].cpu().numpy())
            
            if len(all_images) >= num_samples:
                break
    
    # Visualize
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        image = denormalize(all_images[idx])
        image = torch.clamp(image, 0, 1)
        image = image.permute(1, 2, 0).numpy()
        
        true_label = all_labels[idx]
        pred_label = all_preds[idx]
        proba = all_probas[idx][pred_label]
        
        color = 'green' if true_label == pred_label else 'red'
        ax.imshow(image)
        ax.set_title(f'True: {class_names[true_label]}\n'
                    f'Pred: {class_names[pred_label]} ({proba:.2f})',
                    fontsize=10, color=color)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle('Model Predictions', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(y_true, y_proba, save_path=None):
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

