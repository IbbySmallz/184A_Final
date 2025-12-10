"""
Utility functions for training, evaluation, and visualization.
"""

import os
import shutil
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    print(f"Checkpoint loaded from {filepath}")
    print(f"  Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return epoch, loss, accuracy


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for ROC-AUC)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = 0.0
    
    return metrics


def print_metrics(metrics, split=''):
    """Print metrics in a formatted way."""
    prefix = f"{split} " if split else ""
    print(f"\n{prefix}Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")


def create_confusion_matrix(y_true, y_pred, class_names=['Uninfected', 'Parasitized'], 
                           save_path=None):
    """Create and optionally save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot training history (loss and accuracy curves)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)


def download_dataset(data_dir='data/cell_images', dataset_id='iarunava/cell-images-for-detecting-malaria'):
    """
    Download the malaria cell images dataset from Kaggle using kagglehub.
    
    Args:
        data_dir: Target directory where cell_images folder should be located
        dataset_id: Kaggle dataset identifier (default: 'iarunava/cell-images-for-detecting-malaria')
    
    Returns:
        Path to the cell_images directory if successful, None otherwise
    """
    try:
        import kagglehub
    except ImportError:
        print("Error: kagglehub is not installed. Please install it with:")
        print("  pip install kagglehub")
        print("\nAlternatively, you can manually download the dataset from:")
        print("  https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria")
        return None
    
    # Check if dataset already exists
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        # Check if it contains the expected structure
        expected_classes = ['Parasitized', 'Uninfected']
        if all(os.path.exists(os.path.join(data_dir, cls)) for cls in expected_classes):
            print(f"Dataset already exists at {data_dir}")
            return data_dir
    
    print(f"Downloading dataset from Kaggle: {dataset_id}")
    print("This may take a few minutes depending on your internet connection...")
    
    try:
        # Download dataset using kagglehub
        # kagglehub downloads to a cache directory and returns the path
        dataset_path = kagglehub.dataset_download(dataset_id)
        print(f"Dataset downloaded to: {dataset_path}")
        
        # Find the cell_images directory in the downloaded dataset
        # The structure might be: dataset_path/cell_images/ or dataset_path/.../cell_images/
        cell_images_path = None
        
        # First, check if cell_images is directly in the dataset_path
        potential_path = os.path.join(dataset_path, 'cell_images')
        if os.path.exists(potential_path) and os.path.isdir(potential_path):
            # Check if this contains the class folders directly
            if all(os.path.exists(os.path.join(potential_path, cls)) for cls in expected_classes):
                cell_images_path = potential_path
            else:
                # Check if there's a nested cell_images folder
                nested_path = os.path.join(potential_path, 'cell_images')
                if os.path.exists(nested_path) and os.path.isdir(nested_path):
                    if all(os.path.exists(os.path.join(nested_path, cls)) for cls in expected_classes):
                        cell_images_path = nested_path
        else:
            # Search for cell_images directory recursively
            for root, dirs, files in os.walk(dataset_path):
                if 'cell_images' in dirs:
                    candidate = os.path.join(root, 'cell_images')
                    # Check if this contains the class folders directly
                    if all(os.path.exists(os.path.join(candidate, cls)) for cls in expected_classes):
                        cell_images_path = candidate
                        break
                    # Check for nested structure
                    nested = os.path.join(candidate, 'cell_images')
                    if os.path.exists(nested) and os.path.isdir(nested):
                        if all(os.path.exists(os.path.join(nested, cls)) for cls in expected_classes):
                            cell_images_path = nested
                            break
        
        if cell_images_path is None:
            print(f"Warning: Could not find 'cell_images' directory with expected class structure.")
            print(f"Please check the structure at: {dataset_path}")
            print("You may need to manually extract and organize the dataset.")
            return None
        
        # Create target directory if it doesn't exist
        target_parent = os.path.dirname(data_dir)
        ensure_dir(target_parent)
        
        # If target directory exists but is empty/wrong, remove it
        if os.path.exists(data_dir):
            if not all(os.path.exists(os.path.join(data_dir, cls)) for cls in expected_classes):
                print(f"Removing incomplete dataset at {data_dir}")
                shutil.rmtree(data_dir)
        
        # Copy or move cell_images to target location
        if os.path.exists(data_dir):
            print(f"Dataset already exists at {data_dir}, skipping copy.")
        else:
            print(f"Copying dataset to {data_dir}...")
            shutil.copytree(cell_images_path, data_dir)
            print(f"Dataset successfully copied to {data_dir}")
        
        return data_dir
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you have a Kaggle account")
        print("2. You may need to authenticate with Kaggle API:")
        print("   - Go to https://www.kaggle.com/settings")
        print("   - Create an API token and save kaggle.json to ~/.kaggle/")
        print("3. Alternatively, manually download from:")
        print("   https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria")
        return None

