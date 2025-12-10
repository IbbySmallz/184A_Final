# Deep Learning for Malaria Parasite Detection in Blood Smears

This project implements a CNN-based image classification system to automatically detect malaria infection from microscopy images of human blood smears.

## Dataset

The dataset is included in this repository. The expected structure is:
```
data/
  cell_images/
    Parasitized/
      (images)
    Uninfected/
      (images)
```

**Original Dataset Source:**
- Kaggle Dataset: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
- The dataset has been downloaded and included in the repository for easy access and reproducibility.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The dataset is already included in the repository at `data/cell_images/`

## Running on Google Colab

Follow these steps to run the project on Google Colab with free GPU access:

### Step 1: Open Google Colab
1. Go to https://colab.research.google.com/
2. Create a new notebook or open an existing one

### Step 2: Enable GPU
1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU** (T4 is free)
3. Click **Save**

### Step 3: Clone the Repository
In a new code cell, run:
```python
!git clone https://github.com/IbbySmallz/184A_Final.git
%cd 184A_Final
```

### Step 4: Install Dependencies
```python
!pip install -r requirements.txt
```

### Step 5: Run Training
Now you can run any of the training scripts:

**Quick start (complete workflow):**
```python
!python example_usage.py
```

**Train MobileNetV2:**
```python
!python train.py --model mobilenetv2 --epochs 20 --batch_size 32 --data_dir data/cell_images
```

**Train EfficientNetB0:**
```python
!python train.py --model efficientnetb0 --epochs 20 --batch_size 32 --data_dir data/cell_images
```

**Evaluate a trained model:**
```python
!python evaluate.py --model_path models/mobilenetv2_best.pth --model_type mobilenetv2 --data_dir data/cell_images
```

### Notes for Colab:
- The dataset is included in the repository, so no additional download is needed
- Training typically takes 5-10 minutes with frozen backbone on GPU
- Model checkpoints and results will be saved in `models/` and `results/` directories
- To download files from Colab: Right-click on files in the file browser → Download
- Colab sessions timeout after ~12 hours of inactivity

## Usage

### Quick Start

For a complete example workflow, run:
```bash
python example_usage.py
```

### Training Models

Train MobileNetV2 (default, fastest):
```bash
python train.py --model mobilenetv2 --epochs 20 --batch_size 32 --data_dir data/cell_images
```

Train EfficientNetB0:
```bash
python train.py --model efficientnetb0 --epochs 20 --batch_size 32 --data_dir data/cell_images
```

**Training Options:**
- `--model`: Pretrained model architecture (`mobilenetv2`, `efficientnetb0`) - default: `mobilenetv2`
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--image_size`: Input image size (default: 224)
- `--no_pretrained`: Do not use pretrained weights (default: uses pretrained)
- `--unfreeze_backbone`: Unfreeze backbone for fine-tuning (default: frozen)
- `--patience`: Early stopping patience (default: 10)
- `--data_dir`: Path to cell_images directory
- `--save_dir`: Directory to save models (default: `models`)

**Note:** By default, the backbone is frozen and only the classification head is trained, making training very fast (typically 5-10 minutes on GPU).

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py --model_path models/mobilenetv2_best.pth --model_type mobilenetv2 --data_dir data/cell_images
```

**Evaluation Options:**
- `--model_path`: Path to model checkpoint
- `--model_type`: Model architecture type (`mobilenetv2`, `efficientnetb0`)
- `--data_dir`: Path to cell_images directory
- `--save_dir`: Directory to save results (default: `results`)
- `--num_misclassified`: Number of misclassified samples to visualize (default: 10)

## Features

### Data Preprocessing
- Automatic train/validation/test split (80/10/10)
- Data augmentation for training (rotation, flipping, color jitter, etc.)
- Image normalization using ImageNet statistics
- Support for various image formats (PNG, JPG, JPEG)

### Models
- **MobileNetV2**: Lightweight pretrained model, fast training (~5-10 min on GPU)
- **EfficientNetB0**: More accurate but slightly slower pretrained model
- Both models use frozen backbones by default (only classifier head is trained)

### Training Features
- Cross-entropy loss with Adam optimizer
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping based on validation accuracy
- Automatic checkpoint saving
- Training history visualization

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Misclassified sample visualization

## Project Structure

```
.
├── data_preprocessing.py    # Data loading, preprocessing, and augmentation
├── models.py                 # CNN model architectures
├── train.py                  # Main training script
├── evaluate.py               # Evaluation script with comprehensive metrics
├── visualize.py              # Visualization tools for results and failure cases
├── utils.py                  # Utility functions (metrics, checkpoints, etc.)
├── example_usage.py          # Complete example workflow
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Output Files

After training, the following files are generated in the `models/` directory:
- `{model_name}_best.pth`: Best model checkpoint (based on validation accuracy)
- `{model_name}_epoch_N.pth`: Checkpoint saved every 10 epochs
- `{model_name}_training_history.png`: Training/validation loss and accuracy curves

After evaluation, the following files are generated in the `results/` directory:
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve plot
- `misclassified_samples.png`: Visualization of misclassified samples
- `metrics.txt`: Text file with all evaluation metrics

## Team Members

- **Ibrahim Syed**: Transfer learning model implementation, training pipeline, evaluation scripts, visualizations
- **Dhanush Venna**: Data preprocessing and augmentation, comparative experiments with different pretrained backbones, project report

## Technical Details

### Model Architectures

**MobileNetV2 Transfer:**
- Pretrained MobileNetV2 backbone (frozen by default)
- Binary classification head: 1280 → 512 → 1 (sigmoid output)
- Dropout (0.5) for regularization
- Fast training, good accuracy

**EfficientNetB0 Transfer:**
- Pretrained EfficientNetB0 backbone (frozen by default)
- Binary classification head: 1280 → 512 → 1 (sigmoid output)
- Dropout (0.5) for regularization
- Higher accuracy, slightly slower than MobileNetV2

### Training Details
- **Approach**: Transfer learning with frozen pretrained backbones
- **Loss function**: Binary Cross-Entropy (BCE) for binary classification
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Learning rate scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early stopping**: Patience=10 epochs
- **Data split**: 80% train, 10% validation, 10% test
- **Image size**: 224x224 pixels
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Training time**: ~5-10 minutes on GPU (backbone frozen), ~30-60 minutes if unfrozen

## Citation

If you use this code, please cite the original dataset:
- Kaggle Dataset: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

