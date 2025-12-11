# Deep Learning for Malaria Parasite Detection in Blood Smears

This project implements a CNN-based image classification system to automatically detect malaria infection from microscopy images of human blood smears using transfer learning with pretrained MobileNetV2 and EfficientNetB0 models.

## Quick Start (Demo)

The repository includes a minimal dataset (400 images) that allows you to run the complete demo end-to-end without any additional downloads.

### Step 1: Clone the Repository
```bash
git clone https://github.com/IbbySmallz/184A_Final.git
cd 184A_Final
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Demo
```bash
python example_usage.py
```

This will:
- Load the minimal dataset from `data_minimal/cell_images/`
- Train a MobileNetV2 model for 20 epochs
- Evaluate the trained model on the test set
- Generate visualizations (confusion matrix, ROC curve, training history)
- Save results to `models/` and `results/` directories

**Expected runtime:** ~5-10 minutes on CPU, ~2-3 minutes on GPU

## Dataset

### Minimal Dataset (Included in Repository)

A minimal dataset is included in this repository for demonstration purposes:
- **Location:** `data_minimal/cell_images/`
- **Size:** 400 images (200 Parasitized, 200 Uninfected)
- **Purpose:** Allows the demo to run end-to-end without external downloads
- **Structure:**
  ```
  data_minimal/
    cell_images/
      Parasitized/
        (200 images)
      Uninfected/
        (200 images)
  ```

### Full Dataset (Optional - For Better Results)

If you want to train on the full dataset for better performance, you can download it from Kaggle:

**Original Dataset Source:**
- Kaggle Dataset: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
- **Size:** ~27,558 images (13,779 Parasitized, 13,779 Uninfected)
- **Size on disk:** ~777 MB

**Download Instructions:**

1. **Option A: Using Kaggle API (Recommended)**
   ```bash
   # Install Kaggle API
   pip install kaggle
   
   # Set up Kaggle credentials (download kaggle.json from https://www.kaggle.com/settings)
   # Place kaggle.json in ~/.kaggle/ directory
   chmod 600 ~/.kaggle/kaggle.json
   
   # Download dataset
   kaggle datasets download -d iarunava/cell-images-for-detecting-malaria
   unzip cell-images-for-detecting-malaria.zip -d temp_extract
   
   # Organize dataset structure
   # Find the cell_images folder in the extracted files and move it to data/cell_images/
   # The structure should be: data/cell_images/Parasitized/ and data/cell_images/Uninfected/
   ```

2. **Option B: Manual Download**
   - Visit https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
   - Click "Download" (requires Kaggle account)
   - Extract the zip file
   - Organize the dataset so the structure is:
     ```
     data/
       cell_images/
         Parasitized/
           (images)
         Uninfected/
           (images)
     ```

**Note:** The demo code (`example_usage.py`) uses the minimal dataset by default. To use the full dataset, modify the `data_dir` parameter in the script or use command-line arguments.

## Usage

### Training Models

Train MobileNetV2 (default, fastest) with minimal dataset:
```bash
python train.py --model mobilenetv2 --epochs 20 --batch_size 32 --data_dir data_minimal/cell_images
```

Train MobileNetV2 with full dataset (if downloaded):
```bash
python train.py --model mobilenetv2 --epochs 20 --batch_size 32 --data_dir data/cell_images
```

Train EfficientNetB0:
```bash
python train.py --model efficientnetb0 --epochs 20 --batch_size 32 --data_dir data_minimal/cell_images
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
- `--data_dir`: Path to cell_images directory (default: `data_minimal/cell_images` for demo)
- `--save_dir`: Directory to save models (default: `models`)

**Note:** By default, the backbone is frozen and only the classification head is trained, making training very fast (typically 2-3 minutes on GPU for minimal dataset, 5-10 minutes for full dataset).

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py --model_path models/mobilenetv2_best.pth --model_type mobilenetv2 --data_dir data_minimal/cell_images
```

**Evaluation Options:**
- `--model_path`: Path to model checkpoint
- `--model_type`: Model architecture type (`mobilenetv2`, `efficientnetb0`)
- `--data_dir`: Path to cell_images directory (default: `data_minimal/cell_images`)
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

## Reproducibility

This repository is designed to be fully reproducible. To reproduce our results:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/IbbySmallz/184A_Final.git
   cd 184A_Final
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo:**
   ```bash
   python example_usage.py
   ```

The demo uses the minimal dataset included in the repository and will:
- Train a MobileNetV2 model for 20 epochs
- Save the best model checkpoint
- Evaluate on the test set
- Generate all visualizations

**Note:** Results may vary slightly due to random initialization, but the overall performance should be consistent. For exact reproducibility, the random seed is set to 42 in the data preprocessing code.

## Verification

To verify the installation and that everything works:

```bash
# Check Python version (should be 3.8+)
python --version

# Check PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Verify minimal dataset exists
ls data_minimal/cell_images/Parasitized/ | wc -l  # Should show ~200
ls data_minimal/cell_images/Uninfected/ | wc -l   # Should show ~200

# Run a quick test
python example_usage.py
```

## Citation

If you use this code, please cite the original dataset:
- Kaggle Dataset: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

