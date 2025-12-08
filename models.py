"""
Transfer learning models for malaria detection.
Uses pretrained CNNs (MobileNetV2, EfficientNetB0) with frozen backbones.
"""

import torch
import torch.nn as nn
import torchvision.models as models

# Check for EfficientNet availability (requires torchvision >= 0.13.0)
try:
    from torchvision.models import efficientnet_b0
    try:
        from torchvision.models import EfficientNet_B0_Weights
        EFFICIENTNET_AVAILABLE = True
    except ImportError:
        # Older torchvision versions
        EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False


class MobileNetV2Transfer(nn.Module):
    """
    MobileNetV2-based transfer learning model for malaria detection.
    Uses pretrained MobileNetV2 with frozen backbone and binary classification head.
    """
    
    def __init__(self, pretrained=True, freeze_backbone=True):
        super(MobileNetV2Transfer, self).__init__()
        
        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(weights='DEFAULT' if pretrained else None)
        
        # Freeze backbone by default (only train classifier head)
        if freeze_backbone:
            for param in mobilenet.parameters():
                param.requires_grad = False
        
        # Remove the final classifier
        self.backbone = mobilenet.features
        
        # Binary classification head (single output with sigmoid)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),  # Single output for binary classification
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x.squeeze(1)  # Remove extra dimension


class EfficientNetB0Transfer(nn.Module):
    """
    EfficientNetB0-based transfer learning model for malaria detection.
    Uses pretrained EfficientNetB0 with frozen backbone and binary classification head.
    """
    
    def __init__(self, pretrained=True, freeze_backbone=True):
        super(EfficientNetB0Transfer, self).__init__()
        
        if not EFFICIENTNET_AVAILABLE:
            raise ImportError("EfficientNet requires torchvision >= 0.13.0")
        
        # Load pretrained EfficientNetB0
        if pretrained:
            try:
                efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            except NameError:
                # Fallback for older torchvision versions
                efficientnet = efficientnet_b0(pretrained=True)
        else:
            efficientnet = efficientnet_b0(pretrained=False)
        
        # Freeze backbone by default (only train classifier head)
        if freeze_backbone:
            for param in efficientnet.parameters():
                param.requires_grad = False
        
        # Extract features (remove classifier)
        self.backbone = efficientnet.features
        
        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, 512),  # EfficientNetB0 features are 1280
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),  # Single output for binary classification
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x.squeeze(1)  # Remove extra dimension


def get_model(model_name, pretrained=True, freeze_backbone=True):
    """
    Factory function to get a transfer learning model by name.
    
    Args:
        model_name: 'mobilenetv2' or 'efficientnetb0'
        pretrained: Whether to use pretrained weights (default: True)
        freeze_backbone: Whether to freeze backbone weights (default: True)
    
    Returns:
        Model instance with binary classification output
    """
    model_name = model_name.lower()
    
    if model_name == 'mobilenetv2':
        return MobileNetV2Transfer(pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif model_name == 'efficientnetb0' or model_name == 'efficientnet':
        return EfficientNetB0Transfer(pretrained=pretrained, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'mobilenetv2' or 'efficientnetb0'")

