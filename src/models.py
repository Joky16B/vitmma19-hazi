import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

from utils import setup_logger

logger = setup_logger(__name__)


class MajorityClassBaseline(nn.Module):
    """
    Baseline model that always predicts the majority class from training data.
    Used to establish a performance floor.
    """
    def __init__(self, train_labels, num_classes=3):
        super(MajorityClassBaseline, self).__init__()
        
        self.num_classes = num_classes
        self.majority_class = self._calculate_majority_class(train_labels)
        
        # Dummy parameter to make it a valid nn.Module
        self.dummy_param = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
        logger.info(f"MajorityClassBaseline initialized - always predicts class {self.majority_class}")

    def _calculate_majority_class(self, train_labels):
        """Find the most common class in training labels."""
        if isinstance(train_labels, torch.Tensor):
            train_labels = train_labels.numpy()
        
        counter = Counter(train_labels)
        most_common_class = counter.most_common(1)[0][0]
        return most_common_class
    
    def forward(self, x):
        """Return logits that always predict the majority class."""
        batch_size = x.size(0)
        device = self.dummy_param.device
        
        # Create logits with very low values
        logits = torch.full((batch_size, self.num_classes), -100.0, device=device)
        
        # Set majority class logit to high value
        logits[:, self.majority_class] = 100.0
        
        return logits


class AnkleNet(nn.Module):
    """
    Custom CNN architecture for ankle alignment classification.
    Features deeper blocks with spatial dropout and a multi-stage classifier.
    """
    def __init__(self, num_classes=3, input_size=224):
        super(AnkleNet, self).__init__()
        
        # Block 1: 3 -> 48 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Spatial dropout
            nn.MaxPool2d(2, 2)  # 224 -> 112
        )
        
        # Block 2: 48 -> 96 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),
            nn.MaxPool2d(2, 2)  # 112 -> 56
        )
        
        # Block 3: 96 -> 192 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2, 2)  # 56 -> 28
        )
        
        # Block 4: 192 -> 256 channels
        self.block4 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2, 2)  # 28 -> 14
        )
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def _count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def create_model(model_name, num_classes=3, train_labels=None):
    """
    Factory function to create models.
    
    Args:
        model_name: 'baseline' or 'anklenet'
        num_classes: Number of output classes
        train_labels: Required for baseline model (array of training labels)
    
    Returns:
        PyTorch model
    """
    if model_name.lower() == 'baseline':
        if train_labels is None:
            raise ValueError("train_labels required for baseline model")
        model = MajorityClassBaseline(train_labels, num_classes)
    
    elif model_name.lower() == 'anklenet':
        model = AnkleNet(num_classes)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}. Use 'baseline' or 'anklenet'")
    
    return model
