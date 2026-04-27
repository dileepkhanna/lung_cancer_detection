"""
ResNet18 + LTC Model with Transfer Learning
Optimized for fast training and high accuracy (90-95%)
"""

import torch
import torch.nn as nn
from torchvision import models
from ncps.torch import LTC
from ncps.wirings import AutoNCP


class ResNetLTC(nn.Module):
    """
    ResNet18 backbone + LTC classifier
    Uses transfer learning for fast training
    """
    
    def __init__(self, num_classes=2, hidden_units=64, pretrained=True):
        super(ResNetLTC, self).__init__()
        
        # Load pre-trained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Keep original conv1 for RGB input (3 channels)
        # No modification needed - ResNet18 already expects 3 channels
        
        # Remove the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Feature reducer with BatchNorm for stability
        self.feature_reducer = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # LTC wiring and cell (better with intermediate neurons)
        self.ltc_output_size = 16  # Store output size
        self.wiring = AutoNCP(hidden_units, self.ltc_output_size)
        self.ltc = LTC(
            input_size=256,
            units=self.wiring,
            return_sequences=False
        )
        
        # Final classifier (maps from LTC output to num_classes)
        self.classifier = nn.Linear(self.ltc_output_size, num_classes)
        
    def forward(self, x):
        # ResNet feature extraction
        features = self.resnet(x)
        
        # Reduce features
        reduced = self.feature_reducer(features)
        
        # Add sequence dimension for LTC (batch, seq_len=1, features)
        reduced = reduced.unsqueeze(1)
        
        # LTC processing (returns tuple: output, state)
        ltc_out, _ = self.ltc(reduced)
        
        # Remove sequence dimension safely
        ltc_out = ltc_out.reshape(ltc_out.size(0), -1)
        
        # Final classification
        output = self.classifier(ltc_out)
        
        return output
    
    def freeze_backbone(self):
        """Freeze ResNet layers for initial training"""
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Note: All layers frozen, will train only LTC and feature reducer
    
    def unfreeze_backbone(self):
        """Unfreeze ResNet layers for fine-tuning"""
        for param in self.resnet.parameters():
            param.requires_grad = True


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Better than CrossEntropyLoss for imbalanced datasets
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == '__main__':
    # Test the model
    model = ResNetLTC(num_classes=2, hidden_units=64, pretrained=False)
    x = torch.randn(4, 3, 224, 224)  # RGB input
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model created successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
