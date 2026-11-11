"""
Mobile-Friendly Vision Backbone Architecture
Implements MobileNetV3-Large and EfficientNet-B0 backbones with support for
single-label (Softmax / CrossEntropy) and multi-label (Sigmoid / BCEWithLogits) heads,
as well as visual feature extraction for multimodal fusion.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)
from typing import Optional, Tuple

class CropDiseaseClassifier(nn.Module):
    """
    Mobile-optimized vision model for 20-crop disease diagnosis.
    Provides headless feature extraction for multimodal fusion as well as
    direct single-label and multi-label classification heads.
    """
    def __init__(
        self,
        num_classes: int = 82,
        backbone_name: str = "mobilenet_v3_large",
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        multi_label: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.multi_label = multi_label

        if backbone_name == "mobilenet_v3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base = mobilenet_v3_large(weights=weights)
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.embedding_dim = 960

        elif backbone_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = efficientnet_b0(weights=weights)
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.embedding_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.Hardswish(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def extract_visual_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate visual feature representation vector."""
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        embedding = self.extract_visual_embedding(x)
        logits = self.classifier(embedding)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Computes probabilities using Sigmoid (multi-label) or Softmax (single-label)."""
        logits = self.forward(x)
        if self.multi_label:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

def build_model(
    num_classes: int = 82,
    backbone: str = "mobilenet_v3_large",
    pretrained: bool = True,
    multi_label: bool = False
) -> CropDiseaseClassifier:
    """Factory helper to build and initialize classifier model."""
    return CropDiseaseClassifier(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=pretrained,
        multi_label=multi_label
    )

