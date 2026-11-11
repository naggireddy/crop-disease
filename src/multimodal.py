"""
Multimodal Agronomic Fusion Architecture
Combines CNN visual embeddings with tabular weather metrics and categorical
farmer field surveys through multi-branch deep neural fusion with missing-modality tolerance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from model import CropDiseaseClassifier, build_model

# Categorical encodings for 3-Question Farmer Survey
ROOT_CONDITIONS = ["normal", "discolored", "rotting", "unknown"]
STEM_HEALTHS = ["healthy", "weak", "lesions_spots", "wilting", "unknown"]
CHEMICAL_SPRAYS = ["no", "yes_pesticide", "yes_fungicide", "yes_fertilizer", "unknown"]

class SurveyEncoder:
    """Encodes categorical farmer survey responses into standardized tabular numeric vectors."""
    @staticmethod
    def encode(survey_data: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Encodes survey answers into a 15-dimensional vector:
        - 4-dim one-hot: root_condition
        - 5-dim one-hot: stem_health
        - 5-dim one-hot: recent_chemical_spray
        - 1-dim flag: survey_available (1.0 or 0.0)
        """
        if not survey_data:
            # Neutral / missing survey vector
            vec = np.zeros(15, dtype=np.float32)
            # Default to "unknown" category flags
            vec[ROOT_CONDITIONS.index("unknown")] = 1.0
            vec[4 + STEM_HEALTHS.index("unknown")] = 1.0
            vec[9 + CHEMICAL_SPRAYS.index("unknown")] = 1.0
            vec[14] = 0.0  # is_available = False
            return vec

        vec = np.zeros(15, dtype=np.float32)
        root = str(survey_data.get("root_condition", "unknown")).lower()
        stem = str(survey_data.get("stem_health", "unknown")).lower()
        spray = str(survey_data.get("recent_chemical_spray", "unknown")).lower()

        r_idx = ROOT_CONDITIONS.index(root) if root in ROOT_CONDITIONS else ROOT_CONDITIONS.index("unknown")
        s_idx = STEM_HEALTHS.index(stem) if stem in STEM_HEALTHS else STEM_HEALTHS.index("unknown")
        c_idx = CHEMICAL_SPRAYS.index(spray) if spray in CHEMICAL_SPRAYS else CHEMICAL_SPRAYS.index("unknown")

        vec[r_idx] = 1.0
        vec[4 + s_idx] = 1.0
        vec[9 + c_idx] = 1.0
        vec[14] = 1.0  # is_available = True

        return vec

class MultimodalCropDiseaseModel(nn.Module):
    """
    Multimodal Fusion Network:
    [Visual Features] (MobileNetV3 / EfficientNet)
           +
    [Weather Features] (MLP)
           +
    [Survey Features] (MLP)
           ↓
    Concatenation & Fusion Layer
           ↓
    Disease Prediction Logits
    """
    def __init__(
        self,
        num_classes: int = 82,
        backbone_name: str = "mobilenet_v3_large",
        weather_in_dim: int = 5,
        survey_in_dim: int = 15,
        tabular_embed_dim: int = 32,
        pretrained: bool = True,
        multi_label: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.multi_label = multi_label

        # Visual Branch
        self.visual_backbone = build_model(
            num_classes=num_classes,
            backbone=backbone_name,
            pretrained=pretrained,
            multi_label=multi_label
        )
        visual_dim = self.visual_backbone.embedding_dim

        # Weather Branch MLP
        self.weather_mlp = nn.Sequential(
            nn.Linear(weather_in_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, tabular_embed_dim),
            nn.ReLU()
        )

        # Survey Branch MLP
        self.survey_mlp = nn.Sequential(
            nn.Linear(survey_in_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, tabular_embed_dim),
            nn.ReLU()
        )

        # Joint Multimodal Fusion Head
        fusion_in_dim = visual_dim + tabular_embed_dim + tabular_embed_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(
        self,
        images: torch.Tensor,
        weather_vectors: Optional[torch.Tensor] = None,
        survey_vectors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass combining visual, weather, and survey representations."""
        batch_size = images.size(0)
        device = images.device

        # 1. Visual representation
        vis_emb = self.visual_backbone.extract_visual_embedding(images)

        # 2. Weather representation (handle missing gracefully)
        if weather_vectors is None:
            # Default zero vector with is_available = 0
            weather_vectors = torch.zeros((batch_size, 5), dtype=torch.float32, device=device)
        weather_emb = self.weather_mlp(weather_vectors)

        # 3. Survey representation (handle missing gracefully)
        if survey_vectors is None:
            # Neutral vector
            survey_vectors = torch.zeros((batch_size, 15), dtype=torch.float32, device=device)
            survey_vectors[:, 3] = 1.0  # unknown root
            survey_vectors[:, 8] = 1.0  # unknown stem
            survey_vectors[:, 13] = 1.0 # unknown spray
        survey_emb = self.survey_mlp(survey_vectors)

        # 4. Joint Multimodal Concatenation & Fusion
        fused = torch.cat([vis_emb, weather_emb, survey_emb], dim=1)
        logits = self.fusion_head(fused)

        return logits

    def predict_probabilities(
        self,
        images: torch.Tensor,
        weather_vectors: Optional[torch.Tensor] = None,
        survey_vectors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes probabilities using Sigmoid or Softmax."""
        logits = self.forward(images, weather_vectors, survey_vectors)
        if self.multi_label:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

