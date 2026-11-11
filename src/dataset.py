"""
PyTorch Dataset and DataLoader Engine
Loads image data, tabular weather vectors, and farmer survey responses from
dataset_manifest.csv with class balancing via class weighting and WeightedRandomSampler.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Add parent directory for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "02_Augmentation"))
sys.path.insert(0, str(REPO_ROOT / "04_Weather"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from augmentation import get_train_transforms, get_val_transforms, get_test_transforms
from weather_connector import WeatherConnector
from multimodal import SurveyEncoder
from config import config

class CropDiseaseDataset(Dataset):
    """
    Multimodal PyTorch Dataset for 20-crop disease diagnosis.
    Returns:
        - Image tensor: (3, H, W)
        - Weather feature vector: (5,)
        - Survey feature vector: (15,)
        - Target label: int (single-label) or binary float vector (multi-label)
        - Metadata dictionary
    """
    def __init__(
        self,
        manifest_path: Path = config.MANIFEST_CSV,
        class_labels_path: Path = config.CLASS_LABELS_CSV,
        split: str = "train",
        image_size: int = config.IMAGE_SIZE,
        multi_label: bool = config.MULTI_LABEL,
        transform = None
    ):
        self.split = split
        self.image_size = image_size
        self.multi_label = multi_label

        # Load label mapping
        self.label_to_idx, self.idx_to_label = self._load_classes(class_labels_path)
        self.num_classes = len(self.label_to_idx)

        # Load manifest
        self.df = self._load_manifest(manifest_path, split)

        # Assign transform
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(image_size)
        elif split == "val":
            self.transform = get_val_transforms(image_size)
        else:
            self.transform = get_test_transforms(image_size)

    def _load_classes(self, labels_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
        label_to_idx = {}
        idx_to_label = {}
        if labels_path.exists():
            df_lbl = pd.read_csv(labels_path)
            for _, row in df_lbl.iterrows():
                cid = int(row["class_id"])
                lbl = str(row["canonical_label"]).strip()
                label_to_idx[lbl] = cid
                idx_to_label[cid] = lbl
        return label_to_idx, idx_to_label

    def _load_manifest(self, manifest_path: Path, split: str) -> pd.DataFrame:
        if not manifest_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(manifest_path)
        if len(df) == 0:
            return df

        # Filter by split and validity
        if "split" in df.columns:
            df = df[df["split"] == split]
        if "is_valid" in df.columns:
            df = df[df["is_valid"] == True]
        if "status" in df.columns:
            df = df[df["status"] != "unmapped"]

        # Keep only records whose canonical_label is in label catalog
        df = df[df["canonical_label"].isin(self.label_to_idx)]
        return df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Union[int, torch.Tensor], Dict[str, Any]]:
        row = self.df.iloc[idx]
        image_path = str(row["path"])
        canonical = str(row["canonical_label"])

        # 1. Load image
        img = cv2.imread(image_path)
        if img is None:
            # Fallback black canvas if file moved or missing
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply Albumentations transform
        transformed = self.transform(image=img)
        image_tensor = transformed["image"]

        # 2. Weather vector (synthetic/neutral or real if columns present)
        weather_dict = {
            "temperature_mean": float(row.get("temperature", 25.0)) if "temperature" in row else 25.0,
            "humidity_mean": float(row.get("humidity", 65.0)) if "humidity" in row else 65.0,
            "rainfall_7d": float(row.get("rainfall_7d", 10.0)) if "rainfall_7d" in row else 10.0,
            "rainfall_14d": float(row.get("rainfall_14d", 25.0)) if "rainfall_14d" in row else 25.0,
            "is_available": "temperature" in row
        }
        weather_vec = WeatherConnector.normalize_weather_features(weather_dict)
        weather_tensor = torch.from_numpy(weather_vec).float()

        # 3. Survey vector (synthetic/neutral or real if columns present)
        survey_dict = {
            "root_condition": str(row.get("root_condition", "unknown")),
            "stem_health": str(row.get("stem_health", "unknown")),
            "recent_chemical_spray": str(row.get("recent_chemical_spray", "unknown"))
        }
        survey_vec = SurveyEncoder.encode(survey_dict if "root_condition" in row else None)
        survey_tensor = torch.from_numpy(survey_vec).float()

        # 4. Target Label
        class_idx = self.label_to_idx.get(canonical, 0)
        if self.multi_label:
            target = torch.zeros(self.num_classes, dtype=torch.float32)
            target[class_idx] = 1.0
        else:
            target = class_idx

        meta = {
            "image_id": str(row.get("image_id", f"img_{idx}")),
            "canonical_label": canonical,
            "crop": str(row.get("crop", "")),
            "disease": str(row.get("disease", ""))
        }

        return image_tensor, weather_tensor, survey_tensor, target, meta

    def get_class_weights(self) -> torch.Tensor:
        """
        Computes inverse frequency class weights to balance training loss:
        w_c = Total / (Num_Classes * Count_c)
        """
        counts = np.zeros(self.num_classes, dtype=np.float32)
        if len(self.df) == 0:
            return torch.ones(self.num_classes, dtype=torch.float32)

        for _, row in self.df.iterrows():
            lbl = row["canonical_label"]
            if lbl in self.label_to_idx:
                counts[self.label_to_idx[lbl]] += 1

        total = np.sum(counts)
        weights = np.zeros_like(counts)
        for i in range(self.num_classes):
            if counts[i] > 0:
                weights[i] = total / (self.num_classes * counts[i])
            else:
                weights[i] = 1.0

        # Normalize weights to mean 1.0
        weights = weights / np.mean(weights)
        return torch.from_numpy(weights).float()

    def get_balanced_sampler(self) -> Optional[WeightedRandomSampler]:
        """Creates a WeightedRandomSampler for minority class oversampling."""
        if len(self.df) == 0:
            return None

        counts = {}
        for _, row in self.df.iterrows():
            lbl = row["canonical_label"]
            counts[lbl] = counts.get(lbl, 0) + 1

        sample_weights = []
        for _, row in self.df.iterrows():
            lbl = row["canonical_label"]
            c = counts.get(lbl, 1)
            sample_weights.append(1.0 / c)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return sampler

def create_dataloaders(
    manifest_path: Path = config.MANIFEST_CSV,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    use_balanced_sampler: bool = True
) -> Dict[str, DataLoader]:
    """Factory creating Train, Val, and Test PyTorch DataLoaders."""
    train_dataset = CropDiseaseDataset(manifest_path, split="train")
    val_dataset = CropDiseaseDataset(manifest_path, split="val")
    test_dataset = CropDiseaseDataset(manifest_path, split="test")

    sampler = train_dataset.get_balanced_sampler() if use_balanced_sampler and len(train_dataset) > 0 else None
    shuffle = sampler is None and len(train_dataset) > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    ) if len(train_dataset) > 0 else None

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    ) if len(val_dataset) > 0 else None

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    ) if len(test_dataset) > 0 else None

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "datasets": {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        }
    }

