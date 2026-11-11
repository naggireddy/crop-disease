"""
Real-World Agricultural Augmentation Engine
Simulates complex field conditions: glare, shadows, water droplets, blur,
sensor noise, resolution variation, rotation, perspective, and occlusions
using Albumentations while strictly preserving diagnostic lesion morphology.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import Optional, Union, Dict, Any

# ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Comprehensive training augmentation pipeline simulating diverse field conditions
    without obliterating fine disease symptoms (e.g. pustules, bacterial specks).
    """
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        
        # Spatial / Geometric variations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.12, rotate_limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
        A.Perspective(scale=(0.04, 0.08), keep_size=True, p=0.3),

        # Environmental illumination & field glare
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.4),
            A.RandomShadow(shadow_roi=(0, 0.3, 1, 1), num_shadows_limit=(1, 2), shadow_dimension=5, p=0.4),
        ], p=0.6),

        # Weather / water droplet simulation
        A.OneOf([
            A.RandomRain(slant_lower=-5, slant_upper=5, drop_length=10, drop_width=1, drop_color=(200, 200, 200), rain_type="drizzle", p=0.5),
            A.MotionBlur(blur_limit=5, p=0.5),
        ], p=0.25),

        # Mobile camera sensor noise & resolution degradation
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.5),
            A.ImageCompression(quality_lower=65, quality_upper=95, p=0.5),
            A.Downscale(scale_min=0.75, scale_max=0.95, interpolation=cv2.INTER_NEAREST, p=0.5),
        ], p=0.35),

        # Mild color variation (preserves chlorosis & necrotic discoloration)
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.15, hue=0.05, p=0.35),

        # Field occlusion (debris, intersecting leaves)
        A.CoarseDropout(
            max_holes=4,
            max_height=int(image_size * 0.12),
            max_width=int(image_size * 0.12),
            min_holes=1,
            min_height=int(image_size * 0.04),
            min_width=int(image_size * 0.04),
            fill_value=0,
            p=0.3
        ),

        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Deterministic preprocessing for validation (NO random augmentations)."""
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

def get_test_transforms(image_size: int = 224) -> A.Compose:
    """Deterministic preprocessing for test evaluation (NO random augmentations)."""
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

def get_field_simulation_transforms(image_size: int = 224) -> A.Compose:
    """Stress-test transform pipeline for robust offline field testing."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.25, p=0.8),
        A.MotionBlur(blur_limit=7, p=0.5),
        A.GaussNoise(var_limit=(20.0, 70.0), p=0.6),
        A.RandomShadow(p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

def augment_image_file(input_path: Union[str, Path], output_path: Union[str, Path], transform: Optional[A.Compose] = None):
    """Augments a single image file and writes the resulting RGB image to output_path."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"Unable to read image from {input_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if transform is None:
        # Visual export pipeline without tensor conversion
        export_transform = A.Compose([
            A.Resize(224, 224),
            A.RandomBrightnessContrast(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomShadow(p=0.3),
        ])
        augmented = export_transform(image=img_rgb)["image"]
    else:
        res = transform(image=img_rgb)["image"]
        if hasattr(res, "permute"):
            # Tensor to numpy
            res = res.permute(1, 2, 0).cpu().numpy()
            res = ((res * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)) * 255).astype(np.uint8)
        augmented = res

    augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), augmented_bgr)

