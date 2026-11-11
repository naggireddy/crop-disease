"""
Centralized Configuration Module
Loads settings from environment variables and provides sensible production defaults.
Avoids hardcoding paths, credentials, and thresholds.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from repository root if present
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

class SystemConfig:
    # Project Paths
    ROOT_DIR: Path = REPO_ROOT
    DATA_DIR: Path = ROOT_DIR / "data"
    DATASET_DIR: Path = ROOT_DIR / "01_Dataset"
    MANIFEST_CSV: Path = DATASET_DIR / "dataset_manifest.csv"
    CLASS_LABELS_CSV: Path = DATASET_DIR / "class_labels.csv"
    CROP_CATALOG_JSON: Path = DATASET_DIR / "crop_catalog.json"
    DATASET_SOURCES_JSON: Path = DATASET_DIR / "dataset_sources.json"
    SURVEY_SCHEMA_JSON: Path = ROOT_DIR / "03_Survey" / "survey_schema.json"
    WEATHER_SCHEMA_JSON: Path = ROOT_DIR / "04_Weather" / "weather_schema.json"
    TREATMENT_RULES_JSON: Path = ROOT_DIR / "05_Treatment" / "treatment_rules.json"
    
    MODELS_DIR: Path = ROOT_DIR / "models"
    CHECKPOINTS_DIR: Path = MODELS_DIR / "checkpoints"
    EXPORTED_DIR: Path = MODELS_DIR / "exported"
    FEEDBACK_DIR: Path = DATA_DIR / "feedback"
    LOGS_DIR: Path = DATA_DIR / "logs"

    # Dataset Source Roots (Overridden via environment variables)
    PLANTVILLAGE_ROOT: Path = Path(os.getenv("PLANTVILLAGE_ROOT", str(DATA_DIR / "raw" / "PlantVillage")))
    PLANTDOC_ROOT: Path = Path(os.getenv("PLANTDOC_ROOT", str(DATA_DIR / "raw" / "PlantDoc")))
    PLANTWILD_ROOT: Path = Path(os.getenv("PLANTWILD_ROOT", str(DATA_DIR / "raw" / "PlantWild")))
    FIELD_DATA_ROOT: Path = Path(os.getenv("FIELD_DATA_ROOT", str(DATA_DIR / "raw" / "FieldData")))

    # Model Architecture & Hyperparameters
    BACKBONE: str = os.getenv("MODEL_BACKBONE", "mobilenet_v3_large")  # mobilenet_v3_large or efficientnet_b0
    IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", "224"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    NUM_WORKERS: int = int(os.getenv("NUM_WORKERS", "0"))  # Default 0 on Windows for stability
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.0003"))
    WEIGHT_DECAY: float = float(os.getenv("WEIGHT_DECAY", "0.0001"))
    MAX_EPOCHS: int = int(os.getenv("MAX_EPOCHS", "25"))
    PATIENCE: int = int(os.getenv("PATIENCE", "5"))
    MULTI_LABEL: bool = os.getenv("MULTI_LABEL", "false").lower() == "true"

    # Leaf Gate & Quality Thresholds
    BLUR_THRESHOLD: float = float(os.getenv("BLUR_THRESHOLD", "80.0"))  # Laplacian variance
    MIN_BRIGHTNESS: float = float(os.getenv("MIN_BRIGHTNESS", "35.0"))
    MAX_BRIGHTNESS: float = float(os.getenv("MAX_BRIGHTNESS", "240.0"))
    MIN_IMAGE_DIM: int = int(os.getenv("MIN_IMAGE_DIM", "128"))
    MIN_PLANT_COVERAGE: float = float(os.getenv("MIN_PLANT_COVERAGE", "0.03"))

    # Uncertainty & OOD Safety Gate
    # Target safety margin around 80% boundary as specified, configurable
    OOD_CONFIDENCE_THRESHOLD: float = float(os.getenv("OOD_CONFIDENCE_THRESHOLD", "0.80"))
    MAX_ENTROPY_THRESHOLD: float = float(os.getenv("MAX_ENTROPY_THRESHOLD", "2.20"))

    # Weather API Configuration
    WEATHER_API_URL: str = os.getenv("WEATHER_API_URL", "https://api.open-meteo.com/v1/forecast")
    WEATHER_API_KEY: str = os.getenv("WEATHER_API_KEY", "")

    # Lightweight Leaf Detector Weights (Optional path)
    YOLO_DETECTOR_WEIGHTS: Optional[Path] = Path(os.getenv("YOLO_DETECTOR_WEIGHTS", str(MODELS_DIR / "checkpoints" / "leaf_yolo.onnx"))) if os.getenv("YOLO_DETECTOR_WEIGHTS") else None

    # Version Tracking
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "v1.0.0")
    DATASET_VERSION: str = os.getenv("DATASET_VERSION", "dataset_v1.0")

# Singleton instance
config = SystemConfig()

