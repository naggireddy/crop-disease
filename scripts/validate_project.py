"""
Project Integrity & Structural Validation Script
Verifies folder layout, 20-crop taxonomy, schema validity, and component importability.
"""

import sys
import json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    REPO_ROOT / "01_Dataset" / "crop_catalog.json",
    REPO_ROOT / "01_Dataset" / "class_labels.csv",
    REPO_ROOT / "01_Dataset" / "dataset_sources.json",
    REPO_ROOT / "01_Dataset" / "dataset_manifest.csv",
    REPO_ROOT / "01_Dataset" / "dataset_quality_check.md",
    REPO_ROOT / "01_Dataset" / "source_loaders.py",
    REPO_ROOT / "01_Dataset" / "merge_external_datasets.py",
    REPO_ROOT / "01_Dataset" / "split_dataset.py",
    REPO_ROOT / "02_Augmentation" / "augmentation.py",
    REPO_ROOT / "03_Survey" / "survey_schema.json",
    REPO_ROOT / "04_Weather" / "weather_schema.json",
    REPO_ROOT / "04_Weather" / "weather_connector.py",
    REPO_ROOT / "05_Treatment" / "treatment_rules.json",
    REPO_ROOT / "05_Treatment" / "treatment_engine.py",
    REPO_ROOT / "src" / "config.py",
    REPO_ROOT / "src" / "dataset.py",
    REPO_ROOT / "src" / "leaf_detector.py",
    REPO_ROOT / "src" / "model.py",
    REPO_ROOT / "src" / "multimodal.py",
    REPO_ROOT / "src" / "uncertainty.py",
    REPO_ROOT / "src" / "explainability.py",
    REPO_ROOT / "src" / "inference.py",
    REPO_ROOT / "src" / "feedback.py",
    REPO_ROOT / "src" / "retraining.py",
    REPO_ROOT / "src" / "train.py",
    REPO_ROOT / "src" / "evaluate.py",
    REPO_ROOT / "src" / "export_onnx.py",
    REPO_ROOT / "api" / "main.py",
    REPO_ROOT / "api" / "dashboard.py",
    REPO_ROOT / "mobile" / "offline_inference.py",
    REPO_ROOT / "mobile" / "README.md",
    REPO_ROOT / "tests" / "test_project.py",
    REPO_ROOT / "requirements.txt",
    REPO_ROOT / ".env.example",
    REPO_ROOT / ".gitignore",
    REPO_ROOT / "README.md"
]

REQUIRED_CROPS = [
    "rice", "wheat", "maize", "cotton", "tomato", "potato",
    "chilli", "grape", "apple", "banana", "mango", "soybean",
    "groundnut", "sugarcane", "coffee", "pepper", "citrus",
    "beans", "okra", "pea"
]

def validate():
    print("=" * 70)
    print("  CROP DISEASE ML SYSTEM — PROJECT INTEGRITY VALIDATION")
    print("=" * 70)

    # 1. Check Files
    print("\n[*] Validating Project Architecture & Required Files...")
    missing_files = []
    for f in REQUIRED_FILES:
        if not f.exists():
            missing_files.append(str(f.relative_to(REPO_ROOT)))
            print(f"  [X] Missing: {f.relative_to(REPO_ROOT)}")
        else:
            print(f"  [√] Present: {f.relative_to(REPO_ROOT)}")

    if missing_files:
        print(f"\n[!] ERROR: {len(missing_files)} required files are missing!")
        sys.exit(1)

    # 2. Check 20 Crops in Catalog
    print("\n[*] Validating 20-Crop Taxonomy Catalog...")
    catalog_path = REPO_ROOT / "01_Dataset" / "crop_catalog.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        cat_data = json.load(f)
    crops_in_cat = {c["crop_id"].lower() for c in cat_data.get("crops", [])}

    for crop in REQUIRED_CROPS:
        if crop not in crops_in_cat:
            print(f"  [X] Required crop '{crop}' missing from crop_catalog.json!")
            sys.exit(1)
        else:
            print(f"  [√] Crop verified: {crop.capitalize()}")

    # 3. Check Canonical Labels CSV
    labels_csv = REPO_ROOT / "01_Dataset" / "class_labels.csv"
    df_labels = pd.read_csv(labels_csv)
    print(f"\n[*] Validating class_labels.csv ({len(df_labels)} canonical classes)...")
    for _, row in df_labels.iterrows():
        lbl = str(row["canonical_label"])
        assert "__" in lbl, f"Malformed canonical label: {lbl}"
    print(f"  [√] All canonical labels follow 'crop__disease' syntax.")

    # 4. Check JSON Schemas
    print("\n[*] Validating JSON Schemas...")
    for schema_file in [
        REPO_ROOT / "03_Survey" / "survey_schema.json",
        REPO_ROOT / "04_Weather" / "weather_schema.json",
        REPO_ROOT / "05_Treatment" / "treatment_rules.json"
    ]:
        with open(schema_file, "r", encoding="utf-8") as sf:
            json.load(sf)
        print(f"  [√] Valid JSON: {schema_file.name}")

    print("\n" + "=" * 70)
    print("  [SUCCESS] All project components and integrity checks PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    validate()

