# Crop Disease Diagnosis & Agronomic Advisory ML System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, modular, production-oriented multimodal machine learning system for plant leaf disease diagnosis and verified agronomic advisory across **20 major agricultural crops**.

---

## 1. Project Overview & Workflow

The system implements the complete end-to-end agricultural workflow:

```text
Image Capture (Mobile 3-View Protocol)
           │
           ▼
Image Quality Gate (Blur, Darkness, Dimensions, Foliage Heuristic)
           │
           ▼
Leaf Detection & Isolation (Lightweight YOLO / Saliency Contour)
           │
           ▼
Multimodal Feature Extraction (MobileNetV3 / EfficientNet Visual Embedding)
           │
           ├──────────────────────────┬──────────────────────────┐
           ▼                          ▼                          ▼
    Visual Embedding          Tabular Weather           3-Question Survey
     (960/1280-dim)            (Rain, Temp, RH)          (Root, Stem, Spray)
           └──────────────────────────┼──────────────────────────┘
                                      ▼
                            Multimodal Fusion Network
                                      │
                                      ▼
                 Uncertainty & OOD Safety Boundary Check
                        (Configurable ~80% Threshold)
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        [Confidence >= 0.80]                   [Confidence < 0.80]
             "Accepted"                             "Uncertain"
                   │                                     │
                   ▼                                     ▼
     Grad-CAM Lesion Heatmap Overlay         Withhold Forced Diagnosis
                   │                                     │
                   ▼                                     ▼
     Geo-Fenced Agronomic Treatment         Route to Agronomist Queue
     (Growth Stage & Bounding Box)                       │
                   │                                     ▼
                   ▼                             Expert Clinical Review
         Final JSON Response                             │
                   │                                     ▼
                   ▼                             Trusted Retraining Set
      Mobile Edge Offline Support
```

---

## 2. Supported 20-Crop Taxonomy & Canonical Labels

The system supports a centralized, extensible catalog spanning at least **20 crops**:

1. **Rice** (*Oryza sativa*) — Bacterial leaf blight, Brown spot, Leaf blast, Sheath blight, Healthy
2. **Wheat** (*Triticum aestivum*) — Stripe rust, Leaf rust, Powdery mildew, Septoria leaf blight, Healthy
3. **Maize / Corn** (*Zea mays*) — Common rust, Northern leaf blight, Gray leaf spot, Healthy
4. **Cotton** (*Gossypium hirsutum*) — Bacterial blight, Leaf curl virus, Target spot, Healthy
5. **Tomato** (*Solanum lycopersicum*) — Early blight, Late blight, Leaf mold, Bacterial spot, Septoria leaf spot, Yellow leaf curl virus, Healthy
6. **Potato** (*Solanum tuberosum*) — Early blight, Late blight, Healthy
7. **Chilli** (*Capsicum annuum*) — Anthracnose, Leaf curl, Bacterial spot, Healthy
8. **Grape** (*Vitis vinifera*) — Black rot, Esca (Black measles), Leaf blight, Healthy
9. **Apple** (*Malus domestica*) — Apple scab, Black rot, Cedar apple rust, Healthy
10. **Banana** (*Musa acuminata*) — Panama disease (Fusarium wilt), Black sigatoka, Yellow sigatoka, Healthy
11. **Mango** (*Mangifera indica*) — Anthracnose, Powdery mildew, Dieback, Healthy
12. **Soybean** (*Glycine max*) — Bacterial blight, Frogeye leaf spot, Rust, Healthy
13. **Groundnut** (*Arachis hypogaea*) — Early leaf spot, Late leaf spot, Rust, Healthy
14. **Sugarcane** (*Saccharum officinarum*) — Red rot, Smut, Rust, Healthy
15. **Coffee** (*Coffea arabica*) — Leaf rust, Cercospora leaf spot, Sooty mold, Healthy
16. **Pepper** (*Capsicum annuum*) — Bacterial spot, Anthracnose, Healthy
17. **Citrus** (*Citrus sinensis*) — Canker, Black spot, Greening (HLB), Healthy
18. **Beans** (*Phaseolus vulgaris*) — Angular leaf spot, Rust, Healthy
19. **Okra** (*Abelmoschus esculentus*) — Yellow vein mosaic, Enation leaf curl, Powdery mildew, Healthy
20. **Pea** (*Pisum sativum*) — Powdery mildew, Rust, Downy mildew, Healthy

### Canonical Label Syntax:
All labels strictly follow the `crop__disease` convention (e.g., `tomato__early_blight`, `rice__leaf_blast`, `apple__apple_scab`). Unrecognized or unmapped classes are isolated to `review/unmapped` and never silently guessed.

---

## 3. Directory Layout

```text
Crop_Disease_ML_System/
│
├── 01_Dataset/
│   ├── class_labels.csv             <- Master 82-class canonical taxonomy
│   ├── crop_catalog.json            <- 20-crop biological catalog & symptoms
│   ├── dataset_sources.json         <- PlantVillage, PlantDoc, PlantWild, Field configs
│   ├── dataset_manifest.csv         <- Consolidated dataset manifest with MD5 & dHash
│   ├── dataset_quality_check.md     <- Automated dataset audit and balance report
│   ├── source_loaders.py            <- Scanner, integrity verifier, deduplicator
│   ├── merge_external_datasets.py   <- Ingestion and normalization CLI
│   └── split_dataset.py             <- 70/15/15 Leakage-aware dataset splitting
│
├── 02_Augmentation/
│   └── augmentation.py              <- Albumentations field condition pipelines
│
├── 03_Survey/
│   └── survey_schema.json           <- JSON Schema for 3-Question Farmer Survey
│
├── 04_Weather/
│   ├── weather_schema.json          <- Schema for normalized meteorological features
│   └── weather_connector.py         <- Open-Meteo API connector & offline cache
│
├── 05_Treatment/
│   ├── treatment_rules.json         <- Verified, geo-fenced agronomic rules database
│   └── treatment_engine.py          <- Geo-spatial & growth stage lookup engine
│
├── data/
│   ├── raw/                         <- Mount directory for benchmark datasets
│   ├── processed/                   <- Processed image crops and Grad-CAM overlays
│   └── feedback/                    <- Anonymized logs, review queue, verified labels
│
├── models/
│   ├── checkpoints/                 <- PyTorch model weights (best_model.pt)
│   └── exported/                    <- Exported FP32 and INT8 ONNX edge models
│
├── src/
│   ├── config.py                    <- Centralized settings loaded via .env
│   ├── dataset.py                   <- PyTorch dataset with class balancing
│   ├── leaf_detector.py             <- Two-stage quality gate & leaf detector
│   ├── model.py                     <- MobileNetV3 / EfficientNet vision backbone
│   ├── multimodal.py                <- Multimodal deep fusion architecture
│   ├── uncertainty.py               <- OOD & predictive entropy safety gate
│   ├── explainability.py            <- Grad-CAM hook engine & shortcut detector
│   ├── inference.py                 <- Central end-to-end diagnostic service
│   ├── feedback.py                  <- Diagnostic logger & agronomist review queue
│   ├── retraining.py                <- Active learning curator & drift monitor
│   ├── train.py                     <- Training loop with early stopping & scheduler
│   ├── evaluate.py                  <- Precision, Recall, F1, Confusion Matrix, OOD
│   └── export_onnx.py               <- PyTorch to ONNX & dynamic INT8 quantizer
│
├── api/
│   ├── main.py                      <- FastAPI backend REST application
│   └── dashboard.py                 <- Web-based Agronomist Review Portal
│
├── mobile/
│   ├── offline_inference.py         <- Standalone edge offline diagnosis engine
│   └── README.md                    <- Flutter / React Native 3-view camera guide
│
├── tests/
│   └── test_project.py              <- Comprehensive unit & integration test suite
│
├── scripts/
│   └── validate_project.py          <- Project integrity and taxonomy validator
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 4. Installation & Environment Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Recommended: Virtual environment (`venv` or `conda`)

```bash
# 1. Clone repository and navigate to root
cd Crop_Disease_ML_System_FINAL

# 2. Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy configuration template
cp .env.example .env
```

---

## 5. Dataset Curation, Balancing & Splitting

The system combines **PlantVillage**, **PlantDoc**, **PlantWild**, and field datasets.

### Ingestion & Manifest Generation
Configure dataset paths via `.env` or pass CLI flags:

```bash
python 01_Dataset/merge_external_datasets.py \
  --plantvillage-root /path/to/PlantVillage \
  --plantdoc-root /path/to/PlantDoc \
  --output-manifest 01_Dataset/dataset_manifest.csv \
  --output-report 01_Dataset/dataset_quality_check.md
```

### Leakage-Aware Splitting (70% Train / 15% Val / 15% Test)
Guarantees that identical or near-duplicate images sharing an MD5 or perceptual difference hash (`dHash`) **never** appear across both train and test splits:

```bash
python 01_Dataset/split_dataset.py \
  --manifest 01_Dataset/dataset_manifest.csv \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

---

## 6. Real-World Field Augmentation

Using **Albumentations**, the training pipeline simulates in-field imaging variations:
- Midday sun glare (`RandomSunFlare`, `RandomBrightnessContrast`)
- Tree and cloud canopy shadows (`RandomShadow`)
- Water droplets and dew (`RandomRain`)
- Motion blur from hand tremors (`MotionBlur`)
- Camera sensor noise (`GaussNoise`, `ISONoise`)
- Compression and low resolution (`ImageCompression`, `Downscale`)
- Occlusion by debris or neighboring leaves (`CoarseDropout`)

*Validation and Test splits use purely deterministic resizing and ImageNet normalization.*

---

## 7. Model Training & Evaluation

### Training Loop
Trains the multimodal neural network with Cosine Annealing learning rate scheduling, class-weighted loss, and early stopping:

```bash
# Train on curated dataset
python src/train.py --epochs 25 --batch-size 32 --lr 0.0003 --backbone mobilenet_v3_large

# Or run immediate self-contained smoke test to verify execution mechanics
python src/train.py --smoke-test
```

### Comprehensive Evaluation
Evaluates test accuracy, macro/weighted Precision, Recall, F1-Score, confusion matrix, and OOD rejection rates:

```bash
python src/evaluate.py --checkpoint models/checkpoints/best_model.pt --threshold 0.80
```

---

## 8. Grad-CAM Visual Explainability

Generates gradient-weighted class activation heatmaps on the final convolutional layer of MobileNetV3 / EfficientNet, overlays them on the input foliage, and detects potential background shortcut learning:

```python
from explainability import GradCAM
gradcam = GradCAM(model)
heatmap = gradcam.generate_heatmap(img_tensor, target_class_idx=target_idx)
overlay, shortcut_info = gradcam.overlay_heatmap(heatmap, leaf_crop_bgr)
gradcam.save_explanation(overlay, "data/processed/gradcam/explanation.jpg")
```

---

## 9. Geo-Fenced Treatment Rules Engine

Treatment lookup strictly matches **crop**, **disease**, **growth stage**, and **geographic boundaries**:

```python
from treatment_engine import TreatmentEngine
engine = TreatmentEngine()

rule = engine.lookup(
    crop="tomato",
    disease="early_blight",
    growth_stage="vegetative",
    latitude=17.385,
    longitude=78.486
)
# Returns verified treatment dosage, active ingredients, organic alternatives, and safety PHI notes.
```
If no verified rule exists or confidence is uncertain, returns:
`"No verified treatment rule available."`

---

## 10. End-to-End Command-Line Inference

```bash
python src/inference.py \
  --image path/to/leaf.jpg \
  --lat 17.385 \
  --lon 78.486 \
  --root normal \
  --stem weak \
  --spray fungicide \
  --growth-stage vegetative \
  --output result.json
```

---

## 11. Backend API & Agronomist Web Portal

### Start FastAPI Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive API Swagger Docs:** `http://localhost:8000/docs`
- **Agronomist Review Portal:** `http://localhost:8000/dashboard`

### Key Endpoints:
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health status and model parameters |
| `POST` | `/predict` | Multipart upload for leaf image + GPS + 3-question survey |
| `POST` | `/survey` | Validates 3-question farmer field survey schema |
| `GET` | `/weather` | Retrieves normalized weather context for coordinates |
| `GET` | `/treatment` | Queries verified geo-fenced treatment rules |
| `GET` | `/review-queue` | Retrieves uncertain cases awaiting agronomist review |
| `POST` | `/review` | Agronomist records ground-truth verdict |
| `GET` | `/drift/stats` | Operational drift metrics and retraining triggers |

---

## 12. Edge Model Export & Mobile Offline Inference

Export model to ONNX with dynamic INT8 quantization:

```bash
python src/export_onnx.py --checkpoint models/checkpoints/best_model.pt
```

Run offline edge diagnosis:

```bash
python mobile/offline_inference.py --image path/to/leaf.jpg --root normal --stem healthy --spray no
```

---

## 13. Project Verification & Automated Tests

```bash
# 1. Structural and taxonomy validation
python scripts/validate_project.py

# 2. Run complete test suite
pytest tests/test_project.py -v
```

---

## 14. Known Limitations & Production Readiness

- **External Datasets Required:** Actual image files from PlantVillage, PlantDoc, and PlantWild are subject to third-party distribution terms and must be mounted in `data/raw/` by the operator.
- **Trained Model Weights:** The codebase includes complete training, validation, export, and inference pipelines. Production deployment requires executing `python src/train.py` on the mounted datasets to generate production `best_model.pt` weights.
- **Pesticide Registration:** Chemical recommendations in `treatment_rules.json` reflect standard extension guidelines; operators must ensure local regulatory compliance with respective state/national chemical registration boards.

