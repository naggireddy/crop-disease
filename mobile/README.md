# Mobile Edge Architecture & Client Integration Specification

This document details the cross-platform mobile client architecture (supporting **Flutter** and **React Native**) for the Crop Disease Diagnosis ML System.

---

## 1. System Architecture Overview

```text
┌────────────────────────────────────────────────────────┐
│                   Mobile Client                        │
│                                                        │
│  ┌──────────────────┐         ┌─────────────────────┐  │
│  │ Guided 3-View    │ ──────> │ On-Device Quality   │  │
│  │ Camera Workflow  │         │ Gate (Blur & Light) │  │
│  └──────────────────┘         └─────────────────────┘  │
│                                          │             │
│                                      [Passes]          │
│                                          ▼             │
│  ┌──────────────────┐         ┌─────────────────────┐  │
│  │ Offline SQLite   │ <────── │ Local ONNX Edge     │  │
│  │ Sync Queue       │         │ Model Runtime       │  │
│  └──────────────────┘         └─────────────────────┘  │
│           │                              │             │
│     [Network Up]                   [Treatment DB]      │
│           ▼                              ▼             │
│  ┌──────────────────┐         ┌─────────────────────┐  │
│  │ FastAPI Cloud    │         │ Verified Geo-Fenced │  │
│  │ Sync Engine      │         │ Advice Display      │  │
│  └──────────────────┘         └─────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

## 2. Guided Camera Workflow (3-View Multi-Angle Protocol)

The mobile client enforces a structured **3-view guided photo acquisition protocol** to capture symptomatic foliage alongside systemic whole-plant diagnostic cues:

### View 1 — Leaf Close-Up (Primary Diagnostic View)
- **Objective:** Captures fine lesion margins, fungal fruiting structures (pycnidia/pustules), chlorotic halos, or bacterial water-soaking.
- **On-Screen Reticle:** High-contrast oval bounding guide instructing farmer: *"Place affected leaf flat within the frame. Distance: 15–20 cm."*
- **Camera Configuration:** Macro focus mode with tap-to-focus locking.

### View 2 — Lower Stem and Base (Vascular & Soil Interface)
- **Objective:** Identifies stem cankers, collar rot, basal lesions, sclerotia, and vascular discoloration indicative of wilting pathogens (e.g. *Fusarium*, *Rhizoctonia*).
- **On-Screen Reticle:** Vertical guide box: *"Align camera with the base of the plant where the stem enters the soil."*

### View 3 — Whole Plant Patch (Field Context View)
- **Objective:** Assesses spatial spread patterns (random spots vs. clustered patches), canopy lodging, and overall crop vigor.
- **On-Screen Reticle:** Wide angle landscape guide: *"Step back 1.5–2 meters to capture the full plant and neighboring foliage."*

---

## 3. Real-Time On-Device Camera Quality Gate

Before heavy neural inference or network transmission occurs, the native camera frame buffer is inspected by a real-time C++ / OpenCV quality pipeline:

1. **Blur Detection (Laplacian Variance):**
   ```text
   variance = var(Laplacian(gray_image))
   if (variance < 80.0) -> Alert: "Holding steady... Image is blurry."
   ```
2. **Exposure & Glare Check:**
   ```text
   mean_luminance = mean(grayscale_image)
   if (mean_luminance < 35) -> Alert: "Too dark. Turn on torch/flash."
   if (mean_luminance > 240) -> Alert: "Severe sun glare. Shade the leaf."
   ```
3. **Plant Tissue Verification:**
   - Evaluates green/chlorotic color spectrum coverage.
   - If foliage coverage is below 5%, the shutter button is disabled with prompt: *"Center the plant leaf inside the guide."*

---

## 4. Offline Edge Model Runtime

1. **Model Format:** Quantized INT8 ONNX (`models/exported/crop_disease_model_int8.onnx`).
2. **Runtime Engine:**
   - **Flutter:** `onnxruntime` Flutter package or native TFLite/CoreML bridge.
   - **React Native:** `react-native-onnxruntime` or `react-native-fast-tflite`.
3. **Offline Fallback Matrix:**
   - **Weather unavailable:** Automatically checks local coordinate cache (`weather_cache.json`). If coordinates are unrecorded, proceeds with neutral climatology features and marks weather confidence as unassisted.
   - **Treatment Rules:** Packaged as an embedded asset (`treatment_rules.json`). Only verified rules for the device's GPS region are surfaced.

---

## 5. Offline Queue & Background Cloud Sync Protocol

When operating in remote agricultural fields without cellular reception:
1. Every prediction is persisted in an encrypted local **SQLite** database (`offline_sync_queue.db`).
2. When connectivity is restored (`ConnectivityResult.wifi` or `cellular`), an asynchronous background worker (`WorkManager` on Android, `BGAppRefresh` on iOS) batches pending records and transmits them to:
   - `POST /predict` (for cloud confirmation and Grad-CAM generation)
   - `POST /feedback` (for farmer diagnostic rating logs)
3. Synchronized records are flagged with `synced = 1`.

