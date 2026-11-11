"""
Comprehensive Test Suite for Crop Disease Diagnosis ML System
Covers Dataset integrity, Survey schema, Weather connector, Leaf quality gate,
Model architectures, Uncertainty safety, Treatment rules, and FastAPI endpoints.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2
import torch
import pandas as pd
from fastapi.testclient import TestClient

# Path resolution
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "01_Dataset"))
sys.path.insert(0, str(REPO_ROOT / "02_Augmentation"))
sys.path.insert(0, str(REPO_ROOT / "04_Weather"))
sys.path.insert(0, str(REPO_ROOT / "05_Treatment"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "api"))

from config import config
from source_loaders import LabelNormalizer, compute_md5, compute_dhash, verify_image_integrity
from split_dataset import leakage_aware_split
from weather_connector import WeatherConnector
from treatment_engine import TreatmentEngine
from leaf_detector import LeafQualityGate, LeafDetector
from model import CropDiseaseClassifier, build_model
from multimodal import MultimodalCropDiseaseModel, SurveyEncoder
from uncertainty import UncertaintySafetyGate
from feedback import FeedbackManager
from main import app

REQUIRED_20_CROPS = [
    "rice", "wheat", "maize", "cotton", "tomato", "potato",
    "chilli", "grape", "apple", "banana", "mango", "soybean",
    "groundnut", "sugarcane", "coffee", "pepper", "citrus",
    "beans", "okra", "pea"
]

class TestDatasetAndTaxonomy(unittest.TestCase):
    def test_20_crops_catalog_present(self):
        """Verifies all 20 required crops are present in crop_catalog.json."""
        self.assertTrue(config.CROP_CATALOG_JSON.exists(), "crop_catalog.json does not exist")
        with open(config.CROP_CATALOG_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        present_crops = {c["crop_id"].lower() for c in data.get("crops", [])}
        for crop in REQUIRED_20_CROPS:
            self.assertIn(crop, present_crops, f"Required crop '{crop}' missing from crop_catalog.json")

    def test_canonical_labels_csv_format(self):
        """Verifies class_labels.csv canonical format: crop__disease."""
        self.assertTrue(config.CLASS_LABELS_CSV.exists(), "class_labels.csv does not exist")
        df = pd.read_csv(config.CLASS_LABELS_CSV)
        self.assertGreaterEqual(len(df), 20)
        for _, row in df.iterrows():
            lbl = str(row["canonical_label"])
            self.assertIn("__", lbl, f"Label {lbl} does not follow 'crop__disease' convention")
            c, d = lbl.split("__", 1)
            self.assertEqual(c, row["crop"].lower())

    def test_label_normalizer_behavior(self):
        """Tests that known aliases map to canonical and unknown map to review/unmapped."""
        normalizer = LabelNormalizer()
        crop, disease, canon, status = normalizer.normalize("Tomato___Early_blight", "PlantVillage")
        self.assertEqual(canon, "tomato__early_blight")
        self.assertEqual(status, "mapped_override")

        # Unknown label must NEVER be silently guessed
        crop_u, disease_u, canon_u, status_u = normalizer.normalize("AlienCrop___Space_Fungus", "UnknownSource")
        self.assertEqual(canon_u, "review/unmapped")
        self.assertEqual(status_u, "unmapped")

    def test_leakage_aware_split_integrity(self):
        """Verifies zero hash overlap between train, val, and test splits."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as tmp:
            tmp.write(
                "image_id,source,path,crop,disease,canonical_label,is_valid,status,md5_hash,dhash\n"
                "img1,PV,p1,tomato,early_blight,tomato__early_blight,True,canonical,hashA,dhA\n"
                "img2,PV,p2,tomato,early_blight,tomato__early_blight,True,canonical,hashA,dhA\n"
                "img3,PV,p3,tomato,early_blight,tomato__early_blight,True,canonical,hashB,dhB\n"
                "img4,PV,p4,tomato,early_blight,tomato__early_blight,True,canonical,hashC,dhC\n"
                "img5,PV,p5,potato,late_blight,potato__late_blight,True,canonical,hashD,dhD\n"
                "img6,PV,p6,potato,late_blight,potato__late_blight,True,canonical,hashE,dhE\n"
                "img7,PV,p7,potato,late_blight,potato__late_blight,True,canonical,hashF,dhF\n"
            )
            tmp_path = Path(tmp.name)

        try:
            df_split = leakage_aware_split(tmp_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
            train_hashes = set(df_split[df_split["split"] == "train"]["md5_hash"])
            val_hashes = set(df_split[df_split["split"] == "val"]["md5_hash"])
            test_hashes = set(df_split[df_split["split"] == "test"]["md5_hash"])

            self.assertEqual(len(train_hashes.intersection(val_hashes)), 0)
            self.assertEqual(len(train_hashes.intersection(test_hashes)), 0)
            self.assertEqual(len(val_hashes.intersection(test_hashes)), 0)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

class TestSurveySchema(unittest.TestCase):
    def test_survey_encoder_output_shape(self):
        """Verifies SurveyEncoder produces 15-dimensional numeric vector."""
        survey = {
            "root_condition": "normal",
            "stem_health": "weak",
            "recent_chemical_spray": "yes_fungicide"
        }
        vec = SurveyEncoder.encode(survey)
        self.assertEqual(vec.shape, (15,))
        self.assertEqual(vec[14], 1.0)  # available flag

    def test_missing_survey_encoder(self):
        """Verifies None survey produces neutral vector with available flag = 0."""
        vec = SurveyEncoder.encode(None)
        self.assertEqual(vec.shape, (15,))
        self.assertEqual(vec[14], 0.0)

class TestWeatherConnector(unittest.TestCase):
    def setUp(self):
        self.connector = WeatherConnector()

    def test_invalid_coordinates_handled_gracefully(self):
        """Tests that invalid GPS coordinates return safe fallback without crashing."""
        res = self.connector.get_weather(999.0, 999.0)
        self.assertFalse(res["is_available"])
        self.assertIn("Fallback", res["source"])

    def test_normalized_features_range(self):
        """Tests normalized weather feature vector is within valid [0, 1] range."""
        weather_dict = {
            "temperature_mean": 28.5,
            "humidity_mean": 75.0,
            "rainfall_7d": 40.0,
            "rainfall_14d": 90.0,
            "is_available": True
        }
        vec = WeatherConnector.normalize_weather_features(weather_dict)
        self.assertEqual(vec.shape, (5,))
        self.assertTrue(np.all(vec >= 0.0) and np.all(vec <= 1.0))
        self.assertEqual(vec[4], 1.0)

class TestLeafQualityGate(unittest.TestCase):
    def setUp(self):
        self.gate = LeafQualityGate(min_dim=64)

    def test_blurry_image_rejection(self):
        """Tests that artificially blurred images are rejected with blur warning."""
        # Create solid gray image (Laplacian variance = 0)
        solid_img = np.ones((128, 128, 3), dtype=np.uint8) * 128
        res = self.gate.check(solid_img)
        self.assertFalse(res["accepted"])
        self.assertTrue(any("blurry" in r.lower() for r in res["rejection_reasons"]))

    def test_dark_image_rejection(self):
        """Tests that underexposed black images are rejected."""
        dark_img = np.zeros((128, 128, 3), dtype=np.uint8)
        res = self.gate.check(dark_img)
        self.assertFalse(res["accepted"])
        self.assertTrue(any("dark" in r.lower() for r in res["rejection_reasons"]))

    def test_clear_plant_image_acceptance(self):
        """Tests that well-lit green image passes quality gate."""
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img[:, :] = (35, 160, 45)  # Vibrant green
        # Add high-frequency edge texture
        img[::4, ::4] = (10, 80, 20)
        img[1::4, 1::4] = (60, 220, 80)
        res = self.gate.check(img)
        self.assertTrue(res["accepted"], f"Expected acceptance, got: {res['rejection_reasons']}")

class TestMLModelAndMultimodal(unittest.TestCase):
    def test_vision_backbone_shape(self):
        """Verifies MobileNetV3 visual backbone input and output dimensions."""
        model = build_model(num_classes=82, backbone="mobilenet_v3_large", pretrained=False)
        dummy_x = torch.randn(2, 3, 224, 224)
        out = model(dummy_x)
        self.assertEqual(out.shape, (2, 82))

    def test_multimodal_forward_and_missingness(self):
        """Verifies Multimodal fusion model handles complete and missing modalities."""
        model = MultimodalCropDiseaseModel(num_classes=82, backbone_name="mobilenet_v3_large", pretrained=False)
        dummy_img = torch.randn(2, 3, 224, 224)
        dummy_weather = torch.randn(2, 5)
        dummy_survey = torch.randn(2, 15)

        # 1. Full modalities
        out_full = model(dummy_img, dummy_weather, dummy_survey)
        self.assertEqual(out_full.shape, (2, 82))

        # 2. Missing weather and survey (tolerates None)
        out_missing = model(dummy_img, None, None)
        self.assertEqual(out_missing.shape, (2, 82))

class TestUncertaintySafetyGate(unittest.TestCase):
    def setUp(self):
        self.gate = UncertaintySafetyGate(confidence_threshold=0.80)
        self.labels = ["rice__blast", "tomato__early_blight", "potato__late_blight"]

    def test_high_confidence_accepted(self):
        """Confidence >= 0.80 must be accepted."""
        probs = np.array([0.05, 0.90, 0.05])
        res = self.gate.evaluate(probs, self.labels)
        self.assertEqual(res["status"], "accepted")
        self.assertFalse(res["uncertain"])
        self.assertEqual(res["prediction"], "tomato__early_blight")

    def test_low_confidence_rejected(self):
        """Confidence < 0.80 must be marked uncertain and prediction withheld."""
        probs = np.array([0.35, 0.40, 0.25])
        res = self.gate.evaluate(probs, self.labels)
        self.assertEqual(res["status"], "uncertain")
        self.assertTrue(res["uncertain"])
        self.assertIsNone(res["prediction"])

class TestTreatmentEngine(unittest.TestCase):
    def setUp(self):
        self.engine = TreatmentEngine()

    def test_verified_treatment_lookup(self):
        """Verifies verified treatment lookup for tomato early blight in South Asia."""
        res = self.engine.lookup(
            crop="tomato",
            disease="early_blight",
            growth_stage="vegetative",
            latitude=17.385,
            longitude=78.486
        )
        self.assertTrue(res["matched"])
        self.assertEqual(res["approval_status"], "verified")
        self.assertIn("Mancozeb", res["dosage"])

    def test_healthy_plant_guidance(self):
        """Verifies healthy leaf queries return standard non-chemical maintenance protocol."""
        res = self.engine.lookup(crop="rice", disease="healthy")
        self.assertTrue(res["matched"])
        self.assertIn("healthy", res["recommended_action"].lower())

    def test_unverified_fallback_notice(self):
        """Verifies that unknown or unverified disease queries return conservative fallback."""
        res = self.engine.lookup(crop="cotton", disease="fictional_space_rot")
        self.assertFalse(res["matched"])
        self.assertTrue(res["is_fallback"])
        self.assertIn("No verified treatment rule available", res["message"])

class TestFastAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("supported_classes", data)

    def test_weather_endpoint(self):
        res = self.client.get("/weather?latitude=17.385&longitude=78.486")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("temperature_mean", data)

    def test_treatment_endpoint(self):
        res = self.client.get("/treatment?crop=potato&disease=late_blight&growth_stage=vegetative")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["matched"])

    def test_survey_validation_endpoint(self):
        payload = {
            "root_condition": "normal",
            "stem_health": "healthy",
            "recent_chemical_spray": "no"
        }
        res = self.client.post("/survey", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["valid"])

    def test_predict_endpoint_with_image(self):
        # Create a synthetic image in memory
        img = np.zeros((160, 160, 3), dtype=np.uint8)
        img[:, :] = (35, 160, 45)  # Green leaf tone
        img[::3, ::3] = (50, 200, 70)
        _, img_encoded = cv2.imencode(".jpg", img)

        files = {"image": ("test_leaf.jpg", img_encoded.tobytes(), "image/jpeg")}
        data = {
            "latitude": "17.385",
            "longitude": "78.486",
            "root_condition": "normal",
            "stem_health": "healthy",
            "recent_chemical_spray": "no",
            "growth_stage": "vegetative"
        }
        res = self.client.post("/predict", files=files, data=data)
        self.assertEqual(res.status_code, 200)
        resp_json = res.json()
        self.assertIn("status", resp_json)
        self.assertIn("confidence", resp_json)
        self.assertIn("treatment_rule", resp_json)

if __name__ == "__main__":
    unittest.main()

