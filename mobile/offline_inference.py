"""
Edge & Mobile Offline Inference Engine
Executes complete crop disease diagnosis without continuous internet access.
Utilizes ONNX Runtime for hardware-accelerated edge inference, local weather cache,
on-device 3-question survey encoding, local treatment database queries,
and local diagnostic queueing for later cloud synchronization.
"""

import os
import sys
import json
import sqlite3
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "04_Weather"))
sys.path.insert(0, str(REPO_ROOT / "05_Treatment"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from config import config
from weather_connector import WeatherConnector
from treatment_engine import TreatmentEngine
from uncertainty import UncertaintySafetyGate
from multimodal import SurveyEncoder

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class OfflineDiagnosisEngine:
    """
    Independent offline diagnosis runner suitable for mobile edge runtimes.
    """
    def __init__(
        self,
        onnx_model_path: Optional[Path] = None,
        treatment_rules_path: Optional[Path] = None,
        offline_db_path: Optional[Path] = None
    ):
        self.onnx_path = onnx_model_path or (config.EXPORTED_DIR / "crop_disease_model_int8.onnx")
        if not self.onnx_path.exists():
            # Try FP32 fallback
            self.onnx_path = config.EXPORTED_DIR / "crop_disease_model.onnx"

        self.treatment_engine = TreatmentEngine(treatment_rules_path)
        self.weather_connector = WeatherConnector()
        self.uncertainty_gate = UncertaintySafetyGate(confidence_threshold=config.OOD_CONFIDENCE_THRESHOLD)

        self.offline_db_path = offline_db_path or (config.DATA_DIR / "feedback" / "offline_sync_queue.db")
        self._init_offline_db()

        self.class_labels = self._load_class_labels()
        self.session = self._init_onnx_session()

    def _load_class_labels(self) -> List[str]:
        if config.CLASS_LABELS_CSV.exists():
            import pandas as pd
            df = pd.read_csv(config.CLASS_LABELS_CSV)
            return df["canonical_label"].tolist()
        return [f"class_{i}" for i in range(82)]

    def _init_onnx_session(self):
        """Initializes ONNX Runtime session if model exists."""
        if self.onnx_path.exists():
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
                print(f"[+] Offline ONNX session initialized from {self.onnx_path}")
                return session
            except Exception as e:
                print(f"[!] Warning: Could not initialize ONNX session ({e}).")
        else:
            print(f"[-] ONNX edge model not found at {self.onnx_path}. Use export_onnx.py to compile edge model.")
        return None

    def _init_offline_db(self):
        """Initializes local SQLite database to queue offline diagnoses for cloud sync."""
        self.offline_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.offline_db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS offline_inferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                image_path TEXT,
                prediction TEXT,
                confidence REAL,
                status TEXT,
                synced INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

    def preprocess_image(self, image_bgr: np.ndarray, target_size: int = 224) -> np.ndarray:
        """Standardized edge image preprocessing: resize, RGB, normalize to NCHW."""
        img_resized = cv2.resize(image_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
        # Transpose HWC -> CHW, add batch dimension
        nchw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(nchw, axis=0).astype(np.float32)

    def diagnose_offline(
        self,
        image_input: Union[str, Path, np.ndarray],
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        survey_dict: Optional[Dict[str, Any]] = None,
        growth_stage: str = "vegetative",
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs on-device offline diagnosis."""
        if isinstance(image_input, (str, Path)):
            img_path = str(Path(image_input).resolve())
            img = cv2.imread(img_path)
            if img is None:
                return {"status": "error", "message": f"Cannot read image from {image_input}"}
        else:
            img = image_input
            img_path = "offline_capture.jpg"

        # Preprocess image
        input_tensor = self.preprocess_image(img, target_size=config.IMAGE_SIZE)

        # Handle weather offline: use cached or neutral fallback
        if latitude is not None and longitude is not None:
            weather_data = self.weather_connector.get_weather(latitude, longitude)
        else:
            weather_data = self.weather_connector.get_fallback_weather(0.0, 0.0, reason="offline_no_gps")

        weather_vec = WeatherConnector.normalize_weather_features(weather_data)
        weather_tensor = np.expand_dims(weather_vec, axis=0).astype(np.float32)

        # Survey vector
        survey_vec = SurveyEncoder.encode(survey_dict)
        survey_tensor = np.expand_dims(survey_vec, axis=0).astype(np.float32)

        # Execute model
        if self.session is not None:
            outputs = self.session.run(
                None,
                {
                    "images": input_tensor,
                    "weather_vectors": weather_tensor,
                    "survey_vectors": survey_tensor
                }
            )
            logits = outputs[0][0]
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
        else:
            # Fallback mock distribution if ONNX artifact not yet exported
            probs = np.ones(len(self.class_labels), dtype=np.float32) / len(self.class_labels)

        # Uncertainty Decision
        decision = self.uncertainty_gate.evaluate(probs, self.class_labels)

        pred_label = decision.get("prediction") or decision.get("candidate_prediction")
        crop_name, disease_name = pred_label.split("__", 1) if pred_label and "__" in pred_label else ("unknown", "unknown")

        # Offline Treatment Lookup
        if decision["status"] == "accepted":
            treatment_info = self.treatment_engine.lookup(
                crop=crop_name,
                disease=disease_name,
                growth_stage=growth_stage,
                latitude=latitude,
                longitude=longitude,
                region=region
            )
        else:
            treatment_info = {
                "matched": False,
                "message": "Offline diagnosis uncertain. Do not apply unverified chemicals.",
                "approval_status": "unverified",
                "is_fallback": True
            }

        # Queue locally for sync
        self._queue_for_sync(img_path, pred_label, decision["confidence"], decision["status"])

        return {
            "mode": "edge_offline",
            "crop": crop_name if decision["status"] == "accepted" else None,
            "disease": disease_name if decision["status"] == "accepted" else None,
            "confidence": decision["confidence"],
            "status": decision["status"],
            "uncertain": decision["uncertain"],
            "weather_cached": weather_data.get("source", "").startswith("Cached"),
            "treatment_rule": treatment_info,
            "message": decision["message"],
            "queued_for_cloud_sync": True
        }

    def _queue_for_sync(self, img_path: str, prediction: str, confidence: float, status: str):
        """Queues record in local SQLite database."""
        try:
            from datetime import datetime, timezone
            conn = sqlite3.connect(str(self.offline_db_path))
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO offline_inferences (timestamp, image_path, prediction, confidence, status, synced) VALUES (?, ?, ?, ?, ?, 0)",
                (datetime.now(timezone.utc).isoformat(), img_path, prediction, confidence, status)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run edge offline crop disease inference.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--root", type=str, default="normal")
    parser.add_argument("--stem", type=str, default="healthy")
    parser.add_argument("--spray", type=str, default="no")

    args = parser.parse_args()
    engine = OfflineDiagnosisEngine()
    survey = {"root_condition": args.root, "stem_health": args.stem, "recent_chemical_spray": args.spray}
    res = engine.diagnose_offline(args.image, survey_dict=survey)
    print(json.dumps(res, indent=2))

