"""
Central End-to-End Multimodal Inference Service
Coordinates the complete diagnostic pipeline:
Quality Gate → Leaf Detection → Multimodal Classification →
Uncertainty / OOD Check → Grad-CAM Attribution → Geo-Fenced Treatment Lookup →
Anonymized Diagnostic Logging.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union
import cv2
import numpy as np
import torch
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "02_Augmentation"))
sys.path.insert(0, str(REPO_ROOT / "04_Weather"))
sys.path.insert(0, str(REPO_ROOT / "05_Treatment"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from config import config
from leaf_detector import LeafQualityGate, LeafDetector
from weather_connector import WeatherConnector
from treatment_engine import TreatmentEngine
from uncertainty import UncertaintySafetyGate
from explainability import GradCAM
from multimodal import MultimodalCropDiseaseModel, SurveyEncoder
from feedback import FeedbackManager
from augmentation import get_test_transforms

class InferencePipeline:
    """
    Central production inference orchestrator integrating all modular stages.
    """
    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        confidence_threshold: float = config.OOD_CONFIDENCE_THRESHOLD,
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = checkpoint_path or (config.CHECKPOINTS_DIR / "best_model.pt")

        # Initialize modular components
        self.quality_gate = LeafQualityGate()
        self.leaf_detector = LeafDetector()
        self.weather_connector = WeatherConnector()
        self.treatment_engine = TreatmentEngine()
        self.uncertainty_gate = UncertaintySafetyGate(confidence_threshold=confidence_threshold)
        self.feedback_manager = FeedbackManager()
        self.transform = get_test_transforms(config.IMAGE_SIZE)

        # Load class labels
        self.class_labels = self._load_class_labels()
        self.num_classes = len(self.class_labels)

        # Initialize model
        self.model, self.is_trained = self._load_model()
        self.gradcam = GradCAM(self.model) if self.model is not None else None

    def _load_class_labels(self) -> list:
        if config.CLASS_LABELS_CSV.exists():
            df = pd.read_csv(config.CLASS_LABELS_CSV)
            return df["canonical_label"].tolist()
        return [f"class_{i}" for i in range(82)]

    def _load_model(self) -> Tuple[nn.Module, bool]:
        """Loads trained model checkpoint if available, or initializes untrained architecture."""
        model = MultimodalCropDiseaseModel(
            num_classes=self.num_classes,
            backbone_name=config.BACKBONE,
            pretrained=True,
            multi_label=config.MULTI_LABEL
        ).to(self.device)

        is_trained = False
        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                model.load_state_dict(ckpt["model_state_dict"])
                is_trained = True
                print(f"[+] Loaded trained model weights from: {self.checkpoint_path}")
            except Exception as e:
                print(f"[!] Warning: Could not load checkpoint ({e}). Operating with baseline initialized weights.")
        else:
            print(f"[-] Checkpoint {self.checkpoint_path} not found. Operating with base ImageNet initialization.")

        model.eval()
        return model, is_trained

    def diagnose(
        self,
        image_input: Union[str, Path, np.ndarray],
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        survey_data: Optional[Dict[str, Any]] = None,
        growth_stage: Optional[str] = "vegetative",
        region: Optional[str] = None,
        save_gradcam: bool = True
    ) -> Dict[str, Any]:
        """
        Executes end-to-end diagnostic workflow on an image input.
        """
        # Read image
        if isinstance(image_input, (str, Path)):
            img_path = str(Path(image_input).resolve())
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                return {
                    "status": "error",
                    "message": f"Unable to read image from path: {image_input}"
                }
        else:
            img_bgr = image_input
            img_path = "in_memory_image.jpg"

        # -------------------------------------------------------------
        # Stage 1: Quality Gate
        # -------------------------------------------------------------
        gate_result = self.quality_gate.check(img_bgr)
        if not gate_result["accepted"]:
            return {
                "crop": None,
                "disease": None,
                "prediction": None,
                "confidence": 0.0,
                "status": "rejected_quality_gate",
                "quality_gate": gate_result,
                "message": "Image quality check failed. " + " ".join(gate_result["rejection_reasons"]),
                "action_required": "Please retake the photo following on-screen camera guidelines."
            }

        # -------------------------------------------------------------
        # Stage 2: Leaf Detection & Crop
        # -------------------------------------------------------------
        leaf_crop, detector_meta = self.leaf_detector.detect_and_crop(img_bgr)

        # -------------------------------------------------------------
        # Stage 3 & 4: Weather Retrieval & Survey Encoding
        # -------------------------------------------------------------
        if latitude is not None and longitude is not None:
            weather_dict = self.weather_connector.get_weather(latitude, longitude)
        else:
            weather_dict = self.weather_connector.get_fallback_weather(0.0, 0.0, reason="no_coordinates_provided")

        weather_vec = WeatherConnector.normalize_weather_features(weather_dict)
        survey_vec = SurveyEncoder.encode(survey_data)

        # Convert to PyTorch tensors
        crop_rgb = cv2.cvtColor(leaf_crop, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=crop_rgb)
        img_tensor = transformed["image"].unsqueeze(0).to(self.device)
        weather_tensor = torch.from_numpy(weather_vec).unsqueeze(0).to(self.device)
        survey_tensor = torch.from_numpy(survey_vec).unsqueeze(0).to(self.device)

        # -------------------------------------------------------------
        # Stage 5: Multimodal Forward Pass
        # -------------------------------------------------------------
        with torch.no_grad():
            probs = self.model.predict_probabilities(img_tensor, weather_tensor, survey_tensor)
            probs_np = probs.cpu().numpy().flatten()

        # -------------------------------------------------------------
        # Stage 6: Uncertainty / OOD Safety Check
        # -------------------------------------------------------------
        decision = self.uncertainty_gate.evaluate(probs_np, self.class_labels)

        # Target class for explainability
        target_class_idx = int(np.argmax(probs_np))

        # -------------------------------------------------------------
        # Stage 7: Grad-CAM Explainability
        # -------------------------------------------------------------
        gradcam_path = None
        shortcut_warning = None
        try:
            cam_map = self.gradcam.generate_heatmap(
                img_tensor,
                target_class_idx=target_class_idx,
                weather_tensor=weather_tensor,
                survey_tensor=survey_tensor
            )
            overlay, shortcut_info = self.gradcam.overlay_heatmap(cam_map, leaf_crop)
            shortcut_warning = shortcut_info.get("warning")

            if save_gradcam:
                out_dir = config.DATA_DIR / "processed" / "gradcam"
                out_name = f"gradcam_{Path(img_path).stem}_{target_class_idx}.jpg"
                gradcam_path = self.gradcam.save_explanation(overlay, out_dir / out_name)
        except Exception as e:
            gradcam_path = f"Grad-CAM generation error: {str(e)}"

        # -------------------------------------------------------------
        # Stage 8: Geo-Fenced Treatment Rule Lookup
        # -------------------------------------------------------------
        pred_label = decision.get("prediction") or decision.get("candidate_prediction")
        crop_name, disease_name = pred_label.split("__", 1) if pred_label and "__" in pred_label else ("unknown", "unknown")

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
                "message": "Diagnosis uncertain. Treatment prescription withheld until verified by an accredited agronomist.",
                "approval_status": "unverified",
                "is_fallback": True
            }

        # -------------------------------------------------------------
        # Stage 9: Diagnostic Logging & Agronomist Review Queueing
        # -------------------------------------------------------------
        pred_id = self.feedback_manager.log_prediction(
            prediction_result=decision,
            image_path=img_path,
            weather_available=weather_dict.get("is_available", False),
            survey_available=bool(survey_data and survey_data.get("root_condition")),
            latitude=latitude,
            longitude=longitude,
            gradcam_path=gradcam_path
        )

        # -------------------------------------------------------------
        # Stage 10: Final Response
        # -------------------------------------------------------------
        return {
            "prediction_id": pred_id,
            "crop": crop_name if decision["status"] == "accepted" else None,
            "disease": disease_name if decision["status"] == "accepted" else None,
            "canonical_label": pred_label if decision["status"] == "accepted" else None,
            "confidence": decision["confidence"],
            "entropy": decision["entropy"],
            "status": decision["status"],
            "uncertain": decision["uncertain"],
            "candidates": decision["candidates"],
            "message": decision["message"],
            "weather_context": weather_dict,
            "survey_context": survey_data or {},
            "explanation": {
                "gradcam_path": gradcam_path,
                "shortcut_warning": shortcut_warning
            },
            "treatment_rule": treatment_info,
            "leaf_detector_meta": detector_meta,
            "is_trained_model": self.is_trained
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end multimodal crop disease diagnosis.")
    parser.add_argument("--image", type=str, required=True, help="Path to input plant/leaf image.")
    parser.add_argument("--lat", type=float, default=None, help="GPS latitude.")
    parser.add_argument("--lon", type=float, default=None, help="GPS longitude.")
    parser.add_argument("--root", type=str, default="normal", help="Root condition (normal, discolored, rotting, unknown).")
    parser.add_argument("--stem", type=str, default="healthy", help="Stem health (healthy, weak, lesions_spots, wilting, unknown).")
    parser.add_argument("--spray", type=str, default="no", help="Recent chemical spray (no, yes_pesticide, yes_fungicide, yes_fertilizer, unknown).")
    parser.add_argument("--growth-stage", type=str, default="vegetative", help="Crop growth stage.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path.")

    args = parser.parse_args()

    pipeline = InferencePipeline()
    survey = {
        "root_condition": args.root,
        "stem_health": args.stem,
        "recent_chemical_spray": args.spray
    }

    result = pipeline.diagnose(
        image_input=args.image,
        latitude=args.lat,
        longitude=args.lon,
        survey_data=survey,
        growth_stage=args.growth_stage
    )

    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

