"""
Anonymized Diagnostic Logging and Agronomist Feedback Queue
Logs operational predictions without PII, manages the agronomist review queue,
and stores verified human-in-the-loop labels for retraining.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from config import config

class FeedbackManager:
    """
    Manages anonymized diagnostic logging and human agronomist review queues.
    Only reviews verified by accredited agronomists are promoted to retraining.
    """
    def __init__(self, feedback_dir: Optional[Path] = None):
        self.feedback_dir = feedback_dir or config.FEEDBACK_DIR
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.logs_file = self.feedback_dir / "diagnostic_logs.jsonl"
        self.queue_file = self.feedback_dir / "review_queue.jsonl"
        self.verified_file = self.feedback_dir / "verified_labels.jsonl"

    def log_prediction(
        self,
        prediction_result: Dict[str, Any],
        image_path: str,
        weather_available: bool,
        survey_available: bool,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        gradcam_path: Optional[str] = None
    ) -> str:
        """
        Logs an anonymized prediction record.
        Coordinates are rounded to 2 decimal places to protect farm privacy.
        """
        prediction_id = str(uuid.uuid4())
        
        # Privacy-preserving coordinate rounding (~1.1 km)
        anon_lat = round(latitude, 2) if latitude is not None else None
        anon_lon = round(longitude, 2) if longitude is not None else None

        record = {
            "prediction_id": prediction_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "crop": prediction_result.get("crop"),
            "disease": prediction_result.get("disease"),
            "prediction": prediction_result.get("prediction"),
            "confidence": prediction_result.get("confidence"),
            "status": prediction_result.get("status"),
            "uncertain": prediction_result.get("uncertain", False),
            "image_path": str(image_path),
            "gradcam_path": str(gradcam_path) if gradcam_path else None,
            "latitude": anon_lat,
            "longitude": anon_lon,
            "weather_available": weather_available,
            "survey_available": survey_available
        }

        # Append to master log
        with open(self.logs_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # If uncertain or low confidence, enqueue for agronomist review
        if record["uncertain"] or record["status"] == "uncertain":
            self._enqueue_for_review(record)

        return prediction_id

    def _enqueue_for_review(self, record: Dict[str, Any]):
        """Pushes ambiguous or OOD prediction into the agronomist review queue."""
        queue_item = {
            "prediction_id": record["prediction_id"],
            "timestamp": record["timestamp"],
            "image_path": record["image_path"],
            "gradcam_path": record.get("gradcam_path"),
            "candidate_crop": record.get("crop"),
            "candidate_disease": record.get("disease"),
            "candidate_prediction": record.get("prediction"),
            "confidence": record.get("confidence"),
            "review_status": "pending_agronomist_review"
        }
        with open(self.queue_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(queue_item) + "\n")

    def get_pending_review_queue(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves items awaiting agronomist verification."""
        if not self.queue_file.exists():
            return []

        pending = []
        with open(self.queue_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if item.get("review_status") == "pending_agronomist_review":
                        pending.append(item)
                        if len(pending) >= limit:
                            break
        return pending

    def submit_agronomist_review(
        self,
        prediction_id: str,
        verdict: str,
        verified_crop: Optional[str] = None,
        verified_disease: Optional[str] = None,
        agronomist_id: str = "agronomist_expert",
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Records expert agronomist verdict.
        Verdicts: 'correct', 'incorrect', 'unknown', 'new_disease', 'needs_more_data'.
        If 'correct' or 'incorrect' with verified crop/disease, stores verified record for retraining.
        """
        valid_verdicts = {"correct", "incorrect", "unknown", "new_disease", "needs_more_data"}
        if verdict not in valid_verdicts:
            raise ValueError(f"Invalid review verdict: {verdict}. Must be one of {valid_verdicts}")

        # Find queued item
        target_item = None
        remaining_items = []
        if self.queue_file.exists():
            with open(self.queue_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    if item.get("prediction_id") == prediction_id:
                        target_item = item
                    else:
                        remaining_items.append(item)

        # Rewrite queue without resolved item
        with open(self.queue_file, "w", encoding="utf-8") as f:
            for item in remaining_items:
                f.write(json.dumps(item) + "\n")

        review_record = {
            "prediction_id": prediction_id,
            "verdict": verdict,
            "verified_crop": verified_crop or (target_item.get("candidate_crop") if target_item else None),
            "verified_disease": verified_disease or (target_item.get("candidate_disease") if target_item else None),
            "canonical_label": f"{verified_crop}__{verified_disease}" if verified_crop and verified_disease else None,
            "agronomist_id": agronomist_id,
            "notes": notes or "",
            "image_path": target_item.get("image_path") if target_item else None,
            "reviewed_at": datetime.now(timezone.utc).isoformat()
        }

        # Only trustworthy labels with defined crop & disease are recorded for retraining
        if verdict in ("correct", "incorrect") and review_record["canonical_label"]:
            with open(self.verified_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(review_record) + "\n")

        return review_record

