"""
Model Evaluation and Metrics Module
Calculates Top-1 accuracy, Precision, Recall, F1-Score (macro & weighted),
Confusion Matrix, per-class performance, and OOD rejection rates.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from config import config
from multimodal import MultimodalCropDiseaseModel
from dataset import create_dataloaders, CropDiseaseDataset

def evaluate_model(
    checkpoint_path: Path = config.CHECKPOINTS_DIR / "best_model.pt",
    manifest_path: Path = config.MANIFEST_CSV,
    output_report_path: Path = config.MODELS_DIR / "evaluation_report.json",
    confidence_thresh: float = config.OOD_CONFIDENCE_THRESHOLD
) -> Dict[str, Any]:
    """Evaluates model against test dataset partition and logs metrics."""
    if not checkpoint_path.exists():
        print(f"[!] Checkpoint not found at: {checkpoint_path}. Run training or supply trained model weights.")
        return {"error": "checkpoint_not_found"}

    loaders = create_dataloaders(manifest_path)
    test_loader = loaders["test"]
    if test_loader is None or len(test_loader.dataset) == 0:
        print("[!] No test samples found in dataset_manifest.csv.")
        return {"error": "no_test_samples"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_classes = checkpoint.get("num_classes", len(loaders["datasets"]["test"].label_to_idx))
    backbone = checkpoint.get("backbone", config.BACKBONE)
    multi_label = checkpoint.get("multi_label", False)

    model = MultimodalCropDiseaseModel(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=False,
        multi_label=multi_label
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_confs: List[float] = []

    with torch.no_grad():
        for images, weather, survey, targets, _ in test_loader:
            images = images.to(device)
            weather = weather.to(device)
            survey = survey.to(device)

            probs = model.predict_probabilities(images, weather, survey)
            confs, preds = torch.max(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist() if not multi_label else torch.argmax(targets, dim=-1).cpu().numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_conf = np.array(all_confs)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # OOD rejection metric
    rejected_count = int(np.sum(y_conf < confidence_thresh))
    rejection_rate = float(rejected_count / max(1, len(y_conf)))

    # Filtered accepted accuracy
    accepted_mask = y_conf >= confidence_thresh
    acc_accepted = float(accuracy_score(y_true[accepted_mask], y_pred[accepted_mask])) if np.sum(accepted_mask) > 0 else 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()

    report = {
        "model_version": checkpoint.get("model_version", config.MODEL_VERSION),
        "test_samples": len(y_true),
        "overall_accuracy": round(acc, 4),
        "accepted_accuracy_above_threshold": round(acc_accepted, 4),
        "ood_rejection_rate": round(rejection_rate, 4),
        "rejection_threshold": confidence_thresh,
        "precision_macro": round(float(p_macro), 4),
        "recall_macro": round(float(r_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(p_wt), 4),
        "recall_weighted": round(float(r_wt), 4),
        "f1_weighted": round(float(f1_wt), 4),
        "confusion_matrix": cm
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[+] Evaluation Complete:")
    print(f"    - Overall Accuracy: {acc*100:.2f}%")
    print(f"    - Macro F1-Score:   {f1_macro:.4f}")
    print(f"    - OOD Rejection:    {rejection_rate*100:.2f}% (Threshold: {confidence_thresh*100:.0f}%)")
    print(f"    - Accepted Accuracy: {acc_accepted*100:.2f}%")
    print(f"    - Saved Report to:  {output_report_path}")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate crop disease model.")
    parser.add_argument("--checkpoint", type=Path, default=config.CHECKPOINTS_DIR / "best_model.pt")
    parser.add_argument("--manifest", type=Path, default=config.MANIFEST_CSV)
    parser.add_argument("--output", type=Path, default=config.MODELS_DIR / "evaluation_report.json")
    parser.add_argument("--threshold", type=float, default=config.OOD_CONFIDENCE_THRESHOLD)

    args = parser.parse_args()
    evaluate_model(args.checkpoint, args.manifest, args.output, args.threshold)

