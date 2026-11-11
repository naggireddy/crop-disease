"""
Edge Model Export & Quantization Pipeline
Exports PyTorch Crop Disease model to ONNX format, performs dynamic INT8 quantization,
measures model file size, and benchmarks runtime latency.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from config import config
from multimodal import MultimodalCropDiseaseModel

def export_to_onnx(
    checkpoint_path: Path = config.CHECKPOINTS_DIR / "best_model.pt",
    output_onnx_path: Path = config.EXPORTED_DIR / "crop_disease_model.onnx",
    quantize: bool = True
) -> Dict[str, Any]:
    """
    Exports PyTorch model to ONNX and optionally applies dynamic INT8 quantization.
    """
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")  # Export on CPU for edge compatibility

    num_classes = 82
    backbone = config.BACKBONE
    multi_label = config.MULTI_LABEL

    model = MultimodalCropDiseaseModel(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=False,
        multi_label=multi_label
    ).to(device)

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[+] Loaded model weights from {checkpoint_path}")
    else:
        print("[-] Checkpoint not found. Exporting initialized architecture graph for edge benchmarking.")

    model.eval()

    # Create dummy tensors for 3 modalities
    dummy_img = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    dummy_weather = torch.randn(1, 5, device=device)
    dummy_survey = torch.randn(1, 15, device=device)

    # 1. Export FP32 ONNX
    print(f"[*] Exporting FP32 ONNX model to: {output_onnx_path}...")
    torch.onnx.export(
        model,
        (dummy_img, dummy_weather, dummy_survey),
        str(output_onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["images", "weather_vectors", "survey_vectors"],
        output_names=["logits"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "weather_vectors": {0: "batch_size"},
            "survey_vectors": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )

    fp32_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"[+] FP32 ONNX export complete. Size: {fp32_size_mb:.2f} MB")

    quantized_path = None
    quant_size_mb = None

    # 2. Dynamic INT8 Quantization (if onnxruntime is available)
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantized_path = output_onnx_path.with_name(f"{output_onnx_path.stem}_int8.onnx")
            print(f"[*] Applying dynamic INT8 quantization -> {quantized_path}...")
            quantize_dynamic(
                model_input=str(output_onnx_path),
                model_output=str(quantized_path),
                weight_type=QuantType.QInt8
            )
            quant_size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
            print(f"[+] INT8 Quantized model complete. Size: {quant_size_mb:.2f} MB (Compression: {(1 - quant_size_mb / fp32_size_mb) * 100:.1f}%)")
        except Exception as e:
            print(f"[!] Warning: Quantization could not be performed ({e}). Preserving FP32 artifact.")

    # 3. Latency Benchmark via ONNXRuntime
    benchmark_results = {}
    try:
        import onnxruntime as ort
        target_eval_model = str(quantized_path if quantized_path and quantized_path.exists() else output_onnx_path)
        session = ort.InferenceSession(target_eval_model, providers=["CPUExecutionProvider"])
        
        inp_img = np.random.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).astype(np.float32)
        inp_weather = np.random.randn(1, 5).astype(np.float32)
        inp_survey = np.random.randn(1, 15).astype(np.float32)

        # Warmup
        for _ in range(5):
            session.run(None, {"images": inp_img, "weather_vectors": inp_weather, "survey_vectors": inp_survey})

        # Measure 20 runs
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            session.run(None, {"images": inp_img, "weather_vectors": inp_weather, "survey_vectors": inp_survey})
            times.append((time.perf_counter() - t0) * 1000)

        benchmark_results = {
            "eval_model": target_eval_model,
            "mean_latency_ms": round(float(np.mean(times)), 2),
            "p95_latency_ms": round(float(np.percentile(times, 95)), 2)
        }
        print(f"[+] Benchmark: Mean Latency = {benchmark_results['mean_latency_ms']} ms | P95 = {benchmark_results['p95_latency_ms']} ms")
    except Exception as e:
        benchmark_results = {"error": f"Benchmark failed: {str(e)}"}

    return {
        "fp32_onnx_path": str(output_onnx_path),
        "fp32_size_mb": round(fp32_size_mb, 2),
        "quantized_int8_path": str(quantized_path) if quantized_path else None,
        "quantized_size_mb": round(quant_size_mb, 2) if quant_size_mb else None,
        "benchmark": benchmark_results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch Crop Model to ONNX and Quantize.")
    parser.add_argument("--checkpoint", type=Path, default=config.CHECKPOINTS_DIR / "best_model.pt")
    parser.add_argument("--output", type=Path, default=config.EXPORTED_DIR / "crop_disease_model.onnx")
    parser.add_argument("--no-quantize", action="store_true", help="Disable INT8 quantization.")

    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output, quantize=not args.no_quantize)

