# Exported Edge Models Directory

Stores edge-optimized model artifacts:

- `crop_disease_model.onnx` — Full FP32 ONNX model graph.
- `crop_disease_model_int8.onnx` — Dynamically quantized INT8 edge model for mobile and offline field inference.

To generate exported ONNX models:

```bash
python src/export_onnx.py --checkpoint models/checkpoints/best_model.pt
```

