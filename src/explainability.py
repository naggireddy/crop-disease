"""
Grad-CAM Visual Explainability Module
Generates gradient-weighted class activation mapping (Grad-CAM) heatmaps overlaid
on input plant leaf imagery to visualize symptom attribution and detect background shortcut learning.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

class GradCAM:
    """
    Computes Grad-CAM heatmaps for vision models to explain disease predictions.
    Registers forward and backward hooks on the terminal convolutional feature layer.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or self._find_default_target_layer()

        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _find_default_target_layer(self) -> nn.Module:
        """Locates the final convolutional layer of MobileNetV3 or EfficientNet backbone."""
        # Check if wrapped in multimodal model
        base = getattr(self.model, "visual_backbone", self.model)
        if hasattr(base, "features") and len(base.features) > 0:
            return base.features[-1]
        raise ValueError("Could not automatically locate final feature layer. Provide target_layer explicitly.")

    def _register_hooks(self):
        """Registers backward and forward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks.extend([h1, h2])

    def remove_hooks(self):
        """Cleans up hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
        weather_tensor: Optional[torch.Tensor] = None,
        survey_tensor: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Executes backward pass for selected class and computes normalized Grad-CAM heatmap.
        """
        self.model.zero_grad()

        # Forward pass
        if hasattr(self.model, "visual_backbone"):
            logits = self.model(input_tensor, weather_tensor, survey_tensor)
        else:
            logits = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=-1).item())

        # Target score for class
        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        # Global average pooling of gradients
        # gradients shape: [batch, channels, H, W]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)  # Guided ReLU

        cam_np = cam.squeeze().cpu().numpy()
        # Normalize to [0, 1]
        if np.max(cam_np) > np.min(cam_np):
            cam_np = (cam_np - np.min(cam_np)) / (np.max(cam_np) - np.min(cam_np))
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image_bgr: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resizes heatmap to original image size and overlays onto leaf image.
        Assesses potential background shortcut learning.
        """
        h, w = original_image_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

        overlay = cv2.addWeighted(color_heatmap, alpha, original_image_bgr, 1 - alpha, 0)

        # Background Shortcut Learning Check
        # Check whether high activation energy (>0.6) falls heavily near the image perimeter
        border_mask = np.zeros((h, w), dtype=bool)
        border_thickness = int(min(h, w) * 0.12)
        border_mask[:border_thickness, :] = True
        border_mask[-border_thickness:, :] = True
        border_mask[:, :border_thickness] = True
        border_mask[:, -border_thickness:] = True

        total_activation = np.sum(heatmap_resized)
        border_activation = np.sum(heatmap_resized[border_mask])
        border_ratio = float(border_activation / max(1e-6, total_activation))

        shortcut_detected = border_ratio > 0.45

        return overlay, {
            "border_activation_ratio": round(border_ratio, 3),
            "potential_background_shortcut": shortcut_detected,
            "warning": "Warning: Model appears to focus heavily on image perimeter/background rather than leaf center." if shortcut_detected else None
        }

    def save_explanation(
        self,
        overlay: np.ndarray,
        output_path: Union[str, Path]
    ) -> str:
        """Saves Grad-CAM overlay image to disk."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), overlay)
        return str(out_p.resolve())

