"""
Grad-CAM Explainability Module for Eye Classifier
=================================================
Computes Gradient-weighted Class Activation Mapping (Grad-CAM) to explain
model predictions on eye crops. Shows which spatial regions (eyelids, sclera,
pupil, lashes) drove the network to classify eyes as Closed vs. Open.
"""

from __future__ import annotations

import base64
import logging
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("drishtidrive.explainability.gradcam")


class GradCAM:
    """Computes Grad-CAM saliency maps for convolutional neural networks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target convolutional layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self._handles.extend([h_fwd, h_bwd])

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute Grad-CAM heatmap array normalized between 0.0 and 1.0.

        Args:
            input_tensor: Shape (1, C, H, W) normalized tensor.
            target_class: Class index to explain (defaults to argmax predicted class).

        Returns:
            Tuple of (2D heatmap numpy float array, predicted_class_idx, confidence_prob).
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(probs, dim=1).item())

        confidence = float(probs[0, target_class].item())

        # Backward pass with respect to target class logit
        score = logits[0, target_class]
        score.backward()

        assert self.gradients is not None and self.activations is not None

        # Global average pooling of gradients: weights alpha_k
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps followed by ReLU
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, target_class, confidence

    def overlay_on_image(
        self,
        bgr_image: np.ndarray,
        heatmap_2d: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Resize heatmap to image dimensions and overlay with Jet colormap."""
        h, w = bgr_image.shape[:2]
        resized_cam = cv2.resize(heatmap_2d, (w, h))
        uint_cam = np.uint8(255 * resized_cam)
        colored_cam = cv2.applyColorMap(uint_cam, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(colored_cam, alpha, bgr_image, 1.0 - alpha, 0)
        return overlay

    def explain_crop_b64(
        self,
        crop_bgr: np.ndarray,
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate complete Grad-CAM explanation dictionary with Base64 visualizations.
        """
        # Preprocess input crop (Resize to 64x64, normalize ImageNet stats)
        resized = cv2.resize(crop_bgr, (64, 64))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose((2, 0, 1))).float() / 255.0
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        input_tensor = tensor.unsqueeze(0)

        heatmap_2d, pred_class, confidence = self.generate_heatmap(input_tensor, target_class)
        overlay = self.overlay_on_image(crop_bgr, heatmap_2d)

        # Base64 encodings
        _, raw_buf = cv2.imencode(".jpg", crop_bgr)
        _, overlay_buf = cv2.imencode(".jpg", overlay)

        class_names = {0: "Closed", 1: "Open"}
        return {
            "predicted_class": class_names.get(pred_class, str(pred_class)),
            "class_id": pred_class,
            "confidence": round(confidence, 4),
            "raw_crop_b64": base64.b64encode(raw_buf).decode("ascii"),
            "overlay_b64": base64.b64encode(overlay_buf).decode("ascii"),
            "focus_intensity": round(float(np.mean(heatmap_2d)), 4),
        }

    def close(self) -> None:
        """Remove PyTorch hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
