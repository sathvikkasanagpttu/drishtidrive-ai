"""Domain-specific augmentations for Night and Infrared (IR) driver monitoring.

Truck cabins frequently exhibit extreme darkness, oncoming headlight glare,
and active 850nm/940nm infrared illumination. These transforms simulate:
1. Low-light gamma attenuation and sensor shot noise.
2. Active Infrared (IR) spectral monochrome conversion and specular highlights.
3. Glare/bloom from oncoming highway headlights.
4. Automatic illumination and contrast normalization via CLAHE.
"""
from __future__ import annotations

import random
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance


class RandomLowLight:
    """Simulates extreme nighttime cabin illumination and sensor noise."""

    def __init__(self, p: float = 0.5, gamma_range: Tuple[float, float] = (1.8, 3.2)) -> None:
        """Initialize low-light transform probability and gamma dimming range."""
        self.p = p
        self.gamma_range = gamma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply gamma darkening and low-light sensor noise."""
        if random.random() > self.p:
            return img

        np_img = np.array(img, dtype=np.float32) / 255.0
        gamma = random.uniform(*self.gamma_range)
        darkened = np.power(np_img, gamma)

        # Add simulated sensor shot noise
        noise = np.random.normal(0.0, 0.02, darkened.shape)
        noisy = np.clip(darkened + noise, 0.0, 1.0)
        return Image.fromarray((noisy * 255.0).astype(np.uint8))


class RandomIRSimulation:
    """Simulates active infrared (850nm/940nm) dashcam illumination."""

    def __init__(self, p: float = 0.4) -> None:
        """Initialize IR simulation probability."""
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Convert RGB image to simulated NIR monochrome with high specular contrast."""
        if random.random() > self.p:
            return img

        np_img = np.array(img)
        # Convert to single-channel luminance proxy
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        # In IR, skin is lighter and pupils/corneas have distinct specular reflections
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        ir_chan = clahe.apply(gray)
        # Return 3-channel duplicate for standard CNN backbones
        ir_rgb = cv2.merge([ir_chan, ir_chan, ir_chan])
        return Image.fromarray(ir_rgb)


class RandomHeadlightGlare:
    """Simulates oncoming high-beam headlight glare across the windshield."""

    def __init__(self, p: float = 0.3) -> None:
        """Initialize glare transform probability."""
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Inject localized Gaussian brightness flare."""
        if random.random() > self.p:
            return img

        np_img = np.array(img, dtype=np.float32)
        h, w = np_img.shape[:2]
        center_x = random.randint(0, w)
        center_y = random.randint(0, h)
        radius = random.randint(w // 4, w // 2)

        y, x = np.ogrid[:h, :w]
        dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
        glare_mask = np.exp(-dist_sq / (2.0 * (radius ** 2)))
        glare_3d = np.repeat(glare_mask[:, :, np.newaxis], 3, axis=2) * random.uniform(80.0, 160.0)

        glared = np.clip(np_img + glare_3d, 0.0, 255.0)
        return Image.fromarray(glared.astype(np.uint8))


def get_robust_transforms(
    image_size: int = 96,
    include_night_ir: bool = True,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return robust training and evaluation transform pipelines."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_steps = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
    ]

    if include_night_ir:
        train_steps.extend([
            RandomLowLight(p=0.4),
            RandomIRSimulation(p=0.35),
            RandomHeadlightGlare(p=0.25),
        ])

    train_steps.extend([
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return transforms.Compose(train_steps), eval_transform
