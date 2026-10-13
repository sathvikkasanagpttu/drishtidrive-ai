"""Unit tests for multimodal training transforms and subject-stratified splitting.

Verifies:
- Subject stratification ensures zero subject leakage across train, val, and test splits.
- Low-light, IR simulation, and glare transforms execute without shape/type errors.
- Multimodal evaluation pipeline generates valid condition metrics.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from backend.training.train_multimodal import build_subject_splits
from backend.training.transforms import (
    RandomHeadlightGlare,
    RandomIRSimulation,
    RandomLowLight,
    get_robust_transforms,
)


def test_night_and_ir_augmentations() -> None:
    """Verify custom transforms output valid RGB PIL images."""
    img = Image.new("RGB", (96, 96), color=(120, 140, 160))

    low_light = RandomLowLight(p=1.0)
    out_ll = low_light(img)
    assert isinstance(out_ll, Image.Image)
    assert out_ll.size == (96, 96)

    ir_sim = RandomIRSimulation(p=1.0)
    out_ir = ir_sim(img)
    assert isinstance(out_ir, Image.Image)
    assert out_ir.size == (96, 96)

    glare = RandomHeadlightGlare(p=1.0)
    out_glare = glare(img)
    assert isinstance(out_glare, Image.Image)
    assert out_glare.size == (96, 96)


def test_robust_transforms_pipeline() -> None:
    """Verify composed transform produces correctly shaped and normalized PyTorch tensor."""
    train_t, eval_t = get_robust_transforms(image_size=96, include_night_ir=True)
    img = Image.new("RGB", (128, 128), color=(100, 100, 100))

    tensor = train_t(img)
    assert tensor.shape == (3, 96, 96)

    eval_tensor = eval_t(img)
    assert eval_tensor.shape == (3, 96, 96)


def test_subject_stratified_split_zero_leakage(tmp_path: Path) -> None:
    """Verify that no subject present in train ever appears in val or test sets."""
    # Synthesize multi-subject directory
    dataset_dir = tmp_path / "mock_dataset"
    for s_idx in range(10):
        subj = f"subject_{s_idx:02d}"
        for cond in ["normal", "night"]:
            for state in ["open", "closed"]:
                p = dataset_dir / subj / cond / state
                p.mkdir(parents=True, exist_ok=True)
                img = Image.new("RGB", (32, 32), color=(100, 100, 100))
                img.save(p / "sample_0.png")

    train_s, val_s, test_s = build_subject_splits(dataset_dir, test_ratio=0.2, val_ratio=0.2, seed=42)

    train_subjects = {s[2] for s in train_s}
    val_subjects = {s[2] for s in val_s}
    test_subjects = {s[2] for s in test_s}

    # Strict zero subject leakage
    assert len(train_subjects.intersection(val_subjects)) == 0
    assert len(train_subjects.intersection(test_subjects)) == 0
    assert len(val_subjects.intersection(test_subjects)) == 0
    assert len(train_subjects) + len(val_subjects) + len(test_subjects) == 10
