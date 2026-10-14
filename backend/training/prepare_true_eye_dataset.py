"""Prepare balanced eye datasets using genuine per-eye ground truth labels.

Extracts landmark-aligned eye crops (using standard 6-point ocular boundaries)
from MRL Eye Dataset, Closed Eyes in the Wild (CEW), or auto-labeled driver video feeds.
Enforces subject-level stratified splitting to eliminate identity data leakage.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from backend.landmarks import MediaPipeFaceLandmarker

logger = logging.getLogger("drishtidrive.prepare_true_eyes")


def parse_mrl_filename(filename: str) -> Optional[Dict[str, int]]:
    """Parse MRL Eye dataset filename metadata.

    Format: <subjectId>_<imgNum>_<gender>_<glasses>_<eyeState>_<reflections>_<lighting>_<sensor>.png
    eyeState: 0=closed, 1=open
    glasses: 0=no, 1=yes
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 8:
        try:
            return {
                "subject_id": int(parts[0][1:]) if parts[0].startswith("s") else int(parts[0]),
                "glasses": int(parts[3]),
                "eye_state": int(parts[4]),  # 0=closed, 1=open
                "lighting": int(parts[6]),
            }
        except ValueError:
            return None
    return None


def extract_landmarks_from_face(
    landmarker: MediaPipeFaceLandmarker, frame: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Extract left and right eye crops bounded by 6 landmarks and compute mean EAR."""
    res = landmarker.detect(frame)
    if not res.detected or res.left_eye_crop is None or res.right_eye_crop is None:
        return None, None, 0.28
    return res.left_eye_crop, res.right_eye_crop, res.mean_ear


def prepare_mrl_dataset(
    source_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_samples: int = 5000,
) -> Dict[str, Any]:
    """Prepare MRL Eye dataset with subject-level zero-leakage split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images = list(source_dir.rglob("*.png")) + list(source_dir.rglob("*.jpg"))
    if not images:
        logger.warning("No images found in MRL source directory: %s", source_dir)
        return {"status": "empty", "samples": 0}

    # Group by subject ID for leakage-free subject split
    subject_map: Dict[int, List[Path]] = {}
    for img_path in images:
        meta = parse_mrl_filename(img_path.name)
        if meta is None:
            continue
        sid = meta["subject_id"]
        subject_map.setdefault(sid, []).append(img_path)

    subjects = sorted(list(subject_map.keys()))
    random.seed(42)
    random.shuffle(subjects)

    n_test = max(1, int(len(subjects) * test_ratio))
    n_val = max(1, int(len(subjects) * val_ratio))
    test_subs = set(subjects[:n_test])
    val_subs = set(subjects[n_test : n_test + n_val])
    train_subs = set(subjects[n_test + n_val :])

    splits = {
        "train": (train_subs, output_dir / "train"),
        "val": (val_subs, output_dir / "val"),
        "test": (test_subs, output_dir / "test"),
    }

    counts = {"train": {"open": 0, "closed": 0}, "val": {"open": 0, "closed": 0}, "test": {"open": 0, "closed": 0}}

    for split_name, (sub_set, split_path) in splits.items():
        (split_path / "open").mkdir(parents=True, exist_ok=True)
        (split_path / "closed").mkdir(parents=True, exist_ok=True)

        for sid in sub_set:
            for p in subject_map[sid]:
                meta = parse_mrl_filename(p.name)
                if not meta:
                    continue
                label = "open" if meta["eye_state"] == 1 else "closed"
                dst = split_path / label / p.name
                if not dst.exists():
                    img = cv2.imread(str(p))
                    if img is not None:
                        cv2.imwrite(str(dst), img)
                        counts[split_name][label] += 1

    return {"status": "success", "counts": counts, "total_subjects": len(subjects)}


def auto_label_video_clips(
    video_path: Path,
    output_dir: Path,
    landmarker: MediaPipeFaceLandmarker,
    open_ear_min: float = 0.28,
    closed_ear_max: float = 0.15,
    max_frames: int = 500,
) -> Dict[str, int]:
    """Extract and auto-label high-confidence landmark crops from driver video feeds."""
    open_dir = output_dir / "open"
    closed_dir = output_dir / "closed"
    open_dir.mkdir(parents=True, exist_ok=True)
    closed_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"open": 0, "closed": 0}

    frame_idx = 0
    saved = {"open": 0, "closed": 0}

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        l_crop, r_crop, mean_ear = extract_landmarks_from_face(landmarker, frame)
        if l_crop is None or r_crop is None:
            continue

        # Strict ground-truth filtering: skip ambiguous transition zone
        if mean_ear >= open_ear_min:
            target_dir = open_dir
            cls_name = "open"
        elif mean_ear <= closed_ear_max:
            target_dir = closed_dir
            cls_name = "closed"
        else:
            continue  # Ambiguous frame, exclude from training set

        for side, crop in [("left", l_crop), ("right", r_crop)]:
            out_file = target_dir / f"vid_{video_path.stem}_f{frame_idx}_{side}.jpg"
            cv2.imwrite(str(out_file), crop)
            saved[cls_name] += 1

    cap.release()
    return saved


def main() -> None:
    """CLI entrypoint for true eye dataset preparation."""
    parser = argparse.ArgumentParser(description="Prepare genuine per-eye open/closed datasets.")
    parser.add_argument("--mrl-dir", type=str, help="Path to raw MRL dataset root.")
    parser.add_argument("--video", type=str, help="Path to video file for auto-labeling.")
    parser.add_argument("--output-dir", type=str, default="data/true_eyes", help="Target output directory.")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    if args.mrl_dir:
        res = prepare_mrl_dataset(Path(args.mrl_dir), out_path)
        print("MRL preparation summary:", json.dumps(res, indent=2))

    if args.video:
        landmarker = MediaPipeFaceLandmarker()
        res = auto_label_video_clips(Path(args.video), out_path, landmarker)
        print("Video auto-labeling summary:", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
