"""Guided live interactive evaluation for driver eye state ground truth validation.

Executes structured interactive phases:
1. "Keep eyes OPEN for 10s" (ground truth = OPEN)
2. "CLOSE eyes for 5s"       (ground truth = CLOSED)
3. "Blink naturally for 10s"  (ground truth = natural dynamics)
4. Optional glasses test

Computes empirical open recall, closed recall, false-alarm rate,
time-to-detect closure, and state flicker rate.
Saves results to reports/live_eye_test_<timestamp>.json and confusion matrix.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

# Ensure project root is available
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.camera import CameraConfig, CameraService
from backend.config import PipelineConfig
from backend.pipeline import EyeRegionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drishtidrive.live_eye_test")


def draw_debug_overlay(
    frame: np.ndarray,
    res: Any,
    phase_name: str,
    phase_instruction: str,
    time_rem: float,
) -> np.ndarray:
    """Render rich diagnostic telemetry overlay with landmark points and metrics."""
    disp = frame.copy()
    h, w = disp.shape[:2]

    # Draw phase banner at top
    banner_color = (40, 40, 40)
    cv2.rectangle(disp, (0, 0), (w, 55), banner_color, -1)
    prompt_txt = f"[{phase_name}] {phase_instruction} ({time_rem:.1f}s remaining)"
    cv2.putText(disp, prompt_txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    # If landmarks are available, draw eye points
    if res.landmarks_result and res.landmarks_result.detected:
        lr = res.landmarks_result
        pt_color = (0, 255, 0) if res.overall_state == "OPEN" else (0, 0, 255)
        for pt in lr.left_eye_points:
            cv2.circle(disp, (int(pt[0]), int(pt[1])), 2, pt_color, -1)
        for pt in lr.right_eye_points:
            cv2.circle(disp, (int(pt[0]), int(pt[1])), 2, pt_color, -1)

        # Telemetry panel at bottom
        cv2.rectangle(disp, (0, h - 70), (w, h), (20, 20, 20), -1)
        diag1 = f"STATE: {res.overall_state} ({res.overall_confidence:.2f}) | EAR: {lr.mean_ear:.3f} | BLINK: {lr.mean_blink:.2f}"
        diag2 = f"REASON: {res.status_message[:65]}"
        cv2.putText(disp, diag1, (15, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(disp, diag2, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return disp


def run_live_test(
    camera_index: int = 0,
    video_source: Optional[str] = None,
    open_sec: float = 10.0,
    close_sec: float = 5.0,
    blink_sec: float = 10.0,
    headless: bool = False,
    assert_thresholds: bool = False,
) -> Dict[str, Any]:
    """Execute guided ground-truth protocol and collect evaluation telemetry."""
    cam_cfg = CameraConfig(camera_index=camera_index, video_source=video_source, target_fps=30)
    pipeline = EyeRegionPipeline(config=PipelineConfig(camera=cam_cfg))
    pipeline.start()

    records: List[Dict[str, Any]] = []
    closure_detection_latency: Optional[float] = None
    first_close_timestamp: Optional[float] = None

    phases = [
        ("OPEN_PHASE", "Keep eyes OPEN normally looking ahead", open_sec, "OPEN"),
        ("CLOSE_PHASE", "CLOSE your eyes gently and hold", close_sec, "CLOSED"),
        ("BLINK_PHASE", "Blink normally and look around", blink_sec, "NATURAL"),
    ]

    t_global_start = time.perf_counter()
    logger.info("Starting live eye ground-truth validation session...")

    try:
        for phase_name, instruction, duration, ground_truth in phases:
            logger.info("Entering phase: %s (%s)", phase_name, instruction)
            p_start = time.perf_counter()

            while (time.perf_counter() - p_start) < duration:
                now = time.time()
                res = pipeline.step()
                frame, _ = pipeline.camera.read_frame()

                elapsed_phase = time.perf_counter() - p_start
                rem = max(0.0, duration - elapsed_phase)

                rec = {
                    "phase": phase_name,
                    "ground_truth": ground_truth,
                    "predicted": res.overall_state,
                    "confidence": res.overall_confidence,
                    "mean_ear": res.landmarks_result.mean_ear if res.landmarks_result else 0.28,
                    "mean_blink": res.landmarks_result.mean_blink if res.landmarks_result else 0.0,
                    "timestamp": round(now, 4),
                    "reason": res.status_message,
                }
                records.append(rec)

                # Track time to first confirmed closure in CLOSE_PHASE
                if phase_name == "CLOSE_PHASE":
                    if first_close_timestamp is None:
                        first_close_timestamp = p_start
                    if res.overall_state == "CLOSED" and closure_detection_latency is None:
                        closure_detection_latency = round(time.perf_counter() - first_close_timestamp, 3)

                if not headless and frame is not None and frame.size > 0:
                    overlay = draw_debug_overlay(frame, res, phase_name, instruction, rem)
                    cv2.imshow("DrishtiDrive AI — Ground Truth Validation", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Early termination requested by user.")
                        break
    finally:
        pipeline.stop()
        if not headless:
            cv2.destroyAllWindows()

    # Compute empirical metrics from logged ground truth records
    open_records = [r for r in records if r["ground_truth"] == "OPEN"]
    closed_records = [r for r in records if r["ground_truth"] == "CLOSED"]

    true_open = sum(1 for r in open_records if r["predicted"] == "OPEN")
    false_closed_in_open = sum(1 for r in open_records if r["predicted"] == "CLOSED")
    open_recall = round((true_open / max(1, len(open_records))) * 100.0, 2)
    false_alarm_rate = round((false_closed_in_open / max(1, len(open_records))) * 100.0, 2)

    true_closed = sum(1 for r in closed_records if r["predicted"] == "CLOSED")
    closed_recall = round((true_closed / max(1, len(closed_records))) * 100.0, 2)

    # Compute state transition flicker rate (transitions per second in steady phases)
    steady_preds = [r["predicted"] for r in open_records] + [r["predicted"] for r in closed_records]
    transitions = sum(1 for i in range(1, len(steady_preds)) if steady_preds[i] != steady_preds[i - 1])
    total_steady_sec = (len(open_records) + len(closed_records)) / 30.0
    flicker_rate = round(transitions / max(0.1, total_steady_sec), 2)

    metrics = {
        "session_timestamp": datetime.datetime.now().isoformat(),
        "total_frames_collected": len(records),
        "open_phase_frames": len(open_records),
        "closed_phase_frames": len(closed_records),
        "open_recall_pct": open_recall,
        "closed_recall_pct": closed_recall,
        "false_alarm_rate_pct": false_alarm_rate,
        "time_to_detect_closure_sec": closure_detection_latency or 0.0,
        "flicker_rate_per_sec": flicker_rate,
        "confusion_matrix": {
            "ground_truth_open": {
                "pred_open": true_open,
                "pred_closed": false_closed_in_open,
                "pred_uncertain": sum(1 for r in open_records if r["predicted"] == "UNCERTAIN"),
            },
            "ground_truth_closed": {
                "pred_closed": true_closed,
                "pred_open": sum(1 for r in closed_records if r["predicted"] == "OPEN"),
                "pred_uncertain": sum(1 for r in closed_records if r["predicted"] == "UNCERTAIN"),
            },
        },
    }

    # Save timestamped report
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"live_eye_test_{date_str}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved ground-truth report to %s", report_file)
    logger.info("SUMMARY: Open Recall: %.1f%% | Closed Recall: %.1f%% | FAR: %.1f%% | Latency: %.2fs",
                open_recall, closed_recall, false_alarm_rate, closure_detection_latency or 0.0)

    if assert_thresholds:
        if open_recall < 95.0 or closed_recall < 95.0:
            logger.error("Recall threshold assertion failed: Open=%.1f%%, Closed=%.1f%% (< 95.0%%)",
                         open_recall, closed_recall)
            sys.exit(1)

    return metrics


def main() -> None:
    """CLI runner for live eye test."""
    parser = argparse.ArgumentParser(description="Live eye detection ground truth validation.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--video", type=str, default=None, help="Optional video file for headless verification.")
    parser.add_argument("--open-sec", type=float, default=10.0, help="Open eye phase duration in seconds.")
    parser.add_argument("--close-sec", type=float, default=5.0, help="Closed eye phase duration in seconds.")
    parser.add_argument("--blink-sec", type=float, default=10.0, help="Natural blink phase duration in seconds.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window.")
    parser.add_argument("--assert-thresholds", action="store_true", help="Exit non-zero if recall < 95%.")
    args = parser.parse_args()

    run_live_test(
        camera_index=args.camera,
        video_source=args.video,
        open_sec=args.open_sec,
        close_sec=args.close_sec,
        blink_sec=args.blink_sec,
        headless=args.headless,
        assert_thresholds=args.assert_thresholds,
    )


if __name__ == "__main__":
    main()
