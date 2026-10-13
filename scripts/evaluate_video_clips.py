"""Evaluation runner for real-world driver video clips in DrishtiDrive AI.

Evaluates end-to-end pipeline on real video clips, reporting per-clip metrics:
- Processed frame count and measured FPS
- Latency breakdown (average and 95th percentile)
- Driver eye state distribution (OPEN, CLOSED, UNCERTAIN)
- Detection quality (face and eye localization success rate)
- Alert escalation events dispatched (Stage 1, Stage 2, Stage 3)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure repository root is on path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

from backend.alert_manager import AlertManager
from backend.audio_backend import MockAudioBackend
from backend.camera import CameraConfig, CameraService
from backend.config import PipelineConfig
from backend.pipeline import EyeRegionPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("drishtidrive.video_eval")


def evaluate_single_clip(
    video_path: Path,
    max_frames: int = 150,
    config: PipelineConfig | None = None,
) -> Dict[str, Any]:
    """Execute full pipeline across a single video clip and aggregate metrics."""
    logger.info("Evaluating clip: %s (max_frames=%d)", video_path, max_frames)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cam_cfg = CameraConfig(video_source=str(video_path), target_fps=30)
    camera = CameraService(cam_cfg)
    mock_audio = MockAudioBackend()
    alert_mgr = AlertManager(audio_backend=mock_audio)
    cfg = config or PipelineConfig()
    pipeline = EyeRegionPipeline(config=cfg, camera=camera, alert_manager=alert_mgr)

    pipeline.start()
    latencies: List[float] = []
    detection_latencies: List[float] = []
    inference_latencies: List[float] = []
    states: Dict[str, int] = {"OPEN": 0, "CLOSED": 0, "UNCERTAIN": 0}
    successful_detections: int = 0
    t_start = time.perf_counter()
    processed_count = 0

    try:
        for _ in range(max_frames):
            res = pipeline.step()
            processed_count += 1
            tot_ms = res.latency_breakdown.get("total_ms", 0.0)
            det_ms = res.latency_breakdown.get("detection_ms", 0.0)
            inf_ms = res.latency_breakdown.get("inference_ms", 0.0)

            latencies.append(tot_ms)
            if det_ms > 0:
                detection_latencies.append(det_ms)
            if inf_ms > 0:
                inference_latencies.append(inf_ms)

            states[res.overall_state] = states.get(res.overall_state, 0) + 1
            if res.detection_result.is_valid:
                successful_detections += 1
    finally:
        pipeline.stop()

    total_time = time.perf_counter() - t_start
    measured_fps = processed_count / max(0.001, total_time)
    det_success_rate = (successful_detections / max(1, processed_count)) * 100.0

    return {
        "clip_name": video_path.name,
        "clip_path": str(video_path.relative_to(repo_root) if video_path.is_relative_to(repo_root) else video_path),
        "frames_processed": processed_count,
        "wall_clock_sec": round(total_time, 2),
        "measured_fps": round(measured_fps, 1),
        "avg_total_latency_ms": round(float(np.mean(latencies)) if latencies else 0.0, 2),
        "p95_total_latency_ms": round(float(np.percentile(latencies, 95)) if latencies else 0.0, 2),
        "avg_detection_latency_ms": round(float(np.mean(detection_latencies)) if detection_latencies else 0.0, 2),
        "avg_inference_latency_ms": round(float(np.mean(inference_latencies)) if inference_latencies else 0.0, 2),
        "detection_success_rate_pct": round(det_success_rate, 1),
        "state_distribution": states,
        "uncertain_percentage": round((states.get("UNCERTAIN", 0) / max(1, processed_count)) * 100.0, 1),
        "alerts_dispatched": alert_mgr.total_alerts_emitted,
    }


def run_video_evaluation_suite(
    video_paths: List[Path],
    output_report_path: Path = Path("reports/video_clip_benchmark.json"),
    max_frames: int = 150,
) -> Dict[str, Any]:
    """Evaluate a batch of real driver clips and compile evaluation report."""
    results: List[Dict[str, Any]] = []

    for v_path in video_paths:
        try:
            clip_res = evaluate_single_clip(v_path, max_frames=max_frames)
            results.append(clip_res)
        except Exception as err:
            logger.error("Failed to evaluate %s: %s", v_path, err)
            results.append({
                "clip_name": v_path.name,
                "clip_path": str(v_path),
                "error": str(err),
            })

    report = {
        "benchmark_timestamp": time.time(),
        "total_clips_evaluated": len(results),
        "clips": results,
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(json.dumps(report, indent=2))
    logger.info("Video benchmark report saved to %s", output_report_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("REAL VIDEO EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Clip Name':<28} | {'Frames':<6} | {'FPS':<6} | {'Det %':<6} | {'Uncertain %':<11} | {'Avg Lat (ms)':<12}")
    print("-" * 80)
    for c in results:
        if "error" in c:
            print(f"{c['clip_name']:<28} | ERROR: {c['error']}")
        else:
            print(
                f"{c['clip_name']:<28} | {c['frames_processed']:<6} | {c['measured_fps']:<6} | "
                f"{c['detection_success_rate_pct']:<6} | {c['uncertain_percentage']:<11} | "
                f"{c['avg_total_latency_ms']:<12}"
            )
    print("=" * 80 + "\n")
    return report


def main() -> None:
    """CLI entry point for video clip evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate DrishtiDrive AI on real video clips")
    parser.add_argument("--video", action="append", help="Specific video path to evaluate (can repeat)")
    parser.add_argument("--dir", default="reports", help="Directory to scan for video clips (.mp4, .avi)")
    parser.add_argument("--max-frames", type=int, default=60, help="Max frames per clip to evaluate")
    parser.add_argument("--output", default="reports/video_clip_benchmark.json", help="Output JSON path")
    args = parser.parse_args()

    videos: List[Path] = []
    if args.video:
        videos = [Path(v) for v in args.video]
    else:
        v_dir = Path(args.dir)
        if v_dir.exists():
            videos = sorted([p for p in v_dir.glob("*.*") if p.suffix.lower() in {".mp4", ".avi", ".mkv"}])

    if not videos:
        logger.warning("No video files found to evaluate.")
        sys.exit(1)

    run_video_evaluation_suite(
        videos,
        output_report_path=Path(args.output),
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
