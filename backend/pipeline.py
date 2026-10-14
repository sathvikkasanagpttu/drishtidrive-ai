"""Integrated Camera and Eye-Region Pipeline for DrishtiDrive AI.

Coordinates camera capture, MediaPipe 3D face landmarks, YOLO pose detection,
preprocessing, CNN classification, multi-modal sensor fusion, and temporal alerts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.alert_manager import AlertManager
from backend.alert_visual import VisualAlertNotification
from backend.camera import CameraService
from backend.calibration import DriverCalibrationService
from backend.config import PipelineConfig
from backend.distraction import DistractionDetector, DistractionStatus
from backend.eye_detection import BaseEyeDetector, DetectionStatus, EyeDetectionResult, PoseEyeDetector
from backend.eye_state_fusion import EyeFusionResult, fuse_eye_states
from backend.fusion import FusionResult, MultiSignalFusion
from backend.head_pose import HeadPoseEstimator, HeadPoseResult
from backend.landmarks import EyeLandmarksResult, MediaPipeFaceLandmarker
from backend.model_inference import EyeClassifier, EyeStatePrediction
from backend.preprocessing import EyePreprocessor
from backend.temporal_machine import TemporalDrowsinessDetector
from backend.temporal_state import TemporalStatus

logger = logging.getLogger("drishtidrive.pipeline")


@dataclass
class PipelineFrameResult:
    """Consolidated outputs and diagnostic telemetry for a single processed frame."""

    timestamp: float
    frame_index: int
    overall_state: str  # "OPEN", "CLOSED", or "UNCERTAIN"
    overall_confidence: float
    detection_status: DetectionStatus
    detection_result: EyeDetectionResult
    left_eye_prediction: Optional[EyeStatePrediction] = None
    right_eye_prediction: Optional[EyeStatePrediction] = None
    latency_breakdown: Dict[str, float] = field(default_factory=dict)
    camera_fps: float = 0.0
    consecutive_closed_frames: int = 0
    drowsiness_declared: bool = False
    status_message: str = ""
    temporal_status: Optional[TemporalStatus] = None
    visual_alert: Optional[VisualAlertNotification] = None
    alert_status: Optional[Dict[str, Any]] = None
    head_pose: Optional[HeadPoseResult] = None
    distraction: Optional[DistractionStatus] = None
    fusion: Optional[FusionResult] = None
    landmarks_result: Optional[EyeLandmarksResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame result to serializable dictionary."""
        return {
            "timestamp": round(self.timestamp, 4),
            "frame_index": self.frame_index,
            "overall_state": self.overall_state,
            "overall_confidence": round(self.overall_confidence, 4),
            "detection_status": self.detection_status.value,
            "latency_breakdown": self.latency_breakdown,
            "camera_fps": self.camera_fps,
            "consecutive_closed_frames": self.consecutive_closed_frames,
            "drowsiness_declared": self.drowsiness_declared,
            "status_message": self.status_message,
            "temporal_status": self.temporal_status.to_dict() if self.temporal_status else None,
            "visual_alert": self.visual_alert.to_dict() if self.visual_alert else None,
            "alert_status": self.alert_status,
            "head_pose": asdict(self.head_pose) if self.head_pose else None,
            "distraction": asdict(self.distraction) if self.distraction else None,
            "fusion": asdict(self.fusion) if self.fusion else None,
            "landmarks": {
                "detected": True, "left_ear": self.landmarks_result.left_ear,
                "right_ear": self.landmarks_result.right_ear, "mean_ear": self.landmarks_result.mean_ear,
                "mean_blink": self.landmarks_result.mean_blink,
                "left_eye_points": self.landmarks_result.left_eye_points,
                "right_eye_points": self.landmarks_result.right_eye_points,
            } if self.landmarks_result and self.landmarks_result.detected else None,
        }


class EyeRegionPipeline:
    """End-to-end coordinator for the OpenCV Camera and Eye-Region Pipeline."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        camera: Optional[CameraService] = None,
        detector: Optional[BaseEyeDetector] = None,
        preprocessor: Optional[EyePreprocessor] = None,
        classifier: Optional[EyeClassifier] = None,
        temporal_detector: Optional[TemporalDrowsinessDetector] = None,
        alert_manager: Optional[AlertManager] = None,
        landmarker: Optional[MediaPipeFaceLandmarker] = None,
    ) -> None:
        """Initialize pipeline with modular components and shared configuration."""
        self.config = config or PipelineConfig()
        self.camera = camera or CameraService(self.config.camera)
        self.preprocessor = preprocessor or EyePreprocessor(self.config.preprocessing)
        self.classifier = classifier or EyeClassifier(self.config.inference)
        self.detector = detector or PoseEyeDetector(self.config.detection)
        self.landmarker = landmarker or MediaPipeFaceLandmarker()
        self.temporal_detector = temporal_detector or TemporalDrowsinessDetector(self.config.temporal)
        self.alert_manager = alert_manager or AlertManager(self.config.alert)
        self.head_pose_estimator = HeadPoseEstimator()
        self.distraction_detector = DistractionDetector()
        self.fusion_engine = MultiSignalFusion()
        self.calibration_service = DriverCalibrationService()

        self.frame_index: int = 0
        self.consecutive_closed_frames: int = 0
        self._is_active: bool = False

    def start(self) -> None:
        """Start the camera capture service and pipeline session."""
        self.camera.start()
        self.frame_index = 0
        self.consecutive_closed_frames = 0
        self.temporal_detector.reset()
        self._is_active = True
        logger.info("EyeRegionPipeline started.")

    def stop(self) -> None:
        """Stop camera capture and release resources gracefully."""
        self._is_active = False
        self.camera.stop()
        self.alert_manager.stop()
        logger.info("EyeRegionPipeline stopped.")

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> PipelineFrameResult:
        """Execute full detection, preprocessing, and classification on a single frame."""
        t_start = time.perf_counter()
        now = timestamp if timestamp is not None else time.time()
        self.frame_index += 1
        latency: Dict[str, float] = {}

        # 1. Primary: MediaPipe Face Landmarker; Secondary: Pose Eye Detector
        t0 = time.perf_counter()
        landmark_res = self.landmarker.detect(frame)
        det_res = self.detector.detect(frame)
        latency["detection_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # 2. Handle Degraded Observations Safely
        if not det_res.is_valid and not (landmark_res and landmark_res.detected):
            self.consecutive_closed_frames = 0
            temp_status = self.temporal_detector.step("UNCERTAIN", False, now)
            latency["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
            deg_pose = HeadPoseResult(method="none")
            return PipelineFrameResult(
                timestamp=now, frame_index=self.frame_index, overall_state="UNCERTAIN",
                overall_confidence=0.0, detection_status=det_res.status, detection_result=det_res,
                latency_breakdown=latency, camera_fps=self.camera.current_fps,
                status_message=det_res.message, temporal_status=temp_status,
                visual_alert=self.alert_manager.handle_temporal_status(temp_status),
                alert_status=self.alert_manager.get_status(), head_pose=deg_pose,
                distraction=DistractionStatus(message=det_res.message, event_timestamp=now),
                fusion=self.fusion_engine.update("UNCERTAIN", 0.0, 0.14, deg_pose, now),
                landmarks_result=landmark_res,
            )

        # 3. Preprocess and Classify Eye Regions (prefer landmark-aligned crops if available)
        l_crop = (landmark_res.left_eye_crop if landmark_res and landmark_res.detected and landmark_res.left_eye_crop is not None
                  else det_res.left_eye_crop)
        r_crop = (landmark_res.right_eye_crop if landmark_res and landmark_res.detected and landmark_res.right_eye_crop is not None
                  else det_res.right_eye_crop)

        crops = [c for c in (l_crop, r_crop) if c is not None]
        crop_roles = [r for r, c in (("left", l_crop), ("right", r_crop)) if c is not None]

        t0 = time.perf_counter()
        batch_tensor = self.preprocessor.preprocess_crops_batch(crops)
        latency["preprocessing_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        t0 = time.perf_counter()
        preds = self.classifier.predict_batch(batch_tensor)
        latency["inference_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        left_pred: Optional[EyeStatePrediction] = None
        right_pred: Optional[EyeStatePrediction] = None
        for role, pred in zip(crop_roles, preds):
            if role == "left":
                left_pred = pred
            elif role == "right":
                right_pred = pred

        # 4. Multi-Signal Head Pose Estimation
        h, w = frame.shape[:2] if frame is not None and frame.size > 0 else (480, 640)
        head_pose = self.head_pose_estimator.estimate(det_res.keypoints_xy, det_res.keypoints_conf, (h, w))

        # 5. Synthesize Combined Eye State via Decision Fusion Engine
        state, conf, msg = self._combine_eye_states(
            left_pred, right_pred, getattr(det_res, "ear", None), landmark_res, head_pose.pitch
        )

        if state == "CLOSED":
            self.consecutive_closed_frames += 1
        else:
            self.consecutive_closed_frames = 0

        # 6. Distraction, Multi-Signal Fusion & Calibration Updates
        distraction = self.distraction_detector.update(
            head_pose, det_res.keypoints_xy, det_res.keypoints_conf, frame, timestamp=now
        )
        mar_val = getattr(det_res, "mar", 0.14)
        closed_prob = conf if state == "CLOSED" else (1.0 - conf if state == "OPEN" else 0.5)
        fusion = self.fusion_engine.update(state, closed_prob, mar_val, head_pose, timestamp=now)
        ear_val = landmark_res.mean_ear if (landmark_res and landmark_res.detected) else getattr(det_res, "ear", 0.28)
        blink_val = landmark_res.mean_blink if (landmark_res and landmark_res.detected) else 0.0

        self.calibration_service.update(
            state, conf, head_pose.pitch, head_pose.yaw, head_pose.roll,
            is_blink=(state == "CLOSED" and self.consecutive_closed_frames == 1),
            timestamp=now, mean_ear=ear_val, mean_blink=blink_val,
        )

        # 7. Apply Temporal State Machine as Final Safety Layer
        is_obs_valid = (state != "UNCERTAIN")
        temp_status = self.temporal_detector.step(state, is_obs_valid, now, mar=mar_val, ear=ear_val)
        visual_notif = self.alert_manager.handle_temporal_status(temp_status)
        drowsiness_event = (temp_status.alert_stage.value >= 2)

        latency["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)

        return PipelineFrameResult(
            timestamp=now,
            frame_index=self.frame_index,
            overall_state=state,
            overall_confidence=round(conf, 4),
            detection_status=det_res.status,
            detection_result=det_res,
            left_eye_prediction=left_pred,
            right_eye_prediction=right_pred,
            latency_breakdown=latency,
            camera_fps=self.camera.current_fps,
            consecutive_closed_frames=self.consecutive_closed_frames,
            drowsiness_declared=drowsiness_event,
            status_message=msg,
            temporal_status=temp_status,
            visual_alert=visual_notif,
            alert_status=self.alert_manager.get_status(),
            head_pose=head_pose,
            distraction=distraction,
            fusion=fusion,
            landmarks_result=landmark_res,
        )

    def _combine_eye_states(
        self,
        left: Optional[EyeStatePrediction],
        right: Optional[EyeStatePrediction],
        measured_ear: Optional[float] = None,
        landmark_res: Optional[EyeLandmarksResult] = None,
        head_pitch: float = 0.0,
    ) -> Tuple[str, float, str]:
        """Delegate dual-eye combination to hierarchical multi-modal fusion engine."""
        prof = self.calibration_service.profile
        close_th = prof.ear_close_threshold if prof else 0.20
        open_th = prof.ear_open_threshold if prof else 0.23
        lm = landmark_res if landmark_res and landmark_res.detected else None
        res = fuse_eye_states(
            left_ear=lm.left_ear if lm else None, right_ear=lm.right_ear if lm else None,
            mean_ear=lm.mean_ear if lm else measured_ear,
            blink_left=lm.blink_left if lm else None, blink_right=lm.blink_right if lm else None,
            mean_blink=lm.mean_blink if lm else None,
            left_cnn=left, right_cnn=right, head_pitch=head_pitch,
            ear_close_threshold=close_th, ear_open_threshold=open_th,
            prev_closed=(self.consecutive_closed_frames > 0),
        )
        return res.state, res.confidence, res.reason

    def step(self) -> PipelineFrameResult:
        """Fetch next camera frame and process through pipeline."""
        frame, ts = self.camera.read_frame()
        return self.process_frame(frame, timestamp=ts)

    def __enter__(self) -> "EyeRegionPipeline":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()
