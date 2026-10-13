# DrishtiDrive AI — Model Card

**Document:** `model_card.md`  
**Version:** 1.2  
**Target:** Eye-State Binary Classification Models (`open` vs `closed`) & Real-Time Pipeline Integration  
**Last Updated:** October 2026  

---

## 1. Model Overview

DrishtiDrive AI evaluates five Convolutional Neural Network (CNN) architectures for real-time eye-state classification on standard desktop and mobile hardware:

1. **Custom Compact CNN:** Baseline 3-stage Conv2D model with Batch Normalization and Dropout, trained from scratch.
2. **MobileNetV2 (Transfer Learning):** Lightweight depthwise separable architecture optimized for low-latency CPU and edge execution.
3. **EfficientNetB0 (Transfer Learning):** Compound scaled CNN offering high accuracy and strong feature representation.
4. **ResNet50 (Transfer Learning):** Deep residual network baseline for evaluating compute-heavy accuracy bounds.
5. **Depthwise-Separable CNN:** Custom edge-oriented lightweight model trained from scratch with minimal parameter count (Winner).

---

## 2. Intended Use & Integration

- **Primary Application:** Binary prediction of driver eye openness on cropped 96×96 RGB patches.
- **Pipeline Integration:** Operates in conjunction with OpenCV/YOLO pose keypoint localization, a time-based temporal state machine, and a three-stage audio-visual alert engine.
- **Out of Scope:** Direct automated vehicle control (steering, braking), legal liability decisions, or certified commercial collision prevention.

---

## 3. Dataset & Licensing Documentation

### 3.1 Primary Training & Held-Out Test Dataset: MRL Eye Dataset
- **Source:** [MRL Eye Dataset — VSB Technical University of Ostrava](https://mrl.cs.vsb.cz/eyedataset)
- **License / Terms:** Free for research and non-commercial educational evaluation with attribution.
- **Partitioning:** Stratified subject-independent splits (70% train, 15% validation, 15% held-out test).
- **Metadata Encoding:** Each sample encodes subject ID, eye state, eyeglasses (0=no, 1=yes), reflections (0=none, 1=small, 2=bad), and lighting (0=bad, 1=good).
- **Test Set Size:** 600 images (300 open, 300 closed).

### 3.2 Evaluation Sequence Dataset: Driver Feed & Roboflow Driver Dataset
- **Driver Feed Video:** 60-frame continuous test driver video sequence (`reports/test_driver_feed.mp4`).
- **Driver Dataset 2:** Roboflow Universe Driver Drowsiness Detection Dataset (Version 1, License: CC BY 4.0).
- **Face/Pose Model:** Ultralytics YOLO11n-Pose (`yolo11n-pose.pt`, AGPL-3.0 License).

---

## 4. Phase 3 Architecture Comparison (Headline: Dataset 2 Real Driving Conditions)

Evaluated on CPU on 600 held-out real in-vehicle driver eye crops (300 open, 300 closed from Dataset 2, reproducible via `reports/model_comparison.csv`):

| Candidate Model | Test Accuracy (%) | Closed Recall (%) | Open Recall (%) | Macro F1 (%) | Avg CPU Latency (ms) | Size (MB) | Parameters | Benchmark Status |
|---|---|---|---|---|---|---|---|---|
| **Custom Compact CNN** | 67.17% | 70.67% | 63.67% | 67.13% | 1.89 ms | 1.19 MB | 102,082 | Verified Run |
| **Depthwise-Separable CNN** | **72.00%** | **69.67%** | **74.33%** | **71.98%** | **1.88 ms** | **0.38 MB** | **29,522** | **Selected Winner** |
| **MobileNetV2** | 71.83% | 73.00% | 70.67% | 71.83% | 9.36 ms | 9.66 MB | 2,305,986 | Verified Run |
| **EfficientNetB0** | 73.83% | 65.67% | 82.00% | 73.66% | 13.02 ms | 16.52 MB | 4,089,662 | Verified Run |
| **ResNet50** | 71.83% | 66.67% | 77.00% | 71.76% | 25.47 ms | 91.48 MB | 23,639,298 | Verified Run |

**Selection Decision:** Depthwise-Separable CNN was selected as the embedded winner due to balanced 72.00% accuracy, 69.67% closed-eye safety recall, ultra-low 1.88ms CPU latency (>530 FPS), and minimal 0.38 MB footprint.

---

## 5. Phase 8 Independent Integration & Safety Evaluation

Measured empirical performance recorded in [`reports/phase8_integration_benchmark.json`](../reports/phase8_integration_benchmark.json):

### 5.1 Temporal Driving Scenarios (FAR & MER)
- **Attentive Driver (Normal Blinking, 12s, 360 frames @ 30 FPS):**
  - False Alarm Rate (FAR): **0.0%** (0 false alarms out of 360 frames)
  - Missed Event Rate (MER): **0.0%**
- **Prolonged Eye Closure / Microsleep (3.2s sustained closure starting at $t=1.00\text{s}$):**
  - False Alarm Rate (FAR): **0.0%**
  - Missed Event Rate (MER): **0.0%** (Stage 1 reached at 0.83s from onset, Stage 2 at 1.87s, Stage 3 at 3.07s; all within 0.07s of configured thresholds)
- **Repeated Micro-Closures (3x 0.9s closures in 20s):**
  - Escalation to Stage 2 Drowsiness Warning reached at 7.67s.
- **Sensor Dropout / Head Occlusion (3.0s UNCERTAIN):**
  - Safe degradation hold maintained; **0 false alarms**.

### 5.2 Multi-Dataset Evaluation Honesty
- **Headline Benchmark [Dataset 2]:** 72.00% Accuracy, 69.67% Closed Recall, 74.33% Open Recall on real driver cockpit images.
- **Secondary Laboratory Benchmark [MRL Eye State]:** Evaluated separately on isolated cropped eye patches. Previous 100% metrics reflect laboratory patches, not automotive cabin conditions. Every metric in benchmarks and reports is explicitly prefixed with its originating dataset tag.

---

## 6. Known Limitations and Failure Modes

1. **Dark Sunglasses / Opaque Tinted Eyewear:** Heavy polarization and opaque dark lenses prevent visible light transmission, rendering eye keypoint tracking unavailable (`UNCERTAIN` state).
2. **Extreme Head Yaw / Pitch:** Yaw angles exceeding 35 degrees obscure the contralateral eye, reducing dual-eye observation to monocular observation.
3. **Severe Night Underexposure:** Standard RGB webcams without active NIR (near-infrared) illuminators experience severe grain and keypoint dropouts in complete darkness.
4. **Single-Frame Isolation Principle:** Eye closure is an indicator of possible drowsiness, not definitive medical proof. Critical alarms require time-based persistence (>1.8s for Stage 2, >3.0s for Stage 3).

---

## 7. Safety Notice

> [!CAUTION]
> **Experimental Prototype Only:**  
> DrishtiDrive AI is not certified under automotive ISO 26262 functional safety standards. It must not be connected to vehicle actuators (steering, brakes, horn, lights) and cannot guarantee drowsiness detection. Drivers must rest whenever fatigued.
