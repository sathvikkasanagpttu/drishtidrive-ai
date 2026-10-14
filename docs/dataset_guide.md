# External Dataset Acquisition & Multi-Modal Evaluation Guide

DrishtiDrive AI supports evaluation and training across four prominent public driver state benchmarks without bundling large video datasets in the repository.

---

## 1. Supported Public Benchmarks

| Dataset | Primary Focus | Access & Citation |
|---|---|---|
| **NTHU-DDD** | Drowsiness under diverse conditions (normal, glasses, sunglasses, night) | [NTHU Computer Vision Lab](http://cv.cs.nthu.edu.tw/php/callforpaper/category.php?id=4) |
| **UTA-RLDD** | Real-life drowsiness with natural fatigue progression (60 subjects) | [UTA RLDD GitHub](https://github.com/haberdar/uta-rldd) |
| **YawDD** | Yawning & mouth aspect ratio under in-vehicle lighting | [University of Ottawa](https://www.site.uottawa.ca/~shervin/yawning/) |
| **DMD** (Driver Monitoring Dataset) | Distraction (phone, reaching, gazing away) and drowsiness | [DMD Dataset / Vicomtech](https://dmd.vicomtech.org/) |

---

## 2. Directory Layout Setup

Extract downloaded datasets under the `data/external/` directory following this layout:

```text
data/external/
├── nthu_ddd/
│   ├── subject_01/
│   │   ├── normal/
│   │   │   ├── open/
│   │   │   └── closed/
│   │   ├── glasses/
│   │   ├── sunglasses/
│   │   └── night/
│   └── subject_02/
├── uta_rldd/
│   ├── subject_01/
│   └── subject_02/
└── yawdd/
    ├── female/
    └── male/
```

---

## 3. Subject-Stratified Training & Evaluation

To avoid data leakage across train, validation, and test sets, `backend/training/train_multimodal.py` enforces **subject-level partitioning** (subjects in the training set never appear in test or validation sets).

### Running Evaluation

```bash
python -m backend.training.train_multimodal \
  --data-dir data/external/nthu_ddd \
  --report reports/multimodal_training_report.json
```

### Outputs Generated

- **Per-condition metrics**: Accuracy, Closed-eye Recall, Open-eye Recall, and Macro F1 across conditions (Normal, Glasses, Night, Glare).
- **Confusion matrices**: Saved in `reports/multimodal_training_report.json`.
- **Reproducibility**: Run without network access once the dataset folder is placed locally.

---

## 4. True Eye-Level Ground Truth Datasets (MRL & CEW)

To train eye classifiers on real eyelid closure rather than noisy image-level "drowsy/awake" tags:

### A. MRL Eye Dataset
Contains 84,898 infrared and visible eye images across 37 subjects with explicit per-eye open/closed labels, glasses annotations, and subject IDs:
- **Source**: [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset)
- **Prepare**:
  ```bash
  python -m backend.training.prepare_true_eye_dataset \
    --mrl-dir data/external/mrl_raw \
    --output-dir data/true_eyes
  ```

### B. Closed Eyes in the Wild (CEW)
- Contains 4,846 real in-the-wild faces (2,423 open, 2,423 closed) under challenging lighting and occlusions.

### C. Auto-labeling from Driver Feeds
Extracts landmark-bounded eye crops auto-labeled by physical EAR ($EAR \ge 0.28 \to \text{open}$, $EAR \le 0.15 \to \text{closed}$, skipping ambiguous transition frames):
```bash
python -m backend.training.prepare_true_eye_dataset \
  --video reports/test_driver_feed.mp4 \
  --output-dir data/true_eyes
```

---

## 5. MediaPipe Face Landmarker Asset Setup

The MediaPipe Tasks API requires the official Google `face_landmarker.task` bundle (3.6 MB):
```bash
curl -fL -o backend/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```
Do **not** commit large `.task` binary weights to git; `*.task` is ignored by `.gitignore`.

