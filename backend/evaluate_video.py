"""Video clip evaluation entrypoint wrapper for DrishtiDrive AI.

Delegates to scripts/evaluate_video_clips.py for evaluating video feeds.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_video_clips import main

if __name__ == "__main__":
    main()
