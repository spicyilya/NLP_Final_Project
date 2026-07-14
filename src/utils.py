"""Shared constants and helpers for the MELD emotion-classification project.

Imported by every notebook so that the seed, label map, and metric definitions
cannot drift between data preparation, training, and evaluation.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

SEED = 42

# Repo root, resolved from this file so notebooks work regardless of their cwd.
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
CHECKPOINTS = ROOT / "checkpoints"

# Ordered by descending frequency in the MELD train split, so label id 0 is the
# majority class. Fixed here once; notebooks must never rebuild this from a
# split's own value_counts (that would reorder ids per split).
LABEL_NAMES = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}
NUM_LABELS = len(LABEL_NAMES)

MODELS = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
}

MAX_LENGTH = 128  # justified by the token-length distribution in notebook 02


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_metrics(eval_pred):
    """Trainer-compatible metrics: weighted F1 (primary), macro F1, accuracy.

    Weighted F1 is the primary metric because MELD is heavily imbalanced
    (neutral ~47%); macro F1 is reported alongside to expose rare-class
    performance that weighted F1 hides.
    """
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(axis=-1)
    return {
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }


def class_weights_path() -> Path:
    return RESULTS / "class_weights.json"


def load_class_weights() -> np.ndarray:
    """Inverse-frequency class weights written by notebook 02, ordered by label id."""
    with open(class_weights_path(), encoding="utf-8") as fh:
        payload = json.load(fh)
    return np.array([payload["weights"][name] for name in LABEL_NAMES], dtype=np.float32)


def ensure_dirs() -> None:
    for d in (DATA_PROCESSED, RESULTS, FIGURES, CHECKPOINTS):
        d.mkdir(parents=True, exist_ok=True)
