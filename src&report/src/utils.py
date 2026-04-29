from __future__ import annotations

import json
import os
import random
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from model import MLPClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_checkpoint(
    path: str,
    model: MLPClassifier,
    metadata: Dict[str, Any],
    split_payload: Dict[str, np.ndarray],
) -> None:
    arrays = model.state_dict()
    arrays["mean"] = split_payload["mean"].astype(np.float32)
    arrays["std"] = split_payload["std"].astype(np.float32)
    arrays["train_idx"] = split_payload["train_idx"].astype(np.int64)
    arrays["val_idx"] = split_payload["val_idx"].astype(np.int64)
    arrays["test_idx"] = split_payload["test_idx"].astype(np.int64)
    arrays["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(path, **arrays)


def load_checkpoint(path: str) -> Dict[str, Any]:
    checkpoint = np.load(path, allow_pickle=False)
    metadata = json.loads(str(checkpoint["metadata_json"]))
    state = {key: checkpoint[key] for key in checkpoint.files if key.startswith("layer")}
    return {
        "metadata": metadata,
        "state_dict": state,
        "mean": checkpoint["mean"],
        "std": checkpoint["std"],
        "train_idx": checkpoint["train_idx"],
        "val_idx": checkpoint["val_idx"],
        "test_idx": checkpoint["test_idx"],
    }
