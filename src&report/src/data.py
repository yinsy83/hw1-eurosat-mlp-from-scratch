from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class DatasetBundle:
    x: np.ndarray
    y: np.ndarray
    class_names: List[str]
    class_to_idx: Dict[str, int]
    image_shape: Tuple[int, int, int]
    file_paths: List[str]


@dataclass
class SplitBundle:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def load_eurosat_dataset(root: str, sample_limit_per_class: Optional[int] = None) -> DatasetBundle:
    class_names = sorted(
        directory for directory in os.listdir(root) if os.path.isdir(os.path.join(root, directory))
    )
    if not class_names:
        raise FileNotFoundError(f"No class folders found in {root}.")

    images = []
    labels = []
    file_paths: List[str] = []
    image_shape: Optional[Tuple[int, int, int]] = None
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(root, class_name)
        filenames = sorted(
            name for name in os.listdir(class_dir) if name.lower().endswith(IMAGE_EXTENSIONS)
        )
        if sample_limit_per_class is not None:
            filenames = filenames[:sample_limit_per_class]

        for filename in filenames:
            path = os.path.join(class_dir, filename)
            image = Image.open(path).convert("RGB")
            array = np.asarray(image, dtype=np.uint8)
            if image_shape is None:
                image_shape = array.shape
            elif array.shape != image_shape:
                raise ValueError(f"Inconsistent image shape: {path} has {array.shape}, expected {image_shape}.")

            images.append(array.reshape(-1))
            labels.append(class_to_idx[class_name])
            file_paths.append(path)

    return DatasetBundle(
        x=np.stack(images, axis=0),
        y=np.asarray(labels, dtype=np.int64),
        class_names=class_names,
        class_to_idx=class_to_idx,
        image_shape=image_shape if image_shape is not None else (64, 64, 3),
        file_paths=file_paths,
    )


def stratified_split_indices(
    y: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1.")

    rng = np.random.default_rng(seed)
    train_indices: List[np.ndarray] = []
    val_indices: List[np.ndarray] = []
    test_indices: List[np.ndarray] = []

    for label in np.unique(y):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)

        n_total = len(indices)
        n_test = int(round(n_total * test_ratio))
        n_val = int(round(n_total * val_ratio))
        n_train = n_total - n_val - n_test

        if n_train <= 0:
            raise ValueError(f"Split ratios leave no training samples for class {label}.")

        train_indices.append(indices[:n_train])
        val_indices.append(indices[n_train : n_train + n_val])
        test_indices.append(indices[n_train + n_val :])

    train_idx = np.concatenate(train_indices)
    val_idx = np.concatenate(val_indices)
    test_idx = np.concatenate(test_indices)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def compute_normalization(x_train_uint8: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    x_train = x_train_uint8.astype(np.float32) / 255.0
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def build_splits(
    dataset: DatasetBundle,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitBundle:
    train_idx, val_idx, test_idx = stratified_split_indices(
        dataset.y, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    mean, std = compute_normalization(dataset.x[train_idx])
    return SplitBundle(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, mean=mean, std=std)


def normalize_batch(x_uint8: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = x_uint8.astype(np.float32) / 255.0
    return (x - mean) / std


def make_batches(
    x_uint8: np.ndarray,
    y: np.ndarray,
    indices: Sequence[int],
    batch_size: int,
    mean: np.ndarray,
    std: np.ndarray,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    indices = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        x_batch = normalize_batch(x_uint8[batch_idx], mean=mean, std=std)
        y_batch = y[batch_idx]
        yield x_batch, y_batch, batch_idx
