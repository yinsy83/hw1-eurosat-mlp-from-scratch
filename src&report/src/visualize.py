from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from metrics import per_class_accuracy


def plot_training_curves(history: Dict[str, List[float]], output_path: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_accuracy"], color="tab:green", label="Val Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_accuracy(cm: np.ndarray, class_names: Sequence[str], output_path: str) -> None:
    scores = per_class_accuracy(cm)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(len(class_names)), scores, color="tab:blue")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def visualize_first_layer_weights(
    weight_matrix: np.ndarray,
    image_shape: Tuple[int, int, int],
    output_path: str,
    max_units: int = 25,
) -> None:
    num_units = min(weight_matrix.shape[1], max_units)
    grid_size = math.ceil(math.sqrt(num_units))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = np.atleast_1d(axes).reshape(grid_size, grid_size)

    for idx in range(grid_size * grid_size):
        ax = axes.flat[idx]
        ax.axis("off")
        if idx >= num_units:
            continue
        weight = weight_matrix[:, idx].reshape(image_shape)
        weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)
        ax.imshow(weight)
        ax.set_title(f"Unit {idx}", fontsize=8)

    fig.suptitle("First Layer Weights", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_error_cases(
    x_uint8: np.ndarray,
    file_paths: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    image_shape: Tuple[int, int, int],
    output_path: str,
    max_cases: int = 12,
) -> None:
    error_indices = np.where(y_true != y_pred)[0][:max_cases]
    if len(error_indices) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No misclassified samples found.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    cols = 3
    rows = math.ceil(len(error_indices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for plot_idx in range(rows * cols):
        ax = axes.flat[plot_idx]
        ax.axis("off")
        if plot_idx >= len(error_indices):
            continue
        idx = error_indices[plot_idx]
        image = x_uint8[idx].reshape(image_shape)
        ax.imshow(image)
        title = f"T:{class_names[int(y_true[idx])]}\nP:{class_names[int(y_pred[idx])]}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(file_paths[idx].split("\\")[-1], fontsize=8)

    fig.suptitle("Misclassified Test Samples", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
