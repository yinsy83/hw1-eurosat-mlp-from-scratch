from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np

from data import load_eurosat_dataset
from metrics import confusion_matrix
from model import MLPClassifier
from train import evaluate_model
from utils import ensure_dir, load_checkpoint, save_json
from visualize import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    save_error_cases,
    visualize_first_layer_weights,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the best EuroSAT MLP checkpoint on the test split.")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    metadata: Dict[str, object] = checkpoint["metadata"]

    dataset = load_eurosat_dataset(root=args.dataset_root)
    model = MLPClassifier(
        input_dim=int(metadata["input_dim"]),
        hidden_dims=list(metadata["hidden_dims"]),
        num_classes=int(metadata["num_classes"]),
        activation=str(metadata["activation"]),
        seed=int(metadata["seed"]),
    )
    model.load_state_dict(checkpoint["state_dict"])

    metrics = evaluate_model(
        model=model,
        dataset=dataset,
        indices=checkpoint["test_idx"],
        mean=checkpoint["mean"],
        std=checkpoint["std"],
        batch_size=args.batch_size,
    )

    cm = confusion_matrix(
        y_true=metrics["y_true"],
        y_pred=metrics["y_pred"],
        num_classes=len(dataset.class_names),
    )

    ensure_dir(os.path.join(args.output_root, "confusion_matrix"))
    ensure_dir(os.path.join(args.output_root, "weight_viz"))
    ensure_dir(os.path.join(args.output_root, "error_cases"))

    run_name = str(metadata["run_name"])
    cm_path = os.path.join(args.output_root, "confusion_matrix", f"{run_name}_test_cm.png")
    per_class_path = os.path.join(args.output_root, "confusion_matrix", f"{run_name}_per_class_acc.png")
    weights_path = os.path.join(args.output_root, "weight_viz", f"{run_name}_first_layer_weights.png")
    errors_path = os.path.join(args.output_root, "error_cases", f"{run_name}_error_cases.png")
    summary_path = os.path.join(args.output_root, "confusion_matrix", f"{run_name}_test_summary.json")

    plot_confusion_matrix(cm, dataset.class_names, cm_path)
    plot_per_class_accuracy(cm, dataset.class_names, per_class_path)
    visualize_first_layer_weights(
        weight_matrix=model.layers[0].weight.data,
        image_shape=tuple(metadata["image_shape"]),
        output_path=weights_path,
    )

    test_indices = checkpoint["test_idx"]
    save_error_cases(
        x_uint8=dataset.x[test_indices],
        file_paths=[dataset.file_paths[int(idx)] for idx in test_indices],
        y_true=metrics["y_true"],
        y_pred=metrics["y_pred"],
        class_names=dataset.class_names,
        image_shape=tuple(metadata["image_shape"]),
        output_path=errors_path,
    )

    summary = {
        "run_name": run_name,
        "test_accuracy": float(metrics["accuracy"]),
        "test_loss": float(metrics["loss"]),
        "checkpoint": args.checkpoint,
        "confusion_matrix": cm.tolist(),
        "class_names": dataset.class_names,
        "artifacts": {
            "confusion_matrix": cm_path,
            "per_class_accuracy": per_class_path,
            "first_layer_weights": weights_path,
            "error_cases": errors_path,
        },
    }
    save_json(summary_path, summary)

    print(f"Test accuracy: {summary['test_accuracy']:.4f}")
    print("Class order:")
    print(", ".join(dataset.class_names))
    print("Confusion matrix:")
    print(np.array2string(cm, separator=", "))
    print(f"Confusion matrix figure: {cm_path}")
    print(f"Weight visualization: {weights_path}")
    print(f"Error analysis figure: {errors_path}")


if __name__ == "__main__":
    main()
