from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from autograd import Tensor
from data import DatasetBundle, SplitBundle, build_splits, load_eurosat_dataset, make_batches
from losses import cross_entropy_with_logits, l2_regularization
from metrics import accuracy_score
from model import MLPClassifier
from optim import SGD
from utils import ensure_dir, save_checkpoint, save_json, set_seed, timestamp
from visualize import plot_training_curves


def parse_hidden_dims(text: str) -> List[int]:
    dims = [int(piece.strip()) for piece in text.split(",") if piece.strip()]
    if not dims:
        raise ValueError("hidden_dims must contain at least one integer.")
    return dims


def make_run_name(config: Dict[str, object]) -> str:
    dims = "-".join(str(dim) for dim in config["hidden_dims"])
    return (
        f"mlp_{config['activation']}_h{dims}_lr{config['lr']}_wd{config['weight_decay']}"
        f"_bs{config['batch_size']}_{timestamp()}"
    )


def evaluate_model(
    model: MLPClassifier,
    dataset: DatasetBundle,
    indices: Sequence[int],
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
) -> Dict[str, np.ndarray | float]:
    total_loss = 0.0
    total_samples = 0
    predictions = []
    targets = []

    for x_batch, y_batch, _ in make_batches(
        dataset.x,
        dataset.y,
        indices=indices,
        batch_size=batch_size,
        mean=mean,
        std=std,
        shuffle=False,
    ):
        logits = model(Tensor(x_batch, requires_grad=False))
        loss = cross_entropy_with_logits(logits, y_batch)
        batch_size_now = len(y_batch)
        total_loss += float(loss.data) * batch_size_now
        total_samples += batch_size_now
        predictions.append(np.argmax(logits.data, axis=1))
        targets.append(y_batch)

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_experiment(
    dataset: DatasetBundle,
    splits: SplitBundle,
    config: Dict[str, object],
    output_root: str,
    run_name: Optional[str] = None,
    save_artifacts: bool = True,
) -> Dict[str, object]:
    set_seed(int(config["seed"]))
    ensure_dir(output_root)

    model = MLPClassifier(
        input_dim=dataset.x.shape[1],
        hidden_dims=config["hidden_dims"],
        num_classes=len(dataset.class_names),
        activation=str(config["activation"]),
        seed=int(config["seed"]),
    )
    optimizer = SGD(model.parameters(), lr=float(config["lr"]))

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_accuracy": [], "lr": []}
    best_val_accuracy = -1.0
    best_epoch = -1

    if run_name is None:
        run_name = make_run_name(config)

    checkpoint_path = os.path.join(output_root, "checkpoints", f"{run_name}_best.npz")
    curves_path = os.path.join(output_root, "curves", f"{run_name}_curves.png")
    history_path = os.path.join(output_root, "curves", f"{run_name}_history.json")
    ensure_dir(os.path.dirname(checkpoint_path))
    ensure_dir(os.path.dirname(curves_path))

    for epoch in range(1, int(config["epochs"]) + 1):
        lr = float(config["lr"]) * (float(config["lr_decay_gamma"]) ** ((epoch - 1) // int(config["lr_decay_step"])))
        optimizer.lr = lr
        history["lr"].append(lr)

        total_loss = 0.0
        total_samples = 0

        for x_batch, y_batch, _ in make_batches(
            dataset.x,
            dataset.y,
            indices=splits.train_idx,
            batch_size=int(config["batch_size"]),
            mean=splits.mean,
            std=splits.std,
            shuffle=True,
            seed=int(config["seed"]) + epoch,
        ):
            model.zero_grad()
            logits = model(Tensor(x_batch, requires_grad=False))
            loss = cross_entropy_with_logits(logits, y_batch)
            if float(config["weight_decay"]) > 0:
                loss = loss + l2_regularization(model.parameters(), float(config["weight_decay"]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_now = len(y_batch)
            total_loss += float(loss.data) * batch_size_now
            total_samples += batch_size_now

        train_loss = total_loss / max(total_samples, 1)
        val_metrics = evaluate_model(
            model=model,
            dataset=dataset,
            indices=splits.val_idx,
            mean=splits.mean,
            std=splits.std,
            batch_size=int(config["batch_size"]),
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))

        if float(val_metrics["accuracy"]) > best_val_accuracy:
            best_val_accuracy = float(val_metrics["accuracy"])
            best_epoch = epoch
            metadata = {
                "dataset_root": config["dataset_root"],
                "class_names": dataset.class_names,
                "image_shape": list(dataset.image_shape),
                "input_dim": int(dataset.x.shape[1]),
                "num_classes": len(dataset.class_names),
                "hidden_dims": list(config["hidden_dims"]),
                "activation": str(config["activation"]),
                "seed": int(config["seed"]),
                "val_ratio": float(config["val_ratio"]),
                "test_ratio": float(config["test_ratio"]),
                "batch_size": int(config["batch_size"]),
                "epochs": int(config["epochs"]),
                "lr": float(config["lr"]),
                "lr_decay_gamma": float(config["lr_decay_gamma"]),
                "lr_decay_step": int(config["lr_decay_step"]),
                "weight_decay": float(config["weight_decay"]),
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_accuracy,
                "run_name": run_name,
            }
            save_checkpoint(
                checkpoint_path,
                model=model,
                metadata=metadata,
                split_payload={
                    "mean": splits.mean,
                    "std": splits.std,
                    "train_idx": splits.train_idx,
                    "val_idx": splits.val_idx,
                    "test_idx": splits.test_idx,
                },
            )

        print(
            f"[Epoch {epoch:03d}] lr={lr:.5f} "
            f"train_loss={train_loss:.4f} val_loss={float(val_metrics['loss']):.4f} "
            f"val_acc={float(val_metrics['accuracy']):.4f}"
        )

    if save_artifacts:
        plot_training_curves(history, curves_path)
        save_json(
            history_path,
            {
                "run_name": run_name,
                "history": history,
                "best_val_accuracy": best_val_accuracy,
                "best_epoch": best_epoch,
                "config": {key: (list(value) if isinstance(value, list) else value) for key, value in config.items()},
            },
        )

    return {
        "run_name": run_name,
        "checkpoint_path": checkpoint_path,
        "curves_path": curves_path,
        "history_path": history_path,
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a custom MLP on EuroSAT_RGB.")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--hidden-dims", type=str, default="256,128")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5)
    parser.add_argument("--lr-decay-step", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-limit-per-class", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config: Dict[str, object] = vars(args)
    config["hidden_dims"] = parse_hidden_dims(args.hidden_dims)

    dataset = load_eurosat_dataset(
        root=args.dataset_root,
        sample_limit_per_class=args.sample_limit_per_class,
    )
    splits = build_splits(
        dataset=dataset,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    result = train_experiment(
        dataset=dataset,
        splits=splits,
        config=config,
        output_root=args.output_root,
        run_name=args.run_name,
        save_artifacts=True,
    )
    print(f"Best validation accuracy: {result['best_val_accuracy']:.4f}")
    print(f"Best checkpoint saved to: {result['checkpoint_path']}")


if __name__ == "__main__":
    main()
