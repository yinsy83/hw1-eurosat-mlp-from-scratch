from __future__ import annotations

import argparse
import itertools
import os
from typing import Dict, List

from data import build_splits, load_eurosat_dataset
from train import train_experiment
from utils import ensure_dir, save_json


def parse_list(text: str, cast_type):
    return [cast_type(piece.strip()) for piece in text.split(",") if piece.strip()]


def parse_hidden_grid(text: str) -> List[List[int]]:
    groups = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        groups.append([int(piece.strip()) for piece in chunk.split(",") if piece.strip()])
    if not groups:
        raise ValueError("hidden-grid must contain at least one architecture.")
    return groups


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid search for EuroSAT MLP hyperparameters.")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lrs", type=str, default="0.05,0.01")
    parser.add_argument("--weight-decays", type=str, default="0.0001,0.001")
    parser.add_argument("--activations", type=str, default="relu,tanh")
    parser.add_argument("--hidden-grid", type=str, default="256,128;512,256")
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5)
    parser.add_argument("--lr-decay-step", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-limit-per-class", type=int, default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
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

    lrs = parse_list(args.lrs, float)
    weight_decays = parse_list(args.weight_decays, float)
    activations = parse_list(args.activations, str)
    hidden_architectures = parse_hidden_grid(args.hidden_grid)
    total_trials = len(lrs) * len(weight_decays) * len(activations) * len(hidden_architectures)

    results = []
    trial_id = 0
    for lr, weight_decay, activation, hidden_dims in itertools.product(
        lrs, weight_decays, activations, hidden_architectures
    ):
        trial_id += 1
        if args.max_trials is not None and trial_id > args.max_trials:
            break
        config: Dict[str, object] = {
            "dataset_root": args.dataset_root,
            "hidden_dims": hidden_dims,
            "activation": activation,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "lr_decay_gamma": args.lr_decay_gamma,
            "lr_decay_step": args.lr_decay_step,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "sample_limit_per_class": args.sample_limit_per_class,
        }
        run_name = f"search_trial{trial_id:02d}_{activation}_h{'-'.join(map(str, hidden_dims))}_lr{lr}_wd{weight_decay}"
        print(f"=== Trial {trial_id} / {total_trials} ===")
        result = train_experiment(
            dataset=dataset,
            splits=splits,
            config=config,
            output_root=args.output_root,
            run_name=run_name,
            save_artifacts=False,
        )
        results.append(
            {
                "trial_id": trial_id,
                "activation": activation,
                "hidden_dims": hidden_dims,
                "lr": lr,
                "weight_decay": weight_decay,
                "best_val_accuracy": result["best_val_accuracy"],
                "best_epoch": result["best_epoch"],
                "checkpoint_path": result["checkpoint_path"],
            }
        )
        ensure_dir(os.path.join(args.output_root, "search"))
        search_path = os.path.join(args.output_root, "search", "grid_search_results.json")
        partial_results = sorted(results, key=lambda item: item["best_val_accuracy"], reverse=True)
        save_json(
            search_path,
            {
                "dataset_root": args.dataset_root,
                "completed_trials": len(results),
                "planned_trials": total_trials if args.max_trials is None else min(total_trials, args.max_trials),
                "results": partial_results,
                "best_result": partial_results[0] if partial_results else None,
            },
        )

    results = sorted(results, key=lambda item: item["best_val_accuracy"], reverse=True)
    ensure_dir(os.path.join(args.output_root, "search"))
    search_path = os.path.join(args.output_root, "search", "grid_search_results.json")
    save_json(
        search_path,
        {
            "dataset_root": args.dataset_root,
            "results": results,
            "best_result": results[0] if results else None,
        },
    )
    if results:
        print("Best configuration:")
        print(results[0])
    print(f"Search results saved to: {search_path}")


if __name__ == "__main__":
    main()
