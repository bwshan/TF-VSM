"""Evaluate TF-VSM checkpoints on the controlled / held-out benchmark.

The paper's leakage-free protocol is implemented by two steps:
  1) `--split selection` over epoch checkpoints picks the peak epoch;
  2) `--split test --checkpoint <selected>` reports the 80-pair held-out score.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtg.utils.tf_vsm_runtime import (  # noqa: E402
    build_transform,
    correlation_rows,
    evaluate_split,
    load_benchmark_samples,
    load_model,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/train_tf_vsm.yaml")
    parser.add_argument("--benchmark-config", default="configs/evaluation/controlled_benchmark.yaml")
    parser.add_argument("--benchmark-split-json", default=None)
    parser.add_argument("--checkpoint", nargs="+", required=True, help="One or more checkpoint files")
    parser.add_argument("--split", choices=["selection", "test"], default="test")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--select-best", action="store_true", help="Pick checkpoint by selection-split Pearson r")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_cfg = load_config(args.benchmark_config)
    splits = load_benchmark_samples(benchmark_cfg, args.benchmark_split_json)
    samples = splits[args.split]
    transform = build_transform(benchmark_cfg.img_size)

    results = {}
    for checkpoint in args.checkpoint:
        print(f"Loading {checkpoint}")
        model, cfg, device, dtype = load_model(args.config, checkpoint, args.device)
        results[checkpoint] = evaluate_split(model, samples, transform, device, dtype)
        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.select_best:
        def score(item):
            return item[1]["correlations"].get("tf_vsm", {}).get("pearson", float("-inf"))

        best_checkpoint = max(results.items(), key=score)[0]
        write_json(
            output_dir / "selected_checkpoint.json",
            {"checkpoint": best_checkpoint, "split": args.split, "metric": "tf_vsm", "score": score((best_checkpoint, results[best_checkpoint]))},
        )
        print(f"Selected checkpoint: {best_checkpoint}")

    table = []
    for checkpoint, result in results.items():
        for row in correlation_rows(result):
            table.append({"checkpoint": checkpoint, "split": args.split, **row})
    with open(output_dir / f"{args.split}_correlations.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["checkpoint", "split", "metric", "pearson_r", "spearman_rho"])
        writer.writeheader()
        writer.writerows(table)

    write_json(output_dir / f"{args.split}_results.json", results)
    for row in table:
        print(
            f"{Path(row['checkpoint']).name:16s} {row['metric']:28s} "
            f"r={row['pearson_r']:.3f} rho={row['spearman_rho']:.3f}"
        )


if __name__ == "__main__":
    main()
