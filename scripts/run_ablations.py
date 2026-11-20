"""Reproduce the six diagnostic ablations of TF-VSM (paper Section 4.6).

Examples:
  # Ablations 1-3 and 4 (readout variants + schedule length):
  python scripts/run_ablations.py --run main outputs/tf_vsm

  # Ablations 5 (data scale) and 6 (loss balance alpha): repeat training with
  # --num-training-samples / --alpha and pass each run directory here:
  python scripts/run_ablations.py --run half outputs/tf_vsm_half --run a1 outputs/tf_vsm_a1

Only the selection split is used here; the held-out test split stays untouched.
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

from mtg.utils.correspondence import TF_VSM_ABLATION_KEYS  # noqa: E402
from mtg.utils.tf_vsm_runtime import (  # noqa: E402
    build_transform,
    evaluate_split,
    load_benchmark_samples,
    load_model,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/train_tf_vsm_ablations.yaml")
    parser.add_argument("--benchmark-config", default="configs/evaluation/controlled_benchmark.yaml")
    parser.add_argument("--benchmark-split-json", default=None)
    parser.add_argument("--split", choices=["selection", "test"], default="selection")
    parser.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "CKPT_DIR"),
                        help="Repeat for each training run (data scale / alpha sweeps)")
    parser.add_argument("--checkpoint-dir", default=None, help="Shortcut for --run main DIR")
    parser.add_argument("--checkpoint-glob", default="epoch_*.pt")
    parser.add_argument("--output-dir", default="outputs/ablations")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    runs = list(args.run or [])
    if args.checkpoint_dir:
        runs.append(("main", args.checkpoint_dir))
    if not runs:
        raise SystemExit("Pass at least one --run LABEL CKPT_DIR (or --checkpoint-dir).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_cfg = load_config(args.benchmark_config)
    splits = load_benchmark_samples(benchmark_cfg, args.benchmark_split_json)
    samples = splits[args.split]
    transform = build_transform(benchmark_cfg.img_size)

    rows = []
    for label, checkpoint_dir in runs:
        checkpoints = sorted(Path(checkpoint_dir).glob(args.checkpoint_glob))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints matching {args.checkpoint_glob} in {checkpoint_dir}")
        for checkpoint in checkpoints:
            model, cfg, device, dtype = load_model(args.config, checkpoint, args.device)
            result = evaluate_split(model, samples, transform, device, dtype)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for metric in TF_VSM_ABLATION_KEYS:
                corr = result["correlations"].get(metric, {})
                rows.append(
                    {
                        "run": label,
                        "epoch": int(str(checkpoint.stem).split("_")[-1]) if checkpoint.stem.startswith("epoch") else -1,
                        "checkpoint": str(checkpoint),
                        "metric": metric,
                        "pearson_r": corr.get("pearson", float("nan")),
                        "spearman_rho": corr.get("spearman", float("nan")),
                    }
                )
            write_json(output_dir / f"{label}_{args.split}_raw.json", result)

    with open(output_dir / "ablation_table.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["run", "epoch", "checkpoint", "metric", "pearson_r", "spearman_rho"]
        )
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "ablation_results.json", rows)
    print(f"Wrote {len(rows)} ablation rows to {output_dir / 'ablation_table.csv'}")


if __name__ == "__main__":
    main()
