"""Evaluate the global-embedding baselines reported in the TF-VSM paper.

CLIP-I and DINOv2 are fully reproducible from public checkpoints; VLM /
RefVNLI protocols require external APIs and are not bundled here.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtg.utils.global_baselines import clip_i_similarity, dinov2_similarity  # noqa: E402
from mtg.utils.metrics import _spearman_coeff  # noqa: E402
from mtg.utils.tf_vsm_runtime import (  # noqa: E402
    _open_image,
    _open_mask,
    load_benchmark_samples,
    load_config,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", default="configs/evaluation/controlled_benchmark.yaml")
    parser.add_argument("--benchmark-split-json", default=None)
    parser.add_argument("--split", choices=["selection", "test"], default="test")
    parser.add_argument("--baselines", nargs="+", default=["clip_i", "dino_v2"], choices=["clip_i", "dino_v2"])
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_cfg = load_config(args.benchmark_config)
    splits = load_benchmark_samples(benchmark_cfg, args.benchmark_split_json)
    samples = splits[args.split]
    pairs = [
        {
            "image_1": _open_image(sample["image_1_inpainted"]),
            "image_2": _open_image(sample["image_2_original"]),
        }
        for sample in samples
    ]
    oracle = [
        1.0 - np.asarray(_open_mask(sample["image_1_part_mask"])).sum() / max(
            np.asarray(_open_mask(sample["image_1_object_mask"])).sum(), 1.0
        )
        for sample in samples
    ]

    scores = {}
    if "clip_i" in args.baselines:
        scores["clip_i"] = clip_i_similarity(pairs, device)
    if "dino_v2" in args.baselines:
        scores["dino_v2"] = dinov2_similarity(pairs, device)

    table = []
    for name, values in scores.items():
        table.append(
            {
                "metric": name,
                "split": args.split,
                "pearson_r": float(np.corrcoef(values, oracle)[0, 1]),
                "spearman_rho": _spearman_coeff(values, oracle),
            }
        )

    with open(output_dir / f"{args.split}_global_baselines.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "split", "pearson_r", "spearman_rho"])
        writer.writeheader()
        writer.writerows(table)
    write_json(output_dir / f"{args.split}_global_baselines.json", {"results": table, "per_pair_scores": scores})
    for row in table:
        print(f"{row['metric']:10s} r={row['pearson_r']:.3f} rho={row['spearman_rho']:.3f}")


if __name__ == "__main__":
    main()
