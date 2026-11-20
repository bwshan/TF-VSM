"""Train the p2 (TF-VSM) dual-branch aggregation network.

Compared with p1 this script implements the extended schedule of the paper:
14 epochs, AdamW with lr=5e-4, alpha=10, 4,421/491 train/val split, and
checkpoint selection by oracle correlation on a disjoint 20-pair selection
split.  The 80-pair held-out test split is never touched here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import MTG_DATASET_PATH  # noqa: E402
from mtg.data.mtg_dataset import load_dataset  # noqa: E402
from mtg.utils.tf_vsm_runtime import (  # noqa: E402
    build_transform,
    evaluate_split,
    load_backbone,
    load_benchmark_samples,
    load_config,
    load_state_dict,
    resolve_device_dtype,
    write_json,
    build_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/train_tf_vsm.yaml")
    parser.add_argument("--benchmark-config", default="configs/evaluation/controlled_benchmark.yaml")
    parser.add_argument("--benchmark-split-json", default=None, help="Optional fixed 20/80 split file")
    parser.add_argument("--output-dir", default="outputs/tf_vsm")
    parser.add_argument("--dataset-path", default=None, help="Directory containing the 4,912 generated pairs")
    parser.add_argument("--init-ckpt", default=None, help="Warm start from p1/TF-VSM weights")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override config learning rate")
    parser.add_argument("--alpha", type=float, default=None, help="Override visual loss scale alpha")
    parser.add_argument("--num-training-samples", type=int, default=None, help="Ablation 5: nested data subset")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-benchmark-selection", action="store_true", help="Select by val loss instead of oracle r")
    return parser.parse_args()


def configure_dataset_paths(cfg, dataset_path: str | None):
    base = Path(dataset_path) if dataset_path else Path(str(MTG_DATASET_PATH))
    if not base.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {base}. Set MTG_DATASET_PATH or pass --dataset-path."
        )
    for split in ("train", "val"):
        cfg.data[split].dataset_path = str(base)
    return base


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.alpha is not None:
        cfg.model.lr_alpha = args.alpha
    if args.num_training_samples is not None:
        cfg.data.train.num_samples = args.num_training_samples

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved_config.json", OmegaConf.to_container(cfg, resolve=True))

    device, dtype = resolve_device_dtype(cfg, args.device)
    backbone = load_backbone(device)
    model = build_model(cfg, device, dtype, backbone)
    if args.init_ckpt:
        load_state_dict(model, args.init_ckpt, device)
        print(f"Warm-started from {args.init_ckpt}")

    configure_dataset_paths(cfg, args.dataset_path)
    train_cfg, val_cfg = cfg.data.train, cfg.data.val
    from configs.config import fetch_cfg_from_dataset

    train_cfg = fetch_cfg_from_dataset(train_cfg)
    val_cfg = fetch_cfg_from_dataset(val_cfg)

    transform = build_transform(train_cfg.img_size or [768, 768])
    train_ds, val_ds, _, _ = load_dataset(train_cfg, val_cfg, transform, transform)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    scheduler = None
    if cfg.get("scheduler_step"):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.scheduler_step), gamma=0.1)

    benchmark = None
    if not args.no_benchmark_selection:
        benchmark_cfg = load_config(args.benchmark_config)
        splits = load_benchmark_samples(benchmark_cfg, args.benchmark_split_json)
        benchmark = {"samples": splits["selection"], "transform": build_transform(benchmark_cfg.img_size)}
        write_json(
            output_dir / "selection_indices.json",
            {
                "selection": [int(s["_benchmark_index"]) for s in benchmark["samples"]],
                "note": "Indices into the original benchmark; the 80 held-out pairs are never used here.",
            },
        )

    log = []
    best_score, best_epoch = (-float("inf"), -1) if benchmark else (float("inf"), -1)
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        running = {}
        for batch in train_loader:
            losses, *_ = model(batch)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            for key, value in losses.items():
                running.setdefault(key, 0.0)
                running[key] += float(value.detach()) if torch.is_tensor(value) else float(value)
        train_loss = {key: value / max(len(train_loader), 1) for key, value in running.items()}

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                losses, *_ = model(batch)
                val_loss_sum += float(losses["total"].detach())
        val_loss = val_loss_sum / max(len(val_loader), 1)

        selection_result = None
        if benchmark is not None:
            selection_result = evaluate_split(
                model, benchmark["samples"], benchmark["transform"], device, dtype
            )
            selection_r = selection_result["correlations"].get("tf_vsm", {}).get("pearson", float("nan"))
            score = selection_r
            is_better = score > best_score
        else:
            score = val_loss
            is_better = score < best_score

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "selection_tf_vsm_pearson": score if benchmark is not None else None,
        }
        log.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if scheduler is not None:
            scheduler.step()

        ckpt_path = output_dir / f"epoch_{epoch:02d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "selection": selection_result["correlations"] if selection_result else None,
            },
            ckpt_path,
        )
        if is_better:
            best_score, best_epoch = score, epoch
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, output_dir / "best.pt")

    write_json(
        output_dir / "training_log.json",
        {
            "log": log,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "selection_split": "20-pair benchmark selection split" if benchmark else "validation loss",
            "note": "The 80-pair held-out split was never evaluated during training.",
        },
    )
    print(f"Best epoch: {best_epoch} (score={best_score:.4f}); checkpoints in {output_dir}")


if __name__ == "__main__":
    main()
