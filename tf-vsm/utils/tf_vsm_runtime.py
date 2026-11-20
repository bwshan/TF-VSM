"""Runtime helpers for TF-VSM training, checkpoint selection and evaluation.

This module deliberately avoids importing GroundingDINO / SAM: inference only
needs the frozen CleanDIFT backbone and the two aggregation branches.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import CLEANDIFT_CONFIG_PATH, CLEANDIFT_PATH  # noqa: E402
from mtg.models import MindTheGlitchModel  # noqa: E402
from mtg.utils.metrics import compute_vsm_correlations  # noqa: E402


def load_config(path: str | Path) -> DictConfig:
    return OmegaConf.load(str(path))


def resolve_device_dtype(cfg: Mapping, device: Optional[str] = None) -> Tuple[torch.device, torch.dtype]:
    resolved_device = torch.device(device or cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    dtype_name = cfg.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_name, torch.float32)
    return resolved_device, dtype


def load_backbone(device: torch.device) -> torch.nn.Module:
    """Load the frozen SD2.1 CleanDIFT feature extractor.

    The CleanDIFT source tree is a git submodule; run
    ``git submodule update --init --recursive`` before using this function.
    """
    if str(CLEANDIFT_PATH) not in sys.path:
        sys.path.insert(0, str(CLEANDIFT_PATH))
    cfg_model = OmegaConf.load(CLEANDIFT_CONFIG_PATH)["model"]
    model_cfg = hydra.utils.instantiate(cfg_model)
    model = model_cfg.to(device).bfloat16()
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(repo_id="CompVis/cleandift", filename="cleandift_sd21_full.safetensors")
    model.load_state_dict(load_safetensors(ckpt_path), strict=True)
    return model.eval()


def load_state_dict(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        state = load_safetensors(str(checkpoint_path))
    else:
        state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model


def build_model(
    cfg: DictConfig,
    device: torch.device,
    dtype: torch.dtype,
    backbone: Optional[torch.nn.Module] = None,
) -> MindTheGlitchModel:
    """Construct a MindTheGlitchModel from a TF-VSM experiment config."""
    if backbone is None:
        backbone = load_backbone(device)
    model_name = cfg.model.get("name", "MindTheGlitchModel")
    if model_name != "MindTheGlitchModel":
        raise ValueError(f"Unsupported model for TF-VSM: {model_name}")
    return MindTheGlitchModel(backbone, cfg, device, dtype)


def load_model(
    config_path: str | Path,
    checkpoint_path: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> Tuple[MindTheGlitchModel, DictConfig, torch.device, torch.dtype]:
    cfg = load_config(config_path)
    dev, dtype = resolve_device_dtype(cfg, device)
    backbone = load_backbone(dev)
    model = build_model(cfg, dev, dtype, backbone)
    if checkpoint_path is not None:
        load_state_dict(model, checkpoint_path, dev)
    model.to(dev).eval()
    return model, cfg, dev, dtype


def build_transform(img_size: Iterable[int]) -> transforms.Compose:
    return transforms.Compose([transforms.Resize(tuple(img_size)), transforms.ToTensor()])


def _open_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    return Image.open(str(value)).convert("RGB")


def _open_mask(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    return Image.open(str(value)).convert("L")


def split_sample_indices(num_samples: int, selection_pairs: int, seed: int) -> Dict[str, List[int]]:
    """Deterministic 20/80 selection/test split of the 100-pair benchmark."""
    if selection_pairs >= num_samples:
        raise ValueError("selection_pairs must be smaller than the benchmark size")
    indices = np.random.RandomState(seed).permutation(num_samples)
    selection = sorted(int(i) for i in indices[:selection_pairs])
    test = sorted(int(i) for i in indices[selection_pairs:])
    return {"selection": selection, "test": test}


def load_benchmark_samples(
    benchmark_cfg: DictConfig,
    split_json: Optional[str | Path] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load the controlled benchmark from HuggingFace or from a local JSON file.

    Local JSON format: a list of dicts with the keys
    ``image_1_inpainted``, ``image_2_original``, ``image_1_object_mask``,
    ``image_2_object_mask``, ``image_1_part_mask``, ``image_2_part_mask``
    (values are image paths), plus an optional ``oracle``.
    """
    dataset_cfg = benchmark_cfg.dataset
    source = dataset_cfg.get("source", "huggingface")
    if source == "huggingface":
        from datasets import load_dataset

        dataset = load_dataset(dataset_cfg.repo_id, split=dataset_cfg.split)
        n = int(dataset_cfg.get("num_pairs", 100))
        if n > 0:
            dataset = dataset.select(range(min(n, len(dataset))))
        samples = [dict(dataset[i]) for i in range(len(dataset))]
    elif source == "json":
        with open(str(dataset_cfg.json_path), "r", encoding="utf-8") as handle:
            samples = json.load(handle)
        n = int(dataset_cfg.get("num_pairs", -1))
        if n > 0:
            samples = samples[:n]
    else:
        raise ValueError(f"Unknown benchmark source: {source}")

    splits = split_sample_indices(
        len(samples),
        int(dataset_cfg.get("selection_pairs", 20)),
        int(dataset_cfg.get("seed", 42)),
    )
    if split_json is not None:
        with open(str(split_json), "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        for name in splits:
            if name in loaded:
                splits[name] = [int(i) for i in loaded[name]]

    result: Dict[str, List[Dict[str, Any]]] = {}
    for name, idxs in splits.items():
        split_samples = []
        for i in idxs:
            if 0 <= i < len(samples):
                sample = dict(samples[i])
                sample["_benchmark_index"] = i
                split_samples.append(sample)
        result[name] = split_samples
    return result


@torch.inference_mode()
def evaluate_split(
    model: MindTheGlitchModel,
    samples: List[Dict[str, Any]],
    transform: transforms.Compose,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """Run TF-VSM (and legacy VSM/oracle) over a benchmark split."""
    accumulated: Dict[str, List[float]] = {}
    for sample in tqdm(samples, desc="evaluating", leave=False):
        img1 = _open_image(sample["image_1_inpainted"])
        img2 = _open_image(sample["image_2_original"])
        obj1 = _open_mask(sample["image_1_object_mask"])
        obj2 = _open_mask(sample["image_2_object_mask"])
        part1 = _open_mask(sample["image_1_part_mask"])
        part2 = _open_mask(sample.get("image_2_part_mask", sample["image_1_part_mask"]))

        output = model.inference(
            img1,
            img2,
            img1_obj_mask_p=obj1,
            img2_obj_mask_p=obj2,
            img1_part_mask_p=part1,
            img2_part_mask_p=part2,
            transform=transform,
        )
        for key, values in output["metrics"].items():
            accumulated.setdefault(key, []).extend([float(v) for v in values])

    if "oracle" not in accumulated and all("oracle" in sample for sample in samples):
        accumulated["oracle"] = [float(sample["oracle"]) for sample in samples]

    correlations = compute_vsm_correlations(dict(accumulated))
    summary: Dict[str, Dict[str, float]] = {}
    for name, value in correlations.items():
        if name.startswith("pearson_") or name.startswith("spearman_"):
            metric_name = name.replace("pearson_", "").replace("spearman_", "")
            summary.setdefault(metric_name, {})[name.split("_", 1)[0]] = float(value)

    return {"metrics": accumulated, "correlations": summary, "num_pairs": len(samples)}


def correlation_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for metric_name, values in sorted(result.get("correlations", {}).items()):
        rows.append(
            {
                "metric": metric_name,
                "pearson_r": values.get("pearson", float("nan")),
                "spearman_rho": values.get("spearman", float("nan")),
            }
        )
    return rows


def write_json(path: str | Path, payload: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
