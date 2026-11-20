"""Global embedding baselines used in the TF-VSM paper (Tables 1, 2, 4).

CLIP-I uses OpenCLIP ViT-B/32 through the ``openai-clip`` package already
required by p1.  DINOv2 uses the HuggingFace ViT-L/14 checkpoint.  VLM and
RefVNLI protocols require external APIs and are intentionally not bundled.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


@torch.no_grad()
def clip_i_similarity(
    pairs: List[Dict[str, Any]],
    device: torch.device,
    model_name: str = "ViT-B/32",
) -> List[float]:
    import clip

    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    results = []
    for pair in pairs:
        img1 = preprocess(pair["image_1"]).unsqueeze(0).to(device)
        img2 = preprocess(pair["image_2"]).unsqueeze(0).to(device)
        f1 = model.encode_image(img1)
        f2 = model.encode_image(img2)
        f1, f2 = torch.nn.functional.normalize(f1, dim=-1), torch.nn.functional.normalize(f2, dim=-1)
        results.append(float((f1 * f2).sum(-1).item()))
    return results


@torch.no_grad()
def dinov2_similarity(
    pairs: List[Dict[str, Any]],
    device: torch.device,
    model_name: str = "facebook/dinov2-large",
) -> List[float]:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    results = []
    for pair in pairs:
        inputs1 = processor(images=pair["image_1"], return_tensors="pt").to(device)
        inputs2 = processor(images=pair["image_2"], return_tensors="pt").to(device)
        f1 = model(**inputs1).last_hidden_state[:, 0]
        f2 = model(**inputs2).last_hidden_state[:, 0]
        f1, f2 = torch.nn.functional.normalize(f1, dim=-1), torch.nn.functional.normalize(f2, dim=-1)
        results.append(float((f1 * f2).sum(-1).item()))
    return results
