from typing import Optional

import einops
import torch
from torchvision.transforms.functional import to_tensor
from PIL.Image import Image
import numpy as np

from mtg.utils.image_proc import downsample_mask
from mtg.utils.visualization import plot_correspondences


# Keys produced by compute_tf_vsm_metric(..., include_ablation_variants=True).
# They mirror the six diagnostic ablations in TF-VSM (paper Sections 4.6 / App. E).
TF_VSM_ABLATION_KEYS = [
    "tf_vsm",
    "tf_vsm_weight_uniform",
    "tf_vsm_weight_linear",
    "tf_vsm_weight_cubic",
    "tf_vsm_weight_softmax",
    "tf_vsm_hard_ts_0.5",
    "tf_vsm_hard_ts_0.6",
    "tf_vsm_hard_ts_0.7",
    "tf_vsm_hard_ts_0.8",
    "tf_vsm_visual_raw",
    "tf_vsm_visual_sigmoid",
    "tf_vsm_visual_sq",
    "tf_vsm_step_tv_0.5",
]


def _as_1d_float(value) -> torch.Tensor:
    """Convert a numpy array / torch tensor / list to a flat float tensor."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32).reshape(-1)
    return torch.as_tensor(value, dtype=torch.float32).reshape(-1)


def _to_numpy(value) -> np.ndarray:
    """Convert torch tensors (possibly on GPU) or arrays to a CPU numpy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def semantic_confidence_weight(
    semantic_scores: torch.Tensor, kind: str = "quadratic", temperature: float = 0.1
) -> torch.Tensor:
    """w_j = f(s_hat_j) for the TF-VSM semantic confidence.

    The default is the paper's w_j = ReLU(s_hat_j)^2.  The other choices are
    exposed for Ablation 1 and are not used by the default metric.
    """
    scores = _as_1d_float(semantic_scores)
    if kind == "uniform":
        return torch.ones_like(scores)
    if kind == "linear":
        return torch.relu(scores)
    if kind == "quadratic":
        return torch.relu(scores).pow(2)
    if kind == "cubic":
        return torch.relu(scores).pow(3)
    if kind == "softmax":
        return torch.softmax(scores / max(temperature, 1e-8), dim=0)
    raise ValueError(f"Unknown semantic weighting mode: {kind}")


def visual_consistency_transform(
    visual_scores: torch.Tensor, kind: str = "clamp", slope: float = 10.0, center: float = 0.5
) -> torch.Tensor:
    """g(v_hat_j).  The paper sets g(v) = clamp(v, 0, 1) so that the integral
    over r in [0, 1] has the closed form g(v) itself.
    """
    scores = _as_1d_float(visual_scores)
    if kind == "raw":
        return scores
    if kind == "clamp":
        return scores.clamp(0.0, 1.0)
    if kind == "sigmoid":
        return torch.sigmoid(slope * (scores - center))
    if kind == "clamp_sq":
        return scores.clamp(0.0, 1.0).pow(2)
    if kind == "step":
        # Step readout used in the synthetic-oracle diagnostic (Tv=0.5).
        return (scores > 0.5).to(dtype=torch.float32)
    raise ValueError(f"Unknown visual transform: {kind}")


def tf_vsm_score(
    semantic_scores,
    visual_scores,
    target_points_s=None,
    semantic_weight: str = "quadratic",
    visual_transform: str = "clamp",
    semantic_gate: Optional[float] = None,
    deduplicate_targets: bool = False,
    softmax_temperature: float = 0.1,
    eps: float = 1e-12,
    return_per_point: bool = False,
):
    """Closed-form, threshold-free TF-VSM readout (Eq. 7-8 of the paper).

        w_j      = ReLU(s_hat_j)^2
        g(v_j)   = clamp(v_hat_j, 0, 1)
        TF-VSM = sum_j w_j g(v_j) / sum_j w_j

    ``semantic_gate`` and the non-default ``semantic_weight`` / ``visual_transform``
    options are only used by the ablations; the default metric has no scalar
    hyperparameter (Ts = Tv = None).  Setting ``deduplicate_targets=True``
    reproduces p1's one-source-point-per-target-point bookkeeping; the paper's
    formula integrates over every foreground source point, so the default is
    False.
    """
    s = _as_1d_float(semantic_scores)
    v = _as_1d_float(visual_scores)
    if s.numel() != v.numel():
        raise ValueError(f"Semantic/visual score length mismatch: {s.numel()} vs {v.numel()}")
    if s.numel() == 0:
        empty = torch.zeros((), dtype=torch.float32)
        return (empty, empty) if return_per_point else empty

    if deduplicate_targets and target_points_s is not None:
        targets = [tuple(int(x) for x in point) for point in target_points_s]
        best: dict[tuple[int, int], tuple[int, float]] = {}
        for idx, (target, score) in enumerate(zip(targets, s.tolist())):
            if target not in best or score > best[target][1]:
                best[target] = (idx, score)
        keep = torch.tensor([idx for idx, _ in best.values()], dtype=torch.long, device=s.device)
        s, v = s[keep], v[keep]

    if semantic_gate is not None:
        weights = (s > semantic_gate).to(dtype=torch.float32)
    else:
        weights = semantic_confidence_weight(s, semantic_weight, softmax_temperature)

    g = visual_consistency_transform(v, visual_transform)
    denominator = weights.sum().clamp_min(eps)
    per_point = weights * g / denominator
    score = per_point.sum()
    return (score, per_point) if return_per_point else score


def compute_tf_vsm_metric(
    source_points_s,
    target_points_s,
    matching_score_maps_max_s,
    source_points_v,
    target_points_v,
    matching_score_maps_max_v,
    include_ablation_variants: bool = False,
    visualize_matches: bool = False,
    img1: Optional[Image] = None,
    img2: Optional[Image] = None,
):
    """Threshold-free counterpart of ``compute_vsm_metric``.

    Returns a dict of floating point metric values and (optionally) a
    correspondence visualization.  Point arrays are kept in the signature so
    this function is a drop-in replacement in notebooks and evaluation code.
    """
    sem_scores = _as_1d_float(matching_score_maps_max_s)
    vis_scores = _as_1d_float(matching_score_maps_max_v)
    targets = [tuple(int(x) for x in point) for point in target_points_s]

    result = {
        "tf_vsm": tf_vsm_score(sem_scores, vis_scores, targets).item()
    }

    if include_ablation_variants:
        for name in ("uniform", "linear", "cubic", "softmax"):
            result[f"tf_vsm_weight_{name}"] = tf_vsm_score(
                sem_scores, vis_scores, targets, semantic_weight=name
            ).item()
        for threshold in (0.5, 0.6, 0.7, 0.8):
            result[f"tf_vsm_hard_ts_{threshold}"] = tf_vsm_score(
                sem_scores, vis_scores, targets, semantic_gate=threshold
            ).item()
        for name, kwargs in (
            ("raw", {"visual_transform": "raw"}),
            ("sigmoid", {"visual_transform": "sigmoid"}),
            ("sq", {"visual_transform": "clamp_sq"}),
        ):
            result[f"tf_vsm_visual_{name}"] = tf_vsm_score(
                sem_scores, vis_scores, targets, **kwargs
            ).item()
        # Section 4.2 step-readout diagnostic: keeps soft semantic weights but
        # replaces the continuous clamp with a step at Tv=0.5.
        result["tf_vsm_step_tv_0.5"] = tf_vsm_score(
            sem_scores, vis_scores, targets, visual_transform="step"
        ).item()

    correspondence_img = None
    if visualize_matches and img1 and img2:
        conf_scores_s = _to_numpy(sem_scores)
        conf_source_pts_s = _to_numpy(source_points_s)[: sem_scores.numel()]
        conf_target_pts_s = _to_numpy(target_points_s)[: sem_scores.numel()]
        conf_source_pts_v = _to_numpy(source_points_v)[: vis_scores.numel()]
        conf_target_pts_v = _to_numpy(target_points_v)[: vis_scores.numel()]
        correspondence_img = plot_correspondences(
            img1,
            img2,
            conf_source_pts_s,
            conf_target_pts_s,
            conf_scores_s,
            conf_source_pts_v,
            conf_target_pts_v,
            vis_scores.detach().cpu().numpy(),
            score_threshold=0.0,
        )

    return result, correspondence_img


def compute_vsm_metric(
    source_points_s,
    target_points_s,
    matching_score_maps_max_s,
    source_points_v,
    target_points_v,
    matching_score_maps_max_v,
    visual_thresholds=[0.5, 0.6, 0.7],
    semantic_threshold=0.7,
    visualize_matches=False,
    img1: Optional[Image] = None,
    img2: Optional[Image] = None,
):

    # Make sure that every point in source matches to on point in target (with highest confidence)
    best = {}  # key: (x, y) target point -> (src_idx, score)
    for i, (score, tpt) in enumerate(zip(matching_score_maps_max_s, target_points_s)):
        if score <= semantic_threshold:
            continue
        key = (int(tpt[0]), int(tpt[1]))
        if key not in best or score > best[key][1]:
            best[key] = (i, score)

    conf_idx = [i for i, _ in best.values()]

    # if isinstance(matching_score_maps_max_v, torch.Tensor):
    #     matching_score_maps_max_v = matching_score_maps_max_v.float().detach().cpu().numpy()

    conf_scores_v = torch.tensor([matching_score_maps_max_v[i] for i in conf_idx])

    vsm_metric = {
        f"vsm_{threshold}": (torch.sum(conf_scores_v > threshold) / conf_scores_v.shape[0]).item()
        for threshold in visual_thresholds
    }

    correspondence_img = None
    if visualize_matches and img1 and img2:
        # Semantic Correspondences
        conf_source_pts_s = np.array([source_points_s[i] for i in conf_idx])
        conf_target_pts_s = np.array([target_points_s[i] for i in conf_idx])
        conf_scores_s = np.array([matching_score_maps_max_s[i] for i in conf_idx])

        # Visual Correspondences
        conf_source_pts_v = np.array([source_points_v[i] for i in conf_idx])
        conf_target_pts_v = np.array([target_points_v[i] for i in conf_idx])
        correspondence_img = plot_correspondences(
            img1,
            img2,
            conf_source_pts_s,
            conf_target_pts_s,
            conf_scores_s,
            conf_source_pts_v,
            conf_target_pts_v,
            conf_scores_v,
            score_threshold=0.7,
        )

    return vsm_metric, correspondence_img


def reshape_feats(feats, normalize=True):
    # Normalize the features
    if normalize:
        feats = feats / torch.norm(feats, dim=1, keepdim=True)
    return einops.rearrange(feats, "1 c h w -> (h w) c").float().detach().cpu().numpy()


def extract_features(
    img_source: Image, img_target: Image, cleandift_model, item_desc, img_size=(768, 768), feat_key="us6"
):
    """Extract features from two image using CleanDIFT model"""

    img_source_tensor = to_tensor(img_source.resize(img_size))[None].to("cuda") * 2 - 1
    img_target_tensor = to_tensor(img_target.resize(img_size))[None].to("cuda") * 2 - 1

    # GET FEATURES FOR BOTH IMAGES
    feat_key = "us6"
    # print(f"Using feature {feat_key}")

    with torch.no_grad():
        source_features = cleandift_model.get_features(
            img_source_tensor.bfloat16(), [item_desc], t=None, feat_key=feat_key
        )  # [1,C,H,W]
        # print("Got features for source image")
        target_features = cleandift_model.get_features(
            img_target_tensor.bfloat16(), [item_desc], t=None, feat_key=feat_key
        )  # [1,C,H,W]
        # print("Got features for target image")

    return source_features, target_features


def create_coordinate_tensor(h, w):
    # Create a meshgrid for x and y coordinates
    y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    # Stack x and y coordinates to create a tensor
    coordinate_tensor = torch.stack((x_coords, y_coords))
    return coordinate_tensor.reshape(-1, 2)


def match_features_masked(
    source_features: torch.Tensor, target_features: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor
):  # TODO: support batch size > 1
    """Supports a batch size of 1 for now

    Args:
        source_features (torch.Tensor): B x C x H x W
        target_features (torch.Tensor): B x C x H x W
        source_mask (torch.Tensor): B x 1 x H x W
        target_mask (torch.Tensor): B x 1 x H x W

    Returns:
        _type_: _description_
    """
    image_resolution = source_mask.shape[-1]
    source_mask = downsample_mask(source_mask, source_features.shape[-2:])
    target_mask = downsample_mask(target_mask, target_features.shape[-2:])

    # source_points = create_coordinate_tensor(*source_features.shape[-2:])  # Dense Correspondence
    source_points = torch.argwhere(source_mask[0, 0] != 0).flip(-1).to(source_features.device).long()  # Only on object

    # Normalize over the feature dimensions then flatten H,W
    target_mask_flat_t = target_mask[:1].reshape(1, -1).to(target_features.device)
    target_features_upsampled_norm_flat = einops.rearrange(
        target_features / target_features.norm(p=2, dim=1, keepdim=True), "1 c h w -> c (h w)"
    )  # [num_feats, (H*W)]
    target_features_upsampled_norm_flat = target_features_upsampled_norm_flat * target_mask_flat_t

    source_point_feats = source_features[0, :, source_points[:, 1], source_points[:, 0]].T[
        :, None
    ]  # [num_points, 1, num_feats]
    source_point_feats_norm = source_point_feats / source_point_feats.norm(
        p=2, dim=-1, keepdim=True
    )  # [num_points, 1, num_feats]

    matching_score_maps = einops.rearrange(
        source_point_feats_norm @ target_features_upsampled_norm_flat, "b 1 (h w) -> b h w", h=target_features.shape[-2]
    )
    # [num_points, 1, num_feats] @ [num_feats, (H*W)] => [num_point, (H*W)] => [num_points, H, W]
    matching_score_maps_max, matching_score_maps_max_idx = torch.max(
        einops.rearrange(matching_score_maps, "b h w -> b (h w)"), -1
    )
    target_points_matched = torch.stack(
        torch.unravel_index(matching_score_maps_max_idx, matching_score_maps.shape[1:])[::-1]
    ).T

    ### Scale the points to the original image resoution
    scale_factor = image_resolution / target_features.shape[-1]
    source_points_hr = source_points * scale_factor
    target_points_hr = target_points_matched * scale_factor
    return (
        source_points_hr,
        target_points_hr,
        matching_score_maps,
        matching_score_maps_max,
        source_points.cpu().numpy(),
        target_points_matched.cpu().numpy(),
    )
    # Source points in image resolution, Matched points in target image, score maps of the matches, max similarity of the matches
