# TF-VSM

<a href='https://arxiv.org/abs/2611.19780'><img src='https://img.shields.io/badge/ArXiv-2611.19780-red'></a> 

This is the implementation for the paper "A Threshold-Free Visual-Semantic Matching Metric for Evaluating Consistency in Subject-Driven Image Generation"
![f1](fig/f1.jpg)


## Installation

Same dependencies as p1, plus the CleanDIFT submodule:

```bash
pip install -r requirements.txt
git submodule update --init --recursive
export WORKSPACE_PATH=/path/to/workspace
export MTG_DATASET_PATH=/path/to/generated_pairs   # 4,912 supervision pairs
export WANDB_PROJECT_NAME=tf-vsm
```

`external/cleandift` is a git submodule and is empty until initialized. GroundingDINO/SAM are only needed for
regenerating the supervision data, not for TF-VSM inference/evaluation.

## Core implementation

- `mtg/utils/correspondence.py`
  - `tf_vsm_score(...)`: Eq. (7)-(8), the closed-form continuous readout.
  - `compute_tf_vsm_metric(...)`: drop-in replacement for p1's `compute_vsm_metric`.
  - `TF_VSM_ABLATION_KEYS`: all readout variants used by the six ablations.
- `mtg/models/mtg_model.py`: `compute_metrics` records `tf_vsm` alongside legacy `vsm_*` and `oracle`.
- `mtg/utils/metrics.py`: Pearson/Spearman correlation with the inpainting oracle.
- `mtg/utils/tf_vsm_runtime.py`: lightweight model loading and benchmark evaluation helpers.
- `configs/training/train_tf_vsm.yaml`: the extended retraining recipe.
- `configs/evaluation/controlled_benchmark.yaml`: 100-pair benchmark, 20/80 leakage-free split.

Small correctness fixes over p1 are included: the first aggregation layer now receives its learned mixing weight,
and the backbone-freezing loop runs after the trainable heads are built.

## Inference on one pair

```python
from mtg.utils.tf_vsm_runtime import load_model, build_transform

model, cfg, device, dtype = load_model(
    "configs/training/train_tf_vsm.yaml",
    checkpoint_path="outputs/tf_vsm/best.pt",
)
transform = build_transform(cfg.data.train.img_size or [768, 768])
out = model.inference(img1_p, img2_p, obj_mask1, obj_mask2, transform=transform)
print(out["metrics"]["tf_vsm"][0])   # no thresholds, higher = more consistent
```

`vsm_0.5/0.6/0.7` are still returned for comparison, and `notebooks/playground.ipynb` keeps the original
visualization workflow.

## Retraining

```bash
python scripts/train_tf_vsm.py \
  --config configs/training/train_tf_vsm.yaml \
  --dataset-path /path/to/generated_pairs \
  --output-dir outputs/tf_vsm
```

The script evaluates **only the 20-pair selection split** after every epoch, saves `epoch_XX.pt` and `best.pt`, and
writes `training_log.json`. The 80-pair held-out split is never read during training.

## Leakage-free evaluation

```bash
# 1) select the epoch on the 20-pair selection split
python scripts/evaluate_tf_vsm.py \
  --checkpoint outputs/tf_vsm/epoch_*.pt \
  --split selection --select-best --output-dir outputs/evaluation

# 2) report the chosen checkpoint on the strictly held-out 80 pairs
python scripts/evaluate_tf_vsm.py \
  --checkpoint outputs/tf_vsm/best.pt \
  --split test --output-dir outputs/evaluation
```

Outputs include per-pair metric values, Pearson r, and Spearman ρ as JSON and CSV.

Global baselines from Tables 1/2 can be reproduced with public checkpoints:

```bash
python scripts/evaluate_baselines.py --split test --baselines clip_i dino_v2
```

VLM / RefVNLI protocols depend on external APIs and are not bundled.

## Ablations

```bash
# Ablations 1-3 (readout variants) and 4 (schedule length)
python scripts/run_ablations.py --run main outputs/tf_vsm

# Ablation 5 (data scale) and 6 (loss balance alpha): retrain variants and compare
python scripts/train_tf_vsm.py --num-training-samples 2456 --output-dir outputs/tf_vsm_half
python scripts/train_tf_vsm.py --alpha 1 --output-dir outputs/tf_vsm_a1
python scripts/run_ablations.py \
  --run full outputs/tf_vsm --run half outputs/tf_vsm_half --run a1 outputs/tf_vsm_a1
```

The CSV rows are keyed by `run`, `epoch`, and `metric`, so a single table covers all six ablations.

## Data expectations

The repository does not bundle the 4,912 generated supervision pairs or the original HuggingFace benchmark; set
`MTG_DATASET_PATH`/`--dataset-path` to the generated dataset. For custom benchmarks, use a local JSON file in
`configs/evaluation/controlled_benchmark.yaml` with `source: json` and `json_path`, where each entry contains image
paths for `image_1_inpainted`, `image_2_original`, both object masks, and both inpainted-part masks. The oracle is
computed as `1 - |R1| / |O1|` from those masks, exactly as in both papers.

## Citation

If you use this code, please cite the original Mind-the-Glitch paper (Eldesokey et al., NeurIPS 2025) and the
TF-VSM paper (p2.pdf, ICLR 2025 submission).
