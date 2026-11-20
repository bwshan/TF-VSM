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


## Retraining

```bash
python scripts/train_tf_vsm.py \
  --config configs/training/train_tf_vsm.yaml \
  --dataset-path /path/to/generated_pairs \
  --output-dir outputs/tf_vsm
```


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


```bash
python scripts/evaluate_baselines.py --split test --baselines clip_i dino_v2
```


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


## Citation

```bibtex
@article{shan2026tf-vsm,
  title={A Threshold-Free Visual-Semantic Matching Metric for Evaluating Consistency in Subject-Driven Image Generation},
  author={Bowei Shan, Heng Yin},
  journal={arXiv preprint arXiv:2611.19780},
  year={2026}
}
```
