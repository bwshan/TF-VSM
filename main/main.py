%load_ext autoreload
%autoreload 2

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('..')
from pathlib import Path

import torch
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from configs.config import load_dataset_generation_config

mtg_ckpt_pth = Path(hf_hub_download(repo_id="abdo-eldesokey/mind-the-glitch", filename="mtg_weights.safetensors"))
mtg_config = Path(hf_hub_download(repo_id="abdo-eldesokey/mind-the-glitch", filename="experiment_cfg.yaml"))
training_cfg = OmegaConf.load(mtg_config)
dataset_cfg = load_dataset_generation_config("automated")

# Define constants from config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = getattr(torch, training_cfg.dtype, "float32")
MODEL_NAME = training_cfg.model.name
from safetensors.torch import load_file
from torchvision import transforms
import torch

import mtg.models as mtg_models
from mtg.utils.aux_models_manager import ModelManager

# Load Auxiliary models
models_manager = ModelManager(DEVICE, dataset_cfg ,load_cleandift=True, load_groundingsam=True)
groundingdino_model, sam_predictor, cleandift_model = models_manager.groundingdino_model, models_manager.sam_predictor, models_manager.cleandift_model

transform = transforms.Compose([
                                transforms.Resize(dataset_cfg.img_size), 
                                transforms.ToTensor()])    

# Load MTG Model
model = getattr(mtg_models, MODEL_NAME)(cleandift_model, training_cfg, DEVICE, DTYPE)
print(f"Loading checkpoint from {mtg_ckpt_pth}")
if mtg_ckpt_pth.suffix == ".safetensors":
    model.load_state_dict(load_file(mtg_ckpt_pth))
else:
    model.load_state_dict(torch.load(mtg_ckpt_pth, map_location=DEVICE))
model.to(DEVICE)
model.eval()
from datasets import load_dataset

dataset = load_dataset("abdo-eldesokey/mtg-dataset", split="test")

sample_id = 2
sample = dataset[sample_id]

img1_orig = sample["image_1_original"]
img2_orig = sample["image_2_original"]
img1_inpainted = sample["image_1_inpainted"]
img2_inpainted = sample["image_2_inpainted"]
img1_obj_mask = sample["image_1_object_mask"]
img2_obj_mask = sample["image_2_object_mask"]
img1_part_mask = sample["image_1_part_mask"]
img2_part_mask = sample["image_2_part_mask"]
    
    
img1 = img1_inpainted.resize((768, 768))
img2 = img2_orig.resize((768, 768))
model_out_dict = model.inference(img1_inpainted, img2_orig, img1_obj_mask_p=img1_obj_mask,img2_obj_mask_p=img2_obj_mask, img1_part_mask_p=img1_part_mask, img2_part_mask_p=img2_part_mask, transform=transform, return_correspondences=True)
    
metrics = model_out_dict["metrics"]
source_points_s, target_points_s, matching_score_maps_max_s = model_out_dict["source_points_s"], model_out_dict["target_points_s"], model_out_dict["matching_score_maps_max_s"]
source_points_v, target_points_v, matching_score_maps_max_v = model_out_dict["source_points_v"], model_out_dict["target_points_v"], model_out_dict["matching_score_maps_max_v"]
img1_feats_s, img2_feats_s, img1_feats_v, img2_feats_v = model_out_dict["img1_feats_s"], model_out_dict["img2_feats_s"], model_out_dict["img1_feats_v"], model_out_dict["img2_feats_v"]     
from mtg.utils.correspondence import compute_vsm_metric
vsm_new, correspondence_img = compute_vsm_metric(source_points_s, target_points_s, matching_score_maps_max_s, source_points_v, target_points_v, matching_score_maps_max_v,
visualize_matches=True, img1=img1, img2=img2, semantic_threshold=0.6)

from IPython.display import display
from mtg.utils.visualization import plot_consistency_scores
map_1 = plot_consistency_scores(img1, img1_obj_mask, source_points_v, matching_score_maps_max_v)
map_2 = plot_consistency_scores(img2,  img2_obj_mask, target_points_v, matching_score_maps_max_v)
display(map_1)
display(map_2)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA


flat1 = img1_feats_s[0]
flat2 = img2_feats_s[0]

for flat1, flat2, title in ((img1_feats_s[0], img2_feats_s[0], "Semantic Features"), (img1_feats_v[0], img2_feats_v[0], "Visual Feature")):
    # Step 2: Stack and normalize
    all_feats = np.vstack([flat1, flat2])
    scaler = StandardScaler()
    all_feats_scaled = scaler.fit_transform(all_feats)

    # Step 3: PCA on combined features
    pca = PCA(n_components=3)
    all_feats_pca = pca.fit_transform(all_feats_scaled)

    # Step 4: Separate and normalize each back to [0,1]
    pca1 = all_feats_pca[:flat1.shape[0]]
    pca2 = all_feats_pca[flat1.shape[0]:]

    pca1 -= pca1.min(0)
    pca1 /= pca1.max(0)
    pca2 -= pca2.min(0)
    pca2 /= pca2.max(0)

    # Step 5: Reshape to RGB images
    h, w = 48,48
    img1_rgb = pca1.reshape(h, w, 3)
    img2_rgb = pca2.reshape(h, w, 3)

    # Step 6: Display side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img1_rgb)
    plt.axis("off")    
    plt.subplot(1, 2, 2)
    plt.imshow(img2_rgb)
    plt.axis("off")    
    plt.show()


