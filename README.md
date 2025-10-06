# Robust Skin Tone Estimation in Diverse Lighting Conditions

> **Goal.** Predict perceptual color (CIELAB) of skin patches from photos taken under varied lighting/exposure using both image content and camera metadata.

---

## Summary

We study how image metadata (exposure/ISO/camera) helps predict perceived color under adverse conditions. We compare three fusion strategies (concat, **FiLM**, **cross-attention**) and multiple backbones. Our best model reaches **ΔE00 = 0.81** (lower is better) on our internal evaluation, where ΔE00 = 1 represents a _"just barely noticeable difference"_.

---

## Project Overview

### Motivation
As all of us have experienced, photos often fail to reproduce true colors, especially under over/under-exposure or challenging illumination. This project targets robust, human-aligned color prediction for skin tones.

### Problem Definition
We optimize for the perceptual color difference (**ΔE00**) in CIELAB space, which better matches human vision than raw MSE. Our objective is to assess **how much metadata improves robustness** across conditions.

---

## Data & Preprocessing

- **Dataset.** 15 sets × 138 images of different skin tones (**2,070 images**) captured under diverse lighting/exposure, across **3 cameras**.
- **Adaptive cropping.** Gradient-based, outlier-aware cropping to isolate the target patch while **avoiding shadows and borders**; patch is surrounded by a small **background** region to provide local context.
- **Metadata features.** Continuous: brightness / shutter speed / ISO. Categorical: camera ID (embedded).
- **Targets.** Ground-truth CIELAB color per image.

---

## Methods

### Architectures
- **Concat (early fusion):** CNN features (image) concatenated with MLP features (metadata).
- **FiLM (feature-wise linear modulation):** Metadata conditions intermediate CNN activations via per-channel affine transforms.
- **Cross-attention fusion:** Metadata tokens attend to image tokens to exchange information explicitly.

### Encoders & Backbones
- Image backbones include a lightweight SmallCNN and several timm-style CNNs.
- Metadata MLP + optional text embedding (e.g., device strings).

### Losses & Training
- **Primary:** ΔE00 in CIELAB (human-aligned).
- **Auxiliary:** MSE terms in **RGB** and/or **Lab** for smoother convergence.
- **Robustness:** **Huber** loss on auxiliary terms; AdamW / cosine LR; brief trials with SGD on ResNet variants.

---

## Results (abridged)

- **Best overall:** Cross-attention fusion with focused crops **+ background context** achieves **ΔE00 = 0.81** (lower is better).
- **Ablation highlights**
  - **Data is king.** Biggest gains from **focused images**, **adding background context**, and **adaptive cropping**.
  - **Metadata helps most without background.** Clear improvements in **no-BG** scenarios and in harsher exposure regimes.
  - With **background + mean features + larger backbones**, metadata gains **diminish** (convergence/robustness benefits may remain).
  - See `presentation.pdf` for visualizations of feature comparisons.

> **Interpretation.** Local background context plus mean features capture much of what exposure metadata provides; metadata might be valuable in sensitive or low-context setups.

---

## Usage

```bash
# 1) Create env (example)
conda create -n ster python=3.11 -y
conda activate ster
pip install -r requirements.txt

# 2) Train / evaluate (examples — adapt paths)
python -m src.engine.sweep \
  --config ./configs/bg.yaml \
  --auto-summary --analysis --only_full --eval-winners --test_split_file AUTO

# Visualizations and reports
python -m src.analysis.summarize_sweep --ckpt_dir ./checkpoints/sweep/... --out_csv results.csv
python -m src.analysis.visualize_runs --in_csv results.csv --out_dir figs/
