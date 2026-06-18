# Bus-Component Image Classification — DINOv2 & ConvNeXt-V2

Classify a bus-component photo into one of **6 classes** using **pretrained vision
backbones as frozen feature extractors** + a lightweight logistic-regression head.
Two models are provided and compared: **DINOv2** and **ConvNeXt-V2**.

## Classes
`bus_front_side` · `bus_back_side` · `Front_windshield` · `back_windshield` ·
`luggage_compartment` · `Jack and Spare_tyre`

Dataset: ~22,659 images, split 70/15/15 (stratified per class).

---

## Approach

```
image -> [read + resize 224 + normalize] -> frozen backbone -> embedding
      -> L2-normalize -> Logistic Regression -> class + confidence
```

- **Frozen backbone** (no fine-tuning) → fast, runs on a single GPU in minutes.
- A **sweep** tries each backbone weight variant × 3 seeds and keeps the **best**
  (by accuracy); the best `(weights + seed + trained head)` is **saved and reused**.
- Corruption-safe image loading (`ImageFile.LOAD_TRUNCATED_IMAGES = True`).
- Embeddings are cached → re-runs are near-instant.

---

## The two models

| Script | Backbone | Variants swept |
|---|---|---|
| [`dinov2.py`](dinov2.py) | DINOv2 (ViT) | `vit_small_patch14_dinov2`, `vit_base_patch14_dinov2` |
| [`convnextv2.py`](convnextv2.py) | ConvNeXt-V2 | `convnextv2_tiny`, `convnextv2_base` |

Each script has **CELL 1** (sweep + save best) and **CELL 2** (predict one image).

---

## Results (test set ≈ 3,397 images)

| Model | Best variant | Seed | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| **ConvNeXt-V2** | convnextv2_tiny | 123 | **0.9629** | 0.9608 |
| **DINOv2** | vit_base | 123 | **0.9626** | 0.9613 |

**Both models are tied at ~96.3% accuracy / ~0.961 macro-F1.**

Per-class F1 (best of each, ~equal): strongest `bus_front_side` / `Front_windshield`
(~0.97), weakest `luggage_compartment` / `Jack and Spare_tyre` (~0.94).

Full breakdown: [`reports/6class_dinov2_vs_convnextv2_report.md`](reports/6class_dinov2_vs_convnextv2_report.md).

### Key takeaways
- **Accuracy: a tie** — no meaningful difference between the two backbones (frozen).
- **Smaller is better here** — ConvNeXt-V2 *tiny* matched *base*; DINOv2 *small* ≈ *base*.
- **Seed mattered more than model size** — seed 123 was best for both.
- **96% is the frozen-approach ceiling** — to go higher: crop the component (Stage-1)
  + fine-tune, and add data for the two weak classes.

### Recommendation
- **Efficiency** → ConvNeXt-V2 *tiny*.
- **Robustness to real/phone uploads** → DINOv2.

---

## Run (Google Colab)

1. Runtime → Change runtime type → **T4 GPU**.
2. Mount Drive and put the dataset under `/content/new_dataset` (one folder per class).
3. Open `dinov2.py` or `convnextv2.py`, paste **CELL 1** into a cell → Shift+Enter (train + save best),
   then paste **CELL 2** → set an image path → Shift+Enter (predict).

```python
!pip install -q timm      # only extra dependency in Colab
```

Verify the GPU is active before running:
```python
import torch; print(torch.cuda.is_available())   # must be True
```

---

## Requirements
`torch`, `torchvision`, `timm`, `numpy`, `pillow`, `scikit-learn`
(Colab has all but `timm`; install with `pip install timm`).

## Files
```
dinov2.py        DINOv2 6-class classifier (train + predict)
convnextv2.py    ConvNeXt-V2 6-class classifier (train + predict)
reports/         model comparison report
```
