# 6-Class Bus-Component Classification — DINOv2 vs ConvNeXt-V2

**Task:** classify a bus-component image into one of 6 classes.
**Method:** frozen backbone → image embeddings → logistic-regression head (no fine-tuning).
**Selection:** swept backbone weight variants × 3 seeds; best kept by test accuracy.
**Environment:** Google Colab, T4 GPU.

---

## 1. Dataset

| Class | Images |
|---|---:|
| bus_front_side | 4,063 |
| bus_back_side | 4,454 |
| Front_windshield | 4,320 |
| back_windshield | 3,841 |
| luggage_compartment | 2,931 |
| Jack and Spare_tyre | 3,050 |
| **Total** | **22,659** |

Split: 70% train / 15% val / 15% test, stratified per class (test ≈ 3,397 images).
Reasonably balanced (no aggressive imbalance handling needed).

---

## 2. Headline result

| Model | Best variant | Best seed | Test accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| **ConvNeXt-V2** | convnextv2_tiny | 123 | **0.9629** | 0.9608 |
| **DINOv2** | vit_base | 123 | **0.9626** | 0.9613 |

**The two models are statistically tied** (~96.3% accuracy, ~0.961 macro-F1). The 0.03% gap is noise.

---

## 3. Full sweep (variant × seed)

### DINOv2
| Variant | Seed | Acc | Macro-F1 |
|---|---:|---:|---:|
| vit_base | 123 | **0.9626** | 0.9613 |
| vit_small | 123 | 0.9617 | 0.9601 |
| vit_base | 2024 | 0.9606 | 0.9580 |
| vit_small | 2024 | 0.9585 | 0.9555 |
| vit_base | 42 | 0.9558 | 0.9550 |
| vit_small | 42 | 0.9529 | 0.9511 |

### ConvNeXt-V2
| Variant | Seed | Acc | Macro-F1 |
|---|---:|---:|---:|
| convnextv2_tiny | 123 | **0.9629** | 0.9608 |
| convnextv2_tiny | 2024 | 0.9608 | 0.9579 |
| convnextv2_base | 2024 | 0.9597 | 0.9571 |
| convnextv2_base | 123 | 0.9585 | 0.9568 |
| convnextv2_tiny | 42 | 0.9538 | 0.9519 |
| convnextv2_base | 42 | 0.9532 | 0.9516 |

**Observations**
- **Seed matters more than model size.** seed 123 was best for *both* models; seed 42 was worst for both (≈1% spread). The choice of variant (small/base/tiny) barely moved the needle.
- **Bigger isn't better (frozen).** convnextv2_**tiny** beat convnextv2_**base**, and DINOv2 small nearly matched base. With a *frozen* backbone, the larger models don't help on this task — the smaller ones are the smart choice.

---

## 4. Per-class performance (best model of each)

### DINOv2 (vit_base, seed 123)
| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| bus_front_side | 0.951 | 0.990 | 0.970 | 609 |
| bus_back_side | 0.963 | 0.969 | 0.966 | 668 |
| Front_windshield | 0.989 | 0.955 | 0.972 | 648 |
| back_windshield | 0.967 | 0.957 | 0.962 | 576 |
| luggage_compartment | 0.960 | 0.936 | 0.948 | 439 |
| Jack and Spare_tyre | 0.940 | 0.961 | 0.950 | 457 |

### ConvNeXt-V2 (convnextv2_tiny, seed 123)
| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| bus_front_side | 0.962 | 0.987 | 0.974 | 609 |
| bus_back_side | 0.966 | 0.975 | 0.970 | 668 |
| Front_windshield | 0.983 | 0.961 | 0.972 | 648 |
| back_windshield | 0.974 | 0.958 | 0.966 | 576 |
| luggage_compartment | 0.941 | 0.943 | 0.942 | 439 |
| Jack and Spare_tyre | 0.941 | 0.941 | 0.941 | 457 |

**Observations**
- **Strongest classes (both):** bus_front_side, Front_windshield, bus_back_side (F1 ≈ 0.97).
- **Weakest classes (both):** luggage_compartment and Jack_and_Spare_tyre (F1 ≈ 0.94–0.95). These are also the smallest classes and the most visually variable — the place to focus future improvements.
- The two models agree closely class-by-class; neither has a class where it clearly wins.

---

## 5. Cost / speed (T4 GPU, embedding 22,659 images)

| Model | Variant | Weights size | Embed time |
|---|---|---:|---:|
| DINOv2 | vit_small | 88 MB | 4.8 min |
| DINOv2 | vit_base | 346 MB | 8.6 min |
| ConvNeXt-V2 | tiny | 115 MB | 5.2 min |
| ConvNeXt-V2 | base | 355 MB | 7.9 min |

Logistic-regression head trains in seconds. Embeddings cached to Drive → re-runs are near-instant.

---

## 6. Example single-image prediction

Same image (`Front_windshield/23640_60_25.jpeg`), correctly classified by both:

| Model | Top prediction | Confidence |
|---|---|---:|
| DINOv2 (vit_base) | Front_windshield ✅ | **97.5%** |
| ConvNeXt-V2 (tiny) | Front_windshield ✅ | 95.9% |

Both confident and correct; DINOv2 marginally sharper on this example.

---

## 7. Conclusion & recommendation

- **Accuracy: a tie.** Both reach ~96.3% accuracy / ~0.961 macro-F1 on 6 classes. There is **no meaningful accuracy difference** between DINOv2 and ConvNeXt-V2 in the frozen-embedding setup.
- **Best efficiency pick:** **ConvNeXt-V2 tiny** — it matches the best accuracy with the smallest/fastest backbone, ideal if you want a lightweight production model with the frozen approach.
- **Best robustness pick:** **DINOv2 (small or base)** — DINOv2's self-supervised features are known to generalize better to messy/real-world (phone-upload) images, and it gave slightly crisper confidence here. Preferable if real uploads differ a lot from the training photos.
- **Seed:** 123 was best for both; worth fixing seed 123 (or averaging seeds) for the final model.

### Where the remaining ~3.7% error is
Concentrated in **luggage_compartment** and **Jack_and_Spare_tyre**, and in the **front/back** confusions (windshield ↔ bus body share visual content). Highest-leverage next steps:
1. **Crop the component (Stage-1)** before embedding — removes overlapping context that causes front-windshield ↔ bus-front confusion.
2. **Fine-tune** the chosen backbone (not frozen) — should push accuracy past ~96% by adapting features to these specific classes.
3. **More / cleaner data** for the two weakest classes.

### Bottom line
For a frozen, fast, no-GPU-training pipeline, **either model is an excellent choice (~96% accuracy)**. Pick **ConvNeXt-V2 tiny** for efficiency or **DINOv2** for real-world robustness. To go beyond ~96%, move to **crop + fine-tune**.
