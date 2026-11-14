# Results

## Summary
| Version | Breeds | Photos | Test accuracy | Top-5 | Model |
|---|---:|---:|---:|---:|---|
| v2 | 137 (12 cat, 125 dog) | 25,216 | 89.03% | 98.52% | ResNet-50 |
| v1 | 37 (12 cat, 25 dog) | 7,349 | 92.45% | 99.13% | ResNet-50 |


v1 is preserved in the GitHub release `v1_37_breeds` (code + `model.pt`).

---

## v2 (137 breeds)

### Day 1b: Dataset
Built in `notebooks/01b_eda_build_dataset.ipynb`.

**Sources**
| Source | Photos (raw) | Photos (final) | Breeds |
|---|---:|---:|---|
| Stanford Dogs (Kaggle) | 20,580 | 19,168 | 120 dog breeds |
| Oxford-IIIT Pet | 7,390 | 5,947 | 12 cat + 25 dog breeds |
| Indian Pariah (Indie), collected from Pinterest | 101 | 101 | 1 |
| **Total** | **28,071** | **25,216** | **137** (12 cat, 125 dog) |

- 21 breeds exist in both Oxford and Stanford under different names (e.g. Japanese Chin = Japanese spaniel) and were merged into one class each.
- Oxford's American Pit Bull Terrier and Stanford's American Staffordshire Terrier were kept separate (different breeds).

**Cleaning**
| Step | Photos removed | Photos left |
|---|---:|---:|
| Broken images | 0 | 28,071 |
| Exact duplicates (MD5): same-breed copies | 82 | |
| Exact duplicates: conflicting labels (all copies dropped) | 74 | |
| Near-duplicates (perceptual hash): same-breed copies | 108 | |
| Near-duplicates: conflicting labels | 18 | 27,789 |
| Cap of 250 photos per breed | 2,573 | **25,216** |

Photos per breed after cleaning: min 101 (Indie), max 250.

**Split** (stratified by breed, seed 42)
| Split | Photos | Share |
|---|---:|---:|
| Train | 17,648 | 70% |
| Val | 3,784 | 15% |
| Test | 3,784 | 15% |

- Indie photos are split by dog, so no dog appears in two splits: 69 train / 16 val / 16 test.
- Species: 22,853 dog photos (125 breeds), 2,363 cat photos (12 breeds).

**Known limitations**
- Stanford Dogs comes from ImageNet, which the pretrained backbone was trained on, so test accuracy on Stanford breeds may be optimistic. Accuracy will be reported separately for Stanford vs Oxford + Indie photos.
- Indie has the fewest photos (101), so its test score (16 photos) is noisy.
- Indie photos come from Pinterest and are not redistributed in this repo.

### Day 2: Frozen baseline
- EfficientNet-B0 (ImageNet weights), backbone frozen, new Linear(1280→137) head
- Adam lr 1e-3, batch 32, 5 epochs, no augmentation
- Trainable params: 175,497 of ~4.18M (~4%)
- Best val accuracy: **75.50%** (epoch 5), ~112 s/epoch on Colab T4
- Still improving at epoch 5 (v1 plateaued at epoch 2)
- v1 comparison: 88.41% on 37 classes → 75.50% on 137 classes

### Day 3: Fine-tuning
- Same recipe as v1: augmentation, AdamW (backbone lr 1e-4, head lr 1e-3), cosine schedule, label smoothing 0.1, up to 12 epochs, early stopping (patience 3)

| run        | backbone        | frozen | augmentation | val_acc (%) | best_epoch | sec_per_epoch | size_mb |
|:-----------|:----------------|:-------|:-------------|------------:|-----------:|--------------:|--------:|
| Baseline   | efficientnet_b0 | Yes    | No           |       75.50 |          5 |           120 |    17.0 |
| Fine-tuned | efficientnet_b0 | No     | Yes          |       87.71 |         12 |           167 |    17.0 |
| Fine-tuned | resnet50        | No     | Yes          |       88.66 |         12 |           209 |    95.5 |

- Fine-tuning + augmentation: +12.21 points over the frozen baseline (v1: +5.07)
- **Final model: ResNet-50.** +0.95 points over fine-tuned B0 (~36 of 3,784 images), above the 0.5-point threshold
- Neither model early-stopped; most gains came by epoch 4 (B0: 85.7% → 87.7% over epochs 5–12)


### Day 4: Test set (evaluated once)
- Model: resnet50 | Test acc 89.03% | Top-5 98.52% | Macro F1 0.8855 | 3,784 test images
- Cats 94.38% | Dogs 88.48% (reversed from v1, where dogs were higher)
- Baseline (frozen) test acc 74.47% → fine-tuned 89.03% (+14.56 points; v1: +7.74)
- Test acc slightly above val (88.66%), unlike v1 (−2.7 points): larger val set, less selection bias

| Source | Test images | Accuracy | Top-5 |
|---|---:|---:|---:|
| Oxford-IIIT Pet | 915 | 93.77% | 99.23% |
| Stanford Dogs | 2,853 | 87.66% | 98.32% |
| Indie | 16 | 62.50% | 93.75% |

- Stanford is *lower* than Oxford, so no sign that ImageNet pretraining inflates its score. Its breeds are simply harder.
- Hardest breeds: Eskimo Dog and Toy Poodle (59.09%), Siberian Husky (60.71%), Miniature Poodle (60.87%), Indie and Shih-Tzu (62.50%)
- Most confused: Shih-Tzu → Lhasa (9), Siberian Husky → Eskimo Dog (7), poodle sizes (15 combined)
- Confidence threshold 0.7: catches 67.2% of wrong predictions, flags 12.3% of correct ones, 95.6% accuracy on the rest
---

## v1 (37 breeds)

### Day 1: Data
- Dataset: Oxford-IIIT Pet, 37 breeds (12 cats, 25 dogs)
- Official splits: trainval 3,680 / test 3,669 (test untouched until Day 4)
- My split: train 3,128 / val 552, stratified by breed, seed 42
- Images per breed (trainval): min 93, max 100 → balanced, so accuracy is a fair metric

### Day 2: Frozen baseline
- EfficientNet-B0 (ImageNet weights), backbone frozen, new Linear(1280→37) head
- Adam lr 1e-3, batch 32, 5 epochs, no augmentation
- Trainable params: 47,397 of ~4.05M (~1%)
- Best val accuracy: **88.41%** (epoch 2), ~21 s/epoch on Colab T4

### Day 3: Fine-tuning
- Augmentation (train only): RandomResizedCrop(224, scale 0.6–1), HorizontalFlip, ColorJitter (no hue)
- AdamW (backbone lr 1e-4, head lr 1e-3), cosine schedule, label smoothing 0.1,
  up to 12 epochs, early stopping (patience 3)

| run        | backbone        | frozen | augmentation | val_acc (%) | best_epoch | sec_per_epoch | size_mb |
|:-----------|:----------------|:-------|:-------------|------------:|-----------:|--------------:|--------:|
| Baseline   | efficientnet_b0 | Yes    | No           |       88.41 |          2 |            21 |    16.5 |
| Fine-tuned | efficientnet_b0 | No     | Yes          |       93.48 |          5 |            31 |    16.5 |
| Fine-tuned | resnet50        | No     | Yes          |       95.11 |          6 |            39 |    94.7 |

- Fine-tuning + augmentation: +5.07 points over the frozen baseline (EfficientNet-B0)
- **Final model: ResNet-50.** +1.63 points over fine-tuned B0 on validation (~9 of 552 images),
  above my 0.5-point noise threshold. Trade-off: 5.7× larger (94.7 MB vs 16.5 MB), ~25% slower per epoch.

### Day 4: Test set (evaluated once)
- Model: resnet50 | Test acc 92.45% | Top-5 99.13% | Macro F1 0.9239
- Cats 87.91% | Dogs 94.61%
- Baseline (frozen) test acc 84.71% → fine-tuned 92.45% (+7.74 points)
- Most confused: American Pit Bull Terrier → Staffordshire Bull Terrier (26 images), Egyptian Mau → Bengal (16 images)
