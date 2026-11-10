# Results

## Day 1: Data
- Dataset: Oxford-IIIT Pet, 37 breeds (12 cats, 25 dogs)
- Official splits: trainval 3,680 / test 3,669 (test untouched until Day 4)
- My split: train 3,128 / val 552, stratified by breed, seed 42
- Images per breed (trainval): min 93, max 100 → balanced, so accuracy is a fair metric

## Day 2: Frozen baseline
- EfficientNet-B0 (ImageNet weights), backbone frozen, new Linear(1280→37) head
- Adam lr 1e-3, batch 32, 5 epochs, no augmentation
- Trainable params: 47,397 of ~4.05M (~1%)
- Best val accuracy: **88.41%** (epoch 2), ~21 s/epoch on Colab T4

## Day 3: Fine-tuning
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

   ## Day 4: Test set (evaluated once)
   - Model: resnet50 | Test acc 92.45 % | Top-5 99.13 % | Macro F1 0.9239
   - Cats 87.91 % | Dogs 94.61 %
   - Baseline (frozen) test acc 84.71 % → fine-tuned 92.45 %  (+7.74 points)
   - Most confused: American Pit Bull Terrier → 	Staffordshire Bull Terrier (26 images), Egyptian Mau → Bengal (16 images)
