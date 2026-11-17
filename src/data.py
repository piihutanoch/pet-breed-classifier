
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Subset, DataLoader

# ImageNet RGB channel statistics for normalisation
IMAGENET_MEAN= [0.485, 0.456, 0.406]
IMAGENET_STD= [0.229, 0.224, 0.225]


def get_eval_transform():
    """Deterministic preprocessing for validation, testing and inference."""
    return transforms.Compose([
        transforms.Resize(256),       # Resize shorter side while preserving aspect ratio
        transforms.CenterCrop(224),  # Crop a fixed 224x224 region from the centre
        transforms.ToTensor(),       # Convert image to tensor and scale pixels to [0, 1]
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_train_transform():
    """Apply random augmentations to training images."""
    return transforms.Compose([
        # Randomly crop and resize to introduce variation in image framing
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),

        # Flip images horizontally with a 50% probability
        transforms.RandomHorizontalFlip(),

        # Introduce variation in brightness, contrast and saturation
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2
        ),

        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_loaders(data_dir, train_idx, val_idx, batch_size=32):

    # Create dataset objects with separate training and evaluation transforms
    train_base= OxfordIIITPet(
        root=data_dir,
        split="trainval",
        target_types="category",
        transform=get_train_transform()
    )

    val_base= OxfordIIITPet(
        root=data_dir,
        split="trainval",
        target_types="category",
        transform=get_eval_transform()
    )

    # Select the previously saved training and validation indices
    train_ds= Subset(train_base, train_idx)
    val_ds= Subset(val_base, val_idx)

    # Create batches; shuffle training data but keep validation order fixed
    train_loader= DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=2
    )

    val_loader= DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=2
    )

    return train_loader, val_loader
