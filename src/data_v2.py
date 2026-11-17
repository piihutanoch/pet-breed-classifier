from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from src.data import get_eval_transform, get_train_transform   # same preprocessing as v1


class ManifestDataset(Dataset):
    """Reads (photo, label) pairs from the rows of manifest.csv."""

    def __init__(self, rows, roots, transform):
        self.paths = [Path(roots[s]) / p for s, p in zip(rows["source"], rows["path"])]
        self.labels = rows["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")   # some photos are greyscale/RGBA/CMYK
        return self.transform(img), self.labels[i]


def load_manifest(artifacts_dir):
    return pd.read_csv(Path(artifacts_dir) / "manifest.csv")


def get_loaders_v2(manifest, roots, batch_size=32, num_workers=2):
    """Returns train, val and test loaders. Only train uses augmentation."""
    def make(split, transform, shuffle):
        rows = manifest[manifest["split"] == split]
        ds = ManifestDataset(rows, roots, transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return (make("train", get_train_transform(), True),
            make("val",   get_eval_transform(),  False),
            make("test",  get_eval_transform(),  False))
