"""GrainSet manifest adapters and datasets for pool-based active learning.

No labels are exposed by UnlabeledDataset, including by __getitem__.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

N_CLASSES = 8
CLASS_NAMES = ["NOR", "F&S", "SD", "MY", "AP", "BN", "UN", "IM"]


@dataclass(frozen=True)
class Manifest:
    paths: tuple[str, ...]
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.paths)


def read_manifest(data_root: str, split: str) -> Manifest:
    """Reads the same original rice_{split}.txt manifest as GrainSet reader.py.

    Independent parser prevents importing legacy datasets/__init__.py, which
    eagerly imports Albumentations even when no augmentor is requested.
    """
    root = Path(data_root).expanduser().resolve()
    name = f"{root.name}_{split}.txt"
    project_root = Path(__file__).resolve().parents[2]
    candidates = [project_root/'datasets'/'datalist'/name,
                  project_root/'runs'/'datalist'/name]
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"Datalist {name} not found. Searched: {candidates}")
    paths, labels = [], []
    for lineno, line in enumerate(source.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Malformed manifest line {lineno} in {source}: {line}")
        rel, label = parts
        paths.append(str(root/rel))
        labels.append(int(label))
    labels = np.asarray(labels, dtype=np.int64)
    if not paths:
        raise ValueError(f"Empty {split} manifest")
    if not np.all((labels >= 0) & (labels < N_CLASSES)):
        raise ValueError(f"Unexpected labels in {split}: {np.unique(labels)}")
    if len(set(paths)) != len(paths):
        raise ValueError(f"Duplicate paths in {split} manifest")
    print(f"* {split}: {len(paths)} ({source})")
    return Manifest(tuple(paths), labels)


def verify_manifests(train: Manifest, val: Manifest, test: Manifest, check_files=True) -> None:
    sets = [set(m.paths) for m in (train, val, test)]
    for a, b, desc in [(0,1,"train/val"),(0,2,"train/test"),(1,2,"val/test")]:
        common = sets[a] & sets[b]
        if common:
            raise ValueError(f"Exact duplicate file paths across {desc}: {list(common)[:3]}")
    if check_files:
        for name, manifest in (("train",train),("val",val),("test",test)):
            missing = [p for p in manifest.paths if not Path(p).is_file()]
            if missing:
                raise FileNotFoundError(f"{name}: {len(missing)} image files missing. First: {missing[:3]}. "
                                        "The uploaded ZIP has code/manifests only; point --data-root to extracted images.")


def make_transforms(image_size=224):
    """RGB + ImageNet normalization for pretrained ResNet50, not legacy BGR stats."""
    norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
        T.ToTensor(), norm,
    ])
    eval_ = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), norm])
    return train, eval_


class LabeledDataset(Dataset):
    def __init__(self, manifest: Manifest, indices: Sequence[int], transform):
        self.manifest = manifest
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i]); path = self.manifest.paths[idx]
        with Image.open(path) as im:
            image = self.transform(im.convert('RGB'))
        return image, int(self.manifest.labels[idx]), idx


class UnlabeledDataset(Dataset):
    """Important: acquisitions cannot access hidden labels through this dataset."""
    def __init__(self, paths: tuple[str, ...], indices: Sequence[int], transform):
        self.paths = paths
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        with Image.open(self.paths[idx]) as im:
            image = self.transform(im.convert('RGB'))
        return image, idx


def complement(n: int, selected: Sequence[int]) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    mask[np.asarray(selected, dtype=np.int64)] = False
    return np.flatnonzero(mask)


def budgets(n: int, start=2, stop=50, step=2) -> list[tuple[int, int]]:
    if not (0 < start <= stop <= 100 and step > 0):
        raise ValueError("Expected 0 < start <= stop <= 100, step > 0")
    if (stop-start) % step != 0:
        raise ValueError("(stop-start) must be divisible by step")
    return [(p, round(n * p / 100)) for p in range(start, stop + 1, step)]
