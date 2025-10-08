# dataset_anonymizer.py
"""
CacheFaceDataset

Usage:
    ds = CacheFaceDataset("cache_faces")                       # returns tensor only (default)
    ds = CacheFaceDataset("cache_faces", return_path=True)     # returns (tensor, path)
    ds = CacheFaceDataset("cache_faces", emb_map=emb_map_dict) # returns (tensor, emb) or (tensor, emb, path)

Notes:
 - Images are returned as torch.FloatTensor in range [-1, 1] (C, H, W).
 - emb_map (if provided) should be a dict mapping image-path-string -> numpy array embedding.
 - Matching of emb_map keys is done by exact path first, then basename fallback.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def make_default_transform(size: int = 256):
    """
    Returns a torchvision transform matching training pipeline:
      - Resize to (size, size)
      - Random horizontal flip
      - Small color jitter
      - ToTensor (0..1)
      - Normalize to (-1..1)
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
        transforms.ToTensor(),                    # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1,1]
    ])


class CacheFaceDataset(Dataset):
    """
    Dataset over cached aligned face images stored under cache_root/<person>/*.jpg

    Args:
      cache_root: root folder containing per-person subfolders with aligned face images.
      transform: torchvision transform to apply to PIL.Image. If None uses default transform (256).
      min_images_per_person: skip subfolders with fewer than this many images.
      return_path: if True, __getitem__ returns (img_tensor, sample_path) (string Path).
      emb_map: optional dict mapping sample_path string (or basename) -> numpy embedding array.
               If provided, __getitem__ returns (img_tensor, emb) or (img_tensor, emb, path)
               depending on return_path.
      shuffle: whether to shuffle sample list at init.
    """
    def __init__(self,
                 cache_root: str,
                 transform: Optional[transforms.Compose] = None,
                 min_images_per_person: int = 1,
                 return_path: bool = False,
                 emb_map: Optional[Dict[str, Any]] = None,
                 size: int = 256,
                 shuffle: bool = True):
        super().__init__()
        self.root = Path(cache_root)
        if transform is None:
            transform = make_default_transform(size=size)
        self.transform = transform
        self.min_images_per_person = int(min_images_per_person)
        self.return_path = bool(return_path)
        self.emb_map = emb_map  # dict path->np.array or basename->np.array
        self.size = size

        # collect samples
        self.samples: List[str] = []
        if not self.root.exists():
            raise FileNotFoundError(f"Cache root not found: {self.root}")
        for p in sorted(self.root.iterdir()):
            if not p.is_dir():
                continue
            imgs = sorted([str(x) for x in p.glob("*.jpg")] + [str(x) for x in p.glob("*.jpeg")] + [str(x) for x in p.glob("*.png")])
            if len(imgs) < self.min_images_per_person:
                continue
            self.samples.extend(imgs)

        if shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return f"<CacheFaceDataset root={self.root} samples={len(self.samples)} size={self.size} return_path={self.return_path}>"

    def _get_embedding_for_path(self, path: str) -> Optional[np.ndarray]:
        """Return embedding from emb_map by exact path or basename fallback, or None."""
        if self.emb_map is None:
            return None
        # exact match
        if path in self.emb_map:
            return np.asarray(self.emb_map[path], dtype=np.float32)
        # basename match
        bn = os.path.basename(path)
        if bn in self.emb_map:
            return np.asarray(self.emb_map[bn], dtype=np.float32)
        # try encoded key (slashes replaced)
        enc = path.replace(os.sep, "__")
        if enc in self.emb_map:
            return np.asarray(self.emb_map[enc], dtype=np.float32)
        # no match
        return None

    def __getitem__(self, idx: int):
        """
        Returns:
          - default: img_tensor (C,H,W) in [-1,1]
          - if emb_map provided and return_path False: (img_tensor, embedding_np)
          - if emb_map provided and return_path True: (img_tensor, embedding_np, path)
          - if emb_map is None and return_path True: (img_tensor, path)
        """
        sample_path = self.samples[idx]
        img = Image.open(sample_path).convert("RGB")
        img_t = self.transform(img)

        if self.emb_map is not None:
            emb = self._get_embedding_for_path(sample_path)
            if self.return_path:
                return img_t, emb, sample_path
            else:
                return img_t, emb
        else:
            if self.return_path:
                return img_t, sample_path
            return img_t

    def get_all_samples(self) -> List[str]:
        """Return list of sample paths."""
        return list(self.samples)

    def subset_by_person(self, person_name: str) -> "CacheFaceDataset":
        """Return a new dataset instance containing only images for a given person folder name."""
        sub = CacheFaceDataset(self.root, transform=self.transform, min_images_per_person=self.min_images_per_person,
                               return_path=self.return_path, emb_map=self.emb_map, size=self.size, shuffle=False)
        person_dir = self.root / person_name
        if not person_dir.exists():
            raise FileNotFoundError(f"Person folder not found: {person_dir}")
        samples = sorted([str(x) for x in person_dir.glob("*.jpg")] + [str(x) for x in person_dir.glob("*.png")])
        sub.samples = samples
        return sub
