# face_cache.py
"""
Face caching utilities for LFW pipeline.

Usage (CLI):
  # Prepare cache for whole dataset (recurses person subfolders)
  python face_cache.py --data-root /path/to/lfw --cache-root /path/to/cache_faces --force --image-size 256

  # Inspect a single source image and write debug images to /tmp
  python face_cache.py --inspect /path/to/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg

Functions to import:
  - gather_lfw_image_paths(root) -> List[str]
  - cache_aligned_faces(image_paths, out_root, data_root, image_size=256, force=False, device=None, mtcnn=None)
  - inspect_face_tensor(original_image_path, out_dir="/tmp", device=None, mtcnn=None)
"""
from pathlib import Path
import os
import glob
import argparse
from typing import List, Optional, Dict, Tuple

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN

# ---------- Device / singleton MTCNN ----------
_MTNN_SINGLETON = None

def get_mtcnn(image_size: int = 256,
              margin: int = 14,
              keep_all: bool = False,
              post_process: bool = True,
              device: Optional[torch.device] = None) -> MTCNN:
    """
    Return a singleton MTCNN instance for the given device.
    """
    global _MTNN_SINGLETON
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # reuse the same instance if already created for same device & size (simple reuse)
    if _MTNN_SINGLETON is None:
        _MTNN_SINGLETON = MTCNN(image_size=image_size, margin=margin, keep_all=keep_all, post_process=post_process, device=device)
    return _MTNN_SINGLETON

# ---------- Helpers ----------
def gather_lfw_image_paths(root: str) -> List[str]:
    """
    Gather image file paths under `root/<person>/*.jpg|jpeg|png`.
    Returns sorted list.
    """
    patterns = [os.path.join(root, "*", "*.jpg"),
                os.path.join(root, "*", "*.JPG"),
                os.path.join(root, "*", "*.jpeg"),
                os.path.join(root, "*", "*.png")]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat))
    return sorted(paths)

# ---------- Diagnostic ----------
def inspect_face_tensor(original_image_path: str,
                        out_dir: str = "/tmp",
                        device: Optional[torch.device] = None,
                        mtcnn: Optional[MTCNN] = None) -> Optional[Dict]:
    """
    Run MTCNN on one original image, print tensor stats and save debug images:
      - debug_face_saved_by_ToPILImage.jpg
      - debug_face_saved_corrected.jpg
    Returns dictionary with stats or None if detection failed.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mtcnn is None:
        mtcnn = get_mtcnn(device=device)
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    try:
        img = Image.open(original_image_path).convert("RGB")
    except Exception as e:
        print("Failed to open", original_image_path, e)
        return None

    face = mtcnn(img)  # tensor on DEVICE or None
    print("mtcnn returned:", type(face))
    if face is None:
        print("No face detected.")
        return None

    face_cpu = face.cpu()
    print("face_cpu.shape:", face_cpu.shape, "dtype:", face_cpu.dtype)
    mn = float(face_cpu.min()); mx = float(face_cpu.max()); mean = float(face_cpu.mean())
    print(f"face tensor stats: min={mn:.6f}, max={mx:.6f}, mean={mean:.6f}")

    # Save via ToPILImage
    try:
        from torchvision.transforms import ToPILImage
        pil_via_to_pil = ToPILImage()(face_cpu)
        p1 = os.path.join(out_dir, "debug_face_saved_by_ToPILImage.jpg")
        pil_via_to_pil.save(p1, quality=95)
        print("Saved (ToPILImage):", p1)
    except Exception as e:
        p1 = None
        print("ToPILImage save failed:", e)

    # Save corrected version (explicit normalization -> uint8)
    arr = face_cpu.permute(1,2,0).numpy().astype(np.float32)  # H,W,3
    if arr.min() < -0.5 and arr.max() <= 1.5:
        arr = (arr + 1.0) / 2.0
        print("Detected [-1,1] range -> converting to [0,1].")
    elif arr.max() > 50:
        arr = arr / 255.0
        print("Detected large values -> scaling down by 255.")
    arr = np.clip(arr, 0.0, 1.0)
    out_img = (arr * 255.0).astype("uint8")
    p2 = os.path.join(out_dir, "debug_face_saved_corrected.jpg")
    Image.fromarray(out_img).save(p2, quality=95)
    print("Saved (corrected):", p2)

    return {"tensor_min": mn, "tensor_max": mx, "tensor_mean": mean, "pil_path": p1, "corrected_path": p2}

# ---------- Robust caching ----------
def cache_aligned_faces(image_paths: List[str],
                        out_root: str,
                        data_root: str,
                        image_size: int = 256,
                        force: bool = False,
                        device: Optional[torch.device] = None,
                        mtcnn: Optional[MTCNN] = None) -> None:
    """
    Detect+align faces and save to out_root preserving relative structure.
    Handles tensors in ranges [-1,1], [0,1], or 0..255 safely.

    Params:
      - image_paths: list of absolute input image paths
      - out_root: directory to save aligned faces (preserves rel path to data_root)
      - data_root: root directory of originals (to compute relative paths)
      - image_size: aligned face size (passed to MTCNN)
      - force: overwrite existing cached images
      - device / mtcnn: optional for custom device or detector
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mtcnn is None:
        mtcnn = get_mtcnn(image_size=image_size, device=device)

    os.makedirs(out_root, exist_ok=True)
    for p in tqdm(image_paths, desc="Caching faces"):
        try:
            rel = os.path.relpath(p, data_root)
        except Exception:
            rel = os.path.basename(p)
        out_p = os.path.join(out_root, rel)
        out_dir = os.path.dirname(out_p)
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(out_p) and not force:
            continue

        # open original
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            # unable to open; skip
            print("Skipping (open error):", p, e)
            continue

        # MTCNN detect+align -> tensor on device or None
        face = mtcnn(img)
        if face is None:
            # optionally: save original or blank placeholder
            continue

        face_cpu = face.cpu()
        # ensure (3,H,W)
        if face_cpu.ndim == 4 and face_cpu.shape[0] == 1:
            face_cpu = face_cpu.squeeze(0)
        if face_cpu.ndim != 3 or face_cpu.shape[0] != 3:
            print("Unexpected face tensor shape:", face_cpu.shape, "skipping:", p)
            continue

        arr = face_cpu.permute(1,2,0).numpy().astype(np.float32)  # H,W,3
        mn, mx = float(arr.min()), float(arr.max())

        # handle typical ranges:
        if mn < -0.5 and mx <= 1.5:
            # normalized [-1,1]
            arr = (arr + 1.0) / 2.0
        elif mx > 50:
            # likely floats in 0..255
            arr = arr / 255.0

        arr = np.clip(arr, 0.0, 1.0)
        out_img = (arr * 255.0).astype("uint8")  # H,W,3 RGB uint8
        try:
            Image.fromarray(out_img).save(out_p, quality=95)
        except Exception as e:
            print("Failed saving", out_p, e)
            continue

# ---------- CLI entrypoint ----------
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", help="Root folder for source images (LFW)", required=False)
    p.add_argument("--cache-root", help="Where to save cached aligned faces", required=False)
    p.add_argument("--force", action="store_true", help="Overwrite existing cached images")
    p.add_argument("--image-size", type=int, default=256, help="MTCNN crop size")
    p.add_argument("--inspect", type=str, default="", help="Inspect single original image path and write debug images to /tmp")
    return p.parse_args()

def main_cli():
    args = _parse_args()
    if args.inspect:
        print("Running inspect on:", args.inspect)
        res = inspect_face_tensor(args.inspect, out_dir="/tmp", device=None)
        print("Inspect result:", res)
        return

    if not args.data_root or not args.cache_root:
        print("For caching, both --data-root and --cache-root are required. Use --inspect to debug single image.")
        return

    all_paths = gather_lfw_image_paths(args.data_root)
    print(f"Found {len(all_paths)} images under {args.data_root}")
    cache_aligned_faces(all_paths, args.cache_root, data_root=args.data_root, image_size=args.image_size, force=args.force)
    print("Done caching.")

if __name__ == "__main__":
    main_cli()
