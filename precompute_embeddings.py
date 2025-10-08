#!/usr/bin/env python3
"""
precompute_embeddings.py

Create an .npz file mapping dataset sample paths -> ArcFace embeddings.

Output .npz contains:
  - 'paths': array of shape (N,) dtype=object of exact sample paths (strings)
  - 'embeddings': array of shape (N, D) dtype=float32 such that embeddings[i] corresponds to paths[i]

Usage:
  python precompute_embeddings.py --cache-root cache_faces --out cache_faces/embeddings.npz --batch-size 64

Options:
  --arcface-onnx PATH   optionally provide ArcFace ONNX if you don't have insightface installed.
  --device DEVICE       'cuda' or 'cpu' (defaults to cuda if available)
  --fill-missing        if set, will fill missing/no-face embeddings with zeros (otherwise those entries are NaN)
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

def try_import_arcface_wrapper():
    try:
        from arcface_wrapper import load_arcface
        return load_arcface
    except Exception:
        return None

def try_import_insightface():
    try:
        from insightface.app import FaceAnalysis
        return FaceAnalysis
    except Exception:
        return None

def build_embed_fn(onnx_path=None, device='cpu'):
    """
    Return embed_fn(img_list) -> NxD numpy embeddings (L2-normalized),
    where img_list is a list/iterable of numpy uint8 HxWx3 (RGB) or PIL images.
    Tries in this order:
      - arcface_wrapper.load_arcface() if available
      - insightface FaceAnalysis if installed
      - if onnx_path provided, use onnxruntime ONNXArcFace (simple internal impl)
    """
    load_arcface = try_import_arcface_wrapper()
    if load_arcface is not None:
        # arcface_wrapper.load_arcface returns a callable embed_fn
        return load_arcface(onnx_path=onnx_path)

    FaceAnalysis = try_import_insightface()
    if FaceAnalysis is not None:
        # build an insightface-based embed_fn
        fa = FaceAnalysis(allowed_modules=['recognition'])
        # choose GPU if requested and available
        ctx_id = 0 if (device == 'cuda') else -1
        fa.prepare(ctx_id=ctx_id, det_size=(224,224))
        def embed_insight(imgs):
            # imgs: list of numpy HxWx3 uint8 or PIL images
            out = []
            for im in imgs:
                if not isinstance(im, (np.ndarray,)):
                    im = np.array(im)
                res = fa.get(img=im)
                if len(res) == 0:
                    # no face detected
                    out.append(np.zeros((512,), dtype=np.float32))
                else:
                    emb = res[0].embedding
                    emb = emb / (np.linalg.norm(emb) + 1e-10)
                    out.append(emb.astype(np.float32))
            return np.stack(out, axis=0)
        return embed_insight

    # fallback: try ONNXRuntime if onnx_path provided
    if onnx_path is not None and os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())
            input_name = sess.get_inputs()[0].name
            # read input size from model if possible
            inp_shape = sess.get_inputs()[0].shape  # e.g. [None, 3, 112, 112] or ['None', 3, 112, 112]
            # default preprocess to 112
            target_size = 112
            try:
                # find int dims
                for v in inp_shape:
                    if isinstance(v, int) and v > 0:
                        if v in (112, 112):
                            target_size = v
            except Exception:
                pass
            import cv2
            def embed_onnx(imgs):
                arrs = []
                for im in imgs:
                    if not isinstance(im, np.ndarray):
                        im = np.array(im)
                    im_res = cv2.resize(im, (target_size, target_size)).astype('float32') / 255.0
                    # many ArcFace ONNX expect BGR; if your ONNX expects RGB, remove the next line
                    im_res = im_res[..., ::-1]
                    arrs.append(np.transpose(im_res, (2,0,1))[None])
                inp = np.concatenate(arrs, axis=0).astype('float32')
                outs = sess.run(None, {input_name: inp})[0]  # NxD
                # L2 normalize
                norms = np.linalg.norm(outs, axis=1, keepdims=True) + 1e-10
                outs = outs / norms
                return outs.astype(np.float32)
            return embed_onnx
        except Exception as e:
            raise RuntimeError("Failed to use ONNXRuntime for arcface: " + str(e))

    raise RuntimeError("No ArcFace embedder available. Install insightface or provide --arcface-onnx and onnxruntime.")

def collect_sample_paths(cache_root: str):
    root = Path(cache_root)
    if not root.exists():
        raise FileNotFoundError("cache_root not found: " + cache_root)
    samples = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        imgs = sorted([str(x) for x in p.glob("*.jpg")] + [str(x) for x in p.glob("*.jpeg")] + [str(x) for x in p.glob("*.png")])
        samples.extend(imgs)
    return samples

def load_image_as_rgb(path):
    im = Image.open(path).convert("RGB")
    return np.array(im)

def main(args):
    device = args.device if args.device else ("cuda" if (torch_available_and_cuda()) else "cpu")
    print("Using device:", device)
    samples = collect_sample_paths(args.cache_root)
    print(f"Found {len(samples)} samples")

    embed_fn = build_embed_fn(onnx_path=args.arcface_onnx, device=device)
    print("Embedder ready. Running in batches of", args.batch_size)

    paths_out = []
    embs_out = []

    for i in tqdm(range(0, len(samples), args.batch_size), desc="Embedding batches"):
        batch_paths = samples[i:i+args.batch_size]
        imgs = [load_image_as_rgb(p) for p in batch_paths]
        try:
            embs = embed_fn(imgs)  # NxD numpy
        except Exception as e:
            # fallback: try to process images individually to isolate problem
            print("Batch embed failed:", e, "-> trying per-image fallback")
            embs = []
            for im in imgs:
                try:
                    emb = embed_fn([im])[0]
                except Exception:
                    emb = np.zeros((512,), dtype=np.float32)
                embs.append(emb)
            embs = np.stack(embs, axis=0)

        # ensure shape and dtype
        embs = np.asarray(embs, dtype=np.float32)
        for pth, e in zip(batch_paths, embs):
            paths_out.append(pth)
            embs_out.append(e)

    paths_arr = np.array(paths_out, dtype=object)  # object dtype preserves strings
    embs_arr = np.stack(embs_out, axis=0).astype(np.float32)  # NxD

    # handle missing embeddings: if user asked to fill missing, replace NaNs with zeros
    if args.fill_missing:
        # any NaNs -> fill 0
        embs_arr = np.nan_to_num(embs_arr, nan=0.0)

    # save as .npz with two arrays: paths and embeddings
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, paths=paths_arr, embeddings=embs_arr)
    print("Saved embeddings to", args.out)
    print("Paths shape:", paths_arr.shape, "Embeddings shape:", embs_arr.shape)

def torch_available_and_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-root', required=True, help='root of cache faces (cache_faces)')
    parser.add_argument('--out', default='cache_faces/embeddings.npz', help='output .npz file')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--arcface-onnx', default=None, help='optional ArcFace ONNX (if insightface not installed)')
    parser.add_argument('--device', default=None, help='cuda or cpu (default: auto)')
    parser.add_argument('--fill-missing', action='store_true', help='fill missing embeddings with zeros instead of NaN')
    args = parser.parse_args()
    main(args)
