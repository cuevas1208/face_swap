# trainer_helpers.py
"""
Helper utilities for trainer_anonymizer.py

Updates:
 - checkpoint helpers
 - robust embeddings loader
 - landmark helpers
 - landmark-loss batch helper
 - spectral norm helper
 - EMA helpers
"""

import os
import glob
import copy
import numpy as np
import torch
import torch.nn.functional as F

# -----------------------
# Checkpoint helpers
# -----------------------
def save_checkpoint(state, save_dir, it, keep_latest=True):
    os.makedirs(save_dir, exist_ok=True)
    path_iter = os.path.join(save_dir, f'ckpt_iter_{it}.pth')
    torch.save(state, path_iter)
    if keep_latest:
        latest = os.path.join(save_dir, 'latest.pth')
        torch.save(state, latest)
    return path_iter

def find_latest_checkpoint(save_dir):
    latest = os.path.join(save_dir, 'latest.pth')
    if os.path.exists(latest):
        return latest
    pat = os.path.join(save_dir, 'ckpt_iter_*.pth')
    files = glob.glob(pat)
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]

# -----------------------
# Embedding loader (robust)
# -----------------------
def load_embeddings_map(emb_path, dataset_samples):
    """
    Robust loader for embeddings saved as .npz / .npz-like files.

    Supported formats:
      - .npz with keys 'paths' (array of strings) and 'embeddings' (NxD array)
      - .npz with many named arrays where keys may be basenames or encoded paths
      - single-array .npy / .npz where length matches number of dataset samples (maps by order)

    Returns:
      dict mapping exact dataset sample path -> numpy array embedding (float32),
      or None if file missing / not matched.
    """
    if emb_path is None:
        return None
    if not os.path.exists(emb_path):
        print(f"[embeddings] file not found: {emb_path}")
        return None

    try:
        data = np.load(emb_path, allow_pickle=True)
    except Exception as e:
        print("[embeddings] failed to load:", e)
        return None

    emb_map = {}

    # Case A: data has 'paths' and 'embeddings'
    if 'paths' in data.files and 'embeddings' in data.files:
        paths = data['paths']
        embs = data['embeddings']
        for i, p in enumerate(paths):
            pstr = str(p)
            emb_map[pstr] = np.asarray(embs[i], dtype=np.float32)
        print(f"[embeddings] loaded {len(emb_map)} entries from 'paths'+'embeddings'")
        return emb_map

    # Case B: one single array that matches dataset length -> map by order
    if len(data.files) == 1:
        arr = data[data.files[0]]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] == len(dataset_samples):
            print("[embeddings] single-array file matched by dataset order")
            for i, p in enumerate(dataset_samples):
                emb_map[p] = np.asarray(arr[i], dtype=np.float32)
            return emb_map

    # Case C: multiple keys - try to match keys to sample paths or basenames
    keys = list(data.files)
    basename_to_key = {}
    for k in keys:
        bn = os.path.basename(k)
        if bn not in basename_to_key:
            basename_to_key[bn] = k

    for sample in dataset_samples:
        # exact match by sample path as key
        if sample in data:
            emb_map[sample] = np.asarray(data[sample], dtype=np.float32)
            continue
        # encoded form (slashes replaced by "__")
        enc = sample.replace(os.sep, "__")
        if enc in data:
            emb_map[sample] = np.asarray(data[enc], dtype=np.float32)
            continue
        # basename match
        bn = os.path.basename(sample)
        if bn in data:
            emb_map[sample] = np.asarray(data[bn], dtype=np.float32)
            continue
        if bn in basename_to_key:
            key = basename_to_key[bn]
            emb_map[sample] = np.asarray(data[key], dtype=np.float32)
            continue
        # no match -> will be computed on the fly by trainer
    print(f"[embeddings] matched {len(emb_map)} / {len(dataset_samples)} samples (fallbacks attempted)")
    return emb_map if len(emb_map) > 0 else None

# -----------------------
# Landmark helpers
# -----------------------
def landmarks_from_face_alignment(img_np, fa_inst):
    """img_np: HxWx3 uint8 RGB -> return Nx2 float32 or None"""
    try:
        res = fa_inst.get_landmarks_from_image(img_np)
        if res is None or len(res) == 0:
            return None
        return res[0].astype(np.float32)
    except Exception:
        return None

def landmarks_from_mediapipe(img_np, detector):
    """img_np: HxWx3 uint8 RGB -> return Nx2 float32 or None"""
    try:
        results = detector.process(img_np)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0]
        h, w, _ = img_np.shape
        pts = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)
        return pts
    except Exception:
        return None

def compute_landmark_loss_batch(real_imgs, fake_imgs, device, fa_inst=None, mp_detector=None, use_fa=False):
    """
    real_imgs & fake_imgs: tensors in [-1,1], shape (B,3,H,W)
    returns: scalar torch tensor (average MSE across detected landmarks)
    """
    B = real_imgs.size(0)
    loss = torch.tensor(0.0, device=device)
    count = 0
    real_np_list = [((t.detach().cpu().permute(1,2,0).numpy()+1.0)*127.5).astype(np.uint8) for t in real_imgs]
    fake_np_list = [((t.detach().cpu().permute(1,2,0).numpy()+1.0)*127.5).astype(np.uint8) for t in fake_imgs]
    for real_np, fake_np in zip(real_np_list, fake_np_list):
        if use_fa and fa_inst is not None:
            lr = landmarks_from_face_alignment(real_np, fa_inst)
            lf = landmarks_from_face_alignment(fake_np, fa_inst)
        elif mp_detector is not None:
            lr = landmarks_from_mediapipe(real_np, mp_detector)
            lf = landmarks_from_mediapipe(fake_np, mp_detector)
        else:
            lr = None; lf = None
        if lr is None or lf is None:
            continue
        if lr.shape[0] != lf.shape[0]:
            continue
        lr_t = torch.from_numpy(lr).to(device).float()
        lf_t = torch.from_numpy(lf).to(device).float()
        loss = loss + F.mse_loss(lf_t, lr_t)
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=device)
    return loss / count

# -----------------------
# spectral norm helper
# -----------------------
def apply_spectral_norm(module):
    """
    Recursively apply spectral norm to Conv2d and Linear layers in module.
    """
    import torch.nn as nn
    for name, m in module.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            try:
                torch.nn.utils.spectral_norm(m)
            except Exception:
                # if already applied or fails, ignore
                pass

# -----------------------
# EMA helpers
# -----------------------
def init_ema(model):
    """
    Return a deep-copied EMA model (same architecture).
    """
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model

def ema_update(model, ema_model, decay):
    """
    Exponential moving average update: ema = decay * ema + (1-decay) * model
    """
    with torch.no_grad():
        m_params = list(model.parameters())
        e_params = list(ema_model.parameters())
        for p, e in zip(m_params, e_params):
            e.data.mul_(decay).add_(p.data, alpha=(1.0 - decay))
