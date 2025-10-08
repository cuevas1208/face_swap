#!/usr/bin/env python3
"""
inference.py

Face-swap / anonymize inference tool.

Note: load_generator_checkpoint now prefers 'ema_G' weights in a checkpoint if present.
"""

import os
import sys
import argparse
import math
import time
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F

# Try to import insightface/arcface wrapper for detection & embedding
try:
    from arcface_wrapper import load_arcface
except Exception:
    load_arcface = None

try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None

# Import model constructor if available in repo
try:
    from models_anonymizer import GeneratorUNet
except Exception:
    GeneratorUNet = None

# -------------------------
# Utilities
# -------------------------
def to_uint8_image(img):
    """
    Convert a numpy image-like (tensor or array) to HxWx3 uint8 RGB suitable for PIL / cv2.
    Accepts:
      - torch.Tensor CxHxW or HxWxC in ranges [-1,1], [0,1], [0,255]
      - numpy arrays in ranges [-1,1], [0,1], [0,255]
    Returns uint8 HxWx3
    """
    # Torch tensor
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
        # if (C,H,W)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1,2,0))
    else:
        arr = np.array(img)

    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr]*3, axis=2)

    # detect ranges
    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    if mn >= 0.0 and mx <= 1.0:
        arr = (arr * 255.0).astype(np.uint8)
    elif mn >= -1.0 and mx <= 1.0:
        arr = ((arr + 1.0) * 127.5).astype(np.uint8)
    elif arr.dtype != np.uint8:
        # assume already 0..255 but float
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def pil_from_any(img):
    """Return PIL Image from numpy/torch input, ensure RGB"""
    arr = to_uint8_image(img)
    return Image.fromarray(arr)

def bbox_iou(boxA, boxB):
    # boxes in x1,y1,x2,y2
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    if areaA + areaB - inter <= 0:
        return 0.0
    return inter / float(areaA + areaB - inter)

def box_from_xywh(x,y,w,h, imw=None, imh=None):
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    if imw is not None and imh is not None:
        x1 = max(0, min(x1, imw-1))
        y1 = max(0, min(y1, imh-1))
        x2 = max(0, min(x2, imw-1))
        y2 = max(0, min(y2, imh-1))
    return [x1, y1, x2, y2]

def crop_box(img, box):
    x1,y1,x2,y2 = box
    return img[y1:y2, x1:x2].copy()

def resize_preserve_aspect(img, target_h, target_w):
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

def feather_blend(src_img, dst_img, box, strength=0.5):
    """
    Simple alpha-feather blending. src_img & dst_img are HxWx3 uint8, box is region on dst to blend into.
    src_img is assumed to be same size as box.
    """
    x1,y1,x2,y2 = box
    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        return dst_img
    # create alpha mask with smooth circular-ish falloff
    kernel = max(3, int(min(h,w)/8))
    mask = np.ones((h,w), dtype=np.float32)
    # distance from center
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h/2, w/2
    dist = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    maxd = np.sqrt((cy)**2 + (cx)**2)
    alpha = (1.0 - (dist / (maxd + 1e-9))) ** 2
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = alpha * strength
    alpha = alpha[..., None]
    dst = dst_img.copy().astype(np.float32)
    src = src_img.astype(np.float32)
    dst[y1:y2, x1:x2] = (alpha * src + (1-alpha) * dst[y1:y2, x1:x2])
    dst = np.clip(dst, 0, 255).astype(np.uint8)
    return dst

def poisson_blend(src_img, dst_img, box):
    """
    Use OpenCV seamlessClone. src_img and dst_img uint8. box region is where to place center.
    """
    x1,y1,x2,y2 = box
    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        return dst_img
    src_resized = resize_preserve_aspect(src_img, h, w)
    center = (int((x1+x2)/2), int((y1+y2)/2))
    mask = 255 * np.ones((src_resized.shape[0], src_resized.shape[1], 3), src_resized.dtype)
    try:
        out = cv2.seamlessClone(src_resized, dst_img, mask, center, cv2.NORMAL_CLONE)
        return out
    except Exception:
        # fallback to simple replace
        out = dst_img.copy()
        out[y1:y2, x1:x2] = src_resized
        return out

# -------------------------
# Face detector & tracker
# -------------------------
class FaceDetector:
    def __init__(self, device='cpu', det_size=(640,640)):
        self.device = device
        self.det_size = det_size
        self.fa = None
        if FaceAnalysis is not None:
            try:
                self.fa = FaceAnalysis(allowed_modules=['detection','landmark'])
                # choose GPU if available
                ctx_id = 0 if (device=='cuda') else -1
                self.fa.prepare(ctx_id=ctx_id, det_size=self.det_size)
                print("InsightFace FaceAnalysis initialized (detection).")
            except Exception as e:
                print("InsightFace init failed:", e)
                self.fa = None
        if self.fa is None:
            # fallback: Haar cascade (fast, less accurate)
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(haar_path):
                self.haar = cv2.CascadeClassifier(haar_path)
                print("Using Haar cascade for face detection fallback.")
            else:
                self.haar = None
                print("No face detector available! Install insightface or ensure Haar cascades exist.")

    def detect(self, img_rgb):
        """
        img_rgb: HxWx3 uint8 RGB
        returns list of dicts: {'bbox': [x1,y1,x2,y2], 'kps': np.array Nx2 (if available)}
        """
        H,W,_ = img_rgb.shape
        out = []
        if self.fa is not None:
            try:
                res = self.fa.get(img=img_rgb)
                for r in res:
                    x1,y1,x2,y2 = int(r.bbox[0]), int(r.bbox[1]), int(r.bbox[2]), int(r.bbox[3])
                    kps = None
                    if hasattr(r, 'kps') and r.kps is not None:
                        kps = np.asarray(r.kps)
                    out.append({'bbox':[x1,y1,x2,y2], 'kps':kps})
                return out
            except Exception as e:
                print("InsightFace detection error:", e)
        # Haar fallback (expects grayscale)
        if self.haar is not None:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            rects = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
            for (x,y,w,h) in rects:
                out.append({'bbox': box_from_xywh(x,y,w,h, W, H), 'kps': None})
            return out
        return out

class IoUTracker:
    """
    Very simple IoU tracker: stores last bounding boxes per id and matches by IoU threshold.
    Smoothing: optional EMA on latent vectors per track.
    """
    def __init__(self, iou_thresh=0.4, max_age=10, smoothing=0.0):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.smoothing = smoothing
        self.tracks = {}  # id -> {'box':..., 'age':0, 'last_seen':frame_idx, 'latent': np.array}
        self.next_id = 1

    def update(self, detections, frame_idx, latents=None):
        """
        detections: list of bbox [x1,y1,x2,y2]
        latents: list of np arrays same length or None
        Returns: list of assigned ids aligned with detections list
        """
        assigned_ids = [-1] * len(detections)
        used_track_ids = set()
        # match by IoU greedy
        for i, box in enumerate(detections):
            best_id = None
            best_iou = 0.0
            for tid, t in self.tracks.items():
                iou = bbox_iou(box, t['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
            if best_iou >= self.iou_thresh and best_id is not None and best_id not in used_track_ids:
                assigned_ids[i] = best_id
                used_track_ids.add(best_id)
                # update track
                self.tracks[best_id]['box'] = box
                self.tracks[best_id]['age'] = 0
                self.tracks[best_id]['last_seen'] = frame_idx
                if latents is not None and latents[i] is not None:
                    if self.tracks[best_id].get('latent') is None:
                        self.tracks[best_id]['latent'] = latents[i].copy()
                    else:
                        if self.smoothing > 0:
                            self.tracks[best_id]['latent'] = (self.smoothing * self.tracks[best_id]['latent'] +
                                                              (1.0-self.smoothing) * latents[i])
                        else:
                            self.tracks[best_id]['latent'] = latents[i].copy()
            else:
                # new track
                nid = self.next_id
                self.next_id += 1
                assigned_ids[i] = nid
                self.tracks[nid] = {'box': box, 'age': 0, 'last_seen': frame_idx, 'latent': (latents[i].copy() if (latents is not None and latents[i] is not None) else None)}
                used_track_ids.add(nid)
        # age tracks and remove old
        to_del = []
        for tid, t in self.tracks.items():
            if frame_idx - t['last_seen'] > self.max_age:
                to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]
        return assigned_ids

# -------------------------
# Model loader & inference helper
# -------------------------
def load_generator_checkpoint(path, device):
    """
    Load checkpoint into GeneratorUNet (if available) or try to instantiate a model from checkpoint.
    Returns model (in eval mode) on device.
    """
    if not os.path.exists(path):
        raise FileNotFoundError("checkpoint not found: " + path)
    ck = torch.load(path, map_location='cpu')
    model = None
    if GeneratorUNet is not None:
        try:
            # try instantiate with default params; if your model signature differs update here
            model = GeneratorUNet(in_ch=3, base=64, bottleneck_dim=512)
            # Prefer EMA weights if provided in checkpoint
            state_dict_to_load = None
            if isinstance(ck, dict) and 'ema_G' in ck:
                state_dict_to_load = ck['ema_G']
                print("Loaded ema_G weights from checkpoint for inference.")
            elif isinstance(ck, dict) and 'G' in ck:
                state_dict_to_load = ck['G']
            else:
                # ck may itself be a state_dict
                state_dict_to_load = ck
            model.load_state_dict(state_dict_to_load)
            model.to(device)
            model.eval()
            print("Loaded GeneratorUNet checkpoint.")
            return model
        except Exception as e:
            print("GeneratorUNet load failed:", e)
            model = None
    # fallback: try to load full torchscript or module
    try:
        # try torch.jit
        scripted = torch.jit.load(path, map_location=device)
        scripted.eval()
        print("Loaded scripted model from checkpoint.")
        return scripted
    except Exception:
        pass
    # as last resort, try to interpret ck as a state dict for a model you may not have
    raise RuntimeError("Could not create generator model from checkpoint. Adapt load_generator_checkpoint for your model type.")
