#!/usr/bin/env python3
"""
run_anonymizer_with_ui.py

Features:
 - CLI usage for image / folder / video anonymization with IoU face tracking and temporal smoothing.
 - Streamlit UI for drag-and-drop image/video anonymization (upload checkpoint in UI or supply one via CLI before launching Streamlit).

CLI examples:
  python run_anonymizer_with_ui.py --checkpoint anonymizer.pth --image in.jpg --out out.jpg
  python run_anonymizer_with_ui.py --checkpoint anonymizer.pth --video in.mp4 --out out_anon.mp4 --smooth

Streamlit:
  streamlit run run_anonymizer_with_ui.py
  -> Upload checkpoint in the web UI, then upload an image or video and press "Process"

Notes:
 - The script assumes you have an Anonymizer class defined (same as in anonymize_lfw.py). If you keep Anonymizer in a separate module, adjust import accordingly.
 - This file contains a simple IoU-based tracker (no Kalman/optical flow). It matches detections across frames by IoU and assigns stable track IDs for smoothing.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import tempfile
import time
import math

import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

import torch
from torchvision import transforms
from facenet_pytorch import MTCNN

# ---- IMPORT YOUR Anonymizer CLASS HERE ----
# If your Anonymizer class is in anonymize_lfw.py (same folder), import like:
try:
    from anonymize_lfw import Anonymizer
except Exception:
    # fallback: if anonymizer class is not found, raise a clear error
    raise ImportError("Could not import Anonymizer. Make sure anonymize_lfw.py with class Anonymizer is in the same folder.")

# ---- Device ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Utilities ----
def load_anonymizer(checkpoint_path: str, device=DEVICE):
    model = Anonymizer().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    # try common state formats
    if isinstance(state, dict) and any(k.startswith("enc1") or k.startswith("fc") for k in state.keys()):
        model.load_state_dict(state)
    elif isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model

def tensor_to_image(tensor):
    t = tensor.clamp(-1,1).detach().cpu()
    img = ((t + 1.0) * 127.5).permute(1,2,0).numpy().astype(np.uint8)
    return img  # H,W,3 RGB

# ---- IoU Tracker ----
def iou_xyxy(b1, b2) -> float:
    # boxes as [x1,y1,x2,y2]
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
    area2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union

class Track:
    def __init__(self, tid:int, bbox:Tuple[int,int,int,int], frame_idx:int):
        self.id = tid
        self.bbox = bbox  # latest bbox (x1,y1,x2,y2)
        self.last_seen = frame_idx
        self.age = 0
        self.hits = 1
        # store last anonymized crop for smoothing
        self.prev_crop = None

    def update(self, bbox:Tuple[int,int,int,int], frame_idx:int):
        self.bbox = bbox
        self.last_seen = frame_idx
        self.hits += 1

class IoUTracker:
    def __init__(self, iou_threshold:float=0.35, max_age:int=10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, Track] = {}
        self._next_id = 0

    def reset(self):
        self.tracks = {}
        self._next_id = 0

    def step(self, detections: List[Tuple[int,int,int,int]], frame_idx:int) -> Dict[int, Track]:
        """
        detections: list of [x1,y1,x2,y2]
        returns mapping track_id -> Track (updated)
        """
        assigned = {}
        if len(self.tracks) == 0:
            # create new track for every detection
            for det in detections:
                t = Track(self._next_id, det, frame_idx)
                self.tracks[self._next_id] = t
                self._next_id += 1
            return dict(self.tracks)

        # compute IoU between detections and existing tracks
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=float)
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i,j] = iou_xyxy(self.tracks[tid].bbox, det)

        # greedy match: largest iou first
        used_tracks = set()
        used_dets = set()
        pairs = []
        # flatten sorted by iou desc
        idxs = np.dstack(np.unravel_index(np.argsort(-iou_matrix.ravel()), iou_matrix.shape))[0]
        for (ti, di) in idxs:
            if ti in used_tracks or di in used_dets:
                continue
            if iou_matrix[ti,di] < self.iou_threshold:
                continue
            tid = track_ids[ti]
            self.tracks[tid].update(detections[di], frame_idx)
            used_tracks.add(ti); used_dets.add(di)
            pairs.append((tid, di))

        # unmatched detections -> new tracks
        for j, det in enumerate(detections):
            if j in used_dets: continue
            t = Track(self._next_id, det, frame_idx)
            self.tracks[self._next_id] = t
            self._next_id += 1

        # age and remove stale tracks
        remove_ids = []
        for tid, tr in list(self.tracks.items()):
            if frame_idx - tr.last_seen > self.max_age:
                remove_ids.append(tid)
        for tid in remove_ids:
            del self.tracks[tid]

        return dict(self.tracks)

# ---- Anonymize + blend that uses tracker for smoothing ----
def anonymize_and_blend_with_tracker(pil_img: Image.Image, model, mtcnn, transform, tracker:IoUTracker, frame_idx:int, smooth:bool=True, alpha:float=0.75):
    """
    Detect faces, assign track IDs using IoUTracker, anonymize per-face, apply per-track EMA smoothing in pixel space,
    blend back to original using seamlessClone with ellipse mask.
    Returns: out_bgr, list of current tracks (copy)
    """
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    with torch.no_grad():
        faces = mtcnn(pil_img)   # aligned crops (tensor(s) on device) or None
        boxes, probs = mtcnn.detect(pil_img)  # boxes numpy (N,4) or (None)
    if faces is None or boxes is None:
        # update tracker with zero detections to age tracks
        tracker.step([], frame_idx)
        return img_cv, tracker.tracks

    # ensure faces/list aligned with boxes: facenet-pytorch provides aligned crops in same order as boxes
    if isinstance(faces, torch.Tensor):
        faces = [faces]

    detections = []
    for box in boxes:
        # cast to ints
        x1,y1,x2,y2 = [int(v) for v in box]
        # expand slightly for better blending
        pad = int(0.12 * max(x2-x1, y2-y1))
        x1 = max(0, x1-pad); y1 = max(0, y1-pad)
        x2 = min(img_cv.shape[1], x2+pad); y2 = min(img_cv.shape[0], y2+pad)
        detections.append((x1,y1,x2,y2))

    # step tracker -> updates/creates tracks
    tracks = tracker.step(detections, frame_idx)

    # process each detection and map to track id by matching bbox equality
    # (Tracker updated its tracks' bbox to detections when matched)
    # Build mapping detection_index -> track_id by comparing IoU
    det_to_tid = {}
    for j, det in enumerate(detections):
        best_tid = None; best_iou = 0.0
        for tid, tr in tracker.tracks.items():
            val = iou_xyxy(det, tr.bbox)
            if val > best_iou:
                best_iou = val; best_tid = tid
        if best_iou >= 0.0 and best_tid is not None:
            det_to_tid[j] = best_tid

    # Now anonymize and blend per detection, using previous crop per track for smoothing
    for j, face_tensor in enumerate(faces):
        if j not in det_to_tid:
            continue
        tid = det_to_tid[j]
        tr = tracker.tracks.get(tid, None)
        if tr is None:
            continue
        # prepare input
        inp_pil = transforms.ToPILImage()(face_tensor.cpu())
        inp = transform(inp_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(inp)  # (1,3,H,W)
        out_img_rgb = tensor_to_image(out.squeeze(0))  # RGB numpy
        # resize to track bbox
        x1,y1,x2,y2 = tr.bbox
        out_resized_bgr = cv2.resize(out_img_rgb[:,:,::-1], (x2-x1, y2-y1))  # BGR
        # smoothing in pixel space using previous crop stored in track
        if smooth and tr.prev_crop is not None:
            prev = tr.prev_crop
            if prev.shape != out_resized_bgr.shape:
                prev = cv2.resize(prev, (out_resized_bgr.shape[1], out_resized_bgr.shape[0]))
            out_resized_bgr = (alpha * prev.astype(np.float32) + (1-alpha) * out_resized_bgr.astype(np.float32)).astype(np.uint8)
        # store as previous
        tr.prev_crop = out_resized_bgr

        # blending using ellipse mask and seamlessClone
        h_crop, w_crop = out_resized_bgr.shape[:2]
        mask = np.zeros((h_crop, w_crop, 3), dtype=np.uint8)
        cy, cx = h_crop//2, w_crop//2
        axes = (max(1, int(0.45*w_crop)), max(1, int(0.55*h_crop)))
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, (255,255,255), -1)
        center = (x1 + w_crop//2, y1 + h_crop//2)
        try:
            img_cv = cv2.seamlessClone(out_resized_bgr, img_cv, mask, center, cv2.NORMAL_CLONE)
        except Exception:
            # fallback alpha blend
            alpha_mask = (mask.astype(np.float32)/255.0)[...,0:1]
            img_cv[y1:y2, x1:x2] = (alpha_mask * out_resized_bgr + (1-alpha_mask) * img_cv[y1:y2, x1:x2]).astype(np.uint8)

    return img_cv, tracker.tracks

# ---- CLI runners (image/folder/video) using tracker ----
def run_image_cli(input_path, output_path, model, mtcnn, transform):
    pil = Image.open(input_path).convert("RGB")
    tracker = IoUTracker()
    out_cv, _ = anonymize_and_blend_with_tracker(pil, model, mtcnn, transform, tracker, frame_idx=0, smooth=True)
    cv2.imwrite(output_path, out_cv)
    print(f"Saved anonymized image to {output_path}")

def run_folder_cli(input_folder, out_folder, model, mtcnn, transform, smooth=True):
    files = sorted([p for p in Path(input_folder).glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    os.makedirs(out_folder, exist_ok=True)
    tracker = IoUTracker()
    for i, p in enumerate(tqdm(files, desc="Images")):
        pil = Image.open(str(p)).convert("RGB")
        out_cv, _ = anonymize_and_blend_with_tracker(pil, model, mtcnn, transform, tracker, frame_idx=i, smooth=smooth)
        out_path = os.path.join(out_folder, p.name)
        cv2.imwrite(out_path, out_cv)
    print(f"Saved anonymized images to {out_folder}")

def run_video_cli(input_video, output_video, model, mtcnn, transform, smooth=True):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))
    tracker = IoUTracker()
    frame_idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=total, desc="Video frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out_cv, _ = anonymize_and_blend_with_tracker(pil, model, mtcnn, transform, tracker, frame_idx=frame_idx, smooth=smooth)
        writer.write(out_cv)
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release(); writer.release()
    print(f"Saved anonymized video to {output_video}")

# ---- Streamlit UI ----
def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Anonymizer (IoU Tracker)", layout="wide")
    st.title("Face Anonymizer — IoU Tracker + Smooth Blending")
    st.markdown(
        "Upload a trained anonymizer checkpoint (.pth), then upload an image or video. "
        "This UI uses facenet-pytorch's MTCNN for detection, an IoU tracker for stable face IDs, "
        "and OpenCV seamlessClone for blending."
    )

    # Sidebar: model upload or path
    st.sidebar.header("Model checkpoint")
    ckpt_uploader = st.sidebar.file_uploader("Upload anonymizer .pth (optional)", type=["pth","pt"], accept_multiple_files=False)
    ckpt_path_input = st.sidebar.text_input("Or, enter a checkpoint path (on server)", "")
    use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=torch.cuda.is_available())

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    st.sidebar.write(f"Device: {device}")

    model = None
    # load checkpoint if uploaded or path provided
    if ckpt_uploader is not None:
        # save temp and load
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        tfile.write(ckpt_uploader.getbuffer())
        tfile.flush()
        try:
            model = load_anonymizer(tfile.name, device=device)
            st.sidebar.success("Checkpoint loaded from upload.")
        except Exception as e:
            st.sidebar.error(f"Failed to load uploaded checkpoint: {e}")
    elif ckpt_path_input:
        if os.path.exists(ckpt_path_input):
            try:
                model = load_anonymizer(ckpt_path_input, device=device)
                st.sidebar.success("Checkpoint loaded from path.")
            except Exception as e:
                st.sidebar.error(f"Failed to load checkpoint: {e}")

    # allow model-less demo if not provided? Warn user
    if model is None:
        st.warning("No checkpoint loaded. Please upload or provide a checkpoint path to run the anonymizer.")
        if st.button("Show dummy example (no model)"):
            st.image("https://upload.wikimedia.org/wikipedia/commons/8/89/Portrait_Placeholder.png", caption="Dummy placeholder")
        return

    # inputs
    st.header("Process an image or video")
    uploaded = st.file_uploader("Upload image (.jpg/.png) or video (.mp4)", type=["jpg","jpeg","png","mp4"], accept_multiple_files=False)
    smooth = st.checkbox("Temporal smoothing (per-track EMA)", value=True)
    img_size = st.slider("MTCNN crop size", 128, 512, value=256, step=32)

    mtcnn = MTCNN(image_size=img_size, margin=14, keep_all=True, post_process=True, device=device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])

    if uploaded is not None:
        # handle image vs video
        tname = uploaded.name
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(tname).suffix)
        tmp.write(uploaded.getbuffer())
        tmp.flush()
        tmp_path = tmp.name

        if tname.lower().endswith((".jpg",".jpeg",".png")):
            pil = Image.open(tmp_path).convert("RGB")
            tracker = IoUTracker()
            st.info("Processing image...")
            out_cv, _ = anonymize_and_blend_with_tracker(pil, model, mtcnn, transform, tracker, frame_idx=0, smooth=smooth)
            out_path = tmp_path + "_anon.jpg"
            cv2.imwrite(out_path, out_cv)
            st.image(cv2.cvtColor(out_cv, cv2.COLOR_BGR2RGB), caption="Anonymized", use_column_width=True)
            with open(out_path, "rb") as f:
                st.download_button("Download anonymized image", data=f, file_name=f"{Path(tname).stem}_anon.jpg", mime="image/jpeg")
        else:
            # video
            st.info("Processing video — this may take a while depending on length & GPU.")
            out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out_tmp.close()
            try:
                run_video_cli(tmp_path, out_tmp.name, model, mtcnn, transform, smooth=smooth)
                st.video(out_tmp.name)
                with open(out_tmp.name, "rb") as f:
                    st.download_button("Download anonymized video", data=f, file_name=f"{Path(tname).stem}_anon.mp4", mime="video/mp4")
            except Exception as e:
                st.error(f"Video processing failed: {e}")
    else:
        st.info("Upload an image or video to begin.")

# ---- Main: argparsing for CLI ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="Path to anonymizer .pth checkpoint (required for CLI mode)")
    ap.add_argument("--image", help="Single input image path")
    ap.add_argument("--out", help="Output image path (for single image)")
    ap.add_argument("--input-folder", help="Folder with images to anonymize (jpg/png)")
    ap.add_argument("--out-folder", help="Output folder for anonymized images")
    ap.add_argument("--video", help="Input video path")
    ap.add_argument("--out-video", help="Output video path")
    ap.add_argument("--smooth", action="store_true", help="Use per-track smoothing (EMA in pixel space)")
    ap.add_argument("--img-size", type=int, default=256, help="MTCNN crop size")
    ap.add_argument("--web", action="store_true", help="Launch Streamlit web UI (instead of CLI) -- run via `streamlit run` recommended")
    args = ap.parse_args()

    if args.web:
        # If launched normally with python (not streamlit), try to launch streamlit app
        # But recommended to use: streamlit run run_anonymizer_with_ui.py
        print("Launching Streamlit app... (recommended: run `streamlit run run_anonymizer_with_ui.py`)")
        run_streamlit_app()
        return

    # CLI mode: checkpoint required
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        ap.error("For CLI mode you must provide --checkpoint PATH and ensure file exists.")

    model = load_anonymizer(args.checkpoint, device=DEVICE)
    mtcnn = MTCNN(image_size=args.img_size, margin=14, keep_all=True, post_process=True, device=DEVICE)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])

    if args.image:
        outp = args.out if args.out else str(Path(args.image).with_name(Path(args.image).stem + "_anon" + Path(args.image).suffix))
        run_image_cli(args.image, outp, model, mtcnn, transform)
    elif args.input_folder:
        of = args.out_folder if args.out_folder else str(Path(args.input_folder).with_name(Path(args.input_folder).name + "_anon"))
        run_folder_cli(args.input_folder, of, model, mtcnn, transform, smooth=args.smooth)
    elif args.video:
        outv = args.out_video if args.out_video else str(Path(args.video).with_name(Path(args.video).stem + "_anon" + Path(args.video).suffix))
        run_video_cli(args.video, outv, model, mtcnn, transform, smooth=args.smooth)
    else:
        ap.print_help()

if __name__ == "__main__":
    # detect if streamlit is running this file (streamlit imports this module)
    if 'streamlit' in sys.modules:
        # when run with `streamlit run`, streamlit will import this script.
        # run_streamlit_app() will be called by Streamlit UI actions.
        pass
    else:
        main()
