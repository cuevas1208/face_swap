#!/usr/bin/env python3
"""
run_anonymizer_cli.py

Usage examples:
  # single image
  python run_anonymizer_cli.py --checkpoint anonymizer.pth --image in.jpg --out out.jpg

  # folder of images
  python run_anonymizer_cli.py --checkpoint anonymizer.pth --input-folder imgs/ --out-folder imgs_anon/

  # video
  python run_anonymizer_cli.py --checkpoint anonymizer.pth --video in.mp4 --out out_anon.mp4 --smooth

Notes:
- Put this file in the same directory as anonymize_lfw.py (so it can import Anonymizer).
- Requires: torch torchvision facenet-pytorch pillow opencv-python tqdm
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN

# import your Anonymizer class from the training script
# (assumes anonymize_lfw.py is in same folder and defines Anonymizer)
from anonymize_lfw import Anonymizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Helpers
# -------------------------
def load_anonymizer(checkpoint_path: str, device=DEVICE):
    model = Anonymizer().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    # if saved as state_dict or full model - try both
    if isinstance(state, dict) and any(k.startswith("enc1") or k.startswith("fc") for k in state.keys()):
        model.load_state_dict(state)
    else:
        # fallback: maybe the checkpoint is a dict with 'model_state'
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            # attempt directly (may raise)
            model.load_state_dict(state)
    model.eval()
    return model

def tensor_to_image(tensor):
    """tensor in [-1,1], shape (3,H,W) -> HxWx3 uint8 RGB numpy"""
    t = tensor.clamp(-1,1).detach().cpu()
    img = ((t + 1.0) * 127.5).permute(1,2,0).numpy().astype(np.uint8)
    return img

def prepare_mtcnn(img_size=256, keep_all=True, device=DEVICE):
    return MTCNN(image_size=img_size, margin=14, keep_all=keep_all, post_process=True, device=device)

def anonymize_and_blend(pil_img: Image.Image, model, mtcnn, transform, smooth=False, per_face_prev=None, alpha=0.7):
    """
    Detect faces in pil_img, anonymize them with model, and return a BGR numpy with blended faces.
    per_face_prev: list of previous anonymized crop images (BGR numpy) to use for smoothing.
    Returns: out_bgr, new_per_face_prev
    """
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    with torch.no_grad():
        # get aligned faces (tensor(s) on device) and bounding boxes
        faces = mtcnn(pil_img)             # single tensor or list or None
        boxes, probs = mtcnn.detect(pil_img)  # numpy (N,4) or (None)
    if faces is None:
        return img_cv, []  # no faces, nothing changed

    if isinstance(faces, torch.Tensor):
        faces = [faces]

    new_prev = []
    for i, face_tensor in enumerate(faces):
        inp_pil = transforms.ToPILImage()(face_tensor.cpu())
        inp = transform(inp_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(inp)  # (1,3,H,W) in [-1,1]
        out_img_rgb = tensor_to_image(out.squeeze(0))  # RGB numpy uint8

        # location
        if boxes is None:
            # fallback: center
            h, w = img_cv.shape[:2]
            fh, fw = out_img_rgb.shape[:2]
            x1 = max(0, w//2 - fw//2); y1 = max(0, h//2 - fh//2)
            x2, y2 = min(w, x1+fw), min(h, y1+fh)
        else:
            # boxes are [x1,y1,x2,y2]
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            pad = int(0.15 * max(x2-x1, y2-y1))
            x1 = max(0, x1-pad); y1 = max(0, y1-pad)
            x2 = min(img_cv.shape[1], x2+pad); y2 = min(img_cv.shape[0], y2+pad)

        # resize anonymized output to bbox
        out_resized_bgr = cv2.resize(out_img_rgb[:, :, ::-1], (x2-x1, y2-y1))  # RGB->BGR

        # simple temporal smoothing in pixel space if requested
        if smooth and per_face_prev is not None and i < len(per_face_prev) and per_face_prev[i] is not None:
            prev_crop = per_face_prev[i]
            # ensure same size
            if prev_crop.shape[:2] != out_resized_bgr.shape[:2]:
                # resize prev crop
                prev_crop = cv2.resize(prev_crop, (out_resized_bgr.shape[1], out_resized_bgr.shape[0]))
            out_resized_bgr = (alpha * prev_crop.astype(np.float32) + (1-alpha) * out_resized_bgr.astype(np.float32)).astype(np.uint8)

        # soft elliptical mask for blending
        mask = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
        cy, cx = (y2-y1)//2, (x2-x1)//2
        axes = (max(1,int(0.45*(x2-x1))), max(1,int(0.55*(y2-y1))))
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, (255,255,255), -1)

        center = (x1 + (x2-x1)//2, y1 + (y2-y1)//2)
        try:
            img_cv = cv2.seamlessClone(out_resized_bgr, img_cv, mask, center, cv2.NORMAL_CLONE)
        except Exception:
            # fallback alpha blend
            alpha_mask = (mask.astype(np.float32)/255.0)[...,0:1]
            img_cv[y1:y2, x1:x2] = (alpha_mask * out_resized_bgr + (1-alpha_mask) * img_cv[y1:y2, x1:x2]).astype(np.uint8)

        new_prev.append(out_resized_bgr)

    return img_cv, new_prev

# -------------------------
# CLI runners
# -------------------------
def run_image(input_path, output_path, model, mtcnn, transform, smooth=False):
    pil = Image.open(input_path).convert("RGB")
    out_cv, _ = anonymize_and_blend(pil, model, mtcnn, transform, smooth=smooth, per_face_prev=[])
    cv2.imwrite(output_path, out_cv)
    print(f"Saved anonymized image to {output_path}")

def run_folder(input_folder, out_folder, model, mtcnn, transform, smooth=False):
    os.makedirs(out_folder, exist_ok=True)
    files = sorted([p for p in Path(input_folder).glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    prev_cache = []  # keep per-face prev across frames (approx)
    for p in tqdm(files, desc="Batch images"):
        pil = Image.open(str(p)).convert("RGB")
        out_cv, new_prev = anonymize_and_blend(pil, model, mtcnn, transform, smooth=smooth, per_face_prev=prev_cache)
        # update prev_cache to match number of faces (simple heuristic)
        prev_cache = new_prev
        out_path = os.path.join(out_folder, p.name)
        cv2.imwrite(out_path, out_cv)
    print(f"Saved anonymized images to {out_folder}")

def run_video(input_video, output_video, model, mtcnn, transform, smooth=False):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (W,H))
    frame_idx = 0
    prev_cache = []
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), desc="Video frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out_cv, new_prev = anonymize_and_blend(pil, model, mtcnn, transform, smooth=smooth, per_face_prev=prev_cache)
        prev_cache = new_prev  # naive correspondence by face index
        writer.write(out_cv)
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    writer.release()
    print(f"Saved anonymized video to {output_video}")

# -------------------------
# Main & argparsing
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Run anonymizer on image / folder / video")
    ap.add_argument("--checkpoint", required=True, help="Path to anonymizer .pth checkpoint")
    ap.add_argument("--image", help="Single input image path")
    ap.add_argument("--out", help="Output image path (for single image)")
    ap.add_argument("--input-folder", help="Folder with images to anonymize (jpg/png)")
    ap.add_argument("--out-folder", help="Output folder for anonymized images")
    ap.add_argument("--video", help="Input video path")
    ap.add_argument("--out-video", help="Output video path")
    ap.add_argument("--device", default=None, help="cuda or cpu (overrides auto)")
    ap.add_argument("--keep-all", action="store_true", help="mtcnn: detect keep_all faces (default True for folder/video). If false, only first face processed.")
    ap.add_argument("--smooth", action="store_true", help="Use simple temporal smoothing (frame-by-frame EMA of anonymized crop)")
    ap.add_argument("--img-size", type=int, default=256, help="MTCNN crop size (default 256)")
    args = ap.parse_args()

    # device override
    global DEVICE
    if args.device:
        DEVICE = torch.device(args.device)
    print(f"Using device: {DEVICE}")

    # load model
    model = load_anonymizer(args.checkpoint, device=DEVICE)
    mtcnn = prepare_mtcnn(img_size=args.img_size, keep_all=True if args.keep_all else False, device=DEVICE)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    # dispatch
    if args.image:
        outp = args.out if args.out else (Path(args.image).with_name(Path(args.image).stem + "_anon" + Path(args.image).suffix))
        run_image(args.image, str(outp), model, mtcnn, transform, smooth=args.smooth)
    elif args.input_folder:
        of = args.out_folder if args.out_folder else (str(Path(args.input_folder) / "../" / (Path(args.input_folder).name + "_anon")))
        run_folder(args.input_folder, of, model, mtcnn, transform, smooth=args.smooth)
    elif args.video:
        outv = args.out_video if args.out_video else (str(Path(args.video).with_name(Path(args.video).stem + "_anon" + Path(args.video).suffix)))
        run_video(args.video, outv, model, mtcnn, transform, smooth=args.smooth)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
