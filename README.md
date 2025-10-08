# Face Anonymizer / Face Swap (Image & Video)

Friendly, practical README for this repository. This focuses on the *anonymizer / generator* training & inference code that ships in this repo (the `trainer_anonymizer.py` / `inference.py` flow, dataset pipeline, caching, logging, validation, and helpful utilities). **It intentionally does NOT include any StyleGAN + encoder approach.**

---

## Quick summary

This repo trains a generator that can replace / anonymize faces in images & videos. It supports:

- dataset pipeline for LFW (aligned face caching),
- PyTorch training with GAN + perceptual + identity + landmark losses,
- R1 discriminator regularization (AMP-compatible fix included),
- LPIPS & VGG perceptual metrics,
- EMA for smooth inference weights,
- validation loop (LPIPS / identity), best-checkpoint saving,
- inference script supporting batching, blending, and IoU-based face tracking,
- small CLI utilities: `precompute_embeddings.py`, `show_scalar_tags.py`, training report generator.

---

## Important paths / variables

Default dataset and cache roots used by scripts:

```
DATA_ROOT = "data/lfw-funneled/lfw_funneled"   # unzipped LFW dataset root
CACHE_ROOT = "cache_faces"                     # aligned / cached face crops
```

Place the unzipped LFW dataset in `data/lfw-funneled/lfw_funneled` (images commonly in `<person>/<image>.jpg`) or update env/args accordingly.

Kaggle dataset (original): https://www.kaggle.com/datasets/atulanandjha/lfwpeople?resource=download

---

## Requirements

Python 3.8+ (3.12 has worked in tested environments). GPU + CUDA recommended.

Minimal Python packages:

```
torch torchvision tensorboard matplotlib numpy pandas tqdm Pillow
lpips (optional)      # pip install lpips
insightface (optional but recommended) OR onnxruntime + arcface onnx model
face_alignment (optional for landmark loss)
onnxruntime (optional if you use ONNX ArcFace)
albumentations
```

Install (example):

```bash
pip install torch torchvision tensorboard matplotlib numpy pandas tqdm pillow lpips albumentations
# arcface / insightface (optional):
pip install insightface  # optional, can fallback to onnx
pip install onnxruntime  # optional
pip install face-alignment  # optional, required if using landmark loss
```

> If some optional components are missing the code falls back or throws helpful errors (e.g., `face_alignment` missing if you requested landmark loss).

---

## Repository entry points (scripts)

- `trainer_anonymizer.py` — main training script (GAN + perceptual + identity + LPIPS + R1 + EMA + validation).
- `dataset_anonymizer.py` — dataset loader & cache utilities (aligning / cropping faces into `CACHE_ROOT`).
- `precompute_embeddings.py` — create `.npz` of ArcFace embeddings matching dataset filenames (fast lookup during training).
- `inference.py` — inference wrapper to run the anonymizer on images or videos (batching, blending, face-tracking via IoU).
- `show_scalar_tags.py` — export TensorBoard scalars to CSV.
- `training_report.py` (or helper) — produces annotated plots & pdf report from `all_scalars.csv`.
- `trainer_helpers.py`, `trainer_utils_percept.py`, `models_anonymizer.py` — helper modules (networks, perceptual loss, saving/loading).

---

## Data preparation (LFW)

1. Download and unzip the LFW dataset so images are under `data/lfw-funneled/lfw_funneled/<person>/<img>.jpg`.
2. Run the cache utility to detect faces and save aligned crops into `cache_faces/` (the repo includes dataset code to do this).
   - Example (if script exists in repo):
     ```bash
     python dataset_anonymizer.py --data-root data/lfw-funneled/lfw_funneled --cache-root cache_faces --img-size 128
     ```
   - This produces per-person subfolders like `cache_faces/Aaron_Eckhart/Aaron_Eckhart_0001.jpg`.

If your cached faces appear color-distorted (e.g., odd channels, extreme colors), check:
- The image read / write pipeline is using the same normalization expected by the model (we use `ToTensor()` then `Normalize([0.5],[0.5])` → images in `[-1,1]`).
- No double-conversion or BGR/RGB swaps. The caching code saves standard RGB JPGs.

---

## Precompute embeddings

To speed identity loss / evaluation, precompute ArcFace embeddings that map exactly to dataset filenames:

```bash
python precompute_embeddings.py --cache-root cache_faces --out cache_faces/embeddings.npz --batch-size 16 --device cuda
```

This writes `cache_faces/embeddings.npz` that the trainer can load (use `--emb-path cache_faces/embeddings.npz`).

If you don't precompute, the trainer will compute embeddings at runtime using InsightFace or ONNX ArcFace, which is slower.

---

## Training (example)

Start training from scratch:

```bash
python trainer_anonymizer.py   --cache-root cache_faces   --save-dir checkpoints/anony1   --logdir runs/anony1   --batch-size 8   --epochs 40   --lr-g 2e-4 --lr-d 1e-4   --use-lpips --lambda-lpips 0.25   --lambda-adv 0.3 --lambda-recon 1.5 --lambda-perc 1.0   --r1-gamma 5.0   --use-spectral-norm   --ema-decay 0.999   --fake-buffer-size 2000 --fake-buffer-sample-frac 0.5   --pretrain-epochs 2   --val-interval 1 --val-size 256   --emb-path cache_faces/embeddings.npz
```

**Resume training from last checkpoint**

If a checkpoint exists in `--save-dir` it will be auto-detected. Or provide explicit checkpoint:

```bash
python trainer_anonymizer.py ... --resume checkpoints/anony1/latest.pth
```

**Important CLI flags (high impact)**

- `--lambda-adv` / `--lambda-recon` / `--lambda-perc` / `--lambda-lpips` / `--lambda-id` — tune these to balance adversarial vs perceptual vs identity losses.
- `--r1-gamma` — enable R1 regularizer for stability (0 to disable).
- `--r1-subsample` — compute R1 on only first `k` samples to reduce memory cost.
- `--use-spectral-norm` — spectral norm on discriminator for stability.
- `--fake-buffer-size` & `--fake-buffer-sample-frac` — historical fake buffering for D.
- `--use-amp` — enable mixed precision (watch R1 stability).
- `--pretrain-epochs` — pretrain G with recon/perceptual only before adversarial training.
- `--anonymize` — invert identity loss for anonymization (maximize identity distance).
- `--emb-path` — path to precomputed embeddings `.npz` (saves time).

---

## Validation & saving

- Validation runs every `--val-interval` epochs, using `--val-size` samples.
- Validation metrics saved to TensorBoard and used to pick the best checkpoint (by LPIPS/identity/composite).
- Best checkpoint saved as `best_<metric>.pth`.

---

## Inference

`inference.py` supports single image / batch / video processing with blending & IoU-based tracking.

Example (image swap — source face replaced by reference face):

```bash
python inference.py   --anonymizer checkpoints/anony1/latest.pth   --src-image inputs/target_image.jpg   --ref-image inputs/source_face.jpg   --out-image outputs/swapped.jpg   --device cuda   --use-ema
```

Options you can pass:
- `--use-ema` — prefer EMA generator for inference (recommended).
- `--batch-size` — inference batching.
- `--blend-mode` / `--alpha` — blending parameters for seamless composition.
- `--track-iou-thresh` — IoU threshold for face tracking across frames (video).
- `--first-face-only` — process only the first detected face.

**Auto-normalization**: inference auto-detects whether inputs are in `[-1,1]` or `[0,1]` and will convert before `ToPILImage()` to avoid `ValueError: pic should be 2/3 dimensional. Got 4 dimensions.` issues.

---

## Monitoring & analysis

- Logs: TensorBoard logs in `--logdir` (open with `tensorboard --logdir runs/`).
- Export scalars to CSV:
  ```bash
  python show_scalar_tags.py --logdir runs/anony1 --out all_scalars.csv --per-tag-dir per_tag_csvs
  ```
- Generate annotated report (helper):
  ```bash
  python training_report.py --csv all_scalars.csv --out training_report.pdf
  ```

Interpretation tips:
- If `loss/G` dominated by `loss/G/adv`, reduce `--lambda-adv` and increase `--lambda-recon` / `--lambda-perc` / `--lambda-lpips`.
- If `D/real_mean` or `D/fake_mean` jump wildly, enable `--r1-gamma` and `--use-spectral-norm` and consider reducing `--lr-d`.
- For anonymization tasks, set `--anonymize` and increase `--lambda-id`; for identity-preserving swaps increase `--lambda-id` without `--anonymize`.

---

## Common issues & quick fixes

- `RuntimeError: One of the differentiated Tensors does not require grad`  
  **Fix:** R1 penalty attempted to take gradients on an input that lacked `requires_grad=True`. The trainer computes R1 on a detached clone that `requires_grad_(True)`.
- `ModuleNotFoundError: No module named 'face_alignment'`  
  **Fix:** Install `pip install face-alignment` or run without `--use-landmark`.
- `No module named 'onnxruntime'` or ONNX ArcFace errors  
  **Fix:** Install `onnxruntime` or use `insightface` (preferred) for ArcFace embeddings. Or pass `--emb-path` to use precomputed embeddings.
- `ONNXRuntimeError: Got invalid dimensions for input... Expected: 1`  
  **Fix:** ONNX ArcFace model might expect batch dim=1; ensure you feed it correctly-shaped batches or use the insightface wrapper that handles batching.
- **OOM with R1 enabled**  
  **Fix:** reduce `--batch-size`, use `--r1-subsample 2`, or disable R1 temporarily.

---

## Tips to improve convergence

1. Pretrain G on reconstruction/perceptual for a few epochs: `--pretrain-epochs 2`.
2. Reduce adversarial weight: `--lambda-adv 0.2–0.5`.
3. Increase perceptual / LPIPS weights: `--lambda-perc 1.0`, `--lambda-lpips 0.2–0.5`.
4. TTUR: use `--lr-g 2e-4 --lr-d 1e-4`.
5. Use R1 (`--r1-gamma 5.0`) + spectral norm (`--use-spectral-norm`) for stability.
6. Use historical fake buffer: `--fake-buffer-size 2000 --fake-buffer-sample-frac 0.5`.
7. Use EMA (`--ema-decay 0.999`) and prefer EMA weights for inference.

---

## Example workflow (fast start)

1. Prepare dataset & cache:
   ```bash
   python dataset_anonymizer.py --data-root data/lfw-funneled/lfw_funneled --cache-root cache_faces
   ```

2. Precompute embeddings:
   ```bash
   python precompute_embeddings.py --cache-root cache_faces --out cache_faces/embeddings.npz --device cuda --batch-size 16
   ```

3. Train:
   ```bash
   python trainer_anonymizer.py --cache-root cache_faces --save-dir checkpoints/anony1 --logdir runs/anony1        --batch-size 8 --epochs 40 --emb-path cache_faces/embeddings.npz --use-lpips --lr-g 2e-4 --lr-d 1e-4        --lambda-adv 0.3 --lambda-recon 1.5 --lambda-perc 1.0 --r1-gamma 5.0 --ema-decay 0.999
   ```

4. Inference (image swap):
   ```bash
   python inference.py --anonymizer checkpoints/anony1/latest.pth --src-image target.jpg --ref-image source_face.jpg --out-image out.jpg --use-ema
   ```

---

## Project organization (suggested)

```
.
├── dataset_anonymizer.py
├── precompute_embeddings.py
├── trainer_anonymizer.py
├── inference.py
├── trainer_helpers.py
├── trainer_utils_percept.py
├── models_anonymizer.py
├── show_scalar_tags.py
├── README.md      <- you are reading this
└── cache_faces/   <- generated aligned faces
```

---

## License & contact

This repo is provided as-is for experimentation. If you want patches, CI, or a packaged Dockerfile for a reproducible environment, I can produce them.

If you want, I can:
- produce `dockerfile` / `requirements.txt`,
- add a `--mode` flag to `trainer_anonymizer.py` to select adapter vs vanilla training,
- add a small Streamlit demo for drag & drop inference.

Which of those would you like next?
