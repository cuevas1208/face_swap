#!/usr/bin/env python3
"""
trainer_anonymizer.py

Trainer with:
 - R1 fix (compute on cloned inputs that require_grad)
 - R1 subsample option (--r1-subsample)
 - Historical fake buffer (--fake-buffer-size)
 - Input jitter / noise (--input-noise)
 - VGG feature-matching loss (fm) using perceptual features (--fm-weight)
 - EMA, LPIPS, validation, checkpointing, spectral norm, TTUR, gradient clipping
 - Logging of grad norms & learning rates
"""

import os
import glob
import time
import argparse
import copy
import random
from pathlib import Path
from collections import deque

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# project imports (ensure these modules exist in your repo)
from models_anonymizer import GeneratorUNet, PatchDiscriminator
from dataset_anonymizer import CacheFaceDataset
from trainer_utils_percept import VGG16Perceptual, lpips_loss
from arcface_wrapper import load_arcface  # robust wrapper that tries insightface or ONNX

# helpers (assuming trainer_helpers.py present)
from trainer_helpers import (
    save_checkpoint, find_latest_checkpoint, load_embeddings_map,
    compute_landmark_loss_batch, apply_spectral_norm,
    init_ema, ema_update
)

# ----- device & perf -----
torch.backends.cudnn.benchmark = True

# -----------------------
# Dataset wrapper to provide sample path alongside image tensor
# -----------------------
class CacheFaceDatasetWithPaths(Dataset):
    def __init__(self, base_dataset: CacheFaceDataset):
        self.base = base_dataset
        self.samples = getattr(self.base, "samples", None)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img = self.base[idx]
        path = self.samples[idx] if self.samples is not None else None
        return img, path

# -----------------------
# Fake buffer for discriminator (historical fakes)
# -----------------------
class FakeBuffer:
    def __init__(self, max_size=5000):
        self.max_size = max_size
        self.buffer = deque()
    def push_and_sample(self, new_fakes, sample_size):
        """
        new_fakes: Tensor (B, C, H, W) detached on CPU or GPU
        sample_size: how many items to sample back from buffer (<= len(buffer))
        Returns: list of tensors taken from buffer (or []), and we append new_fakes to buffer
        """
        sampled = []
        # sample existing ones
        buf_len = len(self.buffer)
        sample_from_buf = min(sample_size, buf_len)
        if sample_from_buf > 0:
            # random sample indices
            idxs = random.sample(range(buf_len), sample_from_buf)
            for i in idxs:
                sampled.append(self.buffer[i].clone())
        # push new fakes (as CPU tensors) into buffer
        for i in range(new_fakes.size(0)):
            t = new_fakes[i].detach().cpu().clone()
            self.buffer.append(t)
            if len(self.buffer) > self.max_size:
                self.buffer.popleft()
        return sampled

# -----------------------
# loss helpers
# -----------------------
def hinge_d_loss(real_out, fake_out):
    loss_real = torch.mean(F.relu(1.0 - real_out))
    loss_fake = torch.mean(F.relu(1.0 + fake_out))
    return 0.5 * (loss_real + loss_fake)

def hinge_g_loss(fake_out):
    return -torch.mean(fake_out)

def cosine_sim(a, b):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return (a * b).sum(dim=1)

# -----------------------
# Validation helper
# -----------------------
def validate_on_dataset(model, dataset, device, embed_fn=None, vgg=None, lpips_fn=None, args=None):
    """
    Small validation run over a subset of dataset.
    Computes mean LPIPS (if available) and mean identity cosine similarity between real & fake.
    Returns (metric_value, {'lpips':..., 'cos':...})
    """
    model.eval()
    val_size = min(args.val_size, len(dataset))
    if val_size <= 0:
        return float('inf'), {'lpips': float('nan'), 'cos': float('nan')}
    indices = list(range(val_size))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=args.val_batch, shuffle=False, num_workers=2)

    lpips_vals = []
    cos_vals = []
    with torch.no_grad():
        for batch in loader:
            imgs, _paths = batch
            imgs = imgs.to(device)
            try:
                fake, _ = model(imgs, return_latent=True)
            except Exception:
                fake = model(imgs)
            if lpips_fn is not None:
                lp = lpips_fn((fake+1.0)/2.0, (imgs+1.0)/2.0).mean(dim=[1,2,3])
                lpips_vals.extend(lp.detach().cpu().numpy().tolist())
            else:
                feats_real = vgg((imgs+1.0)/2.0)
                feats_fake = vgg((fake+1.0)/2.0)
                per = 0.0
                for fr, ff in zip(feats_real, feats_fake):
                    per = per + torch.mean(torch.abs(fr - ff), dim=[1,2,3])
                lpips_vals.extend(per.detach().cpu().numpy().tolist())

            imgs_np = [((t.detach().cpu().permute(1,2,0).numpy()+1.0)*127.5).astype(np.uint8) for t in imgs]
            fake_np = [((t.detach().cpu().permute(1,2,0).numpy()+1.0)*127.5).astype(np.uint8) for t in fake]
            if embed_fn is None:
                try:
                    embed_fn = load_arcface(onnx_path=args.arcface_onnx)
                except Exception:
                    embed_fn = None
            if embed_fn is not None:
                emb_real = embed_fn(imgs_np)
                emb_fake = embed_fn(fake_np)
                emb_real = np.asarray(emb_real, dtype=np.float32)
                emb_fake = np.asarray(emb_fake, dtype=np.float32)
                def batch_cos(a,b):
                    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
                    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
                    return np.sum(a*b, axis=1)
                cos_batch = batch_cos(emb_real, emb_fake)
                cos_vals.extend(cos_batch.tolist())
            else:
                cos_vals.extend([float('nan')] * imgs.size(0))

    lpips_mean = float(np.nanmean(lpips_vals)) if len(lpips_vals) > 0 else float('nan')
    cos_mean = float(np.nanmean(cos_vals)) if len(cos_vals) > 0 else float('nan')

    metric_choice = args.val_metric_choice
    if metric_choice == 'lpips':
        metric_val = lpips_mean
    elif metric_choice == 'identity':
        if args.anonymize:
            metric_val = cos_mean
        else:
            metric_val = 1.0 - cos_mean
    else:  # composite
        if args.anonymize:
            metric_val = lpips_mean + cos_mean
        else:
            metric_val = lpips_mean + (1.0 - cos_mean)

    model.train()
    return float(metric_val), {'lpips': lpips_mean, 'cos': cos_mean}

# -----------------------
# Training loop
# -----------------------
def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Training device:", device)
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)

    # best validation value (lower is better)
    best_val = None
    best_ckpt_path = None

    # dataset + loader
    base_dataset = CacheFaceDataset(args.cache_root)
    dataset = CacheFaceDatasetWithPaths(base_dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, prefetch_factor=2)

    # model & optim
    G = GeneratorUNet(in_ch=3, base=64, bottleneck_dim=args.bottleneck_dim).to(device)
    D = PatchDiscriminator(in_ch=3, base=64).to(device)

    # optionally apply spectral norm to D
    if args.use_spectral_norm:
        print("[trainer] applying spectral norm to discriminator")
        apply_spectral_norm(D)

    # optimizers (TTUR-friendly)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    # perceptual / lpips / vgg
    vgg = VGG16Perceptual(device=device)
    lpips_fn = lpips_loss(device=device) if args.use_lpips else None

    # feature-matching uses vgg features as stable alternative (fm_weight)
    fm_weight = args.fm_weight if hasattr(args, 'fm_weight') else 0.0

    # embedder (insightface or ONNX)
    embed_fn = None
    if not args.emb_path:
        print("No precomputed embeddings provided; will use ArcFace embedder (insightface or onnx).")
        try:
            embed_fn = load_arcface(onnx_path=args.arcface_onnx)
        except Exception:
            embed_fn = None

    # load embedding map if provided
    emb_map = None
    if args.emb_path:
        emb_map = load_embeddings_map(args.emb_path, dataset.samples)
        if emb_map is None:
            print("[warning] embeddings path provided but no matches found; falling back to runtime embedding")
            try:
                embed_fn = load_arcface(onnx_path=args.arcface_onnx)
            except Exception:
                embed_fn = None

    # EMA init
    ema_G = init_ema(G) if args.ema_decay and args.ema_decay > 0 else None

    # Fake buffer (historical fakes)
    fake_buffer = FakeBuffer(max_size=args.fake_buffer_size) if args.fake_buffer_size > 0 else None

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == 'cuda' else None
    if scaler is not None and args.r1_gamma > 0:
        print("[trainer] Warning: you enabled AMP and R1 gradient penalty. Monitor numeric stability (R1 under AMP is more fragile).")

    # resume if requested or find latest in save_dir
    start_it = 0
    start_epoch = 0
    if args.resume:
        ckpt_path = args.resume if os.path.exists(args.resume) else find_latest_checkpoint(args.save_dir)
    else:
        ckpt_path = find_latest_checkpoint(args.save_dir)
    if ckpt_path:
        print("Resuming from checkpoint:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'G' in ckpt:
            G.load_state_dict(ckpt['G'])
        if 'D' in ckpt:
            D.load_state_dict(ckpt['D'])
        if 'opt_G' in ckpt:
            opt_G.load_state_dict(ckpt['opt_G'])
        if 'opt_D' in ckpt:
            opt_D.load_state_dict(ckpt['opt_D'])
        if ema_G is not None and 'ema_G' in ckpt:
            try:
                ema_G.load_state_dict(ckpt['ema_G'])
            except Exception:
                pass
        start_it = int(ckpt.get('it', 0))
        start_epoch = int(ckpt.get('epoch', 0))
        if scaler is not None and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        if 'best_val' in ckpt:
            best_val = ckpt['best_val']
            best_ckpt_path = ckpt.get('best_ckpt', None)
        print(f"Resumed at epoch {start_epoch}, it {start_it}")

    it = start_it
    accum_steps = max(1, args.accum_steps)
    print("Accumulation steps:", accum_steps)

    # learning rate schedulers could be added; log initial LRs
    try:
        writer.add_scalar('lr/G', args.lr_g, 0)
        writer.add_scalar('lr/D', args.lr_d, 0)
    except Exception:
        pass

    # training loop
    for epoch in range(start_epoch, args.epochs):
        G.train(); D.train()
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")
        for batch_idx, batch in pbar:
            imgs, paths = batch
            imgs = imgs.to(device)
            bs = imgs.size(0)

            # optionally add small input jitter / noise to real images fed to D
            real_for_D = imgs
            if args.input_noise > 0.0:
                noise = torch.randn_like(real_for_D) * args.input_noise
                real_for_D = torch.clamp(real_for_D + noise, -1.0, 1.0)

            # ---------------- Discriminator update ----------------
            # We compute D on real and fake; optionally add R1 penalty on real images
            if scaler is not None:
                # AMP path
                with torch.cuda.amp.autocast():
                    fake_detach, _ = G(imgs, return_latent=True)
                    d_real = D(real_for_D)
                    d_fake = D(fake_detach.detach())
                    loss_D = hinge_d_loss(d_real, d_fake)

                opt_D.zero_grad()
                if args.r1_gamma > 0:
                    # R1: compute on a detached clone (optionally subsampled) that requires grad.
                    # This avoids mutating original imgs and fixes "requires_grad" errors.
                    k = args.r1_subsample if args.r1_subsample > 0 else imgs.size(0)
                    k = min(k, imgs.size(0))
                    imgs_for_r1 = imgs[:k]
                    imgs_grad = imgs_for_r1.detach().clone().requires_grad_(True)
                    # compute D(imgs_grad) inside autocast for AMP
                    with torch.cuda.amp.autocast():
                        d_real_for_r1 = D(imgs_grad)
                    d_real_sum = d_real_for_r1.sum()
                    grads = torch.autograd.grad(outputs=d_real_sum, inputs=imgs_grad, create_graph=True)[0]
                    r1 = (grads.view(grads.size(0), -1).norm(2, dim=1) ** 2).mean()
                    loss_D = loss_D + 0.5 * args.r1_gamma * r1

                # incorporate historical fake buffer: sample some previous fakes, if configured
                if fake_buffer is not None and args.fake_buffer_sample_frac > 0:
                    # sample size from buffer
                    desired = int(round(bs * args.fake_buffer_sample_frac))
                    sampled = fake_buffer.push_and_sample(fake_detach.detach().cpu(), desired)
                    if len(sampled) > 0:
                        # build a new fake batch containing some buffered and some current fakes
                        sampled_t = torch.stack(sampled, dim=0).to(device)
                        # if sampled fewer than desired, pad with current fakes
                        num_buf = sampled_t.size(0)
                        if num_buf < bs:
                            cur_needed = bs - num_buf
                            cur_fakes = fake_detach.detach()[:cur_needed]
                            fake_combined = torch.cat([sampled_t, cur_fakes], dim=0)
                        else:
                            fake_combined = sampled_t[:bs]
                        d_fake = D(fake_combined.detach())
                        # recompute loss_D accordingly (use hinge with new d_fake)
                        loss_D = hinge_d_loss(d_real, d_fake)

                # step D
                scaler.scale(loss_D).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(opt_D)
                    torch.nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
                scaler.step(opt_D)
                scaler.update()
            else:
                # non-AMP
                fake_detach, _ = G(imgs, return_latent=True)
                d_real = D(real_for_D)
                d_fake = D(fake_detach.detach())
                loss_D = hinge_d_loss(d_real, d_fake)

                if args.r1_gamma > 0:
                    # sub-sample R1 if requested
                    k = args.r1_subsample if args.r1_subsample > 0 else imgs.size(0)
                    k = min(k, imgs.size(0))
                    imgs_for_r1 = imgs[:k]
                    imgs_grad = imgs_for_r1.detach().clone().requires_grad_(True)
                    d_real_for_r1 = D(imgs_grad)
                    d_real_sum = d_real_for_r1.sum()
                    grads = torch.autograd.grad(outputs=d_real_sum, inputs=imgs_grad, create_graph=True)[0]
                    r1 = (grads.view(grads.size(0), -1).norm(2, dim=1) ** 2).mean()
                    loss_D = loss_D + 0.5 * args.r1_gamma * r1

                # historical buffer usage (non-AMP)
                if fake_buffer is not None and args.fake_buffer_sample_frac > 0:
                    desired = int(round(bs * args.fake_buffer_sample_frac))
                    sampled = fake_buffer.push_and_sample(fake_detach.detach().cpu(), desired)
                    if len(sampled) > 0:
                        sampled_t = torch.stack(sampled, dim=0).to(device)
                        num_buf = sampled_t.size(0)
                        if num_buf < bs:
                            cur_needed = bs - num_buf
                            cur_fakes = fake_detach.detach()[:cur_needed]
                            fake_combined = torch.cat([sampled_t, cur_fakes], dim=0)
                        else:
                            fake_combined = sampled_t[:bs]
                        d_fake = D(fake_combined.detach())
                        loss_D = hinge_d_loss(d_real, d_fake)

                opt_D.zero_grad()
                loss_D.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
                opt_D.step()

            # ---------------- Generator update ----------------
            do_adv = (epoch >= args.pretrain_epochs)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    fake, z = G(imgs, return_latent=True)
                    g_out = D(fake)
                    loss_g_adv = hinge_g_loss(g_out) if do_adv else torch.tensor(0.0, device=device)

                    # perceptual
                    feats_real = vgg((imgs + 1.0)/2.0)
                    feats_fake = vgg((fake + 1.0)/2.0)
                    loss_perc = 0.0
                    for fr, ff in zip(feats_real, feats_fake):
                        loss_perc += F.l1_loss(ff, fr)

                    loss_lpips = lpips_fn((fake+1)/2.0, (imgs+1)/2.0).mean() if lpips_fn is not None else torch.tensor(0.0, device=device)

                    # feature-matching via vgg features (stable substitute for D-features)
                    loss_fm = torch.tensor(0.0, device=device)
                    if fm_weight > 0.0:
                        fm = 0.0
                        for fr, ff in zip(feats_real, feats_fake):
                            fm = fm + F.l1_loss(ff, fr)
                        loss_fm = fm

                    # identity embeddings
                    if emb_map:
                        emb_real_list = []
                        for pth in paths:
                            pstr = str(pth[0]) if isinstance(pth, (tuple, list)) else str(pth)
                            val = emb_map.get(pstr, None)
                            if val is None:
                                bn = os.path.basename(pstr)
                                val = emb_map.get(bn, None)
                            emb_real_list.append(None if val is None else np.asarray(val, dtype=np.float32))
                        need_compute_idx = [i for i,e in enumerate(emb_real_list) if e is None]
                        if len(need_compute_idx) > 0:
                            if embed_fn is None:
                                embed_fn = load_arcface(onnx_path=args.arcface_onnx)
                            imgs_for_embed = []
                            for i_idx in need_compute_idx:
                                im_np = ((imgs[i_idx].detach().cpu().permute(1,2,0).numpy() + 1.0) * 127.5).astype(np.uint8)
                                imgs_for_embed.append(im_np)
                            embs_new = embed_fn(imgs_for_embed)
                            embs_new = np.asarray(embs_new, dtype=np.float32)
                            for idx_local, emb_arr in zip(need_compute_idx, embs_new):
                                emb_real_list[idx_local] = emb_arr
                        for i_idx, e in enumerate(emb_real_list):
                            if e is None:
                                emb_real_list[i_idx] = np.zeros((args.bottleneck_dim,), dtype=np.float32)
                        emb_real = np.stack(emb_real_list, axis=0).astype(np.float32)
                        emb_real_t = torch.from_numpy(emb_real).to(device).float()
                    else:
                        imgs_np = [((t.detach().cpu().permute(1,2,0).numpy()+1.0)*127.5).astype(np.uint8) for t in imgs]
                        if embed_fn is None:
                            embed_fn = load_arcface(onnx_path=args.arcface_onnx)
                        emb_real_np = embed_fn(imgs_np)
                        emb_real_t = torch.from_numpy(np.asarray(emb_real_np, dtype=np.float32)).to(device).float()

                    fake_np = [((t.detach().cpu().permute(1,2,0).numpy()+1.0)*127.5).astype(np.uint8) for t in fake]
                    if embed_fn is None:
                        embed_fn = load_arcface(onnx_path=args.arcface_onnx)
                    emb_fake_np = embed_fn(fake_np)
                    emb_fake_t = torch.from_numpy(np.asarray(emb_fake_np, dtype=np.float32)).to(device).float()

                    cos = cosine_sim(emb_real_t, emb_fake_t).mean()
                    if args.anonymize:
                        loss_id = cos
                    else:
                        loss_id = (1.0 - cos)

                    loss_land = torch.tensor(0.0, device=device)
                    if args.use_landmark:
                        loss_land = compute_landmark_loss_batch(imgs, fake, device,
                                                               fa_inst=None, mp_detector=None, use_fa=False)
                    loss_recon = F.l1_loss(fake, imgs)

                    adv_term = (args.lambda_adv * loss_g_adv) if do_adv else torch.tensor(0.0, device=device)

                    loss_G_total = (adv_term +
                                  args.lambda_perc * loss_perc +
                                  args.lambda_lpips * loss_lpips +
                                  args.lambda_recon * loss_recon +
                                  args.lambda_id * loss_id +
                                  args.lambda_land * loss_land +
                                  fm_weight * loss_fm)

                    loss_G = loss_G_total / accum_steps

                opt_G.zero_grad()
                scaler.scale(loss_G).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(opt_G)
                    torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
                if ((it + 1) % accum_steps) == 0:
                    scaler.step(opt_G)
                    scaler.update()
                    if ema_G is not None:
                        ema_update(G, ema_G, args.ema_decay)
            else:
                # non-AMP path
                fake, z = G(imgs, return_latent=True)
                g_out = D(fake)
                loss_g_adv = hinge_g_loss(g_out) if do_adv else torch.tensor(0.0, device=device)

                feats_real = vgg((imgs + 1.0)/2.0)
                feats_fake = vgg((fake + 1.0)/2.0)
                loss_perc = 0.0
                for fr, ff in zip(feats_real, feats_fake):
                    loss_perc += F.l1_loss(ff, fr)
                loss_lpips = lpips_fn((fake+1)/2.0, (imgs+1)/2.0).mean() if lpips_fn is not None else torch.tensor(0.0, device=device)

                loss_fm = torch.tensor(0.0, device=device)
                if fm_weight > 0.0:
                    fm = 0.0
                    for fr, ff in zip(feats_real, feats_fake):
                        fm = fm + F.l1_loss(ff, fr)
                    loss_fm = fm

                # identity
                if emb_map:
                    emb_real_list = []
                    for pth in paths:
                        pstr = str(pth[0]) if isinstance(pth, (tuple, list)) else str(pth)
                        val = emb_map.get(pstr, None)
                        if val is None:
                            bn = os.path.basename(pstr)
                            val = emb_map.get(bn, None)
                        emb_real_list.append(None if val is None else np.asarray(val, dtype=np.float32))
                    need_compute_idx = [i for i,e in enumerate(emb_real_list) if e is None]
                    if len(need_compute_idx) > 0:
                        if embed_fn is None:
                            embed_fn = load_arcface(onnx_path=args.arcface_onnx)
                        imgs_for_embed = []
                        for i_idx in need_compute_idx:
                            im_np = ((imgs[i_idx].detach().cpu().permute(1,2,0).numpy() + 1.0) * 127.5).astype(np.uint8)
                            imgs_for_embed.append(im_np)
                        embs_new = embed_fn(imgs_for_embed)
                        embs_new = np.asarray(embs_new, dtype=np.float32)
                        for idx_local, emb_arr in zip(need_compute_idx, embs_new):
                            emb_real_list[idx_local] = emb_arr
                    for i_idx, e in enumerate(emb_real_list):
                        if e is None:
                            emb_real_list[i_idx] = np.zeros((args.bottleneck_dim,), dtype=np.float32)
                    emb_real = np.stack(emb_real_list, axis=0).astype(np.float32)
                    emb_real_t = torch.from_numpy(emb_real).to(device).float()
                else:
                    imgs_np = [((t.detach().cpu().permute(1,2,0).numpy()+1.0)*127.5).astype(np.uint8) for t in imgs]
                    if embed_fn is None:
                        embed_fn = load_arcface(onnx_path=args.arcface_onnx)
                    emb_real_np = embed_fn(imgs_np)
                    emb_real_t = torch.from_numpy(np.asarray(emb_real_np, dtype=np.float32)).to(device).float()

                fake_np = [((t.detach().cpu().permute(1,2,0).numpy()+1.0)*127.5).astype(np.uint8) for t in fake]
                if embed_fn is None:
                    embed_fn = load_arcface(onnx_path=args.arcface_onnx)
                emb_fake_np = embed_fn(fake_np)
                emb_fake_t = torch.from_numpy(np.asarray(emb_fake_np, dtype=np.float32)).to(device).float()

                cos = cosine_sim(emb_real_t, emb_fake_t).mean()
                if args.anonymize:
                    loss_id = cos
                else:
                    loss_id = (1.0 - cos)

                loss_land = torch.tensor(0.0, device=device)
                if args.use_landmark:
                    loss_land = compute_landmark_loss_batch(imgs, fake, device,
                                                           fa_inst=None, mp_detector=None, use_fa=False)
                loss_recon = F.l1_loss(fake, imgs)

                loss_G_total = ( (args.lambda_adv * loss_g_adv if do_adv else torch.tensor(0.0, device=device)) +
                                args.lambda_perc * loss_perc +
                                args.lambda_lpips * loss_lpips +
                                args.lambda_recon * loss_recon +
                                args.lambda_id * loss_id +
                                args.lambda_land * loss_land +
                                fm_weight * loss_fm )

                loss_G = loss_G_total / accum_steps
                opt_G.zero_grad()
                loss_G.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
                if ((it + 1) % accum_steps) == 0:
                    opt_G.step()
                    if ema_G is not None:
                        ema_update(G, ema_G, args.ema_decay)

            # ---- logging ----
            # scalars
            try:
                writer.add_scalar('loss/D', float(loss_D.item()), it)
                writer.add_scalar('loss/G', float(loss_G_total.item()), it)
            except Exception:
                pass

            # subcomponents
            try:
                writer.add_scalar('loss/G/adv', float(loss_g_adv.item()), it)
                writer.add_scalar('loss/G/recon', float(loss_recon.item()), it)
                writer.add_scalar('loss/G/perceptual', float(loss_perc.item()), it)
                writer.add_scalar('loss/G/lpips', float(loss_lpips.item()), it)
                writer.add_scalar('loss/G/id_raw_cos', float(cos.item()), it)
                writer.add_scalar('loss/G/land', float(loss_land.item()), it)
                writer.add_scalar('loss/G/fm', float(loss_fm.item()) if isinstance(loss_fm, torch.Tensor) else float(loss_fm))
            except Exception:
                pass

            # discriminator diagnostics
            try:
                writer.add_scalar('D/real_mean', float(d_real.mean().item()), it)
                writer.add_scalar('D/fake_mean', float(d_fake.mean().item()), it)
            except Exception:
                pass

            # log gradient norms and LR
            try:
                # grad norms (compute on parameters)
                total_norm_G = 0.0
                for p in G.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_G += param_norm.item() ** 2
                total_norm_G = total_norm_G ** 0.5
                writer.add_scalar('grad_norm/G', float(total_norm_G), it)

                total_norm_D = 0.0
                for p in D.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_D += param_norm.item() ** 2
                total_norm_D = total_norm_D ** 0.5
                writer.add_scalar('grad_norm/D', float(total_norm_D), it)
            except Exception:
                pass

            # log learning rates
            try:
                lr_g = opt_G.param_groups[0]['lr']
                lr_d = opt_D.param_groups[0]['lr']
                writer.add_scalar('lr/G', float(lr_g), it)
                writer.add_scalar('lr/D', float(lr_d), it)
            except Exception:
                pass

            # checkpointing (save checkpoint + EMA state)
            if it % args.save_interval == 0 and it > 0:
                state = {
                    'epoch': epoch,
                    'it': it,
                    'G': G.state_dict(),
                    'D': D.state_dict(),
                    'opt_G': opt_G.state_dict(),
                    'opt_D': opt_D.state_dict()
                }
                if ema_G is not None:
                    state['ema_G'] = ema_G.state_dict()
                if scaler is not None:
                    state['scaler'] = scaler.state_dict()
                save_checkpoint(state, args.save_dir, it)

            it += 1
            pbar.set_postfix({'loss_D': float(loss_D.item()), 'loss_G': float(loss_G_total.item())})
        print(f"Epoch {epoch} done, iters {it}")

        # run validation every N epochs (if enabled)
        if args.val_interval > 0 and ((epoch + 1) % args.val_interval == 0):
            val_model = ema_G if (ema_G is not None) else G
            val_score, val_stats = validate_on_dataset(val_model, dataset, device,
                                                       embed_fn=embed_fn,
                                                       vgg=vgg,
                                                       lpips_fn=lpips_fn,
                                                       args=args)
            print(f"[val] epoch {epoch} -> {args.val_metric_choice} = {val_score:.6f} (lpips={val_stats['lpips']:.4f}, cos={val_stats['cos']:.4f})")
            writer.add_scalar('val/' + args.val_metric_choice, float(val_score), epoch)
            writer.add_scalar('val/lpips', float(val_stats['lpips']), epoch)
            writer.add_scalar('val/cosine', float(val_stats['cos']), epoch)

            # if better, save best checkpoint
            is_better = False
            if best_val is None:
                is_better = True
            else:
                if val_score < best_val:
                    is_better = True
            if is_better:
                best_val = float(val_score)
                best_path = os.path.join(args.save_dir, f'best_{args.val_metric_choice}.pth')
                state = {
                    'epoch': epoch,
                    'it': it,
                    'G': G.state_dict(),
                    'D': D.state_dict(),
                    'opt_G': opt_G.state_dict(),
                    'opt_D': opt_D.state_dict(),
                    'best_val': best_val,
                    'best_ckpt': best_path
                }
                if ema_G is not None:
                    state['ema_G'] = ema_G.state_dict()
                if scaler is not None:
                    state['scaler'] = scaler.state_dict()
                torch.save(state, best_path)
                print(f"[val] saved new best checkpoint: {best_path} (best {args.val_metric_choice}={best_val:.6f})")

    writer.close()
    print("Training complete.")

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-root', required=True)
    p.add_argument('--save-dir', default='checkpoints_anony')
    p.add_argument('--logdir', default='runs/anony')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr-g', type=float, default=2e-4)
    p.add_argument('--lr-d', type=float, default=2e-4)
    p.add_argument('--beta1', type=float, default=0.5)
    p.add_argument('--beta2', type=float, default=0.999)
    p.add_argument('--bottleneck-dim', type=int, default=512)
    p.add_argument('--use-lpips', action='store_true')
    p.add_argument('--use-amp', action='store_true')
    p.add_argument('--use-landmark', action='store_true')
    p.add_argument('--arcface-onnx', default=None, help='optional path to ArcFace ONNX (used if insightface not installed)')
    p.add_argument('--emb-path', default=None, help='optional precomputed embeddings (.npz/.npz-like)')
    p.add_argument('--resume', default=None, help='checkpoint to resume from; if omitted will try save-dir/latest.pth')
    p.add_argument('--device', default=None)
    p.add_argument('--log-interval', type=int, default=200)
    p.add_argument('--save-interval', type=int, default=2000)
    p.add_argument('--accum-steps', type=int, default=1, help='gradient accumulation steps')
    p.add_argument('--pretrain-epochs', type=int, default=0, help='number of epochs to pretrain G with no adversarial loss')
    p.add_argument('--use-spectral-norm', action='store_true', help='apply spectral norm to discriminator')
    p.add_argument('--r1-gamma', type=float, default=0.0, help='R1 regularization gamma (0 to disable)')
    p.add_argument('--r1-subsample', type=int, default=0, help='compute R1 only on first k samples of batch (0=full batch)')
    p.add_argument('--r1-subset-fraction', type=float, default=0.0, help='(deprecated) alternative fractional subsample')
    p.add_argument('--ema-decay', type=float, default=0.999, help='EMA decay for generator (0 to disable)')
    p.add_argument('--grad-clip', type=float, default=0.0, help='clip gradient norm (>0 to enable)')
    p.add_argument('--anonymize', action='store_true', help='if set, identity loss will try to reduce similarity (anonymize)')
    # fake buffer
    p.add_argument('--fake-buffer-size', type=int, default=0, help='historical fake buffer size (0 disables)')
    p.add_argument('--fake-buffer-sample-frac', type=float, default=0.5, dest='fake_buffer_sample_frac', help='fraction of fake batch to draw from buffer for D update (0..1)')
    # input noise
    p.add_argument('--input-noise', type=float, default=0.0, help='stddev of gaussian noise added to real images before D (in [-1,1] scale)')
    # validation args
    p.add_argument('--val-interval', type=int, default=0, help='run validation every N epochs (0 disable)')
    p.add_argument('--val-size', type=int, default=128, help='how many samples to use for validation (first N)')
    p.add_argument('--val-batch', type=int, default=8, help='validation batch size')
    p.add_argument('--val-metric-choice', type=str, default='composite', choices=['lpips','identity','composite'], help='which metric to use for best-checkpoint')
    # feature-matching
    p.add_argument('--fm-weight', type=float, default=0.0, help='weight for VGG-based feature matching (stable alternative to D-feature matching)')
    # loss weights
    p.add_argument('--lambda-adv', type=float, default=1.0)
    p.add_argument('--lambda-perc', type=float, default=0.8)
    p.add_argument('--lambda-lpips', type=float, default=0.0)
    p.add_argument('--lambda-recon', type=float, default=1.0)
    p.add_argument('--lambda-id', type=float, default=1.0)
    p.add_argument('--lambda-land', type=float, default=0.5)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # back-compat: if r1_subset_fraction set, convert to r1_subsample
    if hasattr(args, 'r1_subset_fraction') and args.r1_subsample == 0 and args.r1_subset_fraction > 0:
        # convert fraction to number of samples later (done in loop)
        pass
    train(args)
