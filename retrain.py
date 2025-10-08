#!/usr/bin/env python3
# retrain.py
"""
Retrain wrapper for anonymize_lfw.py

Usage example:
  # fresh train (classifier then anonymizer)
  python retrain.py --cache-root cache_faces --batch-size 32 --epochs-clf 6 --epochs-anon 12 --log-dir runs/exp1

  # resume classifier training
  python retrain.py --cache-root cache_faces --resume-clf clf_epoch3.pth

  # resume anonymizer training using a pretrained classifier
  python retrain.py --cache-root cache_faces --resume-clf clf_pretrained.pth --resume-anon anon_epoch5.pth
"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# import things from your anonymize_lfw module
from anonymize_lfw import (
    LFWFaceDataset, build_classifier, Anonymizer, VGGPerceptual,
    train_classifier as train_clf_simple, train_anonymizer as train_anon_simple
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", required=True, help="Path to cached aligned faces (cache_faces)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs-clf", type=int, default=6)
    p.add_argument("--epochs-anon", type=int, default=10)
    p.add_argument("--lr-clf", type=float, default=1e-3)
    p.add_argument("--lr-anon", type=float, default=1e-4)
    p.add_argument("--min-images-per-person", type=int, default=2, help="Filter persons with fewer images")
    p.add_argument("--log-dir", default="runs/anon_experiment", help="TensorBoard logs")
    p.add_argument("--save-dir", default="checkpoints", help="Where to save checkpoints")
    p.add_argument("--resume-clf", default="", help="Path to classifier checkpoint to resume or reuse")
    p.add_argument("--resume-anon", default="", help="Path to anonymizer checkpoint to resume")
    p.add_argument("--device", default=None, help="cuda or cpu (overrides auto)")
    return p.parse_args()

def build_loaders(cache_root, batch_size, min_images):
    transform = torch.nn.Sequential() if False else None
    # Use the same transform that anonymize_lfw expects:
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = LFWFaceDataset(cache_root, transform=transform, min_images_per_person=min_images)
    print(f"Dataset images={len(ds)} persons={len(ds.person2id)}")
    if len(ds) == 0:
        raise RuntimeError("No images found in cache. Re-run cache prepare.")
    # quick split
    n = len(ds)
    n_val = max(1, int(0.05 * n))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, ds

def save_ckpt(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print("Saved checkpoint:", path)

def load_state_dict_maybe(model, ckpt_path, device):
    if not ckpt_path:
        return model
    sd = torch.load(ckpt_path, map_location=device)
    # allow either state_dict or raw model dict
    if isinstance(sd, dict) and any("enc1" in k or "fc" in k or "layer" in k for k in sd.keys()):
        model.load_state_dict(sd)
    elif isinstance(sd, dict) and "model_state" in sd:
        model.load_state_dict(sd["model_state"])
    else:
        model.load_state_dict(sd)
    print("Loaded checkpoint:", ckpt_path)
    return model

def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print("Using device:", device)

    train_loader, val_loader, ds = build_loaders(args.cache_root, args.batch_size, args.min_images_per_person)

    # --- CLASSIFIER TRAINING (or reuse the provided checkpoint) ---
    num_classes = len(ds.person2id)
    classifier = build_classifier(num_classes)
    classifier.to(device)

    clf_ckpt_out = os.path.join(args.save_dir, "classifier_epoch_latest.pth")
    if args.resume_clf:
        classifier = load_state_dict_maybe(classifier, args.resume_clf, device)
        print("Resumed classifier from:", args.resume_clf)
    else:
        # use the simple training helper from anonymize_lfw (it prints progress)
        print("Training classifier for", args.epochs_clf, "epochs")
        classifier = train_clf_simple(classifier, train_loader, epochs=args.epochs_clf, lr=args.lr_clf)
        save_ckpt(classifier.state_dict(), clf_ckpt_out)

    # Optionally, you can also evaluate classifier on val set here (left as an exercise)

    # --- ANONYMIZER TRAINING ---
    anony = Anonymizer()
    anony.to(device)
    vgg = VGGPerceptual(device=device)

    anon_ckpt_out = os.path.join(args.save_dir, "anonymizer_epoch_latest.pth")
    if args.resume_anon:
        anony = load_state_dict_maybe(anony, args.resume_anon, device)
        print("Resumed anonymizer from:", args.resume_anon)
    else:
        print("Training anonymizer for", args.epochs_anon, "epochs")
        # use the training helper from anonymize_lfw
        anony = train_anon_simple(anony, classifier, vgg, train_loader,
                                  epochs=args.epochs_anon, lr=args.lr_anon)
        save_ckpt(anony.state_dict(), anon_ckpt_out)

    print("Training flow complete. Final classifier saved to:", clf_ckpt_out, "anonymizer saved to:", anon_ckpt_out)
    print("If you want to resume later, pass --resume-clf and --resume-anon with those files.")

if __name__ == "__main__":
    main()
