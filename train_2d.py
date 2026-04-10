#!/usr/bin/env python3
"""
Step 2: Train a 2D ResNet on random frames; evaluate video-level accuracy by
averaging softmax predictions over multiple random frames per test video.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .datasets import VideoFrameDataset
from .metrics_plots import print_metrics, save_confusion_matrix
from .models import build_resnet18_2d
from .video_io import imagenet_normalize, read_random_frame


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_frame_level(model, manifest_csv: Path, split: str, device, frame_size: int):
    df = pd.read_csv(manifest_csv)
    df = df[df["split"] == split]
    model.eval()
    correct = 0
    total = 0
    preds = []
    labels = []
    rng = random.Random(config.SEED)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"frame eval {split}", leave=False):
        x = read_random_frame(row["path"], frame_size, rng)
        x = imagenet_normalize(x).unsqueeze(0).to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).item()
        preds.append(pred)
        labels.append(int(row["class_id"]))
        if pred == int(row["class_id"]):
            correct += 1
        total += 1
    return correct / max(total, 1), preds, labels


@torch.no_grad()
def evaluate_video_averaging(
    model,
    manifest_csv: Path,
    split: str,
    device,
    frame_size: int,
    num_frames: int,
    seed: int,
):
    df = pd.read_csv(manifest_csv)
    df = df[df["split"] == split].reset_index(drop=True)
    model.eval()
    preds = []
    labels = []
    rng = random.Random(seed)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"video avg K={num_frames}", leave=False):
        prob_sum = None
        for _ in range(num_frames):
            x = read_random_frame(row["path"], frame_size, rng)
            x = imagenet_normalize(x).unsqueeze(0).to(device)
            prob = model(x).softmax(dim=1)
            prob_sum = prob if prob_sum is None else prob_sum + prob
        prob_avg = prob_sum / num_frames
        preds.append(prob_avg.argmax(dim=1).item())
        labels.append(int(row["class_id"]))
    acc = sum(p == l for p, l in zip(preds, labels)) / max(len(labels), 1)
    return acc, preds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=config.MANIFEST_CSV)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--frame-size", type=int, default=config.FRAME_SIZE)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true", help="ImageNet weights for 2D ResNet")
    parser.add_argument("--out-dir", type=Path, default=config.RUNS_DIR / "2d_cnn")
    parser.add_argument("--frame-repeats", type=int, default=4, help="Samples per video per epoch")
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    num_classes = len(config.CLASS_NAMES)
    model = build_resnet18_2d(num_classes, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = VideoFrameDataset(
        args.manifest,
        "train",
        args.frame_size,
        frames_per_epoch_factor=args.frame_repeats,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, _, _ = evaluate_frame_level(
            model, args.manifest, "val", device, args.frame_size
        )
        history.append({"epoch": epoch, "loss": loss, "val_frame_acc": val_acc})
        print(f"Epoch {epoch}: loss={loss:.4f} val_frame_acc={val_acc:.4f}")
        if val_acc >= best_val:
            best_val = val_acc
            torch.save(model.state_dict(), args.out_dir / "best_2d.pt")

    model.load_state_dict(torch.load(args.out_dir / "best_2d.pt", map_location=device))

    # Test: single random frame per video (baseline)
    test_frame_acc, pred_f, y_f = evaluate_frame_level(
        model, args.manifest, "test", device, args.frame_size
    )
    save_confusion_matrix(
        y_f,
        pred_f,
        config.CLASS_NAMES,
        args.out_dir / "confusion_test_single_frame.png",
        title="2D CNN — test (1 random frame / video)",
    )
    print("\n=== Test (single random frame per video) ===")
    print(print_metrics(y_f, pred_f, config.CLASS_NAMES))

    # Video-level: average over K random frames
    results = {"test_single_frame_acc": test_frame_acc, "video_averaging": {}}
    for k in [1, 3, 5, 8]:
        acc, pred_v, y_v = evaluate_video_averaging(
            model, args.manifest, "test", device, args.frame_size, k, seed=config.SEED + k
        )
        results["video_averaging"][f"k{k}"] = acc
        save_confusion_matrix(
            y_v,
            pred_v,
            config.CLASS_NAMES,
            args.out_dir / f"confusion_test_avg_k{k}.png",
            title=f"2D CNN — test (avg {k} frames)",
        )
        print(f"\n=== Test video-level (average softmax over K={k} frames) acc={acc:.4f} ===")
        print(print_metrics(y_v, pred_v, config.CLASS_NAMES))

    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved metrics to {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
