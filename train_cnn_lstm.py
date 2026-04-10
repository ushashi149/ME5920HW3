

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .datasets import VideoSequenceDataset
from .metrics_plots import print_metrics, save_confusion_matrix
from .models import CNNLSTM


def train_one_epoch(model, loader, optimizer, criterion, device, freeze_backbone: bool):
    if freeze_backbone:
        model.set_backbone_requires_grad(False)
    else:
        model.set_backbone_requires_grad(True)
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in tqdm(loader, desc="train lstm", leave=False):
        # x: (B, T, C, H, W)
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
def evaluate_loader(model, loader, device):
    model.eval()
    preds = []
    labels = []
    for x, y in tqdm(loader, desc="eval lstm", leave=False):
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().tolist()
        preds.extend(pred)
        labels.extend(y.tolist())
    acc = sum(p == l for p, l in zip(preds, labels)) / max(len(labels), 1)
    return acc, preds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=config.MANIFEST_CSV)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--freeze-epochs", type=int, default=5, help="Train LSTM head only first")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--lr-full", type=float, default=1e-4)
    parser.add_argument("--frame-size", type=int, default=config.FRAME_SIZE)
    parser.add_argument("--seq-len", type=int, default=config.LSTM_SEQ_LEN)
    parser.add_argument("--clip-stride", type=int, default=config.CLIP_STRIDE)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=config.RUNS_DIR / "cnn_lstm")
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    num_classes = len(config.CLASS_NAMES)
    model = CNNLSTM(
        num_classes,
        hidden_size=args.hidden,
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    train_ds = VideoSequenceDataset(
        args.manifest,
        "train",
        args.frame_size,
        args.seq_len,
        args.clip_stride,
        repeats=args.repeats,
    )
    val_ds = VideoSequenceDataset(
        args.manifest,
        "val",
        args.frame_size,
        args.seq_len,
        args.clip_stride,
        repeats=1,
    )
    test_ds = VideoSequenceDataset(
        args.manifest,
        "test",
        args.frame_size,
        args.seq_len,
        args.clip_stride,
        repeats=1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = 0.0

    freeze_n = max(0, min(args.freeze_epochs, args.epochs))

    # Phase 1: LSTM + head only
    if freeze_n > 0:
        optimizer = torch.optim.Adam(
            list(model.lstm.parameters()) + list(model.head.parameters()),
            lr=args.lr_head,
        )
        for epoch in range(1, freeze_n + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device, True)
            val_acc, _, _ = evaluate_loader(model, val_loader, device)
            history.append(
                {"epoch": epoch, "phase": "frozen_backbone", "loss": loss, "val_acc": val_acc}
            )
            print(f"[frozen] Epoch {epoch}: loss={loss:.4f} val_acc={val_acc:.4f}")
            if val_acc >= best_val:
                best_val = val_acc
                torch.save(model.state_dict(), args.out_dir / "best_lstm.pt")

    # Phase 2: fine-tune full model
    if args.epochs > freeze_n:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_full)
        for epoch in range(freeze_n + 1, args.epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device, False)
            val_acc, _, _ = evaluate_loader(model, val_loader, device)
            history.append(
                {"epoch": epoch, "phase": "full", "loss": loss, "val_acc": val_acc}
            )
            print(f"[full] Epoch {epoch}: loss={loss:.4f} val_acc={val_acc:.4f}")
            if val_acc >= best_val:
                best_val = val_acc
                torch.save(model.state_dict(), args.out_dir / "best_lstm.pt")

    model.load_state_dict(torch.load(args.out_dir / "best_lstm.pt", map_location=device))
    test_acc, pred_t, y_t = evaluate_loader(model, test_loader, device)
    save_confusion_matrix(
        y_t,
        pred_t,
        config.CLASS_NAMES,
        args.out_dir / "confusion_test.png",
        title="CNN + LSTM — test",
    )
    print("\n=== Test ===")
    print(print_metrics(y_t, pred_t, config.CLASS_NAMES))

    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump({"test_acc": test_acc, "val_best": best_val}, f, indent=2)
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
