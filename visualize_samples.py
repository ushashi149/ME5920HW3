

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from . import config
from .models import build_resnet18_2d
from .video_io import imagenet_normalize, read_random_frame


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=config.MANIFEST_CSV)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frame-size", type=int, default=config.FRAME_SIZE)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--out", type=Path, default=config.RUNS_DIR / "2d_cnn" / "sample_predictions.png")
    parser.add_argument("--n-per-class", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_2d(len(config.CLASS_NAMES), pretrained=args.pretrained).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    df = pd.read_csv(args.manifest)
    df = df[df["split"] == "test"]
    rng = random.Random(config.SEED)

    rows = []
    for cname in config.CLASS_NAMES:
        sub = df[df["class"] == cname]
        if len(sub) == 0:
            continue
        take = sub.sample(n=min(args.n_per_class, len(sub)), random_state=config.SEED)
        rows.append(take)
    if not rows:
        print("No test rows.")
        return
    pick = pd.concat(rows, ignore_index=True)

    n = len(pick)
    cols = min(5, n)
    r = (n + cols - 1) // cols
    fig, ax_arr = plt.subplots(r, cols, figsize=(3 * cols, 3 * r))
    if n == 1:
        axes = [ax_arr]
    else:
        axes = ax_arr.flatten()

    for ax, (_, row) in zip(axes, pick.iterrows()):
        x = read_random_frame(row["path"], args.frame_size, rng)
        x_n = imagenet_normalize(x).unsqueeze(0).to(device)
        pred = model(x_n).argmax(dim=1).item()
        img = x.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.axis("off")
        true = row["class"]
        pred_n = config.CLASS_NAMES[pred]
        color = "green" if pred_n == true else "red"
        ax.set_title(f"T:{true}\nP:{pred_n}", color=color, fontsize=9)

    for j in range(len(pick), len(axes)):
        axes[j].axis("off")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
