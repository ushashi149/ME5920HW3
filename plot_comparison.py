

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from . import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--twod-dir", type=Path, default=config.RUNS_DIR / "2d_cnn")
    parser.add_argument("--threed-dir", type=Path, default=config.RUNS_DIR / "3d_cnn")
    parser.add_argument("--lstm-dir", type=Path, default=config.RUNS_DIR / "cnn_lstm")
    parser.add_argument("--out", type=Path, default=config.RUNS_DIR / "comparison.png")
    args = parser.parse_args()

    names = []
    scores = []

    p2d = args.twod_dir / "metrics.json"
    if p2d.is_file():
        with open(p2d) as f:
            d = json.load(f)
        
        if "video_averaging" in d and d["video_averaging"]:
            k = max(d["video_averaging"], key=lambda x: int(x.replace("k", "", 1)))
            names.append(f"2D CNN ({k} frames avg)")
            scores.append(d["video_averaging"][k])
        else:
            names.append("2D CNN")
            scores.append(d.get("test_single_frame_acc", 0))

    p3d = args.threed_dir / "metrics.json"
    if p3d.is_file():
        with open(p3d) as f:
            d = json.load(f)
        names.append("3D R3D-18")
        scores.append(d.get("test_clip_acc", 0))

    pl = args.lstm_dir / "metrics.json"
    if pl.is_file():
        with open(pl) as f:
            d = json.load(f)
        names.append("CNN + LSTM")
        scores.append(d.get("test_acc", 0))

    if not scores:
        print("No metrics.json files found. Train models first.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, scores, color=["#4C72B0", "#55A868", "#C44E52"][: len(scores)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Sheep activity — model comparison (test set)")
    for i, v in enumerate(scores):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
