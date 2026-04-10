

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def discover_videos(data_roots: list[Path]) -> list[tuple[str, str]]:
    
    rows: list[tuple[str, str]] = []
    for root in data_roots:
        if not root.is_dir():
            print(f"Warning: missing directory {root}", file=sys.stderr)
            continue
        for class_name in config.CLASS_NAMES:
            sub = root / config.CLASS_TO_SUBDIR[class_name]
            if not sub.is_dir():
                continue
            for p in sub.rglob("*"):
                if p.is_file() and p.suffix in config.VIDEO_EXTENSIONS:
                    rows.append((str(p.resolve()), class_name))
    return rows


def build_manifest(
    data_roots: list[Path] | None = None,
    test_size: float = 0.2,
    val_fraction_of_train: float = 0.1,
    seed: int = config.SEED,
) -> pd.DataFrame:
    data_roots = data_roots or config.DEFAULT_DATA_ROOTS
    rows = discover_videos(data_roots)
    if not rows:
        raise RuntimeError("No video files found. Check DEFAULT_DATA_ROOTS in config.py.")

    df = pd.DataFrame(rows, columns=["path", "class"])
    df["class_id"] = df["class"].map({c: i for i, c in enumerate(config.CLASS_NAMES)})

    X = df["path"].values
    y = df["class_id"].values

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction_of_train,
        stratify=y_trainval,
        random_state=seed,
    )

    def pack(split_name: str, paths, labels) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "path": paths,
                "class_id": labels,
                "split": split_name,
            }
        )

    train_df = pack("train", X_train, y_train)
    val_df = pack("val", X_val, y_val)
    test_df = pack("test", X_test, y_test)
    out = pd.concat([train_df, val_df, test_df], ignore_index=True)
    out["class"] = out["class_id"].map(lambda i: config.CLASS_NAMES[int(i)])
    return out


def write_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")
    print(df["split"].value_counts())
    print(df.groupby(["split", "class"]).size().unstack(fill_value=0))


def write_ffmpeg_batch(
    df: pd.DataFrame,
    out_dir: Path,
    scale_long_edge: int = 1000,
    script_path: Path | None = None,
) -> None:
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_path or (config.PROCESSED_DIR / "compress_videos.sh")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'OUT="{out_dir}"',
        "mkdir -p \"$OUT\"",
    ]
    for _, row in df.iterrows():
        src = Path(row["path"])
        rel = f"{row['class']}_{src.stem}".replace(" ", "_")
        dst = out_dir / f"{rel}.mp4"
        vf = (
            f"scale='if(gt(iw,ih),{scale_long_edge},-2)':"
            f"'if(gt(iw,ih),-2,{scale_long_edge})'"
        )
        lines.append(
            f'ffmpeg -y -i "{src}" -vf "{vf}" -c:v libx264 -c:a aac "{dst}" </dev/null'
        )
    lines.append("echo Done.")
    script_path.write_text("\n".join(lines) + "\n")
    script_path.chmod(script_path.stat().st_mode | 0o111)
    print(f"Wrote ffmpeg batch script: {script_path}")


def compress_one_ffmpeg(src: Path, dst: Path, scale_long_edge: int = 1000) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale='if(gt(iw,ih),{scale_long_edge},-2)':"
        f"'if(gt(iw,ih),-2,{scale_long_edge})'"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged manifest and splits.")
    parser.add_argument("--out", type=Path, default=config.MANIFEST_CSV)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of trainval for validation.")
    parser.add_argument("--write-ffmpeg-script", action="store_true")
    parser.add_argument("--compress-one", type=str, default="", help="Optional single file to compress (debug).")
    args = parser.parse_args()

    if args.compress_one:
        src = Path(args.compress_one)
        dst = config.COMPRESSED_DIR / (src.stem + "_small.mp4")
        compress_one_ffmpeg(src, dst)
        print(f"Wrote {dst}")
        return

    df = build_manifest(test_size=args.test_size, val_fraction_of_train=args.val_frac)
    write_manifest(df, args.out)
    if args.write_ffmpeg_script:
        write_ffmpeg_batch(df, config.COMPRESSED_DIR)


if __name__ == "__main__":
    main()
