

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config
from .video_io import (
    imagenet_normalize,
    imagenet_normalize_cthw,
    imagenet_normalize_tchw,
    read_consecutive_clip,
    read_random_frame,
)


class VideoFrameDataset(Dataset):
    

    def __init__(
        self,
        manifest_csv: Path,
        split: str,
        frame_size: int,
        frames_per_epoch_factor: int = 1,
        seed: int | None = None,
    ):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.frame_size = frame_size
        self.factor = max(1, frames_per_epoch_factor)
        self.base_seed = seed if seed is not None else config.SEED
        self._rng = random.Random(self.base_seed)

    def __len__(self) -> int:
        return len(self.df) * self.factor

    def __getitem__(self, idx: int):
        row_idx = idx % len(self.df)
        row = self.df.iloc[row_idx]
        path = row["path"]
        label = int(float(row["class_id"]))
        rng = random.Random(self.base_seed + idx * 10007)
        x = read_random_frame(path, self.frame_size, rng)
        x = imagenet_normalize(x)
        return x, label


class VideoClipDataset(Dataset):
   

    def __init__(
        self,
        manifest_csv: Path,
        split: str,
        frame_size: int,
        clip_length: int,
        stride: int,
        repeats: int = 2,
        seed: int | None = None,
    ):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.frame_size = frame_size
        self.clip_length = clip_length
        self.stride = stride
        self.repeats = max(1, repeats)
        self.base_seed = seed if seed is not None else config.SEED

    def __len__(self) -> int:
        return len(self.df) * self.repeats

    def __getitem__(self, idx: int):
        row_idx = idx % len(self.df)
        row = self.df.iloc[row_idx]
        path = row["path"]
        label = int(float(row["class_id"]))
        rng = random.Random(self.base_seed + idx * 7919)
        clip = read_consecutive_clip(
            path, self.clip_length, self.frame_size, self.stride, rng
        )
        # (T,C,H,W) -> (C,T,H,W)
        clip = clip.permute(1, 0, 2, 3)
        clip = imagenet_normalize_cthw(clip)
        return clip, label


class VideoSequenceDataset(Dataset):
   

    def __init__(
        self,
        manifest_csv: Path,
        split: str,
        frame_size: int,
        seq_len: int,
        stride: int,
        repeats: int = 2,
        seed: int | None = None,
    ):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.frame_size = frame_size
        self.seq_len = seq_len
        self.stride = stride
        self.repeats = max(1, repeats)
        self.base_seed = seed if seed is not None else config.SEED

    def __len__(self) -> int:
        return len(self.df) * self.repeats

    def __getitem__(self, idx: int):
        row_idx = idx % len(self.df)
        row = self.df.iloc[row_idx]
        path = row["path"]
        label = int(float(row["class_id"]))
        rng = random.Random(self.base_seed + idx * 4243)
        clip = read_consecutive_clip(
            path, self.seq_len, self.frame_size, self.stride, rng
        )
        clip = imagenet_normalize_tchw(clip)
        return clip, label
