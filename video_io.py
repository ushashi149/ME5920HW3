
from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch


def open_capture(path: Path | str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def frame_count(cap: cv2.VideoCapture) -> int:
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        # Fallback: read through (slow)
        n = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return n


def read_frame_at(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    # BGR -> RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def read_random_frame(path: Path | str, size: int, rng: random.Random) -> torch.Tensor:
    
    cap = open_capture(path)
    try:
        n = frame_count(cap)
        if n <= 0:
            raise RuntimeError(f"No frames in {path}")
        idx = rng.randint(0, n - 1)
        frame = read_frame_at(cap, idx)
        if frame is None:
            idx = 0
            frame = read_frame_at(cap, idx)
        if frame is None:
            raise RuntimeError(f"Could not read frame from {path}")
        return _resize_to_tensor(frame, size)
    finally:
        cap.release()


def read_consecutive_clip(
    path: Path | str,
    length: int,
    size: int,
    stride: int,
    rng: random.Random,
) -> torch.Tensor:
    
    cap = open_capture(path)
    try:
        n = frame_count(cap)
        if n <= 0:
            raise RuntimeError(f"No frames in {path}")
        max_start = n - (1 + (length - 1) * stride)
        if max_start < 0:
            # Video too short: repeat last frame
            frames = []
            f0 = read_frame_at(cap, 0)
            if f0 is None:
                raise RuntimeError(f"Empty {path}")
            t0 = _resize_to_tensor(f0, size)
            for _ in range(length):
                frames.append(t0.clone())
            return torch.stack(frames, dim=0)
        start = rng.randint(0, max_start)
        frames = []
        for i in range(length):
            idx = start + i * stride
            fr = read_frame_at(cap, idx)
            if fr is None:
                fr = read_frame_at(cap, start)
            frames.append(_resize_to_tensor(fr, size))
        return torch.stack(frames, dim=0)
    finally:
        cap.release()


def read_clip_fixed_indices(
    path: Path | str, indices: list[int], size: int
) -> torch.Tensor:
    cap = open_capture(path)
    try:
        frames = []
        n = frame_count(cap)
        for idx in indices:
            j = min(max(0, idx), n - 1)
            fr = read_frame_at(cap, j)
            if fr is None:
                fr = read_frame_at(cap, 0)
            frames.append(_resize_to_tensor(fr, size))
        return torch.stack(frames, dim=0)
    finally:
        cap.release()


def _resize_to_tensor(frame_bgr_rgb: np.ndarray, size: int) -> torch.Tensor:
    h, w = frame_bgr_rgb.shape[:2]
    if h != size or w != size:
        frame_bgr_rgb = cv2.resize(frame_bgr_rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(frame_bgr_rgb).permute(2, 0, 1).float() / 255.0
    return x


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(
        -1, 1, 1
    )
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(
        -1, 1, 1
    )
    if x.dim() == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    return (x - mean) / std


def imagenet_normalize_cthw(x: torch.Tensor) -> torch.Tensor:
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(
        3, 1, 1, 1
    )
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(
        3, 1, 1, 1
    )
    return (x - mean) / std


def imagenet_normalize_tchw(x: torch.Tensor) -> torch.Tensor:
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(
        1, 3, 1, 1
    )
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(
        1, 3, 1, 1
    )
    return (x - mean) / std
