from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.io import loadmat, wavfile
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset

from datasets.paths import default_paths
from features.tf_repr import (
    LogMel,
    LogMelConfig,
    STFTLogPow,
    STFTLogPowConfig,
    pad_or_crop_1d,
    pad_or_crop_2d,
)


def ensure_cft(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    if x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 1:
        x = x.squeeze(0)

    if x.ndim == 2:
        x = x.unsqueeze(0)

    if x.ndim != 3:
        raise RuntimeError(f"Expected (C,F,T), got {tuple(x.shape)}")

    return x


def _normalize_audio_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x = x.astype(np.float32) / max(abs(info.min), info.max)
    else:
        x = x.astype(np.float32)

    if x.ndim == 2:
        if x.shape[0] < x.shape[1]:
            x = x.mean(axis=0)
        else:
            x = x.mean(axis=1)

    return x.astype(np.float32)


def _robust_read_wav(path: str) -> tuple[np.ndarray, int]:
    try:
        wav_np, sr = sf.read(path, dtype="float32", always_2d=True)
        wav_np = wav_np.mean(axis=1)
        return wav_np.astype(np.float32), int(sr)
    except Exception:
        pass

    sr, wav_np = wavfile.read(path)
    wav_np = _normalize_audio_np(wav_np)
    return wav_np.astype(np.float32), int(sr)


def _resample_if_needed(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wav.astype(np.float32)
    wav_rs = resample_poly(wav, sr_out, sr_in)
    return wav_rs.astype(np.float32)


def _read_cwru_mat(path: str, preferred_channel: str = "DE") -> tuple[np.ndarray, int]:
    mat = loadmat(path)

    order = [preferred_channel]
    for ch in ["DE", "FE", "BA"]:
        if ch not in order:
            order.append(ch)

    chosen = None
    for ch in order:
        suffix = f"{ch}_time"
        for k in mat.keys():
            if k.endswith(suffix):
                chosen = k
                break
        if chosen is not None:
            break

    if chosen is None:
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            arr = np.asarray(v)
            if arr.size > 100 and np.issubdtype(arr.dtype, np.number):
                chosen = k
                break

    if chosen is None:
        raise RuntimeError(f"Could not find a usable signal in {path}")

    sig = np.asarray(mat[chosen]).squeeze().astype(np.float32)
    sr = 12000
    return sig, sr


class GenericTFDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        task_type: str,
        transform,
        target_frames: int,
        audio_sr: int,
        target_seconds: float,
        multilabel_cols: Optional[Sequence[str]] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.task_type = task_type
        self.transform = transform
        self.target_frames = int(target_frames)
        self.audio_sr = int(audio_sr)
        self.target_seconds = float(target_seconds)
        self.multilabel_cols = list(multilabel_cols) if multilabel_cols is not None else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        path = str(r["filepath"])
        domain = str(r.get("domain", ""))
        sid = str(r.get("sample_id", i))

        if path.lower().endswith(".wav"):
            wav_np, sr = _robust_read_wav(path)
            wav_np = _resample_if_needed(wav_np, sr, self.audio_sr)
            wav = torch.from_numpy(wav_np)

        elif path.lower().endswith(".mat"):
            preferred = domain if domain in {"DE", "FE", "BA"} else "DE"
            sig_np, sr = _read_cwru_mat(path, preferred_channel=preferred)            
            sig_np = _resample_if_needed(sig_np, sr, self.audio_sr)
            wav = torch.from_numpy(sig_np)

        else:
            raise RuntimeError(f"Unsupported file type: {path}")

        wav = pad_or_crop_1d(wav, int(self.audio_sr * self.target_seconds))
        x = self.transform(wav)
        x = ensure_cft(x)
        x = pad_or_crop_2d(x, self.target_frames)

        while x.ndim > 3 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise RuntimeError(f"Expected (1,F,T) but got {tuple(x.shape)}")
        x = ensure_cft(x)

        if self.task_type == "single_label":
            y = torch.tensor(int(r["y"]), dtype=torch.long)
        else:
            if self.multilabel_cols is None:
                raise RuntimeError("multilabel_cols is None but task_type is multi_label")
            y = torch.tensor(r[self.multilabel_cols].astype(float).values, dtype=torch.float32)

        return x, y, domain, sid


def load_task(dataset: str) -> Tuple[str, int, int, object, Optional[int], Optional[Sequence[str]]]:
    paths = default_paths(dataset)
    splits = paths.splits

    if dataset == "dcase":
        classes = [l.strip() for l in open(os.path.join(splits, "classes.txt"), encoding="utf-8") if l.strip()]
        cfg = LogMelConfig(sample_rate=44100)
        return "single_label", len(classes), cfg.sample_rate, LogMel(cfg), 100, None

    if dataset == "sonyc":
        labels = [l.strip() for l in open(os.path.join(splits, "labels.txt"), encoding="utf-8") if l.strip()]
        cfg = LogMelConfig(sample_rate=44100)
        return "multi_label", len(labels), cfg.sample_rate, LogMel(cfg), 100, labels

    if dataset == "cwru":
        classes = [l.strip() for l in open(os.path.join(splits, "classes.txt"), encoding="utf-8") if l.strip()]
        cfg = STFTLogPowConfig(sample_rate=12000)
        return "single_label", len(classes), cfg.sample_rate, STFTLogPow(cfg), 200, None

    raise ValueError(dataset)


def make_loaders(dataset: str, batch_size: int = 64, num_workers: int = 0):
    paths = default_paths(dataset)
    task_type, num_outputs, sr, tfm, target_frames, _labels = load_task(dataset)

    def read_multilabel_cols():
        df0 = pd.read_csv(os.path.join(paths.splits, "train.csv"), nrows=1)
        return [c for c in df0.columns if c not in ("filepath", "domain", "sample_id")]

    ml_cols = read_multilabel_cols() if task_type == "multi_label" else None

    def ds(split: str) -> GenericTFDataset:
        return GenericTFDataset(
            os.path.join(paths.splits, f"{split}.csv"),
            task_type=task_type,
            transform=tfm,
            target_frames=target_frames,
            audio_sr=sr,
            target_seconds=1.0,
            multilabel_cols=ml_cols,
        )

    train_ds, val_ds, id_ds, sh_ds = ds("train"), ds("val"), ds("test_id"), ds("test_shift")

    def dl(d: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            d,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )

    return task_type, num_outputs, dl(train_ds, True), dl(val_ds), dl(id_ds), dl(sh_ds)