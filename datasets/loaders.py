from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import pandas as pd
import soundfile as sf
import torch
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
    """
    Force spectrogram into (C,F,T) with C=1.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    # common cases: (F,T) or (1,F,T) or (1,1,F,T)
    if x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 1:
        x = x.squeeze(0)  # (1,1,F,T) -> (1,F,T)

    if x.ndim == 2:
        x = x.unsqueeze(0)  # (F,T) -> (1,F,T)

    if x.ndim != 3:
        raise RuntimeError(f"Expected (C,F,T), got {tuple(x.shape)}")

    # If someone produced C!=1, keep it but still (C,F,T)
    return x


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

        if not path.lower().endswith(".wav"):
            raise RuntimeError(f"Non-wav file encountered: {path}")

        wav_np, sr = sf.read(path, dtype="float32", always_2d=True)  # (T,C)
        wav = torch.from_numpy(wav_np).T  # (C,T)
        wav = wav.mean(dim=0)  # (T,)

        if int(sr) != self.audio_sr:
            raise RuntimeError(
                f"Sample rate mismatch for {path}: got {sr}, expected {self.audio_sr}"
            )

        wav = pad_or_crop_1d(wav, int(self.audio_sr * self.target_seconds))

        x = self.transform(wav)          # expect (1,F,T) or (F,T) or (1,1,F,T)
        x = ensure_cft(x)                # -> (1,F,T)
        x = pad_or_crop_2d(x, self.target_frames)
        # force x to (C,F,T) with C=1 (remove any extra singleton dims)
        while x.ndim > 3 and x.shape[0] == 1:
            x = x.squeeze(0)   # (1,1,F,T)->(1,F,T) or (1,F,T)->(F,T) won't happen here
        if x.ndim == 2:
            x = x.unsqueeze(0) # (F,T)->(1,F,T)

        if x.ndim != 3:
            raise RuntimeError(f"Expected (1,F,T) but got {tuple(x.shape)}")
        x = ensure_cft(x)                # re-check

        # Labels
        if self.task_type == "single_label":
            y = torch.tensor(int(r["y"]), dtype=torch.long)
        else:
            if self.multilabel_cols is None:
                raise RuntimeError("multilabel_cols is None but task_type is multi_label")
            y = torch.tensor(r[self.multilabel_cols].astype(float).values, dtype=torch.float32)

        return x, y, domain, sid


def load_task(dataset: str) -> Tuple[str, int, int, object, int, Optional[Sequence[str]]]:
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

    # CPU windows: pin_memory False
    def dl(d: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(d, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)

    return task_type, num_outputs, dl(train_ds, True), dl(val_ds), dl(id_ds), dl(sh_ds)