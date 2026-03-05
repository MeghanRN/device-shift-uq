from dataclasses import dataclass
import torch
import torchaudio

@dataclass
class LogMelConfig:
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    n_mels: int = 128

class LogMel:
    def __init__(self, cfg: LogMelConfig):
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            power=2.0,
        )

    def __call__(self, wav_1d: torch.Tensor) -> torch.Tensor:
        x = wav_1d.unsqueeze(0)
        m = self.mel(x)
        m = torch.log1p(m)
        return m  # (1,F,T)

@dataclass
class STFTLogPowConfig:
    sample_rate: int = 12000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024

class STFTLogPow:
    def __init__(self, cfg: STFTLogPowConfig):
        self.cfg = cfg
        self.window = torch.hann_window(cfg.win_length)

    def __call__(self, wav_1d: torch.Tensor) -> torch.Tensor:
        X = torch.stft(
            wav_1d, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length, window=self.window.to(wav_1d.device),
            return_complex=True
        )
        P = (X.abs() ** 2.0)
        P = torch.log1p(P)
        return P.unsqueeze(0)

def pad_or_crop_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.numel() == target_len:
        return x
    if x.numel() > target_len:
        return x[:target_len]
    return torch.nn.functional.pad(x, (0, target_len - x.numel()))

def pad_or_crop_2d(x: torch.Tensor, target_T: int) -> torch.Tensor:
    T = x.shape[-1]
    if T == target_T:
        return x
    if T > target_T:
        return x[..., :target_T]
    return torch.nn.functional.pad(x, (0, target_T - T))
