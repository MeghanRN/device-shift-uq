# Device-Shift Uncertainty (DCASE, SONYC, CWRU)

This repo is a minimal, reproducible codebase for studying **uncertainty estimation under device/sensor-induced distribution shift** across three time-domain sensing domains:

- **DCASE 2025 Task 1 (TAU Urban Acoustic Scenes, multi-device)** via Kaggle mirror (`mahdyr/dcase-2025-task1`)
- **SONYC sensor network audio** via Kaggle (`christopheronyiuke/sonyc-data`)
- **CWRU bearing vibration** via Kaggle mirror (`brjapon/cwru-bearing-datasets`)

## Quickstart

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Download datasets (KaggleHub)
```bash
python datasets/download_dcase.py
python datasets/download_sonyc.py
python datasets/download_cwru.py
```

### 3) Train + evaluate (example)
```bash
python training/train.py --dataset dcase --model cpmobile --seed 0
python training/evaluate.py --dataset dcase --model cpmobile --seed 0
```

Outputs are written to `results/<dataset>/<model>/seed_<seed>/`.

## Folder layout
- `datasets/` download + dataset loaders and split builders
- `features/` log-mel / spectrogram transforms
- `models/` architectures with a unified interface (CPMobile, DynaCP, GRU-CNN)
- `training/` train/eval loops
- `uq/` uncertainty scores + shift detection
