import os, glob
import kagglehub
import pandas as pd
import numpy as np
from datasets.paths import default_paths
from utils.io import ensure_dir

DATASET = "christopheronyiuke/sonyc-data"

def main():
    paths = default_paths("sonyc")
    ensure_dir(paths.splits)

    src = kagglehub.dataset_download(DATASET)

    ann = None
    for cand in [os.path.join(src, "annotations.csv")] + glob.glob(os.path.join(src, "**", "annotations.csv"), recursive=True):
        if os.path.exists(cand):
            ann = cand
            break
    if ann is None:
        raise FileNotFoundError("annotations.csv not found")

    wavs = glob.glob(os.path.join(src, "**", "*.wav"), recursive=True)
    if not wavs:
        raise FileNotFoundError("No wavs found; use christopheronyiuke/sonyc-data (contains audio).")

    # map real wav paths by basename
    by_name = {os.path.basename(p): p for p in wavs}

    df = pd.read_csv(ann)
    if "sensor_id" not in df.columns:
        raise ValueError("annotations.csv missing sensor_id")

    label_cols = [c for c in df.columns if c.endswith("_presence")]
    if not label_cols:
        raise ValueError("Could not find *_presence label columns")

    g = df.groupby(["audio_filename", "sensor_id"], as_index=False)[label_cols].max()

    def resolve_fp(fname: str) -> str:
        base = os.path.basename(fname)
        if base not in by_name:
            raise FileNotFoundError(f"WAV not found for {fname}")
        return by_name[base]

    g["filepath"] = g["audio_filename"].apply(resolve_fp)
    g["domain"] = g["sensor_id"].astype(str)
    g["sample_id"] = g["audio_filename"].astype(str)

    sensors = sorted(g["domain"].unique().tolist())
    rng = np.random.default_rng(0)
    rng.shuffle(sensors)

    g1 = set(sensors[:int(0.7 * len(sensors))])
    g2 = set(sensors[int(0.7 * len(sensors)):])

    g1_df = g[g["domain"].isin(g1)].copy()
    g2_df = g[g["domain"].isin(g2)].copy()

    idx = np.arange(len(g1_df))
    rng.shuffle(idx)
    n_train = int(0.7 * len(idx))
    n_val = int(0.15 * len(idx))

    train = g1_df.iloc[idx[:n_train]].copy()
    val = g1_df.iloc[idx[n_train:n_train + n_val]].copy()
    test_id = g1_df.iloc[idx[n_train + n_val:]].copy()
    test_shift = g2_df.copy()

    out_cols = ["filepath", "domain", "sample_id"] + label_cols
    train[out_cols].to_csv(os.path.join(paths.splits, "train.csv"), index=False)
    val[out_cols].to_csv(os.path.join(paths.splits, "val.csv"), index=False)
    test_id[out_cols].to_csv(os.path.join(paths.splits, "test_id.csv"), index=False)
    test_shift[out_cols].to_csv(os.path.join(paths.splits, "test_shift.csv"), index=False)

    with open(os.path.join(paths.splits, "labels.txt"), "w", encoding="utf-8") as f:
        for c in label_cols:
            f.write(c + "\n")

    print("SONYC splits written to", paths.splits)

if __name__ == "__main__":
    main()