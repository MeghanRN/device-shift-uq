import os
import kagglehub
import pandas as pd
import numpy as np
from datasets.paths import default_paths
from utils.io import ensure_dir

DATASET = "mahdyr/dcase-2025-task1"

def main():
    paths = default_paths("dcase")
    ensure_dir(paths.splits)

    src = kagglehub.dataset_download(DATASET)
    top = os.listdir(src)
    base = os.path.join(src, "dataset") if "dataset" in top else src

    train_csv = os.path.join(base, "train.csv")
    test_csv  = os.path.join(base, "test.csv")
    train_dir = os.path.join(base, "train")
    test_dir  = os.path.join(base, "test")

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)
    train_df.columns = [c.strip().lower() for c in train_df.columns]
    test_df.columns  = [c.strip().lower() for c in test_df.columns]

    def add_fields(df, audio_root):
        out = df.copy()
        out["device_id"] = out["device"].astype(str).str.upper()
        out["filepath"] = out["filename"].apply(lambda f: os.path.join(audio_root, f))
        out["location_id"] = out["filename"].astype(str).str.extract(r"^[^-]+-([^-]+)-", expand=False)
        return out

    train_df = add_fields(train_df, train_dir)
    test_df  = add_fields(test_df, test_dir)

    class_names = sorted(train_df["label"].astype(str).unique().tolist())
    lut = {c:i for i,c in enumerate(class_names)}
    train_df["y"] = train_df["label"].astype(str).map(lut).astype(int)
    test_df["y"] = test_df["label"].astype(str).map(lut).fillna(-1).astype(int)

    ABC = {"A","B","C"}
    S16 = {f"S{i}" for i in range(1,7)}

    train_abc = train_df[train_df["device_id"].isin(ABC)].copy()
    # location-heldout val
    rng = np.random.default_rng(0)
    locs = train_abc["location_id"].dropna().astype(str).unique().tolist()
    rng.shuffle(locs)
    val_locs = set(locs[:max(1,int(0.2*len(locs)))])
    val = train_abc[train_abc["location_id"].astype(str).isin(val_locs)].copy()
    train = train_abc[~train_abc["location_id"].astype(str).isin(val_locs)].copy()

    test_id = test_df[test_df["device_id"].isin(ABC)].copy()
    test_shift = test_df[test_df["device_id"].isin(S16)].copy()

    for df in (train,val,test_id,test_shift):
        df["domain"] = df["device_id"].astype(str)
        df["sample_id"] = df["filename"].astype(str)

    out_cols = ["filepath","y","domain","sample_id"]
    train[out_cols].to_csv(os.path.join(paths.splits,"train.csv"), index=False)
    val[out_cols].to_csv(os.path.join(paths.splits,"val.csv"), index=False)
    test_id[out_cols].to_csv(os.path.join(paths.splits,"test_id.csv"), index=False)
    test_shift[out_cols].to_csv(os.path.join(paths.splits,"test_shift.csv"), index=False)

    with open(os.path.join(paths.splits,"classes.txt"),"w",encoding="utf-8") as f:
        for c in class_names: f.write(c+"\n")

    print("DCASE splits written to", paths.splits)

if __name__ == "__main__":
    main()
