import os, glob
import kagglehub
import numpy as np
import pandas as pd
from datasets.paths import default_paths
from utils.io import ensure_dir

DATASET = "sufian79/cwru-mat-full-dataset"

def main():
    paths = default_paths("cwru")
    ensure_dir(paths.splits)

    src = kagglehub.dataset_download(DATASET)
    mats = glob.glob(os.path.join(src,"**","*.mat"), recursive=True)
    csvs = glob.glob(os.path.join(src,"**","*.csv"), recursive=True)
    files = mats if mats else csvs
    if not files:
        raise FileNotFoundError("No .mat/.csv files found in cwru download")

    df = pd.DataFrame({"filepath": files})
    df["sample_id"] = df["filepath"].apply(lambda p: os.path.basename(p))
    name = df["sample_id"].str.lower()
    df["domain"] = name.str.extract(r"(0hp|1hp|2hp|3hp)", expand=False).fillna("unknown")

    def fault_label(n):
        if "normal" in n or "healthy" in n: return "normal"
        if "inner" in n or "ir" in n: return "inner"
        if "outer" in n or "or" in n: return "outer"
        if "ball" in n: return "ball"
        return "unknown"
    df["label_str"] = name.apply(fault_label)
    df = df[df["label_str"]!="unknown"].copy()
    classes = sorted(df["label_str"].unique().tolist())
    lut = {c:i for i,c in enumerate(classes)}
    df["y"] = df["label_str"].map(lut).astype(int)

    domains = sorted(df["domain"].unique().tolist())
    id_domain = "0hp" if "0hp" in domains else domains[0]
    id_df = df[df["domain"]==id_domain].copy()
    sh_df = df[df["domain"]!=id_domain].copy()

    rng = np.random.default_rng(0)
    idx = np.arange(len(id_df)); rng.shuffle(idx)
    n_train=int(0.7*len(idx)); n_val=int(0.15*len(idx))
    train=id_df.iloc[idx[:n_train]].copy()
    val=id_df.iloc[idx[n_train:n_train+n_val]].copy()
    test_id=id_df.iloc[idx[n_train+n_val:]].copy()
    test_shift=sh_df.copy()

    out_cols=["filepath","y","domain","sample_id"]
    train[out_cols].to_csv(os.path.join(paths.splits,"train.csv"), index=False)
    val[out_cols].to_csv(os.path.join(paths.splits,"val.csv"), index=False)
    test_id[out_cols].to_csv(os.path.join(paths.splits,"test_id.csv"), index=False)
    test_shift[out_cols].to_csv(os.path.join(paths.splits,"test_shift.csv"), index=False)

    with open(os.path.join(paths.splits,"classes.txt"),"w",encoding="utf-8") as f:
        for c in classes: f.write(c+"\n")

    print("CWRU splits written to", paths.splits)

if __name__ == "__main__":
    main()
