import os
import re
import requests
import numpy as np
import pandas as pd
from scipy.io import loadmat

from datasets.paths import default_paths
from utils.io import ensure_dir

BASE = "https://engineering.case.edu"
NORMAL_URL = f"{BASE}/bearingdatacenter/normal-baseline-data"
DRIVE12_URL = f"{BASE}/bearingdatacenter/12k-drive-end-bearing-fault-data"


def _fetch_html(url: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def _extract_mat_links(page_url: str):
    html = _fetch_html(page_url)

    # captures href + anchor text for .mat links
    pairs = re.findall(
        r'<a[^>]+href="([^"]+\.mat)"[^>]*>([^<]+)</a>',
        html,
        flags=re.IGNORECASE,
    )

    out = []
    for href, text in pairs:
        href = href.strip()
        text = text.strip()
        if href.startswith("/"):
            href = BASE + href
        out.append((href, text))
    return out


def _label_from_anchor(anchor: str) -> str:
    a = anchor.upper()
    if a.startswith("IR"):
        return "inner"
    if a.startswith("B"):
        return "ball"
    if a.startswith("OR"):
        return "outer"
    return "normal"


def _download_file(url: str, out_path: str, max_retries: int = 5):
    tmp_path = out_path + ".part"

    for attempt in range(max_retries):
        try:
            downloaded = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            headers = {}
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"

            r = requests.get(url, stream=True, timeout=120, headers=headers)
            if r.status_code not in (200, 206):
                r.raise_for_status()

            mode = "ab" if downloaded > 0 and r.status_code == 206 else "wb"
            with open(tmp_path, mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            os.replace(tmp_path, out_path)
            return

        except Exception as e:
            print(f"download failed ({attempt+1}/{max_retries}) for {url}: {e}")
            if attempt == max_retries - 1:
                raise

def _has_channel(mat_path: str, channel: str) -> bool:
    mat = loadmat(mat_path)
    suffix = f"{channel}_time"
    return any(k.endswith(suffix) for k in mat.keys())


def _stratified_split_id(df: pd.DataFrame, seed: int = 0):
    rng = np.random.default_rng(seed)

    trains, vals, tests = [], [], []
    for _, g in df.groupby("y"):
        idx = np.arange(len(g))
        rng.shuffle(idx)

        n = len(idx)
        n_train = max(1, int(0.7 * n))
        n_val = max(1, int(0.15 * n)) if n >= 3 else 0

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        # keep at least one in test if possible
        if len(test_idx) == 0 and n >= 2:
            test_idx = idx[-1:]
            if len(train_idx) > 1:
                train_idx = train_idx[:-1]

        trains.append(g.iloc[train_idx])
        if len(val_idx) > 0:
            vals.append(g.iloc[val_idx])
        if len(test_idx) > 0:
            tests.append(g.iloc[test_idx])

    train = pd.concat(trains, ignore_index=True) if trains else df.iloc[0:0].copy()
    val = pd.concat(vals, ignore_index=True) if vals else df.iloc[0:0].copy()
    test = pd.concat(tests, ignore_index=True) if tests else df.iloc[0:0].copy()
    return train, val, test


def main():
    paths = default_paths("cwru")
    ensure_dir(paths.root)
    ensure_dir(paths.splits)

    raw_dir = os.path.join(paths.root, "raw_official")
    ensure_dir(raw_dir)

    links = []
    links.extend(_extract_mat_links(NORMAL_URL))
    links.extend(_extract_mat_links(DRIVE12_URL))

    # dedupe by URL
    dedup = {}
    for url, anchor in links:
        dedup[url] = anchor
    links = [(u, a) for u, a in dedup.items()]

    print(f"Found {len(links)} official .mat links")

    downloaded = []
    for url, anchor in links:
        fname = os.path.basename(url)
        out_path = os.path.join(raw_dir, fname)
        _download_file(url, out_path)
        downloaded.append((out_path, anchor))

    rows = []
    for mat_path, anchor in downloaded:
        label = _label_from_anchor(anchor)

        # build one row per sensor channel so domain really reflects sensor/device
        for domain in ["DE", "FE"]:
            if _has_channel(mat_path, domain):
                rows.append(
                    {
                        "filepath": mat_path,
                        "label_str": label,
                        "domain": domain,
                        "sample_id": f"{os.path.splitext(os.path.basename(mat_path))[0]}_{domain}",
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable CWRU rows were created from official files")

    classes = ["normal", "inner", "ball", "outer"]
    lut = {c: i for i, c in enumerate(classes)}
    df = df[df["label_str"].isin(lut)].copy()
    df["y"] = df["label_str"].map(lut).astype(int)

    # ID = DE sensor, Shift = FE sensor
    id_df = df[df["domain"] == "DE"].copy()
    shift_df = df[df["domain"] == "FE"].copy()

    train, val, test_id = _stratified_split_id(id_df, seed=0)
    test_shift = shift_df.copy()

    out_cols = ["filepath", "y", "domain", "sample_id"]
    train[out_cols].to_csv(os.path.join(paths.splits, "train.csv"), index=False)
    val[out_cols].to_csv(os.path.join(paths.splits, "val.csv"), index=False)
    test_id[out_cols].to_csv(os.path.join(paths.splits, "test_id.csv"), index=False)
    test_shift[out_cols].to_csv(os.path.join(paths.splits, "test_shift.csv"), index=False)

    with open(os.path.join(paths.splits, "classes.txt"), "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")

    print("CWRU splits written to", paths.splits)
    print("train:", len(train), "val:", len(val), "test_id:", len(test_id), "test_shift:", len(test_shift))
    print("train domains:", sorted(train["domain"].unique()))
    print("shift domains:", sorted(test_shift["domain"].unique()))


if __name__ == "__main__":
    main()