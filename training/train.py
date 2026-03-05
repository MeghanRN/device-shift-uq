import os, argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from utils.seed import set_seed, get_device
from utils.io import ensure_dir, save_json
from datasets.loaders import make_loaders
from models.factory import build_model
from uq.scores import softmax

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dcase","sonyc","cwru"])
    ap.add_argument("--model", required=True, choices=["cpmobile","dynacp","grucnn"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    task_type, num_outputs, train_loader, val_loader, _, _ = make_loaders(args.dataset, batch_size=args.batch_size)

    model = build_model(args.model, num_outputs).to(device)
    loss_fn = nn.CrossEntropyLoss() if task_type=="single_label" else nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = ensure_dir(os.path.join("results", args.dataset, args.model, f"seed_{args.seed}"))
    best = -1.0
    best_path = os.path.join(out_dir, "best.pt")

    for ep in range(1, args.epochs+1):
        model.train()
        for x, y, _, _ in tqdm(train_loader, leave=False, desc=f"train ep{ep:03d}"):

            # ALWAYS fix shape here
            # expected: (B,1,F,T). if you get (B,1,1,F,T) squeeze the extra dim
            if x.ndim == 5:
                x = x.squeeze(2)  # (B,1,1,F,T) -> (B,1,F,T)
            if x.ndim != 4:
                raise RuntimeError(f"Expected x to be 4D (B,1,F,T) but got {tuple(x.shape)}")

            x, y = x.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        # val
        model.eval()
        L,Y = [],[]
        with torch.no_grad():
            for x, y, _, _ in tqdm(val_loader, leave=False, desc=f"val ep{ep:03d}"):
                if x.ndim == 5:
                    x = x.squeeze(2)
                if x.ndim != 4:
                    raise RuntimeError(f"Expected x to be 4D (B,1,F,T) but got {tuple(x.shape)}")
                x = x.to(device)
                L.append(model(x).cpu())
                Y.append(y)
        logits = torch.cat(L,0).numpy()
        y = torch.cat(Y,0).numpy()

        if task_type=="single_label":
            p = softmax(logits)
            pred = p.argmax(1)
            acc = accuracy_score(y, pred)
            mf1 = f1_score(y, pred, average="macro")
            print(f"[ep {ep:03d}] val acc={acc:.4f} macro_f1={mf1:.4f}")
            score = acc
        else:
            pred = (1/(1+np.exp(-logits)) >= 0.5).astype(int)
            mf1 = f1_score(y, pred, average="micro", zero_division=0)
            print(f"[ep {ep:03d}] val micro_f1={mf1:.4f}")
            score = mf1

        if score > best:
            best = score
            torch.save({"model": model.state_dict()}, best_path)

    save_json({"best": float(best), "ckpt": best_path}, os.path.join(out_dir, "train_summary.json"))
    print("Saved:", best_path)

if __name__ == "__main__":
    main()
