import os, argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

from utils.seed import set_seed, get_device
from utils.io import ensure_dir, save_json
from datasets.loaders import make_loaders
from models.factory import build_model
from uq.scores import softmax, uq_single_from_logits, uq_multilabel_from_logits

def auroc_ood(id_scores, ood_scores):
    y = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    s = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y, s))

def aupr_ood(id_scores, ood_scores):
    y = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    s = np.concatenate([id_scores, ood_scores])
    return float(average_precision_score(y, s))

def collect(model, loader, device):
    model.eval()
    L,Y,D,S = [],[],[],[]
    with torch.no_grad():
        for x,y,dom,sid in tqdm(loader, leave=False):
            x = x.to(device)
            L.append(model(x).cpu())
            Y.append(y)
            D.extend(list(dom)); S.extend(list(sid))
    return torch.cat(L,0).numpy(), torch.cat(Y,0).numpy(), np.array(D), np.array(S)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dcase","sonyc","cwru"])
    ap.add_argument("--model", required=True, choices=["cpmobile","dynacp","grucnn"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    task_type, num_outputs, _, _, id_loader, sh_loader = make_loaders(args.dataset, batch_size=args.batch_size)
    model = build_model(args.model, num_outputs).to(device)

    ckpt = os.path.join("results", args.dataset, args.model, f"seed_{args.seed}", "best.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])

    id_logits, id_y, _, _ = collect(model, id_loader, device)
    sh_logits, sh_y, _, _ = collect(model, sh_loader, device)

    out_dir = ensure_dir(os.path.join("results", args.dataset, args.model, f"seed_{args.seed}"))

    if task_type=="single_label":
        id_p = softmax(id_logits); sh_p = softmax(sh_logits)
        id_pred = id_p.argmax(1); sh_pred = sh_p.argmax(1)
        id_metrics = {"acc": float(accuracy_score(id_y, id_pred)), "macro_f1": float(f1_score(id_y, id_pred, average="macro"))}
        sh_metrics = {"acc": float(accuracy_score(sh_y, sh_pred)), "macro_f1": float(f1_score(sh_y, sh_pred, average="macro"))}
        id_u = uq_single_from_logits(id_logits); sh_u = uq_single_from_logits(sh_logits)
    else:
        id_p = 1/(1+np.exp(-id_logits)); sh_p = 1/(1+np.exp(-sh_logits))
        id_pred = (id_p>=0.5).astype(int); sh_pred=(sh_p>=0.5).astype(int)
        id_metrics={"micro_f1": float(f1_score(id_y, id_pred, average="micro", zero_division=0))}
        sh_metrics={"micro_f1": float(f1_score(sh_y, sh_pred, average="micro", zero_division=0))}
        id_u = uq_multilabel_from_logits(id_logits); sh_u = uq_multilabel_from_logits(sh_logits)

    shift_det = {k: {"auroc": auroc_ood(id_u[k], sh_u[k]), "aupr": aupr_ood(id_u[k], sh_u[k])} for k in id_u.keys()}
    save_json({"id": id_metrics, "shift": sh_metrics, "shift_det": shift_det}, os.path.join(out_dir, "eval_uq_summary.json"))
    print("Wrote", os.path.join(out_dir, "eval_uq_summary.json"))

if __name__ == "__main__":
    main()
