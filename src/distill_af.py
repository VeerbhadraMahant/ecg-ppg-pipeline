"""PPG-only student distilled from an ECG+PPG fusion teacher, on the subject-wise
AF task (updates.md 3.6 / 7.3).

Per (seed, fold) of the SAME subject-wise folds as src/af_task.py:
  1. train teacher candidates (cross_attention scratch, cross_attention
     initialised from the alarm checkpoint) on the fit subjects; keep the one
     with the best inner-validation AUC ("best ECG+PPG model for that fold");
  2. soft labels = teacher logits on the fit windows, temperature-scaled;
  3. train PPG-only students (same CNNEncoder, mean-pooled head, ppg_only
     width) with  loss = alpha * BCE(hard) + (1-alpha) * T^2 * KL(teacher_T || student_T)
     alpha=1.0 is the hard-label-only control trained with the identical loop.

Work is cached per (seed, fold) so each invocation can be time-boxed:
    python -m src.distill_af --seeds 3 --budget-min 12     # rerun to resume

Caveat: the teacher has seen the fit labels, so its soft labels on the fit set
are partly memorised; temperature > 1 softens this but does not remove it.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score  # noqa: E402

from src.af_task import (OUT_DIR, fmt, fold_row, load_af_windows, splits, summarize)  # noqa: E402
from src.data.dataset import AlarmWindowDataset  # noqa: E402
from src.metrics import binary_metrics  # noqa: E402
from src.models.classifier import build_model  # noqa: E402
from src.train import get_width_mult, predict, set_seed, train_one_fold  # noqa: E402

UNIT_DIR = OUT_DIR / "distill_units"


def to_logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p)).astype(np.float32)


def train_student(cfg, ppg, y, t_logit, fit, val, te, alpha, temp, device, wm):
    t = cfg["train"]
    model = build_model("ppg_only", cfg, wm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    X = torch.from_numpy(ppg[fit]).float()
    Y = torch.from_numpy(y[fit]).float()
    TL = torch.from_numpy(t_logit)
    n_pos = float(Y.sum())
    pos_w = torch.tensor((len(Y) - n_pos) / max(n_pos, 1.0), device=device)
    val_ds = AlarmWindowDataset(ppg[val], ppg[val], y[val])
    best_f1, best_state, patience = -1.0, None, t["early_stop_patience"]
    bs = t["batch_size"]
    for _ in range(t["epochs"]):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            xb, yb, tb = X[b].to(device), Y[b].to(device), TL[b].to(device)
            logit, _ = model(xb, xb)
            hard = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_w)
            loss = alpha * hard
            if alpha < 1.0:
                pt = torch.sigmoid(tb / temp)
                ls = logit / temp
                # KL(Bern(pt) || Bern(sigmoid(ls))) = BCE(ls, pt) - H(pt)
                ent = -(pt * torch.log(pt.clamp_min(1e-7)) + (1 - pt) * torch.log((1 - pt).clamp_min(1e-7)))
                soft = (F.binary_cross_entropy_with_logits(ls, pt, reduction="none") - ent).mean()
                loss = loss + (1 - alpha) * temp * temp * soft
            opt.zero_grad()
            loss.backward()
            opt.step()
        vp = predict(model, val_ds, device, bs * 2)
        f1 = binary_metrics(y[val], vp)["f1"]
        if f1 > best_f1:
            best_f1, patience = f1, t["early_stop_patience"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_state)
    return predict(model, AlarmWindowDataset(ppg[te], ppg[te], y[te]), device, bs * 2)


def run_unit(cfg, data, seed, fold, fit, val, te, args, device):
    ecg, ppg, y = data["ecg"], data["ppg"], data["label"]
    mk = lambda ix: AlarmWindowDataset(ecg[ix], ppg[ix], y[ix])  # noqa: E731
    wm_t = get_width_mult("cross_attention", ROOT)
    set_seed(cfg["seed"] + seed + 100 * fold)

    best = None
    for cand in args.teachers.split(","):
        c = copy.deepcopy(cfg)
        if cand.endswith("+pretrain"):
            c["train"]["init_from"] = str(ROOT / "runs" / "challenge2015_ppg" / "checkpoints" / f"cross_attention_seed0_fold{fold}.pt")
        m, model, preds = train_one_fold(c, "cross_attention", wm_t, mk(fit), mk(val), mk(te), device, seed)
        try:
            vauc = roc_auc_score(y[val], preds["val_prob"])
        except ValueError:
            vauc = float("nan")
        score = -1 if np.isnan(vauc) else vauc
        if best is None or score > best[0]:
            fit_prob = predict(model, mk(fit), device, cfg["train"]["batch_size"] * 2)
            best = (score, cand, preds["test_prob"], fit_prob)
        del model
    _, tname, t_test, t_fit = best
    t_logit = to_logit(t_fit)

    out = {"teacher": t_test}
    wm_s = get_width_mult("ppg_only", ROOT)
    for a in args.alphas:
        set_seed(cfg["seed"] + seed + 100 * fold + 7)
        out[f"student_a{a:g}"] = train_student(cfg, ppg, y, t_logit, fit, val, te, a, args.temp, device, wm_s)
    return out, tname


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--max-windows", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--teachers", default="cross_attention,cross_attention+pretrain")
    parser.add_argument("--alphas", default="1.0,0.5,0.0", help="weight of the hard-label BCE; 1.0 = hard-only control")
    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--budget-min", type=float, default=12.0, help="stop starting new units after this many minutes")
    args = parser.parse_args()
    args.alphas = [float(a) for a in args.alphas.split(",")]

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["train"]["epochs"] = args.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_af_windows(cfg, args.max_windows)
    y, group = data["label"], data["group"]
    n_folds = cfg["split"]["n_folds"]
    UNIT_DIR.mkdir(parents=True, exist_ok=True)

    t0, todo = time.time(), 0
    for seed in range(args.seeds):
        for fold, fit, val, te in splits(group, y, n_folds, cfg["seed"] + seed):
            f = UNIT_DIR / f"s{seed}_f{fold}.npz"
            meta = UNIT_DIR / f"s{seed}_f{fold}.json"
            if f.exists() and meta.exists():
                continue
            if (time.time() - t0) / 60 > args.budget_min:
                todo += 1
                continue
            out, tname = run_unit(cfg, data, seed, fold, fit, val, te, args, device)
            np.savez(f, test_idx=te, **out)
            meta.write_text(json.dumps({"teacher": tname, "temp": args.temp, "alphas": args.alphas, "epochs": args.epochs}))
            print(f"seed {seed} fold {fold} done, teacher={tname} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if todo:
        print(f"{todo} units remaining; rerun to resume")
        return

    # ---- aggregate
    names = ["teacher"] + [f"student_a{a:g}" for a in args.alphas]
    oof = {n: np.zeros((args.seeds, len(y))) for n in names}
    rows = {n: [] for n in names}
    teachers = []
    for seed in range(args.seeds):
        for fold, fit, val, te in splits(group, y, n_folds, cfg["seed"] + seed):
            z = np.load(UNIT_DIR / f"s{seed}_f{fold}.npz")
            teachers.append(json.loads((UNIT_DIR / f"s{seed}_f{fold}.json").read_text())["teacher"])
            for n in names:
                oof[n][seed, z["test_idx"]] = z[n]
                rows[n].append(fold_row(z[n], y[z["test_idx"]], group[z["test_idx"]], seed, fold))
    results = {"config": {"temp": args.temp, "alphas": args.alphas, "teachers": args.teachers, "epochs": args.epochs,
                          "seeds": args.seeds, "teacher_choice_counts": {t: teachers.count(t) for t in set(teachers)}}}
    for n in names:
        results[n] = {**summarize(rows[n], oof[n], y, group, args.n_boot), "folds": rows[n]}
        print(f"{n:16s} {fmt(results[n])}")
    np.savez(OUT_DIR / "oof_distill.npz", label=y, group=group, **oof)
    (OUT_DIR / "distill_results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"written {OUT_DIR / 'distill_results.json'}")


if __name__ == "__main__":
    main()
