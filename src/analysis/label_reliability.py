"""Label reliability on VTaC (updates.md analysis 6.11).

VTaC v1.1 releases the individual annotator votes. Using the saved test
predictions of a run trained on the official split (runs/<run>/preds):

1. Annotator agreement: fraction of events with unanimous votes vs split votes.
2. Does model uncertainty track annotator disagreement? Spearman correlation
   between predictive entropy and the disagreement score 1 - |2*frac_true - 1|,
   and the AUC of entropy for separating split-vote from unanimous events.
3. Performance on unanimous vs disagreed events (F1/AUC/accuracy against the
   consensus label), averaged over seeds.
4. Optional paired comparison against a run trained with --soft-labels
   (see src/train.py) on the same seeds.

Usage:
    python -m src.analysis.label_reliability --run-name vtac_official --variants cross_attention,ecg_only
    python -m src.analysis.label_reliability --run-name vtac_official --soft-run vtac_official_soft --variants cross_attention
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_processed  # noqa: E402
from src.train import run_dir  # noqa: E402


def entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def load(run: Path, variant: str):
    out = []
    for f in sorted((run / "preds").glob(f"{variant}_seed*_fold0.npz")):
        z = np.load(f)
        out.append((int(f.stem.split("_seed")[1].split("_")[0]), z["test_idx"], z["test_prob"], z["test_y"]))
    return out


def analyse(run: Path, variant: str, data: dict) -> dict:
    frac = data["annotator_frac_true"]
    n_ann = data["n_annotators"]
    seeds = load(run, variant)
    if not seeds:
        raise FileNotFoundError(f"no preds for {variant} in {run}")
    res = {"variant": variant, "n_seeds": len(seeds)}
    rho, auc_dis, f1u, f1d, accu, accd, aucu, aucd = [], [], [], [], [], [], [], []
    for _, idx, p, y in seeds:
        fr = frac[idx]
        ok = ~np.isnan(fr) & (n_ann[idx] >= 2)
        dis = 1 - np.abs(2 * fr - 1)  # 0 unanimous .. 1 evenly split
        split = (dis > 0) & ok
        unan = (dis == 0) & ok
        if split.sum() < 5 or unan.sum() < 5:
            continue
        rho.append(stats.spearmanr(entropy(p[ok]), dis[ok])[0])
        auc_dis.append(roc_auc_score(split[ok], entropy(p[ok])))
        pred = p >= 0.5
        for mask, f, a, u in ((unan, f1u, accu, aucu), (split, f1d, accd, aucd)):
            f.append(f1_score(y[mask], pred[mask], zero_division=0))
            a.append(float((pred[mask] == y[mask]).mean()))
            u.append(roc_auc_score(y[mask], p[mask]) if len(set(y[mask])) > 1 else np.nan)
    m = lambda v: float(np.nanmean(v))  # noqa: E731
    res.update({
        "spearman(entropy, disagreement)": m(rho), "AUC(entropy -> split-vote event)": m(auc_dis),
        "F1_unanimous": m(f1u), "F1_split": m(f1d), "acc_unanimous": m(accu), "acc_split": m(accd),
        "AUC_unanimous": m(aucu), "AUC_split": m(aucd),
    })
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--run-name", default="vtac_official")
    ap.add_argument("--soft-run", default=None)
    ap.add_argument("--variants", default="cross_attention")
    ap.add_argument("--processed", default=str(ROOT / "data" / "processed" / "vtac_windows.npz"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data = load_processed(args.processed)

    frac, n_ann = data["annotator_frac_true"], data["n_annotators"]
    ok = ~np.isnan(frac) & (n_ann >= 2)
    dis = 1 - np.abs(2 * frac - 1)
    print(f"events with >=2 independent votes: {int(ok.sum())} / {len(frac)}")
    print(f"unanimous: {int(((dis == 0) & ok).sum())}  split-vote: {int(((dis > 0) & ok).sum())} "
          f"({100 * ((dis > 0) & ok).sum() / max(ok.sum(), 1):.1f} %)")
    print("vote-count distribution:", dict(zip(*np.unique(n_ann, return_counts=True))))

    rows = [analyse(run_dir(cfg, args.run_name), v, data) for v in args.variants.split(",")]
    df = pd.DataFrame(rows)
    out = run_dir(cfg, args.run_name) / "label_reliability"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "reliability.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(df.round(3).to_string(index=False))

    if args.soft_run:
        print("\nsoft-label training vs consensus-label training (VTaC test split, same seeds):")
        rows = []
        for v in args.variants.split(","):
            hard, soft = load(run_dir(cfg, args.run_name), v), load(run_dir(cfg, args.soft_run), v)
            hs, ss = {s: (i, p, y) for s, i, p, y in hard}, {s: (i, p, y) for s, i, p, y in soft}
            d_f1, d_auc = [], []
            for s in sorted(set(hs) & set(ss)):
                y = hs[s][2]
                d_f1.append(f1_score(y, ss[s][1] >= 0.5) - f1_score(y, hs[s][1] >= 0.5))
                d_auc.append(roc_auc_score(y, ss[s][1]) - roc_auc_score(y, hs[s][1]))
            rows.append({"variant": v, "n_seeds": len(d_f1), "dF1(soft-hard)": np.mean(d_f1), "dF1_sd": np.std(d_f1),
                         "dAUC(soft-hard)": np.mean(d_auc), "dAUC_sd": np.std(d_auc)})
        sd = pd.DataFrame(rows)
        sd.to_csv(out / "soft_vs_hard.csv", index=False)
        print(sd.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
