"""Turn a training run (runs/<run-name>/) into the comparison tables and
analyses called for in proposal.md and updates.md Phase 0:

- headline table incl. the official Challenge 2015 score and false alarms
  suppressed at fixed sensitivity (95 / 99 / 100%, thresholds picked on the
  inner validation set, never on the test fold)
- per-alarm-type and signal-quality breakdowns (out-of-fold predictions)
- speed / size
- pre-registered significance family: Nadeau-Bengio corrected resampled
  t-test + paired record-level bootstrap, Holm-corrected

Usage:
    python -m src.evaluate --run-name challenge2015_ppg
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_processed  # noqa: E402
from src.metrics import binary_metrics, challenge_score, suppression_rate  # noqa: E402
from src.models.classifier import ALL_VARIANTS, build_model, count_params  # noqa: E402
from src.stats import corrected_resampled_ttest, holm, record_bootstrap  # noqa: E402
from src.train import get_width_mult, run_dir  # noqa: E402

OPS = ["t50", "s95", "s99", "s100"]
TABLE_VARIANTS = ALL_VARIANTS + ["multimodal", "gbm_features", "ens_mean", "ens_logit", "ens_stack", "ens_greedy", "ens_topk"]


def _fm(results: dict, variant: str) -> list[dict]:
    return results[variant]["fold_metrics"]


def comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for variant in TABLE_VARIANTS:
        if variant not in results:
            continue
        s = results[variant]
        fm = s["fold_metrics"]
        row = {
            "variant": variant,
            "params": fm[0]["params"] if fm else None,
            "Accuracy": f"{np.mean([(m['tp'] + m['tn']) / m['n'] for m in fm]):.3f}",
            "F1": f"{s['f1_mean']:.3f} +/- {s['f1_std']:.3f}",
            "Sensitivity": f"{s['sensitivity_mean']:.3f} +/- {s['sensitivity_std']:.3f}",
            "Specificity": f"{s['specificity_mean']:.3f} +/- {s['specificity_std']:.3f}",
            "AUC": f"{s['auc_mean']:.3f} +/- {s['auc_std']:.3f}",
            "ChallengeScore": f"{np.nanmean([m['challenge_score'] for m in fm]):.1f}",
        }
        # false alarms suppressed at fixed sensitivity: pooled over folds+seeds
        for op in OPS[1:]:
            tn = sum(m["ops"][op]["tn"] for m in fm)
            fp = sum(m["ops"][op]["fp"] for m in fm)
            tp = sum(m["ops"][op]["tp"] for m in fm)
            fn = sum(m["ops"][op]["fn"] for m in fm)
            sens = tp / (tp + fn) if tp + fn else float("nan")
            row[f"FA_suppressed@{op}"] = f"{suppression_rate(tn, fp):.3f}"
            row[f"realised_sens@{op}"] = f"{sens:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def load_oof(run: Path, data: dict, variant: str) -> dict:
    """Pool out-of-fold predictions over all seeds. Each seed gives exactly one
    prediction per window (from the fold that held it out)."""
    files = sorted((run / "preds").glob(f"{variant}_seed*_fold*.npz"))
    if not files:
        raise FileNotFoundError(f"no predictions for {variant} in {run / 'preds'}")
    seeds = sorted({int(f.stem.split("_seed")[1].split("_fold")[0]) for f in files})
    ys, ps, idxs, seed_ids = [], [], [], []
    for sd in seeds:
        for f in sorted((run / "preds").glob(f"{variant}_seed{sd}_fold*.npz")):
            z = np.load(f)
            idxs.append(z["test_idx"]); ps.append(z["test_prob"]); ys.append(z["test_y"])
            seed_ids.append(np.full(len(z["test_idx"]), sd))
    idx = np.concatenate(idxs)
    return {
        "idx": idx, "prob": np.concatenate(ps), "label": np.concatenate(ys), "seed": np.concatenate(seed_ids),
        "alarm_type": data["alarm_type"][idx], "quality": data["quality"][idx],
    }


def quality_stratified_eval(run: Path, data: dict, variant: str, n_bins: int = 3) -> pd.DataFrame:
    oof = load_oof(run, data, variant)
    y, prob, quality = oof["label"], oof["prob"], oof["quality"]
    bins = np.quantile(data["quality"], np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-6
    bin_idx = np.digitize(quality, bins[1:-1])
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        m = binary_metrics(y[mask], prob[mask])
        rows.append({"quality_bin": b, "n": int(mask.sum() // len(set(oof["seed"]))),
                     "F1": m["f1"], "Sensitivity": m["sensitivity"], "Specificity": m["specificity"]})
    return pd.DataFrame(rows)


def per_alarm_type_eval(run: Path, data: dict, variant: str) -> pd.DataFrame:
    oof = load_oof(run, data, variant)
    y, prob, alarm_type = oof["label"], oof["prob"], oof["alarm_type"]
    n_seeds = len(set(oof["seed"]))
    rows = []
    for at in sorted(set(alarm_type)):
        mask = alarm_type == at
        m = binary_metrics(y[mask], prob[mask])
        # official-score style per-type numbers at 0.5
        rows.append({
            "alarm_type": at, "n": int(mask.sum() // n_seeds), "n_true_alarms": int(y[mask].sum() // n_seeds),
            "F1": m["f1"], "Sensitivity": m["sensitivity"], "Specificity": m["specificity"], "AUC": m["auc"],
            "ChallengeScore": challenge_score(m["tp"], m["tn"], m["fp"], m["fn"]),
        })
    return pd.DataFrame(rows)


def per_type_suppression(run: Path, data: dict, results: dict, variant: str) -> pd.DataFrame:
    """Per-alarm-type false-alarm suppression at the validation-chosen 95/99/100%
    sensitivity thresholds. Re-applies each fold's threshold to its test fold."""
    rows = []
    fm_by_key = {(m["seed"], m["fold"]): m for m in _fm(results, variant)}
    acc = {}
    for (sd, fo), m in fm_by_key.items():
        z = np.load(run / "preds" / f"{variant}_seed{sd}_fold{fo}.npz")
        at = data["alarm_type"][z["test_idx"]]
        for op in OPS[1:]:
            thr = m["ops"][op]["threshold"]
            pred = z["test_prob"] >= thr
            for t in set(at):
                mk = at == t
                d = acc.setdefault((t, op), np.zeros(4))
                y = z["test_y"][mk] == 1
                d += [(pred[mk] & y).sum(), (pred[mk] & ~y).sum(), (~pred[mk] & ~y).sum(), (~pred[mk] & y).sum()]
    for (t, op), (tp, fp, tn, fn) in sorted(acc.items()):
        rows.append({"variant": variant, "alarm_type": t, "operating_point": op,
                     "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
                     "false_alarms_suppressed": tn / (tn + fp) if tn + fp else float("nan"),
                     "n_true": int(tp + fn), "n_false": int(tn + fp)})
    return pd.DataFrame(rows)


def record_tallies(run: Path, data: dict, results: dict, variant: str, op: str) -> np.ndarray:
    """(n_seeds, n_records, 4) per-record [tp, fp, tn, fn] at an operating point."""
    n = len(data["label"])
    seeds = sorted({m["seed"] for m in _fm(results, variant)})
    out = np.zeros((len(seeds), n, 4))
    for m in _fm(results, variant):
        z = np.load(run / "preds" / f"{variant}_seed{m['seed']}_fold{m['fold']}.npz")
        pred = z["test_prob"] >= m["ops"][op]["threshold"]
        y = z["test_y"] == 1
        k = seeds.index(m["seed"])
        out[k, z["test_idx"], 0] = pred & y
        out[k, z["test_idx"], 1] = pred & ~y
        out[k, z["test_idx"], 2] = ~pred & ~y
        out[k, z["test_idx"], 3] = ~pred & y
    return out


def significance_family(cfg: dict, run: Path, data: dict, results: dict) -> pd.DataFrame:
    """Runs exactly the comparisons pre-registered in config `stats.comparisons`."""
    sc = cfg["stats"]
    rows = []
    cache = {}

    def tallies(v, op):
        if (v, op) not in cache:
            cache[(v, op)] = record_tallies(run, data, results, v, op)
        return cache[(v, op)]

    for a, b in sc["comparisons"]:
        if a not in results or b not in results:
            continue
        fa = {(m["seed"], m["fold"]): m for m in _fm(results, a)}
        fb = {(m["seed"], m["fold"]): m for m in _fm(results, b)}
        keys = sorted(set(fa) & set(fb))
        diffs = np.array([fb[k]["f1"] - fa[k]["f1"] for k in keys])
        n_tr = np.mean([fa[k]["n_train"] for k in keys])
        n_te = np.mean([fa[k]["n_test"] for k in keys])
        nb = corrected_resampled_ttest(diffs, n_tr, n_te)
        row = {"baseline": a, "challenger": b, "n_pairs": nb["n_pairs"], "f1_mean_diff": nb["mean_diff"],
               "nb_t": nb["t_stat"], "nb_p": nb["p_value"]}
        for metric, op in [("f1", "t50"), ("challenge_score", "t50"), ("specificity", "s95")]:
            ta, tb = tallies(a, op), tallies(b, op)
            used = (ta.sum(axis=(0, 2)) + tb.sum(axis=(0, 2))) > 0  # only records that were ever in a test fold
            bs = record_bootstrap(ta[:, used], tb[:, used], metric, sc.get("n_boot", 5000))
            tag = f"{metric}@{op}"
            row[f"boot_diff[{tag}]"] = bs["obs_diff"]
            row[f"boot_ci[{tag}]"] = f"[{bs['ci95_low']:+.3f}, {bs['ci95_high']:+.3f}]"
            row[f"boot_p[{tag}]"] = bs["p_value"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        for col in ["nb_p", "boot_p[f1@t50]", "boot_p[challenge_score@t50]", "boot_p[specificity@s95]"]:
            df[col + "_holm"] = holm(df[col].fillna(1.0).tolist())
    return df


def benchmark_speed(cfg: dict, variant: str, device: str, n_warmup: int = 10, n_iters: int = 100) -> dict:
    width_mult = get_width_mult(variant, ROOT)
    model = build_model(variant, cfg, width_mult).to(device)
    model.eval()
    win_len = int(cfg["signal"]["window_seconds"] * cfg["signal"]["target_fs"])
    ecg = torch.randn(1, win_len, device=device)
    ppg = torch.randn(1, win_len, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(ecg, ppg)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iters):
            model(ecg, ppg)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
    ms = 1000 * elapsed / n_iters
    return {"variant": variant, "params": count_params(model), "latency_ms": ms, "throughput_per_sec": 1000 / ms}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--run-name", default="challenge2015_ppg")
    parser.add_argument("--processed", default=None)
    parser.add_argument("--focus", default="cross_attention", help="variant for per-type / quality tables")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    run = run_dir(cfg, args.run_name)
    results = json.loads((run / "results.json").read_text())
    processed = args.processed or (Path(cfg["paths"]["processed_dir"]) / "challenge2015_windows.npz")
    data = load_processed(processed)

    table = comparison_table(results)
    print("\n=== Comparison (test folds; thresholds for @s* chosen on inner validation) ===")
    print(table.to_string(index=False))
    table.to_csv(run / "comparison_table.csv", index=False)

    print(f"\n=== Per alarm type ({args.focus}, out-of-fold, pooled over seeds) ===")
    at = per_alarm_type_eval(run, data, args.focus)
    print(at.to_string(index=False))
    at.to_csv(run / "per_alarm_type.csv", index=False)

    print(f"\n=== Per-type false-alarm suppression at fixed sensitivity ({args.focus}) ===")
    pts = per_type_suppression(run, data, results, args.focus)
    print(pts.to_string(index=False))
    pts.to_csv(run / "per_type_suppression.csv", index=False)

    print(f"\n=== Performance vs signal quality ({args.focus}) ===")
    qt = quality_stratified_eval(run, data, args.focus)
    print(qt.to_string(index=False))
    qt.to_csv(run / "quality_stratified.csv", index=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    speed = pd.DataFrame([benchmark_speed(cfg, v, device) for v in ALL_VARIANTS if v in results])
    print("\n=== Model size / inference speed ===")
    print(speed.to_string(index=False))
    speed.to_csv(run / "speed_benchmark.csv", index=False)

    sig = significance_family(cfg, run, data, results)
    print("\n=== Pre-registered comparisons (Nadeau-Bengio + record bootstrap, Holm-corrected) ===")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(sig.to_string(index=False))
    sig.to_csv(run / "significance.csv", index=False)


if __name__ == "__main__":
    main()
