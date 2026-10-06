"""Build site/assets/leaderboard.js from the run CSVs (read-only inputs).

Inputs : runs/final_leaderboard.csv
         runs/legacy_pre_fix/comparison_table.csv   (before leakage fix)
         runs/challenge2015_ppg/comparison_table.csv (after leakage fix)
Output : site/assets/leaderboard.js  (defines window.SITE_DATA)
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "assets" / "leaderboard.js"


def read(path):
    with open(ROOT / path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_of(cell):
    return float(cell.split("+/-")[0].strip())


def main():
    lb = []
    for r in read("runs/final_leaderboard.csv"):
        lb.append({
            "model": r["model"], "group": r["group"], "dataset": r["dataset"],
            "n_seeds": int(r["n_seeds"]),
            "accuracy": round(float(r["accuracy"]), 4), "f1": round(float(r["f1"]), 4),
            "auc": round(float(r["auc"]), 4), "sens": round(float(r["sens"]), 4),
            "spec": round(float(r["spec"]), 4),
            "delta": r["bootstrap_f1_best_minus_model"],
        })

    want = [("ecg_only", "ECG only"), ("ppg_only", "PPG only"),
            ("concat", "Concatenation fusion"), ("cross_attention", "Cross-attention fusion")]
    old = {r["variant"]: r for r in read("runs/legacy_pre_fix/comparison_table.csv")}
    new = {r["variant"]: r for r in read("runs/challenge2015_ppg/comparison_table.csv")}
    leak = []
    for key, label in want:
        leak.append({"variant": label,
                     "f1_before": mean_of(old[key]["F1"]), "auc_before": mean_of(old[key]["AUC"]),
                     "f1_after": mean_of(new[key]["F1"]), "auc_after": mean_of(new[key]["AUC"])})

    data = {"leaderboard": lb, "leakage": leak}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.SITE_DATA = " + json.dumps(data, indent=1) + ";\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(lb)} leaderboard rows, {len(leak)} leakage rows)")


if __name__ == "__main__":
    main()
