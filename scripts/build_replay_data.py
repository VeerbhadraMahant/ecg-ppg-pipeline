"""Build site/assets/replay.json (+ site/assets/replay_data.js) for the streaming replay demo.

For ~12 diverse test windows the cross_attention model (seed 0, fold whose test
split contains the record) is evaluated on the LAST k seconds of the 10 s alarm
window, zero-padded on the left, for k = 2..10 s. This emulates the window
"filling up" as a real-time monitor would see it.

Latency is measured for a single 10 s window (batch 1): fp32 CPU, int8 CPU
(torch dynamic quantization of nn.Linear layers ONLY -- the Conv1d encoders and
MultiheadAttention stay fp32), and fp32 GPU (cuda.synchronize timing).

Usage: python scripts/build_replay_data.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_processed  # noqa: E402
from src.data.signal_ops import signal_quality_score  # noqa: E402
from src.models.classifier import build_model  # noqa: E402
from src.train import get_width_mult, iter_splits, run_dir  # noqa: E402

VARIANT = "cross_attention"
RUN = "challenge2015_ppg"
FS = 250
OUT_FS = 125
N_WINDOWS = 12
KS = list(range(2, 11))
ATTN_BINS = 40  # over the 10 s window -> 0.25 s per bin


def r3(a):
    return np.round(np.asarray(a, dtype=np.float64), 3).tolist()


def pick_windows(data, test_fold_of, n):
    """Round-robin over (alarm_type, label) groups, one window per record."""
    rng = np.random.RandomState(0)
    types = sorted(np.unique(data["alarm_type"]))
    groups = []
    for t in types:
        for lab in (1.0, 0.0):
            idx = np.where((data["alarm_type"] == t) & (data["label"] == lab))[0]
            rng.shuffle(idx)
            if len(idx):
                groups.append(list(idx))
    chosen, seen = [], set()
    while len(chosen) < n and any(groups):
        for g in groups:
            while g:
                i = g.pop()
                if data["record_id"][i] not in seen:
                    seen.add(data["record_id"][i])
                    chosen.append(int(i))
                    break
            if len(chosen) >= n:
                break
    return chosen


@torch.no_grad()
def run_stream(model, ecg, ppg, device):
    """ecg/ppg: (2500,). Returns dict of per-k lists."""
    T = len(ecg)
    xs_e, xs_p = [], []
    for k in KS:
        n = k * FS
        e = np.zeros(T, np.float32)
        p = np.zeros(T, np.float32)
        e[T - n:] = ecg[T - n:]
        p[T - n:] = ppg[T - n:]
        xs_e.append(e)
        xs_p.append(p)
    xe = torch.from_numpy(np.stack(xs_e)).to(device)
    xp = torch.from_numpy(np.stack(xs_p)).to(device)
    logits, attn = model(xe, xp)  # attn (B, T_e, T_p)
    prob = torch.sigmoid(logits).cpu().numpy()
    attn = attn.cpu().numpy()
    Tp = attn.shape[2]
    # attention each PPG time step receives, averaged over all ECG queries
    col = attn.mean(axis=1)  # (B, T_p)
    edges = np.linspace(0, Tp, ATTN_BINS + 1).astype(int)
    binned = np.stack([[c[edges[b]:edges[b + 1]].mean() for b in range(ATTN_BINS)] for c in col])
    # normalise each row to its own max for display; keep raw too
    return prob, binned


def time_fn(fn, device, n_warm=10, n_rep=60):
    for _ in range(n_warm):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"median_ms": round(float(np.median(ts)), 3), "p95_ms": round(float(np.percentile(ts, 95)), 3),
            "n_runs": n_rep}


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    q = cfg["quality"]
    data = load_processed(ROOT / cfg["paths"]["processed_dir"] / "challenge2015_windows.npz")
    wm = get_width_mult(VARIANT, ROOT)
    ckdir = run_dir(cfg, RUN) / "checkpoints"

    fold_of = {}
    for fold_i, _fit, _val, test_idx in iter_splits(data, cfg, 0, "cv"):
        for i in test_idx:
            fold_of[int(i)] = fold_i
    chosen = pick_windows(data, fold_of, N_WINDOWS)

    models = {}

    def get_model(fold, device="cpu"):
        key = (fold, device)
        if key not in models:
            m = build_model(VARIANT, cfg, wm)
            m.load_state_dict(torch.load(ckdir / f"{VARIANT}_seed0_fold{fold}.pt", map_location="cpu"))
            models[key] = m.eval().to(device)
        return models[key]

    n_total = data["ecg"].shape[1]
    step = FS // OUT_FS
    records = []
    for i in chosen:
        fold = fold_of[i]
        m = get_model(fold)
        ecg, ppg = data["ecg"][i].astype(np.float32), data["ppg"][i].astype(np.float32)
        prob, attn = run_stream(m, ecg, ppg, "cpu")
        qe, qp = [], []
        for k in KS:
            seg = slice(n_total - k * FS, n_total)
            qe.append(signal_quality_score(ecg[seg], q["flatline_std_threshold"], q["clip_fraction_threshold"]))
            qp.append(signal_quality_score(ppg[seg], q["flatline_std_threshold"], q["clip_fraction_threshold"]))
        # row-normalise attention for the heat strip
        attn_n = attn / np.maximum(attn.max(axis=1, keepdims=True), 1e-12)
        records.append({
            "idx": i,
            "record_id": str(data["record_id"][i]),
            "alarm_type": str(data["alarm_type"][i]),
            "label": int(data["label"][i]),
            "fold": int(fold),
            "ecg": r3(ecg[::step]),
            "ppg": r3(ppg[::step]),
            "k": KS,
            "prob": r3(prob),
            "ecg_quality": r3(qe),
            "ppg_quality": r3(qp),
            "attn": [r3(a) for a in attn_n],
        })
        print(f"{records[-1]['record_id']:>6} {records[-1]['alarm_type']:<24} label={records[-1]['label']} "
              f"fold={fold} final_p={prob[-1]:.3f}")

    # ---- latency on one 10 s window (first chosen record) ----
    i0 = chosen[0]
    fold0 = fold_of[i0]
    e1 = torch.from_numpy(data["ecg"][i0].astype(np.float32))[None]
    p1 = torch.from_numpy(data["ppg"][i0].astype(np.float32))[None]
    lat = {"window": "one 10 s window, batch=1, model forward only (no I/O, no sqi)",
           "variant": VARIANT, "torch": torch.__version__}
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    lat["cpu_threads"] = torch.get_num_threads()

    fp32 = get_model(fold0)
    with torch.no_grad():
        lat["cpu_fp32"] = time_fn(lambda: fp32(e1, p1), "cpu")
        p_fp32 = float(torch.sigmoid(fp32(e1, p1)[0]))
        try:
            q8 = torch.ao.quantization.quantize_dynamic(
                _fresh(fold0, cfg, wm, ckdir), {torch.nn.Linear}, dtype=torch.qint8)
            lat["cpu_int8_dynamic"] = time_fn(lambda: q8(e1, p1), "cpu")
            lat["cpu_int8_dynamic"]["note"] = (
                "torch dynamic quantization of nn.Linear layers only; Conv1d encoders and "
                "MultiheadAttention (its fused in_proj) stay fp32")
            lat["int8_prob_abs_diff_vs_fp32"] = round(abs(float(torch.sigmoid(q8(e1, p1)[0])) - p_fp32), 5)
        except Exception as ex:  # noqa: BLE001
            lat["cpu_int8_dynamic"] = {"error": repr(ex)}
        if torch.cuda.is_available():
            g = get_model(fold0, "cuda")
            eg, pg = e1.cuda(), p1.cuda()
            lat["gpu_fp32"] = time_fn(lambda: g(eg, pg), "cuda")
            lat["gpu_name"] = torch.cuda.get_device_name(0)
            lat["gpu_note"] = "GPU was shared with a concurrent training job; timings are indicative only"
    lat["n_params"] = int(sum(p.numel() for p in fp32.parameters()))

    out = {
        "meta": {
            "variant": VARIANT, "seed": 0, "fs": OUT_FS, "window_s": 10, "ks": KS,
            "attn_bins": ATTN_BINS, "threshold": 0.5,
            "prob_meaning": "P(true alarm); label 1 = true alarm",
            "attn_meaning": "ECG->PPG attention mass each PPG time bin receives, averaged over all ECG "
                            "queries, per-k row-normalised to max=1; window zero-padded on the left",
            "stream_note": "at k seconds the model sees only the last k s of the 10 s window (left zero-padded). "
                           "The model was trained on full 10 s windows only, so early-k probabilities are "
                           "out-of-distribution and should not be read as calibrated.",
        },
        "latency": lat,
        "records": records,
    }
    js = json.dumps(out, separators=(",", ":"))
    (ROOT / "site" / "assets").mkdir(parents=True, exist_ok=True)
    (ROOT / "site" / "assets" / "replay.json").write_text(js)
    # file:// cannot fetch() local json, so also emit a script-tag loadable copy
    (ROOT / "site" / "assets" / "replay_data.js").write_text("window.REPLAY_DATA=" + js + ";")
    print(f"wrote replay.json: {len(js) / 1e6:.2f} MB")
    print(json.dumps(lat, indent=1))


def _fresh(fold, cfg, wm, ckdir):
    m = build_model(VARIANT, cfg, wm)
    m.load_state_dict(torch.load(ckdir / f"{VARIANT}_seed0_fold{fold}.pt", map_location="cpu"))
    return m.eval()


if __name__ == "__main__":
    main()
