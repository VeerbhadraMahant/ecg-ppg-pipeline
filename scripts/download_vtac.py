"""Fetch VTaC v1.1 (open access, CC BY-SA 4.0) event windows without
downloading the full 18 GB: WFDB range-reads only the samples needed.

Each record is a 6-minute segment at 250 Hz with the alarm onset at 300 s.
For every annotated event we keep [300 - PRE, 300 + POST] seconds:
  PRE  = 60 s  -> decision-time input + long pre-alarm context experiments
  POST = 30 s  -> time-to-verdict curve (0/5/10/30 s of post-alarm data)
Models trained for the headline task only ever see data up to t = 0 (the
alarm onset); the post-alarm samples exist solely for the time-to-verdict
analysis, to avoid the leakage warned about in updates.md section 2.1.

Usage:
    python scripts/download_vtac.py --workers 16
Resumable: finished events are cached as data/raw/vtac/events/<event>.npz.
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "vtac"
BASE = "https://physionet.org/files/vtac/1.1/"
FS = 250
ALARM_S, PRE_S, POST_S = 300, 60, 30


def fetch_meta() -> None:
    import urllib.request

    OUT.mkdir(parents=True, exist_ok=True)
    for f in ["RECORDS", "event_labels.csv", "benchmark_data_split.csv", "event_label_per_annotator.csv"]:
        if not (OUT / f).exists():
            urllib.request.urlretrieve(BASE + f, OUT / f)


_tls = threading.local()


def _session() -> requests.Session:
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()  # keep-alive: avoids a TLS handshake per request
    return _tls.s


def _parse_header(text: str):
    lines = [l for l in text.splitlines() if l and not l.startswith("#")]
    rec = lines[0].split()
    n_sig, fs, n_samp = int(rec[1]), float(rec[2]), int(rec[3])
    names, gains, bases = [], [], []
    for l in lines[1 : 1 + n_sig]:
        f = l.split()
        if f[1] != "16":
            raise ValueError(f"unsupported WFDB format {f[1]}")
        g = f[2].split("/")[0]
        base = None
        if "(" in g:
            g, b = g.split("(")
            base = float(b.rstrip(")"))
        gain = float(g) or 200.0
        gains.append(gain)
        bases.append(base if base is not None else float(f[4]))  # ADC zero fallback
        names.append(f[-1])
    return n_sig, fs, n_samp, names, np.array(gains), np.array(bases)


def fetch_event(rec_path: str, tries: int = 8) -> str:
    """Two HTTP requests per event: the header, then one byte-range read of
    the wanted samples from the single-file, format-16, interleaved .dat."""
    folder, name = rec_path.rsplit("/", 1)
    dest = OUT / "events" / f"{name}.npz"
    if dest.exists():
        return "cached"
    url = f"https://physionet.org/files/vtac/1.1/{folder}/{name}"
    last = None
    for k in range(tries):
        try:
            s = _session()
            h = s.get(url + ".hea", timeout=60)
            h.raise_for_status()
            n_sig, fs, n_samp, names, gains, bases = _parse_header(h.text)
            a, b = int((ALARM_S - PRE_S) * fs), int((ALARM_S + POST_S) * fs)
            b = min(b, n_samp)
            r = s.get(url + ".dat", headers={"Range": f"bytes={a * n_sig * 2}-{b * n_sig * 2 - 1}"}, timeout=120)
            if r.status_code != 206:
                raise IOError(f"range not honoured: {r.status_code}")
            raw = np.frombuffer(r.content, dtype="<i2")
            if raw.size != (b - a) * n_sig:
                raise IOError(f"short read {raw.size} != {(b - a) * n_sig}")
            raw = raw.reshape(-1, n_sig).astype(np.float64)
            phys = (raw - bases) / gains
            phys[raw == -32768] = 0.0  # WFDB invalid-sample marker
            np.savez(dest, sig=phys.astype(np.float16), names=np.array(names), fs=fs)
            return "ok"
        except Exception as e:  # network hiccups: back off and retry
            last = e
            time.sleep(1 + k)
    return f"FAILED {last}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    fetch_meta()
    (OUT / "events").mkdir(parents=True, exist_ok=True)
    recs = [l.strip() for l in open(OUT / "RECORDS") if l.strip().startswith("annotated_waveforms/")]
    if args.limit:
        recs = recs[: args.limit]
    print(f"{len(recs)} annotated events")

    t0, done, failed = time.time(), 0, []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch_event, r): r for r in recs}
        for f in as_completed(futs):
            res = f.result()
            done += 1
            if res.startswith("FAILED"):
                failed.append((futs[f], res))
            if done % 100 == 0 or done == len(recs):
                rate = done / (time.time() - t0)
                print(f"{done}/{len(recs)}  {rate:.1f} ev/s  eta {(len(recs) - done) / max(rate, 1e-6) / 60:.0f} min  failed={len(failed)}", flush=True)
    (OUT / "failed.json").write_text(json.dumps(failed, indent=2))
    print("done; failed:", len(failed))


if __name__ == "__main__":
    main()
