"""Build data/processed/vtac_windows.npz from downloaded VTaC events.

    python scripts/build_vtac_windows.py [--post-seconds 0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.vtac_loader import build_vtac_windows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post-seconds", type=float, default=0.0, help="0 = decision time (no post-alarm data)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    data = build_vtac_windows(ROOT / "data" / "raw" / "vtac", cfg, post_s=args.post_seconds)
    audit = data.pop("_audit")
    name = "vtac_windows.npz" if args.post_seconds == 0 else f"vtac_windows_post{int(args.post_seconds)}s.npz"
    out = Path(args.out) if args.out else ROOT / cfg["paths"]["processed_dir"] / name
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **data)
    lab = data["label"]
    print(json.dumps(audit))
    splits = dict(zip(*np.unique(data["split"], return_counts=True)))
    print(f"wrote {out}: {len(lab)} events, {lab.mean():.1%} true; patients={len(set(data['record_id']))}; "
          f"splits={splits}; ABP-as-pulse={int(data['pulse_is_abp'].sum())}")
    sig, cnt = np.unique(data["lead_signature"], return_counts=True)
    print("lead signatures (device proxy):", dict(zip(sig, cnt)))


if __name__ == "__main__":
    main()
