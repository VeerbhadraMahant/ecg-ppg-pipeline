"""Find width multipliers so all four baseline-ladder variants land within a
tolerance of each other's parameter count (architecture.md section 5: 'all
four variants are matched in approximate parameter count... single-modality
models are widened to compensate for their missing branch').

Writes the resulting width_mult per variant into configs/width_mult.yaml,
which train.py reads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.classifier import build_model, count_params  # noqa: E402


def params_for(variant: str, cfg: dict, width_mult: float) -> int:
    model = build_model(variant, cfg, width_mult)
    return count_params(model)


def find_width_mult(variant: str, cfg: dict, target: int, tol: float = 0.03) -> float:
    lo, hi = 0.5, 4.0
    for _ in range(40):
        mid = (lo + hi) / 2
        n = params_for(variant, cfg, mid)
        if abs(n - target) / target < tol:
            return mid
        if n < target:
            lo = mid
        else:
            hi = mid
    return mid


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())

    # cross_attention (width_mult=1.0) is the reference; the other three are
    # widened/narrowed to match its parameter count.
    reference_params = params_for("cross_attention", cfg, 1.0)
    print(f"reference (cross_attention, width_mult=1.0): {reference_params:,} params")

    result = {"cross_attention": 1.0}
    for variant in ["concat", "ecg_only", "ppg_only"]:
        wm = find_width_mult(variant, cfg, reference_params)
        n = params_for(variant, cfg, wm)
        result[variant] = round(wm, 4)
        print(f"{variant}: width_mult={wm:.4f} -> {n:,} params ({n / reference_params:.1%} of reference)")

    out_path = ROOT / "configs" / "width_mult.yaml"
    out_path.write_text(yaml.dump(result, sort_keys=False))
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
