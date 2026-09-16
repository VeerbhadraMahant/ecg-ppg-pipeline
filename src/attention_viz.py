"""Extract and plot cross-attention weights for a handful of alarm windows,
checked for physiological plausibility (proposal.md objective 4 / architecture.md
section 3.3): a true alarm's attended PPG region should line up with the
expected pulse-transit delay after the corresponding ECG event.

Usage:
    python -m src.attention_viz --n-examples 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import AlarmWindowDataset, load_processed, record_wise_folds  # noqa: E402
from src.models.classifier import build_model  # noqa: E402
from src.train import get_width_mult  # noqa: E402


def plot_example(ecg: np.ndarray, ppg: np.ndarray, attn: np.ndarray, label: float, fs: float, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=False)

    t_ecg = np.arange(len(ecg)) / fs
    axes[0].plot(t_ecg, ecg, color="C0", linewidth=0.8)
    axes[0].set_title(f"ECG (label={'true' if label else 'false'} alarm)")
    axes[0].set_ylabel("z-norm amplitude")

    t_ppg = np.arange(len(ppg)) / fs
    axes[1].plot(t_ppg, ppg, color="C1", linewidth=0.8)
    axes[1].set_title("PPG")
    axes[1].set_ylabel("z-norm amplitude")
    axes[1].set_xlabel("time (s)")

    im = axes[2].imshow(attn, aspect="auto", origin="lower", cmap="viridis")
    axes[2].set_title("Cross-attention weights (ECG query time -> PPG key time)")
    axes[2].set_xlabel("PPG position")
    axes[2].set_ylabel("ECG position")
    fig.colorbar(im, ax=axes[2])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--n-examples", type=int, default=6)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processed_path = Path(cfg["paths"]["processed_dir"]) / "challenge2015_windows.npz"
    data = load_processed(processed_path)
    ecg, ppg, label, record_id = data["ecg"], data["ppg"], data["label"], data["record_id"]

    width_mult = get_width_mult("cross_attention", ROOT)
    ckpt_dir = ROOT / cfg["paths"]["runs_dir"] / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("cross_attention_seed0_fold*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no cross_attention checkpoints in {ckpt_dir}; run src/train.py first")
    fold_i = len(ckpts) - 1
    _, val_idx = list(record_wise_folds(record_id, cfg["split"]["n_folds"], cfg["seed"]))[fold_i]

    model = build_model("cross_attention", cfg, width_mult).to(device)
    model.load_state_dict(torch.load(ckpt_dir / f"cross_attention_seed0_fold{fold_i}.pt", map_location=device))
    model.eval()

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / cfg["paths"]["runs_dir"] / "attention_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(0)
    chosen = rng.choice(val_idx, size=min(args.n_examples, len(val_idx)), replace=False)

    ds = AlarmWindowDataset(ecg[chosen], ppg[chosen], label[chosen])
    with torch.no_grad():
        e = ds.ecg.to(device)
        p = ds.ppg.to(device)
        _, attn_weights = model(e, p)

    for i in range(len(chosen)):
        plot_example(
            ecg[chosen[i]], ppg[chosen[i]], attn_weights[i].cpu().numpy(),
            label[chosen[i]], cfg["signal"]["target_fs"], out_dir / f"example_{i}_{record_id[chosen[i]]}.png",
        )
    print(f"wrote {len(chosen)} attention plots to {out_dir}")


if __name__ == "__main__":
    main()
