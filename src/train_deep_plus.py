"""Stronger deep recipe for ECG+PPG alarm verification, with an honest search.

Ingredients (each a switch, so each can be ablated):
  aug      training-only augmentation on GPU: random circular time shift of BOTH
           signals together, per-channel amplitude scaling, additive Gaussian
           noise, random 1-s dropout of ONE modality (optional mixup)
  optim    AdamW + one-cycle (warm-up + cosine) schedule, grad clipping
  loss     'focal' (gamma 2, pos_weight) or 'bce_ls' (pos-weighted BCE + label smoothing)
  ema      exponential moving average of the weights (SWA-like), the EMA weights
           at their best INNER-validation epoch are used
  early stopping / best epoch: inner-validation AUC only
  tta      test-time augmentation: mean prob over small circular shifts
  k        multi-seed ensemble (K independently initialised members, mean prob)
  *_feat   early-fusion hybrid: the 53 hand-crafted agreement features of
           src/features.py (standardised with FIT-set statistics) are encoded by
           a small MLP and concatenated with the deep features before the head.

Everything that is used to choose something (epoch, EMA vs raw is *recorded*
both ways, grid winner) uses the inner validation set only. The outer test fold
is predicted once per trained model and is never used for selection; the grid
stage does not even compute test predictions.

Sub-commands
  grid   : pre-declared grid, 3 seeds x 5 folds, CinC PPG cohort, inner-val AUC
  select : summarise the grid (mean inner-val AUC) and write chosen.json
  final  : full protocol for chosen recipe(s) -> runs/<run-name>/preds/...

Every fold file keeps the repo layout (test_idx, val_idx, val_prob, val_y,
test_prob, test_y) plus extra ablation views `val_prob__<view>` /
`test_prob__<view>` (single member, raw vs EMA, TTA, ensemble size ...).
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import warnings
import torch
warnings.filterwarnings("ignore", message="RNN module weights")
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_processed  # noqa: E402
from src.metrics import binary_metrics, challenge_score, operating_points, summarize_across_folds  # noqa: E402
from src.models.baselines import build_zoo_model  # noqa: E402
from src.models.losses import FocalLoss  # noqa: E402
from src.models.reproductions import MousaviAttnCNNRNN  # noqa: E402
from src.train import get_width_mult, iter_splits, set_seed  # noqa: E402

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

FS = 250
DATASETS = {
    "cinc": ("challenge2015_windows.npz", "cv"),
    "cinc_recovered": ("challenge2015_windows_recovered.npz", "cv"),
    "vtac": ("vtac_windows.npz", "official"),
}
BASE_ARCHS = ["resnet1d", "tcn", "inceptiontime", "mousavi_attn_cnn_rnn"]
GRID_ARCHS = BASE_ARCHS + ["resnet1d_feat", "tcn_feat"]

# ---- pre-declared grid (declared before any result existed) ----
GRID = {"arch": GRID_ARCHS, "aug": [0, 1], "loss": ["focal", "bce_ls"], "lr": [3e-4, 1e-3]}
GRID_SEEDS = [100, 101, 102]

DEFAULTS = dict(
    arch="resnet1d", aug=1, loss="focal", lr=1e-3, k=3, mixup=0.0,
    wd=1e-2, epochs=30, patience=8, batch=32, label_smooth=0.05, focal_gamma=2.0,
    max_shift=250, scale_sigma=0.15, noise=0.1, mod_drop=0.3, feat_noise=0.1,
    tta_shifts=(-100, -50, 0, 50, 100), ema_horizon_epochs=3.0, feat_dim=32,
)


def width_for(arch: str) -> float:
    base = arch.replace("_feat", "")
    if base == "mousavi_attn_cnn_rnn":
        p = ROOT / "configs" / "width_mult_reproductions.yaml"
        return float(yaml.safe_load(p.read_text())[base])
    return get_width_mult(base, ROOT)


# --------------------------------------------------------------------- model
class Net(nn.Module):
    """Wraps a zoo model / Mousavi model; optionally fuses hand-crafted features
    before the classification head. forward(x (B,2,T), f (B,F) or None)."""

    def __init__(self, arch: str, cfg: dict, n_feat: int, feat_dim: int = 32):
        super().__init__()
        self.hybrid = arch.endswith("_feat")
        base = arch.replace("_feat", "")
        wm = width_for(arch)
        m = cfg["model"]
        self.mousavi = base == "mousavi_attn_cnn_rnn"
        if self.mousavi:
            self.base = MousaviAttnCNNRNN(wm, m["dropout"], m["head_hidden"])
        else:
            self.base = build_zoo_model(base, cfg, wm)
        head = self.base.head
        first = head.net[0] if hasattr(head, "net") else head[0]
        in_dim = first.in_features
        if self.hybrid:
            self.fenc = nn.Sequential(nn.Linear(n_feat, feat_dim), nn.ReLU(), nn.Dropout(0.2))
            self.head = nn.Sequential(nn.Linear(in_dim + feat_dim, m["head_hidden"]), nn.ReLU(),
                                      nn.Dropout(m["dropout"]), nn.Linear(m["head_hidden"], 1))
        else:
            self.head = head

    def forward(self, x, f=None):
        if self.mousavi:
            z = self.base.features(x[:, 0], x[:, 1])
        else:
            z = self.base.features(x)
        if self.hybrid:
            z = torch.cat([z, self.fenc(f)], dim=-1)
        return self.head(z).squeeze(-1)


# ------------------------------------------------------------- augmentation
def circ_shift(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    T = x.shape[-1]
    idx = (torch.arange(T, device=x.device)[None, :] + s[:, None]) % T
    return x.gather(2, idx[:, None, :].expand(-1, x.shape[1], -1))


def augment(x, f, y, r: dict):
    B, C, T = x.shape
    dev = x.device
    s = torch.randint(-r["max_shift"], r["max_shift"] + 1, (B,), device=dev)
    x = circ_shift(x, s)
    x = x * torch.exp(torch.randn(B, C, 1, device=dev) * r["scale_sigma"])
    x = x + torch.randn_like(x) * (torch.rand(B, 1, 1, device=dev) * r["noise"])
    # drop one random 1-s chunk of ONE modality
    chunk = FS
    do = torch.rand(B, device=dev) < r["mod_drop"]
    ch = torch.randint(0, C, (B,), device=dev)
    st = torch.randint(0, T - chunk + 1, (B,), device=dev)
    pos = torch.arange(T, device=dev)[None, :]
    seg = (pos >= st[:, None]) & (pos < st[:, None] + chunk) & do[:, None]
    chm = F.one_hot(ch, C).bool()[:, :, None]
    x = x.masked_fill(seg[:, None, :] & chm, 0.0)
    if f is not None:
        f = f + torch.randn_like(f) * r["feat_noise"]
    return x, f, y


def mixup(x, f, y, alpha: float):
    lam = float(np.random.beta(alpha, alpha))
    p = torch.randperm(x.shape[0], device=x.device)
    x = lam * x + (1 - lam) * x[p]
    if f is not None:
        f = lam * f + (1 - lam) * f[p]
    return x, f, lam * y + (1 - lam) * y[p]


# --------------------------------------------------------------------- utils
def auc_safe(y, p) -> float:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, p))


@torch.no_grad()
def predict(model, X, Fe, shifts=(0,), bs=256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs]
        fb = None if Fe is None else Fe[i:i + bs]
        ps = []
        for s in shifts:
            xs = xb if s == 0 else circ_shift(xb, torch.full((len(xb),), s, device=xb.device))
            ps.append(torch.sigmoid(model(xs, fb)))
        out.append(torch.stack(ps).mean(0).cpu())
    return torch.cat(out).numpy()


def make_loss(r: dict, pos_weight: float, device):
    pw = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    if r["loss"] == "focal":
        return FocalLoss(gamma=r["focal_gamma"], pos_weight=pw)
    if r["loss"] == "bce_ls":
        eps = r["label_smooth"]

        def fn(logits, y):
            return F.binary_cross_entropy_with_logits(logits, y * (1 - eps) + eps / 2, pos_weight=pw)
        return fn
    raise ValueError(r["loss"])


def train_member(r: dict, cfg: dict, Xfit, Ffit, yfit, Xval, Fval, yval_np, seed: int, device):
    """Returns (raw_state, ema_state, info). Both states are the weights at their
    best INNER-validation AUC epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    n_feat = 0 if Ffit is None else Ffit.shape[1]
    model = Net(r["arch"], cfg, n_feat, r["feat_dim"]).to(device)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    n_pos = float(yfit.sum().item())
    pos_weight = (len(yfit) - n_pos) / max(n_pos, 1.0)
    loss_fn = make_loss(r, pos_weight, device)

    N = len(Xfit)
    bs = r["batch"]
    spe = max(1, N // bs)  # drop_last
    opt = torch.optim.AdamW(model.parameters(), lr=r["lr"], weight_decay=r["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=r["lr"], total_steps=r["epochs"] * spe, pct_start=0.15,
        anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0)
    decay = 1.0 - 1.0 / (r["ema_horizon_epochs"] * spe)

    best = {"raw": (-1.0, None, -1), "ema": (-1.0, None, -1)}
    since = 0
    step = 0
    for epoch in range(r["epochs"]):
        model.train()
        perm = torch.randperm(N, device=device)
        for i in range(spe):
            b = perm[i * bs:(i + 1) * bs]
            xb, fb, yb = Xfit[b], (None if Ffit is None else Ffit[b]), yfit[b]
            if r["aug"]:
                xb, fb, yb = augment(xb, fb, yb, r)
            if r["mixup"] > 0:
                xb, fb, yb = mixup(xb, fb, yb, r["mixup"])
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb, fb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            d = min(decay, (1 + step) / (10 + step))
            with torch.no_grad():
                ep, mp = list(ema.parameters()), list(model.parameters())
                torch._foreach_mul_(ep, d)
                torch._foreach_add_(ep, mp, alpha=1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
        improved = False
        for name, mdl in (("raw", model), ("ema", ema)):
            a = auc_safe(yval_np, predict(mdl, Xval, Fval))
            if a > best[name][0]:
                best[name] = (a, {k: v.detach().clone() for k, v in mdl.state_dict().items()}, epoch)
                improved = True
        since = 0 if improved else since + 1
        if since >= r["patience"]:
            break
    info = {"val_auc_raw": best["raw"][0], "val_auc_ema": best["ema"][0],
            "epoch_raw": best["raw"][2], "epoch_ema": best["ema"][2], "epochs_run": epoch + 1}
    return best["raw"][1], best["ema"][1], info, model


def fit_features(Fall: np.ndarray, fit_idx):
    mu = Fall[fit_idx].mean(0)
    sd = Fall[fit_idx].std(0) + 1e-6
    return np.clip((Fall - mu) / sd, -5, 5).astype(np.float32)


def run_fold(r: dict, cfg: dict, data: dict, Fraw, fit_idx, val_idx, test_idx, seed: int, device,
             eval_test: bool = True):
    hybrid = r["arch"].endswith("_feat")
    X = torch.from_numpy(np.stack([data["ecg"], data["ppg"]], axis=1)).float()
    y = torch.from_numpy(data["label"]).float()
    Fz = torch.from_numpy(fit_features(Fraw, fit_idx)) if hybrid else None

    def g(idx):
        return X[idx].to(device), (None if Fz is None else Fz[idx].to(device)), y[idx].to(device)

    Xf, Ff, yf = g(fit_idx)
    Xv, Fv, yv = g(val_idx)
    yv_np = data["label"][val_idx]
    if eval_test:
        Xt, Ft, _ = g(test_idx)
    shifts = tuple(r["tta_shifts"])
    views_v, views_t, infos = {}, {}, []
    members = []  # per member: dict view->(val, test)
    template = None
    for m in range(r["k"]):
        raw_s, ema_s, info, model = train_member(r, cfg, Xf, Ff, yf, Xv, Fv, yv_np, seed * 1000 + m, device)
        infos.append(info)
        per = {}
        for name, st in (("raw", raw_s), ("ema", ema_s)):
            model.load_state_dict(st)
            for tta, sh in (("", (0,)), ("_tta", shifts)):
                pv = predict(model, Xv, Fv, sh)
                pt = predict(model, Xt, Ft, sh) if eval_test else None
                per[name + tta] = (pv, pt)
        members.append(per)
        del model
    # views
    for name in ("raw", "raw_tta", "ema", "ema_tta"):
        views_v[f"single_{name}"] = members[0][name][0]
        views_v[f"ens{r['k']}_{name}"] = np.mean([mm[name][0] for mm in members], 0)
        if eval_test:
            views_t[f"single_{name}"] = members[0][name][1]
            views_t[f"ens{r['k']}_{name}"] = np.mean([mm[name][1] for mm in members], 0)
    for kk in range(2, r["k"]):
        views_v[f"ens{kk}_ema_tta"] = np.mean([mm["ema_tta"][0] for mm in members[:kk]], 0)
        if eval_test:
            views_t[f"ens{kk}_ema_tta"] = np.mean([mm["ema_tta"][1] for mm in members[:kk]], 0)
    main = f"ens{r['k']}_ema_tta"
    return views_v, views_t, main, infos


# ----------------------------------------------------------------- data glue
def load_dataset(key: str, cfg: dict):
    fname, protocol = DATASETS[key]
    path = ROOT / cfg["paths"]["processed_dir"] / fname
    data = load_processed(path)
    cache = path.with_name(path.stem + "_features.npz")
    n = len(data["label"])
    Fraw = None
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if len(z["X"]) == n:
            Fraw = z["X"]
    if Fraw is None:
        from src.features import feature_matrix
        t0 = time.time()
        Fraw, names = feature_matrix(data["ecg"], data["ppg"])
        np.savez(cache, X=Fraw, names=np.array(names))
        print(f"computed features {Fraw.shape} in {time.time() - t0:.0f}s -> {cache.name}")
    return data, Fraw.astype(np.float32), protocol


def parse_recipe(s: str) -> tuple[str, dict]:
    """'name:arch=resnet1d_feat,aug=1,lr=0.001,k=5'"""
    name, _, rest = s.partition(":")
    r = dict(DEFAULTS)
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=")
        d = DEFAULTS[k]
        r[k] = v if isinstance(d, (str, tuple)) else (int(float(v)) if isinstance(d, int) else float(v))
    return name, r


# ---------------------------------------------------------------------- grid
def cmd_grid(args, cfg, device):
    data, Fraw, protocol = load_dataset("cinc", cfg)
    out = ROOT / "runs" / "deep_plus_grid"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "grid.jsonl"
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            d = json.loads(line)
            done.add((d["config"], d["seed"], d["fold"]))
    archs = args.archs.split(",") if args.archs else GRID["arch"]
    configs = list(itertools.product(archs, GRID["aug"], GRID["loss"], GRID["lr"]))
    for arch, aug, loss, lr in configs:
        cname = f"{arch}|aug{aug}|{loss}|lr{lr:g}"
        r = dict(DEFAULTS, arch=arch, aug=aug, loss=loss, lr=lr, k=1)
        for seed in GRID_SEEDS:
            set_seed(cfg["seed"] + seed)
            for fold_i, fit_idx, val_idx, test_idx in iter_splits(data, cfg, seed, protocol):
                if (cname, seed, fold_i) in done:
                    continue
                t0 = time.time()
                set_seed(cfg["seed"] + seed * 10 + fold_i)
                _, _, _, infos = run_fold(r, cfg, data, Fraw, fit_idx, val_idx, test_idx,
                                          seed * 10 + fold_i, device, eval_test=False)
                rec = dict(config=cname, arch=arch, aug=aug, loss=loss, lr=lr, seed=seed, fold=fold_i,
                           seconds=time.time() - t0, **infos[0])
                with open(log, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{cname} s{seed} f{fold_i} valAUC raw={rec['val_auc_raw']:.3f} "
                      f"ema={rec['val_auc_ema']:.3f} ({rec['seconds']:.0f}s)", flush=True)


def cmd_select(args, cfg, device):
    import pandas as pd
    out = ROOT / "runs" / "deep_plus_grid"
    df = pd.DataFrame([json.loads(x) for x in (out / "grid.jsonl").read_text().splitlines()])
    g = df.groupby(["config", "arch", "aug", "loss", "lr"]).agg(
        n=("val_auc_ema", "size"), val_auc_ema=("val_auc_ema", "mean"),
        val_auc_ema_sd=("val_auc_ema", "std"), val_auc_raw=("val_auc_raw", "mean"),
        epochs=("epoch_ema", "mean")).reset_index()
    g["val_auc_ema_se"] = g["val_auc_ema_sd"] / np.sqrt(g["n"])
    g = g.sort_values("val_auc_ema", ascending=False)
    g.to_csv(out / "grid_summary.csv", index=False)
    print(g.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    top = g.iloc[0]
    deep_only = g[~g["arch"].str.endswith("_feat")].iloc[0]
    chosen = {
        "overall": {k: (v.item() if hasattr(v, "item") else v) for k, v in top.items()},
        "deep_only": {k: (v.item() if hasattr(v, "item") else v) for k, v in deep_only.items()},
        "criterion": "mean inner-validation AUC of the EMA model at its best epoch, 3 seeds x 5 folds, CinC PPG cohort",
    }
    (out / "chosen.json").write_text(json.dumps(chosen, indent=2))
    print("\nCHOSEN overall :", chosen["overall"]["config"], f"{top.val_auc_ema:.4f}")
    print("CHOSEN deep-only:", chosen["deep_only"]["config"], f"{deep_only.val_auc_ema:.4f}")


# --------------------------------------------------------------------- final
def cmd_final(args, cfg, device):
    data, Fraw, protocol = load_dataset(args.data, cfg)
    out = ROOT / "runs" / args.run_name
    assert args.run_name not in ("challenge2015_ppg", "vtac_official")
    (out / "preds").mkdir(parents=True, exist_ok=True)
    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for spec in args.recipe:
        name, r = parse_recipe(spec)
        if args.k:
            r["k"] = args.k
        if args.data == "vtac":
            r.update(epochs=args.epochs or 20, patience=6, batch=64)
        if args.epochs:
            r["epochs"] = args.epochs
        fms = []
        for seed in range(args.seeds):
            set_seed(cfg["seed"] + seed)
            for fold_i, fit_idx, val_idx, test_idx in iter_splits(data, cfg, seed, protocol):
                pp = out / "preds" / f"{name}_seed{seed}_fold{fold_i}.npz"
                mp = pp.with_suffix(".json")
                if pp.exists() and mp.exists():
                    fms.append(json.loads(mp.read_text()))
                    continue
                t0 = time.time()
                set_seed(cfg["seed"] + seed * 10 + fold_i)
                vv, vt, main, infos = run_fold(r, cfg, data, Fraw, fit_idx, val_idx, test_idx,
                                               seed * 10 + fold_i, device)
                val_y, test_y = data["label"][val_idx], data["label"][test_idx]
                vp, tp = vv[main], vt[main]
                m = binary_metrics(test_y, tp)
                m["challenge_score"] = challenge_score(m["tp"], m["tn"], m["fp"], m["fn"])
                m["ops"] = operating_points(val_y, vp, test_y, tp)
                m.update(params=sum(p.numel() for p in Net(r["arch"], cfg, Fraw.shape[1], r["feat_dim"]).parameters()),
                         seed=seed, fold=fold_i, train_seconds=time.time() - t0, n_train=len(fit_idx),
                         n_test=len(test_idx), recipe=r, member_info=infos,
                         val_auc=auc_safe(val_y, vp), main_view=main)
                extras = {f"val_prob__{k}": v for k, v in vv.items()}
                extras.update({f"test_prob__{k}": v for k, v in vt.items()})
                np.savez(pp, test_idx=test_idx, val_idx=val_idx, val_prob=vp, val_y=val_y,
                         test_prob=tp, test_y=test_y, **extras)
                mp.write_text(json.dumps(m, default=float))
                fms.append(m)
                print(f"[{name}] seed={seed} fold={fold_i} f1={m['f1']:.3f} auc={m['auc']:.3f} "
                      f"sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} ({m['train_seconds']:.0f}s)",
                      flush=True)
        s = summarize_across_folds(fms)
        s.update(variant=name, recipe=r, fold_metrics=fms)
        results[name] = s
        results_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"== {name:22s} F1={s['f1_mean']:.3f}+/-{s['f1_std']:.3f} Sens={s['sensitivity_mean']:.3f} "
              f"Spec={s['specificity_mean']:.3f} AUC={s['auc_mean']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--archs", default=None, help="comma list to restrict (for chunked running)")
    sub.add_parser("select")
    f = sub.add_parser("final")
    f.add_argument("--data", required=True, choices=list(DATASETS))
    f.add_argument("--run-name", required=True)
    f.add_argument("--recipe", action="append", required=True, help="name:arch=...,aug=1,loss=focal,lr=0.001,k=5")
    f.add_argument("--seeds", type=int, default=10)
    f.add_argument("--k", type=int, default=None)
    f.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    {"grid": cmd_grid, "select": cmd_select, "final": cmd_final}[args.cmd](args, cfg, device)


if __name__ == "__main__":
    main()
