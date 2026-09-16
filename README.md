# ECG + PPG Cross-Attention Fusion for Arrhythmia Alarm Verification

Does letting a model compare ECG and PPG against each other via cross-attention verify ICU
arrhythmia alarms better than giving it only one signal? See [proposal.md](proposal.md) for
the full problem statement, [architecture.md](architecture.md) for the model design, and
[datasets.md](datasets.md) for data sources.

## Results

Trained on the PhysioNet/CinC Challenge 2015 training set (592 of 750 public records kept after
filtering to a usable PPG channel and sufficient window length), record-wise 5-fold CV x 3 seeds,
on an RTX 4060 laptop GPU.

| Variant | Params | F1 | Sensitivity | Specificity | AUC |
|---|---|---|---|---|---|
| PPG only | 266,001 | 0.660 &plusmn; 0.039 | 0.856 &plusmn; 0.076 | 0.555 &plusmn; 0.106 | 0.747 &plusmn; 0.051 |
| ECG only | 266,001 | 0.736 &plusmn; 0.030 | 0.805 &plusmn; 0.058 | 0.758 &plusmn; 0.126 | 0.818 &plusmn; 0.053 |
| Concatenation fusion | 252,229 | 0.750 &plusmn; 0.030 | 0.804 &plusmn; 0.068 | 0.794 &plusmn; 0.067 | 0.853 &plusmn; 0.031 |
| **Cross-attention fusion** | 261,505 | **0.774 &plusmn; 0.046** | 0.797 &plusmn; 0.062 | 0.831 &plusmn; 0.109 | **0.866 &plusmn; 0.041** |

Monotonic in the predicted direction (PPG-only < ECG-only < concat < cross-attention) at matched
parameter count, and the cross-attention gain over concatenation is **statistically significant**
(paired t-test on matched seed/fold F1, p = 0.041, n = 15). External generalization check on MIMIC
PERform AF (35 subjects, never trained on, a distribution-shifted transfer task): F1 = 0.642 &plusmn; 0.071.

Also reported per proposal.md's methodology: sensitivity/specificity/F1 broken down by all five
alarm types (Tachycardia is high-sensitivity/low-specificity; Asystole and V-Fib/Flutter have too
few true-alarm examples for their per-class numbers to be fully trusted), and inference speed
alongside accuracy (all four variants run in under 10ms/window on the RTX 4060; cross-attention is
the slowest at ~107 windows/sec, still comfortably real-time).

Full write-up, charts, and the actual attention-weight visualizations: [`results.ipynb`](results.ipynb)
and the [results dashboard](site/index.html) (`site/index.html`, open directly or serve the folder).

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
```

## Pipeline

1. **Download data**
   ```bash
   python scripts/download_data.py --dataset all
   ```
2. **Preprocess into fixed-length windows**
   ```bash
   python -m src.data.preprocess
   ```
3. **Match baseline-ladder parameter counts**
   ```bash
   python scripts/match_params.py
   ```
4. **Train all four ladder variants** (record-wise CV, multiple seeds)
   ```bash
   python -m src.train --variant all --seeds 3
   ```
5. **Evaluate / build comparison table**
   ```bash
   python -m src.evaluate
   ```
6. **Attention visualizations**
   ```bash
   python -m src.attention_viz
   ```
7. **External validation on MIMIC PERform AF**
   ```bash
   python -m src.external_validation
   ```

## Project layout

```
configs/            central config (config.yaml) + parameter-matching output (width_mult.yaml)
scripts/            data download, parameter matching
src/data/           signal ops, WFDB/CSV loaders, preprocessing, PyTorch Dataset + record-wise CV
src/models/         CNN encoders, cross-attention / concat fusion, classification head, losses
src/train.py        training loop across the 4-variant baseline ladder
src/evaluate.py      comparison table + performance-vs-signal-quality breakdown
src/attention_viz.py attention-weight extraction and plotting
src/external_validation.py  generalization check on MIMIC PERform AF (never trained on)
```

## Baseline ladder

| Variant | Encoders | Fusion |
|---|---|---|
| `ecg_only` | ECG CNN | none |
| `ppg_only` | PPG CNN | none |
| `concat` | Both CNNs | concatenation, no attention |
| `cross_attention` | Both CNNs | transformer cross-attention (proposed) |

All four are parameter-matched (`scripts/match_params.py`) so any gap between variants reflects
the fusion mechanism, not extra capacity.
