# ECG + PPG Cross-Attention Fusion for Arrhythmia Alarm Verification

Does letting a model compare ECG and PPG against each other via cross-attention verify ICU
arrhythmia alarms better than giving it only one signal? See [proposal.md](proposal.md) for
the full problem statement, [architecture.md](architecture.md) for the model design, and
[datasets.md](datasets.md) for data sources.

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
