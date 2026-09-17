# ECG + PPG Cross-Attention Fusion for Arrhythmia Alarm Verification

Does letting a model compare ECG and PPG against each other via cross-attention verify ICU
arrhythmia alarms better than giving it only one signal? See [proposal.md](proposal.md) for
the full problem statement, [architecture.md](architecture.md) for the model design, and
[datasets.md](datasets.md) for data sources.

## What this project actually does

ICU bedside monitors trigger an audible alarm whenever a patient's vital signs cross a
dangerous threshold (e.g. heart rate too low, no heartbeat detected). In practice, the large
majority of these alarms are **false alarms** — caused by a loose electrode, patient movement,
or normal noise in the electrical signal — not by an actual life-threatening event. Constant
false alarms cause "alarm fatigue": staff become desensitized and start ignoring or silencing
alarms, including the rare real ones.

Most monitors decide to alarm using only the **ECG** (the heart's electrical signal). This
project asks: if the model is also shown the **PPG** (a second, independent signal measuring
actual blood flow), can it verify — after the ECG alarm fires — whether the alarm is real, by
checking whether the electrical event was actually followed by a physical heartbeat?

The core idea: a genuine arrhythmia should show up in **both** signals (an abnormal electrical
pattern *and* a corresponding disturbance in blood flow). A false alarm typically shows up in
only one (the ECG looks abnormal, often due to artifact, but the pulse keeps beating normally).
So a model that can compare the two signals against each other, instead of looking at just one,
should be better at telling true alarms from false ones.

The project builds four versions of the same classifier (a "baseline ladder", see below), all
with a matched number of parameters, so any difference in performance can be attributed to *how*
the signals are combined rather than to one model simply being bigger. The proposed model fuses
ECG and PPG using **cross-attention** (defined below); the other three are baselines that use
only one signal or a naive combination. The results (see below) show the proposed fusion model
does perform best.

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
scripts/            data download, parameter matching, results-notebook build
src/data/           signal ops, WFDB/CSV loaders, preprocessing, PyTorch Dataset + record-wise CV
src/models/         CNN encoders, cross-attention / concat fusion, classification head, losses
src/train.py         training loop across the 4-variant baseline ladder
src/evaluate.py      comparison table + performance-vs-signal-quality breakdown
src/attention_viz.py  attention-weight extraction and plotting
src/external_validation.py  generalization check on MIMIC PERform AF (never trained on)
tests/               unit tests for signal ops, dataset, and models
site/                static HTML results dashboard
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

## How the model works (short version)

1. Each raw ECG and PPG window is resampled, bandpass-filtered, and z-normalized.
2. Two separate 1D-CNN encoders (not weight-shared) turn each signal into a *sequence* of
   feature vectors — one vector per short time-slice, not a single summary vector, so the model
   still knows *when* in the window each feature occurred.
3. In the cross-attention variant, a transformer attention layer lets the ECG feature sequence
   "query" the PPG feature sequence: for every moment in the ECG, it looks up which moments in
   the PPG are relevant and pulls in that information. This lets the model learn the natural
   delay between an electrical heartbeat and the resulting pulse arriving at the sensor (the
   **pulse transit time**), instead of assuming the two signals line up sample-for-sample the
   way concatenation does.
4. The fused representation is pooled and passed through a small classifier head, producing a
   true-alarm-vs-false-alarm probability.

Full design rationale: [architecture.md](architecture.md).

## Glossary / definitions

**Signals**

- **ECG (electrocardiogram)** — records the heart's electrical activity via skin electrodes.
  Shows sharp, spiky waveforms; each heartbeat produces a **QRS complex** (the sharp spike
  corresponding to the heart's main pumping contraction).
- **PPG (photoplethysmogram)** — an optical sensor (e.g. a finger or ear clip) that measures
  blood volume changes in tissue as light absorption. Produces a smooth, rounded pulse wave.
  Confirms that an electrical heartbeat actually resulted in blood being pumped.
- **ABP (arterial blood pressure)** — a direct pressure-based pulsatile signal, sometimes present
  in the dataset instead of PPG; this project filters to records that have PPG specifically.
- **Pulse transit time** — the time delay between an electrical event on the ECG and the arrival
  of the corresponding pressure pulse at the PPG sensor. Varies between patients and over time;
  a fixed/naive alignment (like concatenation) cannot account for it, but attention can learn it.

**Clinical alarm types** (from the PhysioNet/CinC Challenge 2015 dataset)

| Alarm type | Definition |
|---|---|
| Asystole | No QRS complex (no heartbeat) detected for at least 4 seconds |
| Extreme Bradycardia | Heart rate under 40 bpm for 5 consecutive beats |
| Extreme Tachycardia | Heart rate over 140 bpm for 17 consecutive beats |
| Ventricular Tachycardia | 5 or more ventricular beats with heart rate over 100 bpm |
| Ventricular Flutter/Fibrillation | Fibrillatory/flutter/oscillatory waveform lasting 4+ seconds |

- **True alarm** — the alarm condition genuinely occurred (a real arrhythmia).
- **False alarm** — the monitor triggered but the condition was not real (usually caused by
  motion artifact, electrode disconnection, or noise).
- **Alarm fatigue** — the clinical problem this project targets: clinicians become desensitized
  to alarms after being exposed to too many false ones, and may miss or delay response to real
  ones as a result.

**Model / architecture terms**

- **1D-CNN (1-dimensional convolutional neural network) encoder** — a stack of convolution
  layers that slides small filters along a time-series signal to extract local waveform
  features (e.g. "is there a spike here", "how fast does the signal change"), progressively
  shrinking the sequence length while increasing the number of feature channels.
- **Cross-attention** — a mechanism (from the transformer architecture) where one sequence (the
  "queries") looks up relevant information in a second sequence (the "keys" and "values") by
  computing a similarity score between every position in one and every position in the other,
  then taking a weighted combination. Here, ECG timesteps query PPG timesteps to find the
  corresponding pulse activity.
- **Concatenation fusion** — the naive baseline: simply stack the ECG and PPG feature vectors
  together and let the classifier head figure out the relationship, with no explicit
  timestep-to-timestep comparison.
- **Weight-shared vs. not weight-shared** — whether two network branches use the same learned
  parameters. The ECG and PPG encoders here are deliberately *not* shared, because the two
  signals differ in shape, frequency content, and failure modes.
- **Positional encoding** — extra information added to each element of a sequence so the model
  knows its position in time; needed here because attention on its own has no sense of order.
- **Parameter-matched baseline ladder** — building all comparison models (ECG-only, PPG-only,
  concatenation, cross-attention) with roughly the same total number of learnable parameters, so
  that a performance difference reflects the fusion strategy rather than one model simply having
  more capacity.
- **Record-wise cross-validation (CV)** — splitting data into folds by patient/record rather than
  by individual windows, so that data from the same patient never appears in both the training
  and validation portions of a fold (which would let the model "cheat" by memorizing a patient).
- **Focal loss / class weighting** — techniques used during training to counteract class
  imbalance (there are far more false alarms than true ones in the data) without duplicating or
  discarding examples.

**Evaluation metrics**

- **Sensitivity (a.k.a. recall)** — of all the *real* true alarms, the fraction the model
  correctly flags as true. High sensitivity means the model rarely misses a genuine emergency.
- **Specificity** — of all the *real* false alarms, the fraction the model correctly flags as
  false. High specificity means the model successfully filters out noise/artifact alarms.
- **F1 score** — the harmonic mean of precision and sensitivity/recall; a single number
  balancing "did it catch true alarms" against "did it avoid falsely calling things true".
- **AUC (area under the ROC curve)** — measures how well the model ranks true alarms above false
  ones across all possible decision thresholds, independent of any one chosen cutoff. 0.5 is
  random guessing, 1.0 is perfect separation.
- **Paired t-test / statistical significance / p-value** — a statistical test used to check
  whether the improvement of one model over another (measured across matched seeds/folds) is
  larger than would be expected from random variation alone. A p-value under 0.05 is
  conventionally treated as "statistically significant".
- **External validation / generalization** — testing the trained model on a completely different
  dataset (MIMIC PERform AF) that it never saw during training, to check whether its performance
  holds up outside the exact data distribution it was trained on.

## Datasets

Full details, download links, and licensing notes: [datasets.md](datasets.md).

- **PhysioNet/CinC Challenge 2015** — primary dataset, used for training and cross-validation.
  750 public ICU records with paired ECG + pulsatile waveform and expert true/false labels for
  one of the five alarm types above.
- **MIMIC PERform AF** — external-only test set (35 subjects, atrial fibrillation vs. non-AF),
  used solely to check generalization; never used in training.
