# ECG + PPG Cross-Attention Fusion for Arrhythmia Alarm Verification

Does letting a model compare ECG and PPG against each other via cross-attention verify ICU
arrhythmia alarms better than giving it only one signal? See [proposal.md](proposal.md) for
the full problem statement, [architecture.md](architecture.md) for the model design, and
[datasets.md](datasets.md) for data sources.

## What this project actually does:

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

## Results (leak-free; supersedes the first version of this README)

The first version of this project reported F1 0.774 for cross-attention fusion. That number was
optimistically biased: the best epoch was selected on the same fold that was then reported. After
adding an inner validation split, 10 seeds and corrected statistics, **the proposed cross-attention
model is not better than naive concatenation, and both are clearly beaten by simpler models.**
Full protocol: [docs/evaluation_protocol.md](docs/evaluation_protocol.md). Full tables with
confidence intervals: [docs/LEADERBOARD.md](docs/LEADERBOARD.md) and [docs/deep_recipe.md](docs/deep_recipe.md).

| Model (Challenge 2015, 592 records, 5-fold CV x 10 seeds) | Accuracy | F1 | AUC |
|---|---|---|---|
| **Ensemble incl. the hybrid deep model (5 seeds)** | **0.878** | **0.830** | **0.950** |
| Ensemble of 10 models, no hybrid (10 seeds) | 0.876 | 0.827 | 0.948 |
| **Hybrid deep recipe** (attention CNN-GRU + augmentation, EMA, TTA, 3-member ensemble; 10 seeds) | 0.871 | 0.828 | 0.932 |
| XGBoost on engineered features (v2) | 0.871 | 0.818 | 0.939 |
| Extra Trees / Random Forest / SVM / MLP / Logistic regression (features v2) | 0.852 - 0.867 | 0.791 - 0.802 | 0.927 - 0.947 |
| ResNet1D (deep, early fusion) | 0.820 | 0.760 | 0.884 |
| TCN (deep) | 0.810 | 0.751 | 0.887 |
| CNN cross-attention fusion (the original proposal) | 0.765 | 0.699 | 0.841 |
| CNN concatenation fusion | 0.757 | 0.693 | 0.836 |

On VTaC (4,542 VT alarms, official patient-disjoint split, 5 seeds) the best model is the hybrid deep recipe
(accuracy 0.913 / F1 0.858 / AUC 0.966), ahead of the 10-model ensemble (0.910 / 0.839 / 0.955), the best
classical model MLP (0.900 / 0.821 / 0.949) and the best plain CNN (bidirectional cross-attention,
0.857 / 0.775 / 0.933). Differences among the top three models are inside bootstrap noise; the gap to the
originally proposed cross-attention network (0.746 F1) is not (+0.11, CI +0.07 ... +0.15).
Caveat for the hybrid recipe: its configuration was chosen on inner-validation scores of Challenge 2015 folds
(grid in [docs/deep_recipe.md](docs/deep_recipe.md)); those records are test records in other seeds/folds, so a small
optimistic bias is possible. A shuffled-label control gave chance-level AUC and no leakage was found.

Key findings (details in `docs/`):

* **Cross-attention is not better than concatenation** (dF1 = +0.006, paired record-bootstrap 95% CI
  -0.015 ... +0.028). Four attention-style fusion variants are statistically indistinguishable.
* **Fusion models do not use ECG-PPG timing.** Swapping in another patient's same-label PPG or shifting the
  PPG by up to 2 s changes AUC by < 0.015; attention does not track measured pulse transit time
  (Spearman rho = -0.04, permutation p = 0.39). The pulse acts as a patient-agnostic evidence cue.
* **Hand-crafted agreement features beat plain deep models**, and a hybrid deep recipe that combines a stronger
  trainer (augmentation, EMA, TTA, ensembling) with an attention CNN-GRU and the features is the best single model
  on both datasets; ensembles are about equal.
* **Cross-dataset transfer is asymmetric**: Challenge 2015 -> VTaC is weak (F1 ~0.55-0.60), whereas VTaC ->
  Challenge 2015 VT alarms works well (ResNet1D F1 0.773 / AUC 0.927), so single-dataset numbers
  overstate real-world performance.
* **Safety:** validation-chosen sensitivity targets are not met exactly on test folds (realised ~92-97%).
  See the safety layer (`src/safety.py`) for a risk-controlled threshold with an explicit confidence bound.
* Missed true alarms are dominated by ventricular tachycardia ([docs/error_taxonomy.md](docs/error_taxonomy.md)).
* Foundation model (PaPaGei) did not beat from-scratch encoders ([docs/foundation_models.md](docs/foundation_models.md));
  self-supervised pretraining + fine-tuning gives only +0.01 to +0.04 F1.
* AF screening on MIMIC PERform AF: an RR-irregularity GBM beats all neural models ([docs/af_screening.md](docs/af_screening.md)).

Other documents: [consolidated analysis findings](docs/analysis_findings.md), [updates status ledger](docs/updates_status.md), [datasheet](docs/datasheet.md), [classical models](docs/classical_models.md),
[model card](docs/model_card.md), [TRIPOD+AI checklist](docs/tripod_ai_checklist.md), [updates.md](updates.md) (plan).
Live replay demo: `site/replay.html`.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
```

## Pipeline

```bash
python scripts/download_data.py --dataset challenge2015       # PhysioNet; slow host, see scripts/download_vtac.py for the parallel range-request trick
python scripts/download_vtac.py --workers 12                  # VTaC v1.1 windows around the alarm (~1 GB instead of 18 GB)
python -m src.data.preprocess                                 # PPG cohort (592) + recovered cohort (724) + cohort_audit.csv
python scripts/build_vtac_windows.py                          # decision-time windows (+ --post-seconds 5/10/30 for time-to-verdict)
python scripts/match_params.py                                # parameter-match all neural variants (~261k)
python -m src.train --variant all --run-name challenge2015_ppg                       # 10 seeds, record-wise CV, inner validation
python -m src.train --variant ecg_only,concat,resnet1d --protocol official --processed data/processed/vtac_windows.npz --run-name vtac_official
python -m src.baselines_gbm                                   # hand-crafted features + gradient boosting
python -m src.classical --dataset cinc --run-name classical_cinc                     # six classical algorithms on features v2
python -m src.ensemble --run-name challenge2015_ppg --out-run final_cinc --members ...   # leak-free ensembles
python -m src.evaluate --run-name challenge2015_ppg           # tables + corrected statistics
python scripts/final_leaderboard.py                           # docs/LEADERBOARD.md
python -m src.analysis.counterfactual / attention_ptt / faithfulness / modality_subsets / stress_benchmark / error_taxonomy
python -m src.pretrain --objective contrastive; python scripts/run_label_efficiency.py
python -m src.safety --variant cross_attention
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
