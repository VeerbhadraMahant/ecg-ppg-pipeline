# Evaluation protocol

This chapter is the single source of truth for how every number in the project
is produced. It exists because an earlier version of this repository reported
results that were optimistically biased (see "Leakage audit" below).

## 1. Units, splits and leakage controls

| Item | Rule |
|---|---|
| Unit of analysis | One alarm event = one 10 s window. In CinC 2015 and VTaC each patient/record contributes the windows of its own alarm(s); all windows of a record stay on one side of every split. |
| Window position | The window **ends at the alarm onset**. No post-alarm samples enter any model input for the headline task. VTaC records also contain 30 s after onset; they are read only by the time-to-verdict analysis, which varies the allowed post-alarm seconds explicitly. |
| Outer split (CinC) | Record-wise 5-fold CV, repeated for **10 seeds** (seed changes the fold assignment). |
| Outer split (VTaC) | The dataset's official, patient-disjoint train/val/test split (4060/495/482 events) for the headline number; record-wise CV as a secondary check. |
| Inner split | Inside every training fold, 15 % of the records are held out (record-wise) as a **validation set**. It is used for early stopping, best-epoch selection, temperature scaling and threshold selection. **The outer test fold is touched exactly once**, to report metrics with the selected weights. |
| Preprocessing | Per-window filtering and z-normalisation only; nothing is fitted across windows, so there is no train/test statistic leakage. Quality filtering uses only the window's own samples. |
| Model comparison | All variants see identical splits (same seeds and folds), so comparisons are paired. All neural variants are matched to ~261k parameters (`scripts/match_params.py`). |
| Hyper-parameters | Fixed a priori in `configs/config.yaml`; **not tuned per variant**. Reported scores are therefore not selected on test data. |

## 2. Metrics

* **Primary**: F1 of the true-alarm class at threshold 0.5, because it is the metric the project began with and is comparable across runs.
* **Safety-oriented** (the headline for a clinical reader):
  * False alarms suppressed at a fixed sensitivity of 95 / 99 / 100 %. The decision threshold is chosen on the **validation set** of each fold to reach the target sensitivity there, then applied unchanged to the test fold. The sensitivity actually *realised* on the test fold is always reported next to the suppression rate, because validation-chosen thresholds do not transfer exactly (validation sets contain few true alarms).
  * The official PhysioNet/CinC 2015 score, `(TP+TN)/(TP+TN+FP+5·FN)`, in which a missed true alarm costs five false alarms.
  * Per-alarm-type sensitivity, specificity, F1 and suppression.
* **Ranking**: AUC.
* **Calibration** (safety layer): temperature scaling fitted on validation, ECE and Brier on test.

## 3. Statistics

The earlier paired t-test on 15 fold/seed pairs treated overlapping-training-set scores as independent and ignored multiple comparisons. It is replaced by:

1. **Comparison family fixed in advance** (`stats.comparisons` in `configs/config.yaml`, committed before the final results existed). Each comparison is `[baseline, challenger]`.
2. **Nadeau–Bengio corrected resampled t-test** on per-(seed, fold) F1 differences, variance factor `1/J + n_test/n_train`.
3. **Paired record-level bootstrap** (5000 resamples): records are resampled with replacement, the *same* resample is applied to both models, the metric is computed per seed and averaged over seeds. Reported for F1, the Challenge score and specificity at the 95 % sensitivity operating point.
4. **Holm correction** across the family, per metric.

A result is called significant only if it survives Holm correction on the
bootstrap and the sign agrees with the Nadeau–Bengio test; where the two
disagree the result is described as "not robust".

## 4. Leakage audit of the earlier results

The original `train.py` selected the best epoch and stopped training using
F1 measured on the *same* fold it then reported. Re-running with a separate
inner validation set (and 10 instead of 3 seeds) moved the numbers as follows
(archived originals in `runs/legacy_pre_fix/`):

| Variant | F1 before | F1 after |
|---|---|---|
| PPG only | 0.660 | 0.602 |
| ECG only | 0.736 | 0.642 |
| Concatenation | 0.750 | 0.693 |
| Cross-attention | 0.774 | 0.699 |

The cross-attention advantage over concatenation shrank from +0.024 to +0.006
and is no longer distinguishable from zero (bootstrap 95 % CI −0.015 … +0.028).

## 5. What is *not* controlled

* Patient overlap between CinC 2015 and VTaC / MIMIC-derived corpora is not verifiable from the released metadata; it is assumed to be zero for the two labelled sets (different institutions and collection periods) but this is unchecked.
* VTaC does not release hospital or device identifiers. Cross-device analyses use the **lead-set signature** of each record as a proxy for monitor manufacturer and are labelled as proxies.
* Validation sets in CinC folds contain roughly 15–25 true alarms, so validation-chosen operating points are noisy; this is visible in the realised-sensitivity columns.
* No prospective validation; US ICU data only.
