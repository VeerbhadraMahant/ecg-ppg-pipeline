# Classical ML on engineered features (v2)

Code: `src/features_v2.py` (266 features; v1 `src/features.py` untouched), `src/classical.py`
(6 algorithms + 2 ensembles), `src/classical_report.py` (tables). Runs: `runs/classical_cinc`,
`runs/classical_cinc_recovered`, `runs/classical_vtac` (same `preds/*.npz|json` and `results.json`
format as `src/baselines_gbm.py`, plus `accuracy`, `best_params`, `val_thr` in each fold json).
Reproduce: `python -m src.classical --dataset {cinc,cinc_recovered,vtac} --run-name classical_<dataset> --seeds 10`.

## Feature set v2 (266 features, + 5 alarm-type one-hots on CinC)
* `e_*` ECG HRV (SDNN, RMSSD, pNN50/20, RR entropy, RR sample entropy, Poincare SD1/SD2, rates over last 3/4/5 s), QRS width/amplitude stats, T and P amplitude proxies, template correlation.
* `p_*` PPG pulse-interval HRV, pulse amplitude/variability, rise time, area, half-width, dicrotic proxy, up-slope, perfusion proxy.
* `es_*`/`ps_*` Welch band powers, spectral entropy, dominant frequency, centroid, edge; Daubechies-4 wavelet energies (pywt installed).
* `x_*` ECG-PPG agreement: PTT mean/std/median/MAD, beats-with-pulse fraction (all, last 5 s, last 6 beats), HR and beat-count ratios over full / last 5 / last 4 / last 3 s, cross-correlation peak and lag (envelope and R-impulse), coherence, dominant-frequency difference.
* `eq_*`/`pq_*` signal quality both channels (kurtosis, flat/clip fraction, per-second SD stats, SNR, ACF, second-detector agreement bSQI).
* `d_*` alarm-type detectors: longest RR gap and time since last beat (asystole), HR at window end, rolling HR extremes (brady/tachy), sample entropy, dominant-frequency concentration, leakage, 3-7 Hz fraction (VT/VF).

Features use only the window's own samples; imputation, clipping and standardisation are fitted on each fold's fit set.

## Protocol
Splits come from `src.train.iter_splits` (CinC: 5-fold record-wise x 10 seeds with a 15 % record-wise inner validation set; VTaC: official split, 10 seeds). Per fold and per algorithm, 30 random hyper-parameter configs are trained on the fit set and selected by **inner-validation AUC only**; the test fold is predicted once. SVM uses Platt probabilities from the fit set; the MLP is the mean of 5 inits; XGBoost is the boosting model. `clf_stack` is a logistic regression on the logits of the 6 base models, fitted on inner-validation predictions (its own validation predictions are 5-fold cross-validated); `clf_avg` is the plain mean. "val-thr" columns (in `runs/classical_leaderboard.md`) use the F1-maximising threshold chosen on the inner validation set; all other columns use threshold 0.5.

## Results (mean ± std over folds x seeds; VTaC std is over seeds of one fixed test split)
Reference `gbm_features` (v1): CinC F1 0.798 / AUC 0.918 / acc 0.851; VTaC F1 0.735 / AUC 0.914.

### CinC 2015, PPG cohort (592 records, 50 folds)
| Model | Accuracy | F1 | AUC | Sens | Spec | Challenge |
|---|---|---|---|---|---|---|
| logreg | 0.852 ± 0.028 | 0.801 ± 0.043 | 0.928 ± 0.021 | 0.799 | 0.886 | 65.9 |
| svm | 0.852 ± 0.034 | 0.791 ± 0.066 | 0.928 ± 0.025 | 0.762 | 0.909 | 63.6 |
| rf | 0.863 ± 0.029 | 0.795 ± 0.047 | 0.938 ± 0.023 | 0.717 | 0.952 | 61.0 |
| extratrees | 0.867 ± 0.028 | 0.802 ± 0.047 | **0.947 ± 0.020** | 0.725 | 0.953 | 61.8 |
| boost (xgboost) | **0.871 ± 0.031** | 0.818 ± 0.049 | 0.939 ± 0.023 | 0.782 | 0.925 | 66.2 |
| mlp | 0.857 ± 0.030 | 0.802 ± 0.047 | 0.927 ± 0.023 | 0.778 | 0.906 | 64.8 |
| **stack** | 0.860 ± 0.033 | **0.820 ± 0.042** | 0.944 ± 0.020 | 0.856 | 0.864 | **71.2** |
| avg | **0.871 ± 0.028** | 0.819 ± 0.043 | 0.945 ± 0.019 | 0.783 | 0.925 | 66.2 |

### CinC recovered (724 records, 50 folds)
| Model | Accuracy | F1 | AUC | Sens | Spec | Challenge |
|---|---|---|---|---|---|---|
| logreg | 0.849 ± 0.032 | 0.808 ± 0.046 | 0.932 ± 0.019 | 0.811 | 0.874 | 65.8 |
| svm | 0.864 ± 0.026 | 0.823 ± 0.040 | 0.939 ± 0.018 | 0.808 | 0.901 | 66.9 |
| rf | 0.861 ± 0.029 | 0.809 ± 0.043 | 0.942 ± 0.018 | 0.750 | 0.934 | 62.2 |
| extratrees | 0.876 ± 0.027 | 0.831 ± 0.039 | **0.953 ± 0.015** | 0.776 | 0.941 | 65.1 |
| boost | **0.877 ± 0.025** | 0.839 ± 0.034 | 0.947 ± 0.015 | 0.813 | 0.919 | 68.0 |
| mlp | 0.857 ± 0.028 | 0.816 ± 0.039 | 0.934 ± 0.018 | 0.808 | 0.889 | 66.1 |
| **stack** | 0.875 ± 0.028 | **0.844 ± 0.037** | **0.953 ± 0.014** | 0.859 | 0.886 | **72.0** |
| avg | 0.872 ± 0.026 | 0.833 ± 0.037 | 0.951 ± 0.014 | 0.816 | 0.908 | 67.9 |

### VTaC (official split, 10 seeds)
| Model | Accuracy | F1 | AUC | Sens | Spec | Challenge |
|---|---|---|---|---|---|---|
| logreg | 0.873 ± 0.002 | 0.770 ± 0.004 | 0.937 ± 0.000 | 0.754 | 0.920 | 68.4 |
| svm | 0.894 ± 0.008 | 0.810 ± 0.015 | 0.948 ± 0.001 | 0.807 | 0.928 | 73.5 |
| rf | 0.884 ± 0.006 | 0.777 ± 0.016 | 0.931 ± 0.003 | 0.721 | 0.948 | 67.4 |
| extratrees | 0.886 ± 0.008 | 0.783 ± 0.017 | 0.936 ± 0.004 | 0.733 | 0.946 | 68.2 |
| boost | 0.892 ± 0.005 | 0.809 ± 0.009 | 0.940 ± 0.004 | 0.816 | 0.921 | 73.9 |
| mlp | 0.901 ± 0.007 | 0.823 ± 0.013 | **0.949 ± 0.002** | 0.818 | 0.934 | 74.8 |
| stack | 0.880 ± 0.010 | 0.808 ± 0.015 | 0.946 ± 0.003 | 0.895 | 0.874 | **78.7** |
| **avg** | **0.904 ± 0.004** | **0.827 ± 0.007** | 0.948 ± 0.001 | 0.810 | 0.941 | 74.5 |

## Best configuration
Boosting, ExtraTrees and the two ensembles are consistently on top. By F1 and challenge score the **stack** wins on both CinC cohorts (F1 0.820 / 0.844, challenge 71-72); by accuracy and AUC the plain **average**, ExtraTrees and boosting are equal within noise. On VTaC **avg** (accuracy 0.904, F1 0.827, AUC 0.948) and the MLP/SVM are best; the stack has the highest challenge score (78.7) by trading specificity for sensitivity (0.895 / 0.874). Versus the v1 GBM baseline: F1 +0.02 / AUC +0.03 on CinC, F1 +0.09 / AUC +0.035 on VTaC. Differences among the top 3-4 models are well inside one fold-std; no paired significance test was run (the preds are in the standard format, so the `src.evaluate` tooling can do it).

## Feature importance
Permutation importance (validation-AUC drop) of the selected models on the inner validation set of seed-0 folds (CinC 3 folds, VTaC 1 split): `runs/classical_*/importance_*.json`, tables in `runs/classical_leaderboard.md`. On CinC the alarm-type indicator (`alarm_tachy`) dominates, followed by wavelet level-1 energy (`es_wav1`), ECG P/T proxy ratios, PPG kurtosis and PPG interval regularity; on VTaC the ECG sample entropy over the last 5 s, per-second SD stationarity features, wavelet energy and template-correlation spread. Importances are small (< 0.04 AUC) and noisy (validation sets of ~70-110 records on CinC), so the rank order beyond the top 3-5 is unreliable.

## Caveats
* Small n: CinC has 592/724 records and fold std is 0.02-0.05 in F1, so most pairwise algorithm differences are not significant. Seeds re-use the same data and are not independent.
* VTaC "official" is one fixed test split of 482 events; the seed std reflects only model/search randomness, not test-set sampling uncertainty.
* Domain knowledge is built in: detectors are tied to the alarm definitions (asystole gap > 2.5/4 s, brady < 50, tachy > 120), and on CinC the **alarm type is an input feature** (known at alarm time, but a strong prior; `alarm_tachy` is the top feature). `--no-alarm-type` drops it; that ablation was not run. VTaC contains only VT alarms, so no alarm-type feature is used there.
* Hyper-parameter selection uses only the inner validation set (15 % of training records), so choosing by AUC among 30 configs is noisy and best-of-30 validation AUC is optimistic; the test fold is never used. The stacker and thresholds are also fitted on that same validation set, so validation predictions are optimistic relative to test.
* Headline F1/accuracy use threshold 0.5, which disadvantages unweighted models (RF/ET/MLP have low sensitivity at 0.5). The validation-chosen-threshold columns gain up to ~0.02 F1 for ET/RF on CinC, and are noisy.
* Beat detectors (Pan-Tompkins variant, PPG peak picking) are imperfect on noisy windows; undefined features (too few beats) are median-imputed from the fit set.
* No feature selection was applied (optional item skipped); all 266 (+5) features go to every model.
* The pipeline z-normalises windows, which removes absolute amplitude information (e.g. a true perfusion index).
