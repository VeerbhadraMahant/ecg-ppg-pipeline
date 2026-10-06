# AF screening on MIMIC PERform AF (updates.md 3.6, 7.3)

**Read this first: n = 35 subjects (19 AF, 16 non-AF). Everything below is low-powered; confidence intervals are wide and most between-model differences are not distinguishable.**

## Setup

- **Data.** Zenodo record 15906524 (`mimic_perform_af_csv.zip`, `mimic_perform_non_af_csv.zip`) was downloaded with `scripts/download_data.py`. The extracted layout (`data/external/mimic_perform_af/{af,non_af}/<name>_csv/*.csv`) works with `iter_subjects` through `rglob`, so the loader was not edited. The `*_fix.txt` sidecars give the original MIMIC subject IDs: 35 distinct IDs, so the 35 files are 35 different patients.
- **CSV format.** Columns `Time,PPG,ECG,resp`, 125 Hz. Each subject has 150,001 samples (20 min), i.e. 120 non-overlapping 10 s windows, 4,200 in total (AF 2,280, non-AF 1,920). Some subjects have NaN rows (the loader interpolates them). `non_af_012` has 1,944 NaN rows (about 1.3%).
- **Windows used.** `build_external_windows` resamples to 250 Hz, band-passes (ECG 0.5-40 Hz, PPG 0.5-8 Hz) and z-normalises per 10 s window. 60 evenly spaced windows per subject were used (2,100 windows, 54.3% AF) to bound training time. Only one in two non-overlapping windows is used, so neighbouring used windows are adjacent in time. They are highly correlated.
- **Splits.** 5 outer folds, subject-wise (grouped by original subject ID), stratified by AF label (7 subjects per fold). Inner validation is about 20% of the training subjects (at least one per class) for early stopping and for best-epoch and teacher selection. 3 seeds change the fold assignment and the initialisation. The same folds and seeds are used by every model, including the GBM baselines and the distillation runs.
- **Models.**
  - Ladder variants `ecg_only`, `ppg_only`, `concat` and `cross_attention`, trained from scratch (`src/af_task.py`, config focal loss, up to 30 epochs, patience 10).
  - `cross_attention+pretrain`: initialised from `runs/challenge2015_ppg/checkpoints/cross_attention_seed0_fold{k}.pt`, then fully fine-tuned (via `cfg.train.init_from`, which `train_one_fold` already supports).
  - Hand-crafted baselines: HistGradientBoosting (depth 3, 150 iterations) on RR irregularity features from `detect_r_peaks` (`gbm_ecg_rr`). The features are mean RR, CV, RMSSD, normalised RMSSD, pNN50, pNN20, sample entropy, median absolute successive difference and range. `gbm_ppg_pp` uses the same features on PPG pulse-peak intervals.
- **Metrics.** Threshold 0.5 throughout. Window level: F1, AUC, sensitivity and specificity. Subject level: mean window probability per subject, then the same metrics (positive class = AF).
  - Main tables report the pooled out-of-fold (OOF) metric for each seed over all 35 subjects, averaged over 3 seeds. The 95% CI (in brackets) is a subject-level cluster bootstrap (1000 resamples, stratified by class, metric averaged over seeds per replicate).
  - The seed-to-seed SD of the pooled metric and the per-fold mean±SD (7 subjects per fold) are given in the second table.
  - Per-fold values and OOF probabilities are in `runs/af_task/`.

## Results: subject-wise CV (pooled OOF, mean over 3 seeds, [95% subject-bootstrap CI])

| model | win F1 | win AUC | subj F1 | subj AUC | subj sens | subj spec |
|---|---|---|---|---|---|---|
| gbm_ecg_rr (RR features) | 0.968 [0.95-0.99] | 0.977 [0.95-1.00] | 1.000 [1.00-1.00] | 1.000 [1.00-1.00] | 1.000 | 1.000 |
| gbm_ppg_pp (pulse-interval features) | 0.898 [0.84-0.95] | 0.933 [0.86-0.98] | 0.950 [0.88-1.00] | 0.939 [0.83-1.00] | 1.000 | 0.875 [0.69-1.00] |
| ppg_only (scratch) | 0.655 [0.58-0.72] | 0.602 [0.46-0.74] | 0.634 [0.55-0.72] | 0.612 [0.45-0.76] | 0.754 [0.61-0.88] | 0.271 [0.10-0.46] |
| ecg_only (scratch) | 0.737 [0.63-0.82] | 0.815 [0.70-0.91] | 0.724 [0.61-0.82] | 0.826 [0.71-0.92] | 0.667 [0.54-0.79] | 0.812 [0.62-0.94] |
| concat (scratch) | 0.762 [0.67-0.84] | 0.787 [0.67-0.89] | 0.761 [0.65-0.85] | 0.795 [0.67-0.90] | 0.807 [0.65-0.95] | 0.625 [0.44-0.79] |
| cross_attention (scratch) | 0.803 [0.72-0.88] | 0.876 [0.80-0.94] | 0.809 [0.71-0.89] | 0.896 [0.82-0.96] | 0.825 [0.68-0.95] | 0.750 [0.58-0.90] |
| cross_attention + alarm-pretrain | 0.856 [0.79-0.92] | 0.916 [0.84-0.97] | 0.887 [0.81-0.95] | 0.934 [0.86-0.99] | 0.877 [0.77-0.96] | 0.875 [0.75-0.98] |

Per-fold mean±SD (5 folds x 3 seeds = 15 values; folds have only 7 subjects, so these are very noisy):

| model | win F1 | win AUC | subj F1 | subj AUC | subj sens | subj spec |
|---|---|---|---|---|---|---|
| gbm_ecg_rr | 0.97±0.02 | 0.98±0.03 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |
| gbm_ppg_pp | 0.90±0.05 | 0.94±0.06 | 0.96±0.05 | 0.94±0.09 | 1.00±0.00 | 0.87±0.16 |
| ppg_only | 0.64±0.14 | 0.65±0.19 | 0.60±0.22 | 0.70±0.24 | 0.77±0.33 | 0.28±0.32 |
| ecg_only | 0.70±0.20 | 0.82±0.14 | 0.69±0.22 | 0.83±0.14 | 0.65±0.26 | 0.81±0.16 |
| concat | 0.75±0.13 | 0.80±0.15 | 0.76±0.16 | 0.80±0.15 | 0.81±0.19 | 0.62±0.23 |
| cross_attention | 0.80±0.08 | 0.88±0.10 | 0.81±0.10 | 0.91±0.11 | 0.83±0.15 | 0.76±0.23 |
| cross_attention+pretrain | 0.85±0.09 | 0.91±0.08 | 0.88±0.13 | 0.93±0.10 | 0.88±0.16 | 0.88±0.15 |

Seed-to-seed SD of the pooled subject AUC: ppg_only 0.034, ecg_only 0.046, concat 0.041, cross_attention 0.036, pretrain 0.031. The GBM baselines are essentially deterministic.

Paired subject-level bootstrap on differences (subject AUC / window F1):

- cross_attention - ppg_only: +0.279 [+0.116, +0.439] / +0.147 [+0.045, +0.248]. Clear.
- pretrain - scratch (cross_attention): +0.038 [-0.013, +0.103] / +0.053 [+0.002, +0.104]. The window-F1 CI barely excludes 0, the subject-AUC CI includes 0. **Suggestive only.**
- pretrain - gbm_ecg_rr: -0.067 [-0.145, -0.016] / -0.113 [-0.182, -0.057]. The RR baseline is significantly better.

Observations:

1. **The simple RR-irregularity gradient-boosting model is the best, and perfect at subject level (subject AUC 1.00 in all folds and seeds).** AF is defined by RR irregularity and these subjects are clean, so this is a strong baseline that no neural model beats here. A deep model's value on this task would have to come from something other than raw accuracy (for example multi-modal robustness). This dataset cannot show that.
2. Among the neural models, `cross_attention` from the alarm-verification checkpoint is best. It improves on scratch cross_attention by roughly 0.04 subject AUC and 0.05 window F1, which is inside the noise at n = 35. Fusion beats single modality: `ppg_only` from scratch is near chance (AUC about 0.61), while `ecg_only` reaches about 0.83.
3. A hand-crafted PPG pulse-interval model reaches AUC 0.94 while the PPG-only CNN stays near 0.6. The CNN is data-starved here (about 28 training subjects, 60 correlated windows each). It is not evidence that PPG is uninformative.

## Distillation: PPG-only student from the ECG+PPG teacher (`src/distill_af.py`)

- Per fold and seed (same folds and seeds as above), two teacher candidates are trained on the fit subjects: scratch cross_attention and alarm-pretrained cross_attention. The one with higher inner-validation AUC is the teacher for that fold. It was scratch 10 times and pretrained 5 times out of 15.
- The student is `ppg_only` (same CNNEncoder, mean-pooled head, same width multiplier as the ppg_only baseline). Its loss is `alpha * BCE(hard, pos-weighted) + (1-alpha) * T^2 * KL(teacher_T || student_T)` with T = 2. The soft targets are teacher logits on the fit windows.
- Compared: alpha = 1.0 (hard labels only, same loop, a clean control), alpha = 0.5 (distilled), and the teacher itself. The focal-loss `ppg_only` baseline from the table above is also relevant. alpha = 0 (pure soft labels) was not run, to save time.

| model | win F1 | win AUC | subj F1 | subj AUC | subj sens | subj spec |
|---|---|---|---|---|---|---|
| teacher (best ECG+PPG per fold) | 0.828 [0.76-0.89] | 0.899 [0.83-0.95] | 0.834 [0.75-0.91] | 0.916 [0.85-0.97] | 0.877 [0.75-0.96] | 0.729 [0.58-0.88] |
| student alpha=0.5 (distilled) | 0.658 [0.58-0.73] | 0.611 [0.49-0.72] | 0.668 [0.56-0.76] | 0.618 [0.47-0.76] | 0.754 [0.61-0.89] | 0.396 [0.21-0.58] |
| PPG-only hard labels, same loop (alpha=1.0) | 0.671 [0.58-0.76] | 0.614 [0.47-0.75] | 0.655 [0.54-0.76] | 0.623 [0.45-0.78] | 0.719 [0.56-0.88] | 0.438 [0.23-0.65] |
| PPG-only hard labels, focal (table above) | 0.655 | 0.602 | 0.634 | 0.612 | 0.754 | 0.271 |

Per-fold mean±SD: teacher subject AUC 0.92±0.13, student alpha=0.5 0.64±0.25, alpha=1.0 0.67±0.26.

Paired subject bootstrap:

- student (0.5) - hard-only (1.0): subject AUC -0.007 [-0.106, +0.095], window F1 -0.014 [-0.054, +0.025].
- student - focal ppg_only: +0.004 [-0.096, +0.118].
- teacher - student: +0.295 [+0.133, +0.465].

**Result: distillation did not help.** The student is indistinguishable from PPG-only trained on hard labels, and it recovers essentially none of the teacher's advantage. A likely reason is that the teacher is trained on the same fit labels, so its soft labels on the fit windows are partly memorised copies of the hard labels. The student's bottleneck is learning PPG rhythm features from about 28 subjects, which neither signal supplies. Untested options, which might help but were not run: teacher outputs on held-out or augmented windows, feature-level distillation, a higher T, an unlabeled larger PPG corpus, and PPG pre-training.

## Limitations

- **Low power.** 35 subjects; each outer fold has 7 test subjects. Subject-level CIs span about ±0.1 AUC, and fold-level SDs are 0.1-0.3. Differences of a few points between neural variants (including pretrain vs scratch) should not be read as established. Only the large gaps (RR-GBM vs neural models, fusion/ECG vs PPG-only CNN, teacher vs student) are supported.
- **Optimistic baseline and ceiling.** The RR-feature result (perfect subject-level separation) shows the label is mostly an RR-irregularity label in clean recordings, so the dataset has little headroom for differentiating models. It is also not a ceiling for noisy ICU conditions.
- **Window correlation.** Windows within a subject are strongly correlated (windows are from a contiguous 20 min recording, using every other 10 s window). Window-level metrics are inflated in apparent sample size. Subject-level metrics and the cluster bootstrap account for this. Window-level CIs shown are also subject-cluster bootstrap.
- **Subject-level threshold and aggregation.** Subject decisions use the mean window probability at 0.5, with no calibration. Neural models were not threshold-tuned, which affects sensitivity and specificity (ppg_only has low specificity).
- **Task difference from alarm verification.** AF vs non-AF in annotated segments is not the same task as true/false alarm verification. The alarm checkpoints transfer as an initialisation only. Gains from pretraining here say nothing directly about alarm verification performance.
- **MIMIC-derived data and possible overlap.** MIMIC PERform AF comes from MIMIC-III waveform data. The PhysioNet/CinC 2015 alarm data and several other corpora in this project (including VTaC-style ICU sets and pretraining sources) are also ICU-monitor recordings, some potentially MIMIC-derived. Patient overlap between the pretraining/alarm data and these 35 subjects was **not checked** and cannot be excluded. If overlapping, the pretrain gain would be optimistic.
- **Selection and tuning.** Early stopping, teacher selection and hyperparameters (T = 2, alpha, 30 epochs, GBM settings) were set once, without a search; no tuning on test folds. Teacher choice is by inner-validation AUC from only about 6 validation subjects, which is noisy.
- **Only 3 seeds**, and 60 windows per subject (of 120 available).

## Reproduce

```
python scripts/download_data.py --dataset mimic_perform_af
python -m src.af_task --variants ecg_only            --seeds 3   # each ~9-12 min on a shared GPU; run variants one at a time
python -m src.af_task --variants ppg_only,concat,cross_attention,cross_attention+pretrain,gbm_ecg_rr,gbm_ppg_pp --seeds 3
python -m src.distill_af --seeds 3 --alphas 1.0,0.5 --budget-min 8   # rerun until it prints the aggregate
```

Outputs: `runs/af_task/af_task_results.json`, `runs/af_task/distill_results.json`, OOF probabilities in `runs/af_task/oof_*.npz`, per-unit distillation caches in `runs/af_task/distill_units/`, and a window cache at `data/processed/mimic_af_windows.npz`.
