# Error taxonomy and case studies (CinC 2015, updates.md 6.12)

Produced by `python -m src.analysis.error_taxonomy` (CPU only, reads the saved
out-of-fold predictions; nothing existing is modified). Outputs live in
`runs/challenge2015_ppg/error_taxonomy/`:
`per_record_errors.csv`, `category_counts.csv`, `breakdowns.csv`,
`subgroup_tests.csv`, `annotation_tag_rates.csv`, `summary.json`,
`features.csv`, `figures/` (19 PNGs).

## 1. Method

* **Data**: 592 alarm windows (223 true, 369 false), one window per record, so
  record = window. Alarm types: VT 270, Tachycardia 111, Asystole 93,
  Bradycardia 73, VFlutter/Fib 45.
* **Consensus error rate** of a record under a variant: fraction of the 10 seeds
  in which its single out-of-fold prediction at threshold 0.5 is wrong
  (values in steps of 0.1).
* **Categories** (per variant): *persistent FN* = true alarm, rate >= 0.8;
  *persistent FP* = false alarm, rate >= 0.8; *unstable* = 0.2 < rate < 0.8
  (split by label); *stable-correct* = rate <= 0.2. The 0.8/0.2 cut-offs are a
  convention, not tuned.
* **Subgroups**: alarm type; signal-quality tercile (`quality`, `ecg_quality`,
  `pulse_quality`; these are saturated near 0.99 so terciles are by value with
  ties kept together, plus a binary degraded flag `quality < 0.9`, n = 136);
  detector disagreement (`hr_abs_diff >= 15 bpm`, `ecg_beats_with_pulse_frac <
  0.5`, `ppg_acf_peak < 0.3`, all from `extract_features`); heart-rate regime
  (ECG-detector median-RR HR: <50, 50-100, 100-140, >=140, or <3 beats);
  pulse source.
* **Tests**: for each (variant, error type) the persistent-error rate of each
  subgroup level vs the rest, among the relevant class (FN among true alarms,
  FP among false alarms), Fisher exact; **Holm** over the 29 tests of that
  (variant, error type) family. A logistic regression (L2, C = 0.1, standardised
  hand-crafted features) with 10x repeated stratified 5-fold record-level CV
  estimates how predictable persistent errors are from signal descriptors.
* **Annotations** are rule-based heuristics on the hand-crafted features
  (HF-power above the 90th percentile or poor beat-template match, quality
  indices < 0.9, ppg ACF < 0.3, HR difference >= 15 bpm, pulse absent but ECG
  regular). They describe signal properties; they are **not** causal
  explanations (see 5).

## 2. Counts per category (n records; % of the class)

| Category | gbm_features (F1 0.798) | resnet1d | cross_attention (F1 0.699) | ecg_only | ppg_only |
|---|---|---|---|---|---|
| persistent FN (of 223 true) | 29 (13.0%) | 27 (12.1%) | 32 (14.3%) | 39 (17.5%) | 22 (9.9%) |
| unstable true alarm | 34 (15.2%) | 45 (20.2%) | 52 (23.3%) | 50 (22.4%) | 44 (19.7%) |
| persistent FP (of 369 false) | 20 (5.4%) | 11 (3.0%) | 38 (10.3%) | 61 (16.5%) | 156 (42.3%) |
| unstable false alarm | 36 (9.8%) | 72 (19.5%) | 77 (20.9%) | 88 (23.9%) | 58 (15.7%) |
| stable-correct (of 592) | 473 (79.9%) | 437 (73.8%) | 393 (66.4%) | 354 (59.8%) | 312 (52.7%) |

Observations:

* The proposed model has about as many persistent FNs as the best model
  (32 vs 29) but twice as many persistent FPs (38 vs 20) and roughly 2x the
  unstable records (129 vs 70). Many of its mean OOF probabilities lie in
  0.4-0.7 (see `shared_vs_unique_errors.png`, middle), so a fixed 0.5 threshold
  flips from seed to seed; much of its deficit is instability/calibration
  rather than systematic misses. This is a description, not a tested claim.
* Errors are largely model-specific: the persistent-FN sets of gbm_features and
  cross_attention overlap in 14 records (Jaccard 0.30), the persistent-FP sets
  in 7 (Jaccard 0.14). 21 records are persistent errors in both.
* Records wrong (rate >= 0.5) under all 5 variants: 17 (8 FN, 9 FP), 6 of them
  persistent (>= 0.8) under all five; 13 of the 17 are VT, Asystole or
  Bradycardia (VT 13, Asystole 3, Bradycardia 1). Records where only
  cross_attention is wrong (others < 0.5): 6, all FN (Asystole 1, Bradycardia 2,
  Tachycardia 2, VT 1); only gbm_features wrong: 10 (8 FN, 2 FP). The proposed model is wrong
  on 99 records where at least one of ecg_only / ppg_only is right, i.e. the
  fusion does not reliably recover the better modality.

## 3. Where the errors concentrate

### Persistent false negatives (missed true alarms)

* **Ventricular tachycardia dominates**: gbm 18/29 (62%), cross_attention
  14/32 (44%) of persistent FNs, although VT is only 59/223 (26%) of true
  alarms. VT true alarms are missed persistently 30.5% (gbm) and 23.7%
  (x-attn) of the time. gbm: VT vs rest 30.5% vs 6.7%, Holm-adjusted p =
  4e-4 (the only Holm-significant positive association for gbm among alarm
  types).
* **Tachycardia true alarms are rarely persistently missed** (gbm 2.8%, resnet
  1.9%, x-attn 5.7%; Holm p = 4e-4, 1e-4, 0.013 respectively). Hence
  `HR >= 140` is also protective for FN (gbm 3.9% vs 20.7%, Holm p = 0.006); the
  two are collinear (most Tachycardia alarms are HR >= 140).
* **Asystole and Bradycardia** true alarms: persistent FN 21-29% (Asystole,
  n = 14) and 8-18% (Bradycardia, n = 39); cross_attention misses 4/14 and 7/39.
  Small n; only ecg_only/Asystole survives Holm (8/14 = 57%, p_Holm = 0.016).
* **Pulse not matching the ECG** (`ECG_beats_with_pulse < 0.5`, n = 41):
  persistent FN 31.7% vs 7.7% for resnet1d (Holm p = 0.004) and 41% vs 12% for
  ecg_only (p = 0.001). For cross_attention the direction agrees (26.8% vs
  11.5%) but is not significant after Holm.
* Signal quality is **not** a clear driver of FN: for gbm, persistent FN rate is
  higher in the *highest* quality tercile (19.8%) than the lowest (11.7%)
  (unadjusted only); degraded vs clean windows 12.3% vs 13.3%. Most missed true
  alarms are therefore not low-quality windows.
* Heuristic tags on persistent FNs (gbm / x-attn): PPG artefact or non-periodic
  pulse 45% / 44%; detectors disagree 45% / 34%; ECG artefact 28% / 28%; no
  flag raised 31% / 31%; pulse-absent-ECG-regular 3% / 6%. See caveat in 5:
  the same tags occur in 25-45% of *correctly* classified records
  (`annotation_tag_rates.csv`), so they are weakly discriminative.

### Persistent false positives (false alarms not suppressed)

* **VT false alarms** give most of them (gbm 13/20, x-attn 21/38), proportional
  to their share of false alarms (211/369 = 57%), plus **Asystole** (6/20 and
  7/38; 7.6-8.9% of 79 asystole false alarms). Because VT is the most frequent
  false-alarm class, per-class rates are similar (6-10%); no alarm-type FP
  effect survives Holm for gbm or cross_attention. (Tachycardia false alarms: n = 5
  only; 3 persistent FP for x-attn, p_Holm = 0.24, uninformative.)
* **Non-periodic pulse** (`ppg_acf < 0.3`, n = 89 false alarms): persistent FP
  12.4% vs 3.2% (gbm, p_raw = 0.002, p_Holm = 0.06), 19.1% vs 7.5% (x-attn,
  p_Holm = 0.12), 9.0% vs 1.1% (resnet1d, p_Holm = 0.022). Consistent direction in
  three models, formally significant only for resnet1d.
* **HR >= 140 false alarms** (gbm: 7/38 = 18.4% vs 3.9%, p_Holm = 0.059). Not significant after Holm.
* **ppg_only** shows the large, Holm-significant quality effect (persistent FP
  79% in the lowest pulse-quality tercile vs 28%, p_Holm < 1e-17), i.e. a
  PPG-only model fails to suppress false alarms caused by corrupted pulse
  waveforms. The fused models do not show this dependence.
* ecg_only: false alarms with detector agreement (HR diff < 15 bpm) are
  persistently missed *more* (24% vs 7%, p_Holm = 5e-4) - when the ECG looks
  regular and the PPG agrees, an ECG-only model cannot reject the alarm.

### Predictability from signal descriptors (record-level CV AUC, mean +/- sd over 50 folds)

| Variant | FN among true alarms (n_err/223) | FP among false alarms (n_err/369) |
|---|---|---|
| gbm_features | 0.74 +/- 0.09 (29) | 0.82 +/- 0.08 (20) |
| resnet1d | 0.80 +/- 0.09 (27) | 0.72 +/- 0.16 (11) |
| cross_attention | 0.80 +/- 0.08 (32) | 0.81 +/- 0.07 (38) |
| ecg_only | 0.80 +/- 0.10 (39) | 0.78 +/- 0.06 (61) |
| ppg_only | 0.80 +/- 0.08 (22) | 0.93 +/- 0.02 (156) |

Persistent errors are moderately predictable (AUC ~0.7-0.8) from the
hand-crafted descriptors, so they are partly systematic, not purely random;
the sd is large for the small error sets. For gbm_features this is partly
circular since the model is trained on the same descriptors.

## 4. Representative records (figures in `figures/`)

Case figures show ECG with detected R peaks, PPG with peaks and feet, label,
alarm type, mean OOF probability and error rate for all five variants, and the
heuristic annotation. Four cases per group:

| Group | Records (annotation, heuristic) |
|---|---|
| gbm persistent FN | b832s Bradycardia (ECG 40 vs PPG 124 bpm), v534s VT (no flag), t717l Tachycardia (ECG+PPG artefact), f544s VFlutter/Fib (artefact tags) |
| x-attn persistent FN | v597l VT (PPG HR 119 vs ECG 78), a443l Asystole (ECG artefact, low pulse quality), b839l Bradycardia (ECG 32 vs PPG 77), t678s Tachycardia (no flag) |
| gbm persistent FP | v119l VT (ECG 146 vs PPG 77), a315l Asystole (low pulse quality), b487l Bradycardia (ECG 45 vs PPG 97), v162s VT (ECG 179 vs PPG 100) |
| x-attn persistent FP | v160s VT, t504s Tachycardia (ECG artefact), a105l Asystole (ECG 41 vs PPG 134), f281l VFlutter/Fib (ECG artefact) |

Other figures: `shared_vs_unique_errors.png` (counts by alarm type for
errors shared by all variants, unique to cross_attention, unique to gbm; per-record
probability and error-rate scatter), `shared_vs_unique_examples.png`
(traces of three shared and three cross_attention-only errors),
`taxonomy_overview.png`. Cases are chosen by highest error rate, diversified
over alarm types and over groups; they are illustrative, not random samples.
Records v159l, v206s, v221l, b313l are persistent FN in both gbm and x-attn
(first ranks by error rate) and are worth manual review.

## 5. Clinical interpretation, safety, limitations

* **Safety implication of FNs.** A persistent FN is a true life-threatening
  arrhythmia the model would suppress in every re-draw of the CV split. At
  threshold 0.5, 13-14% of true alarms fall in this category for the best and
  proposed models, and a further 15-23% flip between seeds. Each missed true
  alarm costs 5x a false alarm in the Challenge score. The shares in VT (the
  persistent-FN rate in true VT is about 24-31%), Asystole and Bradycardia mean
  that the most dangerous classes are the least reliable ones in our data; the
  small n (Asystole 14, VFlutter/Fib 5 true alarms) means rates here have very
  wide intervals. A deployment would need the higher-sensitivity operating
  points (see `runs/.../safety/`) rather than 0.5, and a fail-safe that never
  suppresses when the signals are inconclusive. This analysis does not show
  that such a rule would work.
* **Plausible mechanisms (hypotheses only)**: (i) VT episodes with a regular,
  moderately fast rhythm and a pulse that continues (haemodynamically tolerated
  VT) look like normal-rhythm windows in a model that mostly checks
  ECG-PPG coherence; (ii) pulse-poor or uncoupled true alarms (e.g. low-output
  rhythms) are missed when PPG is treated as evidence against the alarm; (iii) false
  alarms with corrupted PPG or with ECG artefact that mimics VT/asystole
  keep alarming. We did not verify any of these against annotations of the
  underlying rhythm.
* **Limitations.**
  1. Small n: 223 true / 369 false alarms; 14 true Asystole, 5 true VFlutter/Fib,
     5 false Tachycardia. Subgroup percentages have wide CIs; Holm over 29 tests
     per family gives low power, and only the strongest effects survive. Results
     with p_Holm > 0.05 are directions, not findings.
  2. Seeds change fold assignment only, so the 10 predictions per record are not
     independent samples of noise; "persistent" means robust to training-set
     composition, not to measurement noise. Records are single windows, and
     patient identity is only partly controlled (see the evaluation protocol).
  3. Error rates are at a fixed 0.5 threshold, which favours well-calibrated
     models; the neural models' probabilities are compressed near 0.5, which
     inflates "unstable" for them. Operating-point-specific analyses are in the
     safety results.
  4. The annotation heuristics are crude (R-peak detector is a simple
     Pan-Tompkins variant that can lock onto negative QRS deflections, e.g. in
     v597l; thresholds are percentile-based) and are not validated against
     clinical review. Tag rates in correctly classified records are similar
     (PPG artefact tag 36-37%, detector disagreement 40-44%), so a tag does not by itself explain an error; for 31% of persistent FNs (both models)
     no tag fires at all, and 26% of x-attn persistent FPs.
  5. The signal-quality indices are saturated near 0.99 (only 136/592 windows
     below 0.9), so "terciles" mostly distinguish a few degraded windows.
  6. Pulse source: `pulse_is_abp` is False for all 592 records (PPG is the
     pulse channel everywhere), so the pulse-source breakdown is degenerate. ABP
     recorded in addition (206 records) was used as a context flag and
     shows no Holm-significant effect for the best or proposed models.
  7. The gbm_features analysis is circular with the descriptors used in the
     subgroup definitions (hr_abs_diff, ppg_acf, etc. are model inputs).
  8. Analysis is post hoc and exploratory (the 0.8/0.2 cut-offs and the
     15 bpm / 0.3 / 0.5 thresholds were fixed before looking at p-values but not
     pre-registered).
