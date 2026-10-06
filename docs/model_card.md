# Model card: ICU arrhythmia alarm verifier

**Not a medical device. Research prototype. No prospective validation.**

## Intended use
Rank / suppress bedside-monitor arrhythmia alarms (ECG-triggered) as likely true or likely false, using the
10 s of ECG and pulse signal (PPG, or arterial pressure) that **end at the alarm onset**. Intended to support
research on alarm fatigue. Not for unsupervised suppression of alarms in clinical care.

## Recommended models
1. **Hybrid deep recipe** (`src/train_deep_plus.py`, `docs/deep_recipe.md`): attention CNN-GRU on ECG+PPG with
   augmentation, focal loss, EMA weights, test-time augmentation and a 3-member ensemble. Best single model.
2. **Ensemble (logit mean or LR stacker)** of the hybrid model, six classical algorithms on 266 engineered
   features (XGBoost, Extra Trees, Random Forest, SVM, MLP, logistic regression) and ResNet1D/TCN/InceptionTime
   (`src/ensemble.py`, `src/classical.py`, `src/features_v2.py`): about equal accuracy, more stable.
The originally proposed CNN cross-attention network is **not** recommended.

| Data (threshold 0.5) | Model | Accuracy | F1 | AUC | Sens | Spec |
|---|---|---|---|---|---|---|
| Challenge 2015 (592 records, 5-fold CV x 10 seeds) | hybrid deep | 0.871 | 0.828 | 0.932 | 0.827 | 0.898 |
| Challenge 2015 (5 seeds) | ensemble incl. hybrid (mean) | 0.878 | 0.830 | 0.950 | 0.795 | 0.928 |
| VTaC official test split (5 seeds) | hybrid deep | 0.913 | 0.858 | 0.966 | 0.933 | 0.905 |
| VTaC official test split (5 seeds) | ensemble incl. hybrid (logit mean) | 0.912 | 0.844 | 0.957 | 0.842 | 0.940 |

See `docs/LEADERBOARD.md` for confidence intervals and all comparison models. The hybrid recipe was selected
on Challenge 2015 inner-validation scores (small possible optimistic bias, see `docs/deep_recipe.md`).

## Data
PhysioNet/CinC 2015 (US ICUs, five alarm types) and VTaC v1.1 (VT alarms, three US hospitals,
three monitor makers). Details, cohort flow and exclusions: `docs/datasheet.md`.

## Evaluation
Record/patient-wise splits; hyper-parameters, stacking weights and thresholds fitted on inner validation sets
only; corrected resampled t-test and paired record bootstrap with Holm correction: `docs/evaluation_protocol.md`.

## Known limitations and risks
* **Sensitivity is not near 100 %.** At the default threshold ~80-83 % of true alarms are retained. Validation-chosen
  thresholds aimed at 95-100 % sensitivity realised ~92-97 % on test folds. Missing a true alarm is the
  safety-critical error; ventricular tachycardia is where misses concentrate (`docs/error_taxonomy.md`).
* **Asymmetric cross-dataset transfer** (Challenge 2015 -> VTaC F1 ~0.55-0.60; VTaC -> Challenge 2015 VT alarms
  F1 up to 0.77). Expect a drop at a new site or
  monitor type.
* Engineered features encode the alarm definitions (gap since last beat, rate cut-offs); on Challenge 2015 the
  alarm type is an input. This is domain knowledge, not learned physiology.
* The deep networks do not use ECG-to-pulse timing (counterfactual tests), so they should not be described as
  verifying "pulse after beat" agreement.
* Small data (592 / 724 records for Challenge 2015): fold-to-fold SD of F1 ~0.05; many differences between top
  models are inside noise.
* US ICU data only; demographic and skin-tone effects on optical PPG not assessed; VTaC device/site ids not
  released (lead-set signature used as an unverified device proxy).
* Robustness: AUC degrades with heavy noise on both signals (see `runs/*/stress/`); zeroing the pulse sensor
  at inference (no retraining) costs about 0.10-0.12 AUC for the CNN fusion models and 0.27 for ResNet1D.

## Safety layer (optional)
`src/safety.py`: temperature calibration, a risk-controlled suppression threshold with an exact Clopper-Pearson
bound on the miss rate (abstains when validation positives are too few to certify the target), a deferral band
and per-100-alarm utility numbers. Alarm volume per bed-day is a user-supplied assumption, not data.
