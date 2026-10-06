# Datasheet

All numbers below were produced by `src/data/preprocess.py`,
`scripts/build_vtac_windows.py` and the audit files they write
(`data/processed/cohort_audit.csv`, `cohort_flow.json`).

## 1. PhysioNet/CinC Challenge 2015 (training set)

* **Source / licence**: PhysioNet, open access. 750 public ICU records; the 500-record test set was never released, so evaluation is record-wise cross-validation on these 750.
* **Content**: ECG (several leads), arterial blood pressure (ABP) and/or photoplethysmogram (PPG, `PLETH`); one expert true/false label for one of five alarm types per record. Pre-resampled to 250 Hz. The alarm fires at the 5-minute mark (300 s) in every record.
* **Window**: the 10 s that **end at the alarm onset** (no post-alarm samples).
* **Cohort flow** (750 records):

| Step | Records |
|---|---|
| Records in training set | 750 |
| with PPG (`PLETH`) | 627 |
| with ABP | 343 |
| with a second ECG lead | 750 |
| Original **PPG cohort** (usable PPG window, quality ≥ 0.05) | **592** (37.7 % true alarms) |
| **Recovered cohort** (PPG, else ABP substituted as the pulse channel) | **724** (39.6 % true) |
| of which recovered through ABP | 132 |
| Excluded from the recovered cohort (window too short / near-total flat-line or clipping) | 26 |

* **Alarm types**: Asystole, Extreme Bradycardia, Extreme Tachycardia, Ventricular Tachycardia, Ventricular Flutter/Fibrillation. Asystole (14 true of 93) and V-Flutter/Fib (5 true of 45) have too few true alarms for per-type conclusions.
* **Known caveats**: one window per record, so record-wise and window-wise splits coincide; labels are for the alarm as raised by the bedside algorithm in a single institution-mix (US).

## 2. VTaC v1.1 (ventricular tachycardia alarms)

* **Source / licence**: PhysioNet, open access, CC BY-SA 4.0 (Lehman et al., NeurIPS 2023 Datasets & Benchmarks). v1.1 (Sept 2026) adds `event_label_per_annotator.csv`.
* **Content**: 5,037 annotated VT alarm events from ICUs of three US hospitals and three monitor manufacturers; 6-minute records at 250 Hz with the alarm onset at 300 s; ≥ 2 ECG leads plus PLETH and/or ABP. At least two independent expert annotators per event; consensus label in `event_labels.csv`.
* **Downloaded subset**: only `[onset − 60 s, onset + 30 s]` of each event (`scripts/download_vtac.py`, ~1 GB instead of 18 GB). **Decision-time windowing**: the headline task window is the 10 s ending exactly at the onset; the 30 s after the onset are read only by the time-to-verdict analysis (`vtac_windows_post{5,10,30}s.npz`), never by the headline models.
* **Cohort flow** (5,037 events):

| Step | Events |
|---|---|
| Annotated events | 5,037 |
| No pulse channel (neither PPG nor ABP) | 292 |
| Quality < 0.05 | 203 |
| **Kept** | **4,542** (28.0 % true alarms, 2,138 patients) |
| ABP used as pulse (no PPG) | 62 |
| Official split (patient-disjoint) | train 3,662 / val 453 / test 427 |

* **Hospital / device identifiers are not released.** Cross-device analyses use the **lead-set signature** (the set of ECG lead names in a record) as a *proxy* for monitor manufacturer. Two signatures dominate: `II|aVR|V` (2,737 events, 60 %) and `I|II|III|V` (1,191 events, 26 %); the remaining 14 % are 35 small groups. This proxy is unverified.
* **Label reliability**: per-annotator votes are released in v1.1 (`annotator_frac_true`, `n_annotators` are stored in the processed file) and are used for the label-reliability analysis.

## 3. Pretraining / auxiliary corpora

| Corpus | Status |
|---|---|
| VTaC pre-alarm segments (train + val patients only, 60 s before onset) | Used for self-supervised pretraining. Test patients are excluded. |
| VTaC `waveforms/` (unlabeled) | Contains 123 patients not in the labelled set; small. Not used yet. |
| MIT-BIH Noise Stress Test (bw, em, ma) | Used for synthetic noise injection (`data/raw/nstdb`). |
| MIMIC PERform AF | Used for the fine-tuned AF screening task, subject-wise. |
| PulseDB, VitalDB, MIMIC-III waveform | **Not used** — access / size; pending credentials. |

## 4. Overlap and licence checks still open

* Patient overlap between CinC 2015, VTaC and MIMIC-derived corpora cannot be checked from released identifiers. VTaC and MIMIC-derived data come from different institutions in at least part of their ranges, but this is unverified.
* Each dataset's licence permits research use; redistribution of derived windows follows the stricter of CC BY-SA 4.0 (VTaC) and the PhysioNet data-use terms.
