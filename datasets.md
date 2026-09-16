# Datasets

## Primary — PhysioNet/CinC Challenge 2015

Role: training and validation for all four ladder variants.

- Landing page: https://www.physionet.org/content/challenge-2015/1.0.0/
- Download: https://www.physionet.org/content/challenge-2015/1.0.0/training.zip (322.9 MB)
- Terminal mirror: `wget -r -N -c -np https://physionet.org/files/challenge-2015/1.0.0/`

750 public ICU records, real-time subset, WFDB format. Two ECG leads plus one or more pulsatile waveforms (PPG and/or arterial blood pressure) per record. Expert true/false label for one of five life-threatening alarm types per record:

| Alarm type | Coding string | Definition |
|---|---|---|
| Asystole | `Asystole` | No QRS for at least 4 seconds |
| Extreme Bradycardia | `Bradycardia` | HR < 40 bpm for 5 consecutive beats |
| Extreme Tachycardia | `Tachycardia` | HR > 140 bpm for 17 consecutive beats |
| Ventricular Tachycardia | `Ventricular_Tachycardia` | 5+ ventricular beats, HR > 100 bpm |
| Ventricular Flutter/Fibrillation | `Ventricular_Flutter_Fib` | Fibrillatory/flutter/oscillatory waveform for 4+ seconds |

Notes:
- Only the 750-record training set is public. The 500-record test set is held privately by PhysioNet for competition scoring and was never released — record-wise cross-validation on the training set is the correct approach here, not a held-out competition split.
- Not every record necessarily contains PPG specifically (some carry ABP instead) — filter to records with a usable PPG channel before building ECG+PPG pairs.
- Labels are separate from the signal files; confirm the exact annotation file structure after unzipping.
- Read with the `wfdb` Python package (`pip install wfdb`), not manual WFDB parsing.
- Signals are pre-resampled to 250 Hz, 12-bit, with FIR bandpass [0.05-40Hz] and mains notch filtering already applied.

## External test only — MIMIC PERform AF

Role: generalization check after training on PhysioNet 2015. Never train on this.

- Landing page: https://zenodo.org/records/15906524
- AF subjects (CSV): https://zenodo.org/records/15906524/files/mimic_perform_af_csv.zip?download=1 (27.2 MB, 19 subjects)
- Non-AF subjects (CSV): https://zenodo.org/records/15906524/files/mimic_perform_non_af_csv.zip?download=1 (23.2 MB, 16 subjects)

Download both zips — they're two halves of one 35-subject set (19 AF + 16 non-AF), and the non-AF file is the negative class needed for binary evaluation.

Notes:
- CSV format chosen over Matlab/WFDB for direct `pandas.read_csv` loading in a Python/Colab pipeline.
- This dataset does not include ABP — only ECG, PPG, and in some subsets respiration. Don't assume ABP is present when adapting the PhysioNet-2015 loading code.
- After unzipping, concatenate both classes into one dataframe/directory with a label column (AF=1, non-AF=0) before preprocessing.
- This is a Zenodo record maintained by Peter Charlton et al.; cite Charlton PH et al., "Detecting beats in the photoplethysmogram: benchmarking open-source algorithms," Physiological Measurement 2022, DOI: 10.1088/1361-6579/ac826d.