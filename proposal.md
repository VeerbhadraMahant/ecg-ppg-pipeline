# Project Proposal Summary

## Aim

Build a model that reads ECG (electrical activity) and PPG (peripheral pulse) at the same time and decides whether an arrhythmia alarm is real. The question being tested: does letting the model compare the two signals against each other via attention work better than giving it only one signal?

## Problem statement

ICU monitors raise a large volume of arrhythmia alarms from ECG, and a large share are false — caused by movement, electrode slippage, or muscle artifact rather than a genuine cardiac event. Verifying an alarm from ECG alone doesn't work, because the same artifacts that trigger a false alarm also corrupt the signal being used to check it. PPG offers an independent channel: it reflects whether a heartbeat actually produced mechanical blood flow, and it fails under different conditions (perfusion loss, probe movement) than ECG does. This independence is what makes fusion meaningful rather than redundant.

If alarm windows were labeled by reading the ECG alone, an ECG-only model could never really be beaten, since the answer would already be baked into the ECG. Alarm verification avoids that trap: the true/false label depends on ECG and PPG agreeing or disagreeing, not on the ECG by itself.

## Objectives

1. Build the fusion model — dual 1D-CNN encoders + transformer cross-attention layer + classification head.
2. Prove the attention is doing the work — via the parameter-matched baseline ladder (ECG-only, PPG-only, naive concatenation, cross-attention).
3. Show a measurable improvement — sensitivity and F1, with record-wise cross-validation and error bars across folds/seeds, not a single run.
4. Explain the decisions — extract and visualize attention weights, check whether they line up with real ECG-to-pulse timing physiology.
5. Keep the whole pipeline free — public datasets only, free-tier Colab/Kaggle compute only.

## Methodology (condensed)

- **Preprocessing**: common sample rate, per-signal bandpass filtering, per-window z-normalization, fixed-length windows aligned so ECG and PPG cover the same moment, signal-quality scoring per window, record-wise (not window-wise) train/test split.
- **Model**: two non-weight-shared 1D-CNN encoders (each outputs a feature *sequence*, not a pooled vector) → transformer cross-attention layer (ECG queries, PPG keys/values) with positional encodings → shallow classification head.
- **Comparison ladder**: ECG-only, PPG-only, concatenation fusion, cross-attention fusion — all parameter-matched.
- **Evaluation**: record-wise cross-validation with error bars across folds/seeds; sensitivity/specificity/F1 per alarm class; performance vs. signal-quality analysis; model size/speed reported alongside accuracy.
- **Compute**: entirely within free-tier Google Colab / Kaggle GPU time.

## How this differs from existing work

The nearest existing work learns a shared ECG/PPG representation for atrial fibrillation detection using contrastive learning on large private hospital datasets. This project differs by: using attention to align the two signals directly instead of contrastive learning; targeting alarm verification (label depends on both signals) instead of a task where ECG alone gives away the answer; and running entirely on open data and free compute.

## Outcome / deliverables

- Trained fusion model that calls alarms true/false and degrades gracefully under signal noise.
- Comparison table across all four ladder variants under the same test conditions — the core result, reported regardless of which model wins.
- Analysis of when fusion helps vs. doesn't, based on signal quality.
- Attention visualizations showing what the model relied on, checked for physiological plausibility.
- External validation on MIMIC PERform AF.
- Fully reproducible code, runnable from public data with no paid resources.