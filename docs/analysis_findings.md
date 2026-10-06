# Analysis findings (consolidated)

Numbers come from the `runs/` outputs named in each section. All models were trained with the leak-free
protocol in `evaluation_protocol.md`; seeds-based uncertainty is reported where available.

## 1. Do fusion models use ECG-to-pulse timing? (counterfactual tests)
`runs/challenge2015_ppg/counterfactual/` (seed-0 checkpoints, 5 test folds). One modality is perturbed while
the other is intact.

* Replacing the PPG with another patient's **same-label** PPG (real waveform, unrelated timing) changes AUC by
  only 0.00-0.03 for every model (cross-attention 0.002, concat 0.015, ResNet1D 0.021, TCN 0.024).
* Circular shifts of up to 2 s change AUC by <= 0.005 for the CNN fusion models and by 0.03-0.06 for ResNet1D,
  TCN, InceptionTime and BiGRU (slightly shift-sensitive; none depends on a precise lag).
* Replacing the pulse with an **opposite-label** pulse collapses AUC (cross-attention 0.842 -> 0.535), and noise
  costs 0.07-0.21: the models use the pulse's *content* (morphology/quality) as evidence.
* Conclusion: the models do not verify "beat followed by pulse"; they use each signal as a patient-agnostic
  cue. The PTT justification in the original proposal is not supported.

## 2. Does attention track measured pulse transit time?
`runs/challenge2015_ppg/attention_ptt/`. Attention-implied lag vs measured PTT (R-peak to PPG foot):
window-level Spearman rho = -0.039 (permutation p = 0.39), beat-level rho = 0.017, attention mass in the
physiological lag range 5.3 % vs 5.6 % for uniform attention. **No relationship.**

## 3. Explanation faithfulness (deletion / insertion, 0.5 s blocks, 40 windows per fold)
`runs/challenge2015_ppg/faithfulness/`. Paired difference vs a random block ordering (deletion: lower is more
faithful; insertion: higher is better):

| Method | PPG deletion | PPG insertion | ECG deletion | ECG insertion |
|---|---|---|---|---|
| Occlusion | -0.041 | +0.041 | -0.036 | +0.034 |
| Integrated gradients | -0.016 | +0.010 | -0.002 | +0.003 |
| Cross-attention weights | -0.007 | +0.011 | n/a | n/a |

Occlusion is the most faithful; attention weights are barely better than random ordering and should not be
presented as an explanation.

## 4. Label reliability (VTaC v1.1 per-annotator votes)
`runs/vtac_official/label_reliability/`. 7.7 % of events (351 / 4,542) have split annotator votes.

* Accuracy on unanimous vs split-vote events: cross-attention 0.854 vs 0.704, ResNet1D 0.863 vs 0.689.
* Model uncertainty (entropy) tracks disagreement only weakly (Spearman 0.13-0.18; AUC for detecting split-vote
  events 0.66-0.71).
* **Training on annotator vote fractions (soft labels)** vs consensus labels, same seeds, 5 seeds:
  cross-attention dF1 = +0.038 (SD 0.012), dAUC = +0.015; ResNet1D dF1 = +0.015 (SD 0.036), inconclusive.

## 5. Time-to-verdict (VTaC, window of 10 s ending t seconds after the alarm onset)
`runs/vtac_ttv_post{5,10,30}/` (3 seeds, official split; F1 / AUC):

| t | ECG-only | PPG-only | cross-attention | ResNet1D |
|---|---|---|---|---|
| 0 s (decision time) | 0.694 / 0.893 | 0.550 / 0.742 | 0.740 / 0.911 | 0.733 / 0.909 |
| +5 s | 0.667 / 0.861 | 0.542 / 0.741 | 0.713 / 0.892 | 0.719 / 0.901 |
| +10 s | 0.546 / 0.743 | 0.454 / 0.587 | 0.515 / 0.724 | 0.580 / 0.761 |
| +30 s | 0.461 / 0.652 | 0.445 / 0.535 | 0.451 / 0.681 | 0.542 / 0.718 |

**Caveat:** this slides a fixed 10 s window past the trigger, so the event that raised the alarm leaves the
window; it measures "how stale can the input be", not "does waiting help". A window that keeps the
pre-alarm 10 s and appends the post-alarm seconds was not run. There is no evidence here that waiting helps.

## 6. Cross-dataset transfer (`runs/cross_dataset/`)
Trained on Challenge 2015 (recovered cohort), tested on the VTaC test split: F1 0.56-0.60 / AUC 0.73-0.78
(ECG-only 0.603 / 0.776; PPG-only 0.432 / 0.541, near chance). Compare in-distribution VTaC F1 0.71-0.78.
The reverse direction (VTaC -> Challenge 2015 VT alarms) is in `runs/cross_dataset/vtac_to_cinc_vt/results.json`.

## 7. Label efficiency (`runs/labeleff/`, `runs/labeleff_single/`)
Fine-tuning from self-supervised encoders gives modest paired gains over scratch (+0.01 to +0.04 F1; ECG-only
+0.03 to +0.04); frozen encoders are worse than scratch for fusion. Only 1 of 72 paired contrasts has a
Nadeau-Bengio p < 0.05, so the effect is suggestive, not established. Single-signal models use width 1.0 for
both arms (width-matched models cannot load width-1.0 pretrained weights).

## 8. Modality subsets (`runs/challenge2015_recovered/modality_subsets/`)
On the 191 windows with all four sensors, AUC rises from 0.70-0.73 (one sensor) to 0.82 (all four); Shapley
shares of AUC above chance: ECG 0.102, second ECG lead 0.089, PPG 0.073, ABP 0.058. Confidence intervals are
about +-0.07 AUC (small n).

## 9. Robustness (`runs/challenge2015_ppg/stress/`)
ResNet1D is the most noise-robust (0.907 -> 0.844 at 0 dB on both signals); cross-attention falls
0.856 -> 0.696 and concat 0.833 -> 0.720. Zeroing the PPG at inference costs cross-attention 0.12 AUC, concat
0.10, ResNet1D 0.27. Constant PPG delays of 0-400 ms barely change fusion models (consistent with section 1).
