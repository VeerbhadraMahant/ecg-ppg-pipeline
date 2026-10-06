# Foundation-model track (updates.md 5.3)

## What was obtained

| Model | Status | Source | Licence |
|---|---|---|---|
| PaPaGei-S (ResNet1D-MoE, 512-d) and PaPaGei-P (ResNet1D, 512-d) | **Obtained and used** | Code: github.com/Nokia-Bell-Labs/papagei-foundation-model (only `models/resnet.py` + README/LICENSE vendored). Weights: Zenodo 13983110 (`papagei_s.pt` 23 MB, `papagei_p.pt` 20 MB), plain open download, no credentials. Stored in `data/external/papagei/`. | Code: BSD-3-Clause-Clear. Weights: the Zenodo record has **no licence field** (access: open); it only lists the pretraining data licences (VitalDB CC-BY 4.0, MIMIC-III waveform ODbL, MESA/NSRR under its data-use terms). Treat weight reuse as "open, licence unstated"; cite Pillai et al., ICLR 2025 (arXiv 2410.20542). |
| ECG-FM (wav2vec2+CMSC+RLM, 12-lead) | Weights downloaded (`data/external/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt`, 1.1 GB, HF `wanglab/ecg-fm`, MIT, not gated) but **not evaluated** | see blockers | MIT |

### ECG-FM blockers
1. Loading requires `fairseq-signals`, which compiles C++/Cython extensions; this machine has no C/C++ compiler (no cl.exe/gcc), so it cannot be installed with pip in minutes (a pip dry-run would also pull transformers 5 / a newer huggingface_hub and disturb the shared environment).
2. The model expects **12-lead, 500 Hz, ~5 s** input. We have one ECG lead (lead unspecified) at 250 Hz; feeding one lead into a 12-lead model (other leads zeroed, upsampled) is far outside its training distribution, so any result would be hard to interpret.
3. A pure-PyTorch re-implementation of the wav2vec2 trunk is plausible (the checkpoint is a plain 4-layer conv extractor + 12-layer 768-d transformer, 215 tensors) but cannot be numerically validated against fairseq here, so it was not attempted. Needed: a machine with MSVC/gcc (or Linux) to `pip install fairseq-signals`, plus a multi-lead cohort (e.g. the Challenge 2015 second ECG lead `ecg2` is available only for some records).

## Input conventions (PaPaGei)
Model input: 10 s, **125 Hz** (1250 samples), per-window z-score, shape (B,1,1250). Our windows are 10 s at 250 Hz, already band-passed 0.5-8 Hz and z-scored. `src/models/foundation.py` decimates 250 -> 125 Hz with the same Kaiser FIR as `scipy.signal.resample_poly(x,1,2)` (reimplemented in torch; unit-tested to agree to <5e-3), then re-z-scores. No cropping/padding is needed. Output: final feature map (B, 3, 512) as a token sequence (the trunk downsamples 1250 -> 3), or the 512-d embedding (`outputs[0]` of the upstream API, or the mean-pooled trunk feature). Note that the "PPG" channel in Challenge 2015 / VTaC is sometimes an ABP trace (`pulse_is_abp`), which PaPaGei never saw.

## Variants (`src/train_foundation.py`, results in `runs/foundation/<run>/`)
- `pg_linear_*`: frozen embedding -> standardise -> logistic regression (balanced class weights, C picked on inner-val F1). PPG alone.
- `pg_frozen_ppg_head`: frozen tokens -> repo MLP head. PPG alone.
- `pg_frozen_concat` / `pg_frozen_cross_attention`: scratch ECG CNN + frozen PaPaGei-S tokens -> repo ConcatFusion / CrossAttentionFusion + head.
- `pg_ft_*`: last residual block + final BN of PaPaGei-S fine-tuned (lr 1e-4; rest 1e-3). **LoRA was not used**: `peft` imports `transformers`, which is not installed, and it was not worth altering the shared environment; last-block fine-tuning is the fallback the task specified.
Protocol is identical to the repo (`iter_splits`, inner validation, focal loss, patience 10, best inner-val F1 checkpoint, same metrics/ops/preds format; `results.json` is compatible with `src/evaluate.py`, though the variant names are not in its table list). Because the training loop is a copy of `train_one_fold` (to allow a separate lr), it is not literally the same code.

## Results

### Challenge 2015 PPG cohort (592 records, 5 folds x 5 seeds = 25 evaluations; scratch rows restricted to the same seeds 0-4 / identical splits)
| Variant | F1 | Sens | Spec | AUC | Challenge score | trainable params | dF1 vs scratch cross_attn (paired) |
|---|---|---|---|---|---|---|---|
| scratch ecg_only | 0.653 | 0.743 | 0.682 | 0.780 | 51.0 | 261k | |
| scratch ppg_only | 0.613 | 0.825 | 0.480 | 0.717 | 48.8 | 261k | |
| scratch concat | 0.691 | 0.741 | 0.754 | 0.839 | 54.5 | 256k | |
| **scratch cross_attention** | **0.713** | 0.754 | 0.787 | 0.842 | 57.1 | 262k | 0 |
| PaPaGei-S linear probe | 0.583 | 0.608 | 0.717 | 0.733 | 43.1 | 513 | -0.130 |
| PaPaGei-P linear probe | 0.565 | 0.622 | 0.659 | 0.706 | 41.4 | 513 | -0.148 |
| PaPaGei-S linear (pooled feat.) | 0.588 | 0.618 | 0.714 | 0.730 | 43.7 | 513 | -0.125 |
| PaPaGei-S frozen + MLP head | 0.599 | 0.690 | 0.637 | 0.739 | 45.8 | 33k | -0.114 |
| ECG CNN + frozen PaPaGei, concat | 0.675 | 0.723 | 0.756 | 0.832 | 53.4 | 126k | -0.038 |
| ECG CNN + frozen PaPaGei, cross-attn | 0.676 | 0.726 | 0.755 | 0.815 | 53.7 | 234k | -0.037 |
| PaPaGei-S last-block FT, PPG only | 0.597 | 0.703 | 0.616 | 0.724 | 45.6 | 1.6M | -0.116 |
| ECG CNN + FT PaPaGei, cross-attn | 0.659 | 0.731 | 0.711 | 0.791 | 52.0 | 1.8M | -0.054 |

Fold-level F1 SD is about 0.05-0.08 (paired dF1 SD 0.05-0.09, n=25 correlated folds), so differences of <0.03 are not distinguishable. The full-10-seed scratch numbers quoted elsewhere (ppg_only 0.602, ecg_only 0.642, cross_attention 0.699) agree with the 5-seed subset above to within ~0.01.

### VTaC official split (4542 VT events, 3 seeds on the single official train/val/test split; seeds only vary initialisation, so SDs are tiny and understate uncertainty)
| Variant | F1 | Sens | Spec | AUC |
|---|---|---|---|---|
| scratch ecg_only (10 runs, existing) | 0.709 | | | 0.897 |
| scratch ppg_only (10 runs) | 0.566 | | | 0.751 |
| scratch concat / cross_attention (5 runs each, existing) | 0.744 / 0.746 | | | 0.909 / 0.915 |
| PaPaGei-S linear | 0.581 | 0.642 | 0.779 | 0.790 |
| PaPaGei-P linear | 0.503 | 0.633 | 0.655 | 0.688 |
| PaPaGei-S frozen + MLP head | 0.598 | 0.692 | 0.758 | 0.801 |
| ECG CNN + frozen PaPaGei, concat | 0.735 | 0.800 | 0.852 | 0.913 |
| ECG CNN + frozen PaPaGei, cross-attn | 0.708 | 0.822 | 0.805 | 0.886 |
| PaPaGei-S last-block FT, PPG only | 0.616 | 0.628 | 0.840 | 0.811 |
| ECG CNN + FT PaPaGei, cross-attn | 0.715 | 0.811 | 0.821 | 0.897 |

(Scratch VTaC numbers are taken from `runs/vtac_official/results.json` as they exist; counts of runs differ from the foundation runs.)

## Conclusion
- PaPaGei is easy to obtain and use, but on ICU alarm verification it does **not beat** the from-scratch encoders. As a PPG-alone feature extractor it is at best on par with the scratch 261k-parameter CNN: linear probe F1 0.583 / AUC 0.733 vs scratch ppg_only 0.613 / 0.717 on Challenge 2015 (higher AUC, lower F1 at the fixed 0.5 threshold); on VTaC it is better than scratch ppg_only (F1 0.58-0.62 vs 0.57, AUC 0.79-0.81 vs 0.75).
- In the fusion setting, replacing the scratch PPG encoder with frozen PaPaGei features costs about 0.04 F1 on Challenge 2015 (0.675 vs 0.713 cross-attention) and fine-tuning the last block did not recover it (0.659); on VTaC the frozen-PaPaGei fusion models (0.708-0.735) are close to but not above scratch fusion (0.744-0.746). The ECG branch carries almost all of the fusion gain; PaPaGei adds roughly +0.06 F1 over scratch ppg_only only when combined with the ECG encoder (which a scratch PPG encoder also provides).
- Plausible reasons (not tested): pretraining was on wearable/perioperative/sleep PPG at 125 Hz with clean signals, while alarm windows are artefact-laden bedside signals (sometimes ABP); the trunk collapses 10 s to only 3 tokens, which limits cross-attention; and the cohort is tiny (592 records), so conclusions at F1 differences < 0.03 are not supportable. Gains for foundation models might appear at lower label fractions, which was not run.
- Not done: ECG-FM (blockers above), LoRA (peft needs transformers), label-fraction sweeps, more than 3 seeds on VTaC, significance tests.
