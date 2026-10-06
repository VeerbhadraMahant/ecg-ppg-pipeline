# Strongest deep recipe: search and results

Code: `src/train_deep_plus.py` (trainer, grid, selection, final protocol), `src/deep_plus_report.py` (tables).
Outputs: `runs/deep_plus_grid/` (grid.jsonl, grid_summary.csv, chosen.json), `runs/deep_plus_cinc`, `runs/deep_plus_cinc_recovered`, `runs/deep_plus_vtac` (preds in the `src/train.py` layout plus extra ablation views `val_prob__<view>` / `test_prob__<view>`; `results.json` per variant).

## Recipe ingredients (all switchable)

* Training-only augmentation on GPU: per-sample circular time shift (+-1 s) applied to both signals together, per-channel amplitude scaling (log-normal, sigma 0.15), additive Gaussian noise (sigma up to 0.1), a random 1-s dropout of ONE randomly chosen modality (p = 0.3); optional mixup (implemented, not used in the grid).
* AdamW (wd 1e-2), one-cycle schedule (15 % warm-up, cosine decay), gradient clipping 1.0, batch 32 (64 on VTaC), at most 30 epochs (20 on VTaC), early stopping on **inner-validation AUC** (patience 8 / 6).
* Loss: `focal` (gamma 2, pos_weight) or `bce_ls` (pos-weighted BCE, label smoothing 0.05).
* EMA of weights (horizon about 3 epochs); the EMA weights at their best inner-val epoch are used (raw best-epoch weights are stored as a separate view).
* Test-time augmentation: mean probability over circular shifts {-100,-50,0,+50,+100} samples.
* Ensemble: K = 3 independently initialised members, mean probability.
* Hybrids `resnet1d_feat`, `tcn_feat`: the 53 hand-crafted agreement features (`src/features.py`) are standardised with fit-set statistics (clipped to +-5), encoded by a 53-32 MLP and concatenated with the deep features before the head (Gaussian noise 0.1 on features during augmentation). Features for the recovered cohort were computed and cached in `data/processed/challenge2015_windows_recovered_features.npz`.
* Model widths are the repo's parameter-matched widths (about 261k parameters).

## Search (inner validation only)

Pre-declared grid, written into the code before any run: architecture {resnet1d, tcn, inceptiontime, mousavi_attn_cnn_rnn, resnet1d_feat, tcn_feat} x augmentation {off, on} x loss {focal, bce_ls} x lr {3e-4, 1e-3} = 48 configs, each trained on 3 seeds (100-102) x 5 folds of the CinC PPG cohort (15 trainings, single member, EMA). Score = mean over the 15 trainings of the inner-validation AUC of the EMA model at its best epoch. The grid stage never computes test predictions. Full table: `runs/deep_plus_grid/grid_summary.csv`.

Top configs (mean inner-val AUC, SE about 0.01):

| rank | config | inner-val AUC |
|---|---|---|
| 1 | mousavi_attn_cnn_rnn, aug, focal, lr 1e-3 | 0.9362 |
| 2 | mousavi_attn_cnn_rnn, aug, bce_ls, lr 1e-3 | 0.9337 |
| 3 | mousavi_attn_cnn_rnn, no aug, focal, lr 1e-3 | 0.9332 |
| 4 | mousavi_attn_cnn_rnn, no aug, bce_ls, lr 1e-3 | 0.9315 |
| 5 | tcn_feat, aug, focal, lr 1e-3 | 0.9254 |
| 6-8 | mousavi lr 3e-4 variants, tcn_feat no aug | 0.923-0.924 |
| best resnet1d | no aug, focal, 1e-3 | 0.9041 |
| best inceptiontime | no aug, focal, 1e-3 | 0.9035 |
| best tcn | no aug, focal, 1e-3 | 0.9127 |
| best resnet1d_feat | no aug, bce_ls, 1e-3 | 0.9200 |

Marginals (mean over the other factors): architecture mousavi 0.928 > tcn_feat 0.918 > tcn 0.907 ~ resnet1d_feat 0.905 > inceptiontime 0.895 ~ resnet1d 0.893; lr 1e-3 0.914 vs 3e-4 0.901; focal 0.909 vs bce_ls 0.906; augmentation off 0.908 vs on 0.907 (no inner-val effect for a single member). Features add +0.01 to +0.02 inner-val AUC to tcn / resnet1d. The differences among the top four configs are within one SE; the architecture and lr effects are clear.

Chosen (single winner, also the best non-hybrid): `mousavi_attn_cnn_rnn | aug | focal | lr 1e-3`, final recipe = this + AdamW/one-cycle + EMA + TTA + K = 3 ensemble. The same recipe (no re-tuning) was applied to the recovered cohort and VTaC.

## Final results (test folds; mean +- std over seed x fold)

| dataset | protocol | n runs | accuracy | F1 | AUC | sensitivity | specificity |
|---|---|---|---|---|---|---|---|
| CinC PPG cohort (592 rec.) | 5-fold x 10 seeds | 50 | 0.871 +- 0.029 | 0.828 +- 0.037 | 0.932 +- 0.028 | 0.827 +- 0.057 | 0.898 +- 0.036 |
| CinC recovered cohort (724) | 5-fold x 10 seeds | 50 | 0.862 +- 0.031 | 0.828 +- 0.037 | 0.933 +- 0.018 | 0.839 +- 0.051 | 0.878 +- 0.047 |
| VTaC official split (427 test events) | 5 seeds | 5 | 0.913 +- 0.005 | 0.858 +- 0.008 | 0.966 +- 0.001 | 0.933 +- 0.012 | 0.905 +- 0.006 |

Existing leak-free references on CinC: resnet1d 0.760 / 0.884 (F1 / AUC), tcn 0.751 / 0.887, hand-crafted GBM 0.798 / 0.918. The deep+ recipe is +0.03 F1 / +0.014 AUC above the GBM and +0.07 F1 above resnet1d. VTaC reference: ecg_only 0.709 / 0.897 (a different, ECG-only model; this one uses ECG+PPG, so the +0.15 F1 is not an apples-to-apples recipe comparison). Note that the VTaC test split in `vtac_windows.npz` has 427 events (3662/453/427), not the 482 in the protocol document.

## Ablations (same seeds, same trained models unless noted)

CinC PPG cohort, 10 seeds; first block uses stored views of the same trained models (paired, delta F1 against the full recipe, SE over seeds):

| ingredient removed | F1 | AUC | dF1 (SE) |
|---|---|---|---|
| full (K=3, EMA, TTA) | 0.828 | 0.932 | 0 |
| no TTA (K=3, EMA) | 0.824 | 0.931 | -0.003 (0.002) |
| no EMA (K=3, raw best epoch, TTA) | 0.819 | 0.932 | -0.008 (0.002) |
| neither EMA nor TTA (K=3) | 0.819 | 0.930 | -0.008 (0.003) |
| ensemble size 2 (EMA, TTA) | 0.828 | 0.930 | 0.000 (0.002) |
| single member (EMA, TTA) | 0.815 | 0.925 | -0.012 (0.004) |
| single member, raw, no TTA | 0.793 | 0.922 | -0.035 (0.010) |

Recovered cohort shows the same order (single raw -0.022, single EMA -0.003, ensemble/TTA about 0). On VTaC every ingredient is within about 0.01 F1 (single raw 0.849 vs full 0.858), i.e. small and mostly within noise with 5 seeds.

Trained-variant ablations on CinC (5 seeds, paired against the full recipe on those seeds):

| variant | F1 | AUC | dF1 (SE) | dAUC (SE) |
|---|---|---|---|---|
| augmentation off (mousavi) | 0.818 | 0.928 | -0.0135 (0.0011) | -0.004 (0.002) |
| tcn_feat (hybrid, aug) | 0.807 | 0.921 | -0.0245 (0.0067) | -0.011 (0.003) |
| tcn without features (aug) | 0.786 | 0.906 | -0.0458 (0.0066) | -0.026 (0.004) |

Reading: augmentation +0.014 F1; EMA is the largest single post-training ingredient (+0.008 at K=3, +0.022 for a single member, because it stabilises best-epoch selection on a tiny, noisy validation set); TTA about +0.003; ensembling K=1 to 3 about +0.012, with K=2 already capturing it; hand-crafted features raise tcn from 0.786 to 0.807 F1 (+0.021, AUC +0.015) but do not beat the Mousavi-style attention CNN-GRU, which has no feature input. A mousavi+features hybrid was not in the declared grid and was not run. The tcn vs tcn_feat comparison is the clean test of the features (same architecture); the hybrid-vs-mousavi comparison is confounded by architecture.

## Caveats

* The grid was selected on inner-validation AUC from the CinC cohort with seeds 100-102; these inner-val sets are subsets of records that are test records in other final seeds/folds, so the selection used labels of records that appear as outer-test elsewhere. Only the choice of 6 hyper-parameters was made this way, but a small optimistic bias is possible. The recipe was then applied unchanged to VTaC (not re-tuned there), which is the cleaner external check.
* Per-epoch selection uses inner-val AUC with only about 15-25 true alarms per CinC fold, so selection is noisy; the validation scores in the grid are maxima over epochs and therefore optimistic in absolute terms.
* Compute reductions made because of GPU sharing: epochs capped at 30 (20 on VTaC) with short patience, ensemble K = 3 (not 5), VTaC 5 seeds, the ablation trainings on 5 seeds. Stronger training could change the ranking.
* Baselines in the repo use a different trainer (Adam, F1-based early stopping on inner-val, batch 16). Part of the gain is the optimiser/regularisation/ensembling and not the architecture; mousavi was not re-run with the old trainer here, and the earlier sanity run reported F1 about 0.79.
* A shuffled-label control (labels of fit and val sets permuted, one fold) gave test AUC 0.57 on 118 records (chance), and a split check confirmed record-disjoint fit/val/test, so there is no evidence of leakage, but the F1 jump (0.76 to 0.83) is large and deserves independent re-running.
* Differences between the top grid configs, and ensemble size 2 vs 3, are within noise. Std is over seed x fold and is not a confidence interval (folds share training data); the Nadeau-Bengio / bootstrap tests of `src/stats.py` were not run for these variants. The results.json variant names (`deepplus`, ...) are not in `src/evaluate.py`'s TABLE_VARIANTS list, so that script will ignore them unless extended; `fold_metrics` fields are compatible.
* Hand-crafted features are computed on the unshifted window while augmentation shifts the signals, a small train-time mismatch for end-of-window features.
* VTaC: 5 runs of the same official split differ only by seed, so the std (0.005 F1) reflects training randomness only, not sampling uncertainty of the 427 test events (120 positives).
