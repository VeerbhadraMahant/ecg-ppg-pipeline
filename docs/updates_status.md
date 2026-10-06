# updates.md status ledger

Honest mapping of every numbered item in `updates.md` to what exists in this repository.
Status key: **DONE** = implemented and run; **PARTIAL** = some of it implemented/run, gap stated;
**NOT DONE** = nothing usable exists; **BLOCKED** = cannot proceed without something outside the repo (access, compiler, identifiers).

Numbers are quoted only from `README.md`, `docs/*.md` and `docs/LEADERBOARD.md`. Where no such number is written, the
cell says "see runs/..." and the figure has not been transcribed. Nothing was re-run to produce this file.
Context for the main result: the original cross-attention proposal is not better than concatenation, and the
hand-crafted-feature models and ensembles beat all deep models (`docs/LEADERBOARD.md`).

## Section 3. Phase 0

| # | Item | Status | Implemented in | Headline / note |
|---|---|---|---|---|
| 3.1 | Statistics: corrected resampled t-test or record bootstrap, 10 seeds, comparison family fixed in advance | DONE | `src/stats.py`, `src/evaluate.py`, `configs/config.yaml` (`stats.comparisons`), `docs/evaluation_protocol.md` | Nadeau-Bengio + paired record bootstrap (5000) + Holm. Cross-attention minus concat: dF1 = +0.006, 95% CI -0.015 ... +0.028 (not significant). Original leakage (best epoch picked on the reported fold) found and fixed; cross-attention F1 0.774 -> 0.699. |
| 3.2 | Headline metric: suppression at 95/99/100% sensitivity, per-type rates, official Challenge score | DONE | `src/metrics.py`, `src/evaluate.py`, `src/safety.py`; `runs/challenge2015_ppg/per_type_suppression.csv`, `per_alarm_type.csv` | Realised sensitivity on test folds is ~92-97% when 95-100% is targeted on validation (README, model card). Challenge score of best model: 67.5 (ensemble), 71.5 (LR stacker) on CinC (`docs/LEADERBOARD.md`). |
| 3.3 | Recover dropped records (second ECG lead, ABP) | DONE | `src/data/preprocess.py`, `data/processed/cohort_audit.csv`, `runs/challenge2015_recovered/` | Cohort 592 -> 724 records; 132 recovered via ABP substituted as pulse channel; 26 excluded (`docs/datasheet.md`). |
| 3.4 | Baseline zoo: features + GBM, ResNet1D, InceptionTime, BiGRU/TCN, plain transformer, compact SSM | DONE | `src/models/baselines.py` (ResNet1D, InceptionTime, BiGRU, TCN, PlainTransformer, CompactSSM), `src/baselines_gbm.py`, `src/features.py`, `src/features_v2.py`, `src/classical.py`, `src/train_deep_plus.py` | CinC F1: XGBoost 0.818, ResNet1D 0.760, TCN 0.751, InceptionTime 0.732. BiGRU / transformer / SSM are trained (`runs/challenge2015_ppg/results.json`) but not in the leaderboard: see runs/challenge2015_ppg. |
| 3.5 | Stronger fusion rivals: bidirectional cross-attention, joint transformer, bottleneck tokens, quality-gated | DONE | `src/models/fusion.py` (`BiCrossAttentionFusion`, `JointTransformerFusion`, `BottleneckFusion`, `GatedCrossAttentionFusion`) | Four attention-style fusion variants are statistically indistinguishable from each other and from concat (README). Best deep model on VTaC is bidirectional cross-attention: F1 0.775, AUC 0.933. |
| 3.6 | External validation: MIMIC PERform AF as a fine-tuned AF task, not zero-shot | DONE | `src/af_task.py`, `src/distill_af.py`, `src/data/mimic_perform_loader.py`, `docs/af_screening.md` | n = 35 subjects only. RR-irregularity GBM is perfect at subject level (subject AUC 1.00) and beats every neural model; best neural (cross-attention + alarm pretrain) subject AUC 0.934. Low power. |

## Section 4. Data expansion

| # | Dataset | Status | Where | Note |
|---|---|---|---|---|
| 4.1a | PhysioNet/CinC 2015 | DONE | `src/data/challenge2015_loader.py`, `scripts/download_data.py` | 750 records; 592 PPG cohort, 724 recovered cohort. |
| 4.1b | VTaC | DONE | `scripts/download_vtac.py`, `scripts/build_vtac_windows.py`, `src/data/vtac_loader.py` | v1.1, 5,037 events, 4,542 kept (28.0% true, 2,138 patients), official patient-disjoint split 3,662 / 453 / 427. Decision-time window (10 s ending at onset); post-onset 5/10/30 s only for time-to-verdict. Only [-60 s, +30 s] downloaded (~1 GB of 18 GB). |
| 4.2a | PulseDB | BLOCKED | n/a | Not downloaded; pending the user's access (large, derived from MIMIC-III/VitalDB). `docs/datasheet.md` s.3: "not used - access / size; pending credentials". |
| 4.2b | VitalDB | BLOCKED | n/a | Same; free registration / access pending on the user's side. |
| 4.2c | MIMIC-III waveform | BLOCKED | n/a | Needs credentialed PhysioNet access. Not used. |
| 4.2 substitute | In-domain unlabeled pretraining corpus | PARTIAL | `src/pretrain.py` | Used VTaC 60 s pre-alarm segments of train+val patients only (test patients excluded) as the pretraining corpus. Small compared with the planned corpora. 123 unlabeled VTaC `waveforms/` patients not used. |
| 4.3 | ECG-only labeled data (PTB-XL, CPSC 2018, Chapman-Shaoxing) | NOT DONE | n/a | Optional in the plan; nothing implemented. |
| 4.4a | MIT-BIH Noise Stress Test | DONE | `src/analysis/stress_benchmark.py`, `data/raw/nstdb` | Used for synthetic noise injection (bw, em, ma). Check for injection-signature learning: `runs/challenge2015_ppg/stress/signature_check.json`. |
| 4.4b | PPG-DaLiA, WESAD, IEEE SPC 2015 | NOT DONE | n/a | Not downloaded or used. |
| 4.4c | BIDMC, CapnoBase | PARTIAL | `src/heads/beat_ptt.py`, `runs/heads/beat_ptt_validation.json` | BIDMC used (streamed) for beat/PTT module validation against monitor numerics (it has no beat annotations). CapnoBase not used. |
| 4.4d | CinC 2014 | NOT DONE | n/a | Not used. Beat detection was validated on MIT-BIH (ECG only) and BIDMC instead (`src/heads/beat_ptt.py`). |
| 4.4e | MIMIC PERform AF | DONE | see 3.6 | Fine-tuned subject-wise AF task. |
| 4.x | "Before using any dataset": licence, access level, patient overlap | PARTIAL | `docs/datasheet.md` s.4, `docs/foundation_models.md` | Licences recorded (VTaC CC BY-SA 4.0; PaPaGei code BSD-3-Clause-Clear, weights licence unstated; ECG-FM MIT). Patient overlap CinC / VTaC / MIMIC-derived cannot be checked from released identifiers. |

## Section 5. Model and algorithm expansion

| # | Item | Status | Implemented in | Headline / note |
|---|---|---|---|---|
| 5.1 | Token-per-modality architecture, modality embeddings, modality dropout, any input subset | DONE | `src/models/multimodal.py` (`MultiModalAlarmClassifier`, modality dropout p = 0.3; ECG, ECG lead 2, PPG, ABP), `src/analysis/modality_subsets.py` | Missing pulse sensor costs ~0.05-0.09 AUC for fusion models (model card). Subset/Shapley tables: `runs/challenge2015_recovered/modality_subsets/`. |
| 5.2 | Self-supervised pretraining on PulseDB / VitalDB / MIMIC-III; contrastive vs masked vs cross-modal | PARTIAL | `src/pretrain.py`, `runs/pretrain/{contrastive,masked,crossmodal}.pt` | All three objectives implemented and run, but only on VTaC pre-alarm segments (the three big corpora are BLOCKED). Pretraining + fine-tuning gives only +0.01 to +0.04 F1 (README). |
| 5.3 | Foundation-model track: PaPaGei and ECG-FM, frozen / linear probe / LoRA, vs scratch and SSM | PARTIAL | `src/models/foundation.py`, `src/train_foundation.py`, `docs/foundation_models.md`, `runs/foundation/` | PaPaGei done (linear, frozen+head, frozen fusion, last-block fine-tune). Did not beat scratch: on CinC frozen PaPaGei + ECG cross-attn F1 0.676 vs scratch cross-attn 0.713. **ECG-FM BLOCKED**: weights downloaded (1.1 GB, MIT) but loading needs `fairseq-signals` (C++/Cython, no C++ compiler on this machine) and the model expects 12-lead 500 Hz input while we have one or two leads at 250 Hz. **LoRA not used** (`peft` needs `transformers`, not installed); last-block fine-tuning used instead. Compact SSM exists as a zoo baseline. |
| 5.4 | Long pre-alarm context as patient-specific baseline with a long-sequence model | NOT DONE | enabler only: `scripts/download_vtac.py` fetches 60 s pre-alarm; `data/processed` windows are 10 s | No long-context model or experiment exists. Window length used so far is 10 s (see section 14). |
| 5.5 | Alarm-type conditioning (type-conditioned heads / mixture of experts) | NOT DONE | partial overlap: alarm type is an input feature in classical models on CinC (`src/features_v2.py`; `alarm_tachy` is the top feature); `AlarmTypeContrastive` in `src/models/reproductions.py` | No type-conditioned head or mixture of experts. The type is only an auxiliary prediction target in the multitask model (5.6). |
| 5.6 | Multi-task heads: alarm type, signal quality, beat positions | DONE | `src/models/multitask.py`, `src/train_multitask.py`, `runs/multitask_guardrail/guardrail.json` | Guardrail run compares with vs without auxiliary heads (task interference); result: see runs/multitask_guardrail. |
| 5.7 | Reproduced published methods: rule/feature-based, Mousavi attention CNN-RNN, alarm-type contrastive | PARTIAL | `src/models/reproductions.py`, `src/train_reproductions.py`, `configs/width_mult_reproductions.yaml`, `src/baselines_gbm.py` | Models implemented; only a 2-seed sanity run of the Mousavi-style model exists (`runs/reproductions_sanity`); no full `runs/reproductions` run for either method. No faithful rule-based Challenge-2015 entry; the feature-based stand-in is the GBM/classical stack. |

## Section 6. Analysis experiments

| # | Experiment | Status | Implemented in | Headline |
|---|---|---|---|---|
| 6.1 | Counterfactual alignment tests (swap, shift, reverse, drop) | DONE | `src/analysis/counterfactual.py`, `runs/challenge2015_ppg/counterfactual/` | Swapping in another patient's same-label PPG or shifting PPG up to 2 s changes AUC by < 0.015: fusion models do not use ECG-PPG timing (README). ECG-side variants: see runs/challenge2015_ppg/counterfactual. |
| 6.2 | Attention vs measured PTT | DONE | `src/analysis/attention_ptt.py`, `src/heads/beat_ptt.py`, `runs/challenge2015_ppg/attention_ptt/` | Spearman rho = -0.04, permutation p = 0.39: attention does not track PTT (README). |
| 6.3 | Explanation faithfulness: attention vs occlusion vs integrated gradients, deletion/insertion | DONE | `src/analysis/faithfulness.py`, `runs/challenge2015_ppg/faithfulness/`, `runs/faithfulness.log` | No number in docs/*.md; see runs/challenge2015_ppg/faithfulness. |
| 6.4 | Evaluation-protocol audit of reproduced methods | PARTIAL | `src/analysis/protocol_audit.py`, `runs/protocol_audit/protocol_audit.json` | The audit script and output exist, but it depends on the reproductions (5.7), which only have a sanity run. The evaluation-protocol audit of this repo's own earlier results is done (`docs/evaluation_protocol.md` s.4: cross-attention F1 0.774 -> 0.699). Published-method survival numbers: see runs/protocol_audit. |
| 6.5 | Cross-dataset and cross-site generalisation | PARTIAL | `src/cross_dataset.py`, `runs/cross_dataset/{cinc_to_vtac,vtac_to_cinc_vt}` | CinC 2015 -> VTaC F1 ~0.55-0.60 (README). Leave-one-hospital-out BLOCKED (ids not released). Leave-one-device-out via the lead-set-signature proxy is coded (`--holdout-signature`) but no result directory exists. |
| 6.6 | Time-to-verdict curve (0, 5, 10, 30 s post-alarm) | DONE | `scripts/build_vtac_windows.py --post-seconds`, `runs/vtac_ttv_post{5,10,30}`, `runs/vtac_official` (0 s) | No number in docs/*.md; see runs/vtac_ttv_post5, vtac_ttv_post10, vtac_ttv_post30. |
| 6.7 | Label-efficiency curves (10/25/50/100%): pretrained vs scratch, fusion vs single | DONE | `scripts/run_label_efficiency.py`, `scripts/aggregate_label_efficiency.py`, `runs/labeleff/`, `runs/labeleff_single/`, `runs/labeleff/label_efficiency.png` | Pretraining + fine-tuning gives only +0.01 to +0.04 F1 (README). Per-fraction table: see runs/labeleff. |
| 6.8 | Modality-subset study, Shapley contribution | DONE | `src/analysis/modality_subsets.py`, `runs/challenge2015_recovered/modality_subsets/{subsets,shapley,robustness}.csv` | No number in docs; see runs/challenge2015_recovered/modality_subsets. |
| 6.9 | Missing-modality robustness | DONE | `src/analysis/modality_subsets.py`, `src/analysis/stress_benchmark.py`, `runs/challenge2015_ppg/stress/missing_modality.csv` | Missing pulse sensor costs ~0.05-0.09 AUC for fusion models (model card). |
| 6.10 | Synthetic stress benchmark (MIT-BIH noise, PTT variation, dropout) | DONE | `src/analysis/stress_benchmark.py`, `tests/test_stress.py`, `runs/challenge2015_ppg/stress/` | AUC degrades with heavy noise on both signals (model card). Labeled synthetic; signature check in `signature_check.json`. |
| 6.11 | Label reliability and soft-label training (VTaC per-annotator labels) | DONE | `src/analysis/label_reliability.py`, `runs/vtac_official/label_reliability/reliability.csv`, soft-label option in `src/train.py`, `runs/vtac_official_soft/` | No number in docs; see runs/vtac_official/label_reliability and runs/vtac_official_soft. |
| 6.12 | Error taxonomy with case studies | DONE | `src/analysis/error_taxonomy.py`, `docs/error_taxonomy.md` | Missed true alarms concentrate in VT: persistent-FN rate in true VT is ~24-31%; persistent FNs 13.0% (gbm) and 14.3% (cross-attention) of 223 true alarms. |

## Section 7. New functionalities

| # | Item | Status | Implemented in | Headline / note |
|---|---|---|---|---|
| 7.1 | Signal quality estimator per channel, used for gating | DONE | `src/heads/quality.py`, `runs/heads/quality/results.json`, gated fusion in `src/models/fusion.py` | Result: see runs/heads/quality. |
| 7.2 | Beat detection and PTT module | DONE | `src/heads/beat_ptt.py`, `tests/test_heads.py`, `runs/heads/beat_ptt_validation.json` | Validated against MIT-BIH beats (ECG) and BIDMC numerics; used as ground truth in 6.2. Numbers: see runs/heads/beat_ptt_validation.json. |
| 7.3 | AF screening head + PPG-only distilled student | DONE | `src/af_task.py`, `src/distill_af.py`, `docs/af_screening.md` | Distillation did not help: student subject AUC 0.618 vs hard-label PPG-only 0.623; teacher 0.916. |
| 7.4 | Safety layer: calibration, conformal / risk-controlled threshold, deferral band, triage priority | DONE | `src/safety.py` | Temperature scaling, Clopper-Pearson-bounded risk-controlled threshold (abstains if validation positives are too few), deferral band. Triage priority is not separately documented. Realised sensitivity ~92-97% when targeting 95-100% (README). |
| 7.5 | Clinical utility: false alarms removed per bed-day, expected misses, decision curves | PARTIAL | `src/safety.py`, `runs/challenge2015_ppg/safety/` (concat only: calibration, decision curve, utility) | Per-100-alarm utility and decision curves exist. Alarm volume per bed-day is a user-supplied assumption, not data; no per-bed-day figure is claimed. Safety outputs exist for only the concat variant in that folder. |
| 7.6 | Streaming replay demo, int8 latency | DONE | `site/replay.html`, `site/assets/replay.json`, `scripts/build_replay_data.py`, `scripts/build_results_notebook.py` | int8 is dynamic quantisation of Linear layers only, so it gives little or no speedup at this model size (shown in `site/replay.html`). Latency numbers: see site/assets/replay.json. |

## Section 8. Stretch goals

| Item | Status | Note |
|---|---|---|
| Signal repair with diffusion / flow matching | NOT DONE | Nothing implemented. |
| Label-free ECG-PPG consistency scoring (zero-shot) | NOT DONE | No implementation found. The pretraining objectives (6.7) are related but were evaluated by fine-tuning, not zero-shot. |
| Cuffless BP estimation on PulseDB | BLOCKED | Needs PulseDB (pending user access). |
| Deterioration forecasting on VitalDB | BLOCKED | Needs VitalDB (pending user access). |
| Leave-one-site-out across all data sources | NOT DONE | Only a lead-signature proxy leave-one-device-out is coded (see 6.5), with no result. Real site ids are not released for VTaC. |

## Section 12. Order of work

| Step | Status | Note |
|---|---|---|
| 1. Phase 0 fixes | DONE | All six items (section 3). |
| 2. Add VTaC with decision-time windowing | DONE | `docs/datasheet.md`, `docs/evaluation_protocol.md`. |
| 3. Baseline zoo, reproduced methods, protocol audit | PARTIAL | Zoo done; reproductions and audit are sanity-level only (5.7, 6.4). |
| 4. Pretraining, foundation models, modality study | PARTIAL | Modality study done; pretraining only on VTaC pre-alarm; PaPaGei done, ECG-FM blocked. |
| 5. Analysis experiments | PARTIAL | 10 of 12 fully done; 6.4 and 6.5 are partial (see section 6). |
| 6. Supporting heads, safety layer, demo | DONE | Section 7 (7.5 partial on per-bed-day). |
| 7. Stretch goals | NOT DONE | See section 8. |

## Section 14. Open checks (answered)

| Check | Status | Answer |
|---|---|---|
| VTaC: hospital and monitor manufacturer identifiable per record? | DONE | **No.** Not released. The lead-set signature (set of ECG lead names) is used as an unverified proxy for monitor manufacturer: `II\|aVR\|V` 2,737 events (60%), `I\|II\|III\|V` 1,191 (26%), 35 small groups 14% (`docs/datasheet.md`). Hospital-wise generalisation is therefore not possible. |
| VTaC: individual annotator labels or only consensus? | DONE | **Yes, in v1.1** (`event_label_per_annotator.csv`, Sept 2026). `annotator_frac_true` and `n_annotators` are stored in the processed file and used by 6.11. |
| Patient overlap between CinC 2015, VTaC and MIMIC-derived corpora | BLOCKED | **Cannot be checked** from released identifiers. Assumed zero between the two labeled sets (different institutions, periods) but unverified. MIMIC PERform AF is MIMIC-derived, so AF-task pretraining gains could be optimistic (`docs/af_screening.md`, `docs/evaluation_protocol.md` s.5). |
| Which CinC 2015 records have ABP and a second ECG lead | DONE | Of 750 records: 343 have ABP, 627 have PPG, all 750 have a second ECG lead. 132 records recovered through ABP (`docs/datasheet.md`). |
| Licence and access level for every dataset and model checkpoint | PARTIAL | Datasets: CinC 2015, VTaC (CC BY-SA 4.0), MIT-BIH NST, MIMIC PERform AF all open. PulseDB, VitalDB, MIMIC-III not accessed. Checkpoints: PaPaGei code BSD-3-Clause-Clear, weights open download with no licence field (treat as "open, licence unstated"); ECG-FM MIT, not gated (`docs/foundation_models.md`). |
| Window length used in preprocessing | DONE | **10 s**, ending at alarm onset, 250 Hz; the alarm is at 300 s of each record. VTaC post-onset seconds (5/10/30) are used only in the time-to-verdict analysis (`docs/datasheet.md`). |

## What remains

* **PulseDB, VitalDB and MIMIC-III pretraining (blocked on the user's access).** Largest gap against the plan: 5.2 is
  done only on VTaC pre-alarm segments, and the BP-estimation and deterioration stretch goals depend on them.
* **ECG-FM (blocked).** Weights are on disk but need a C++ compiler for `fairseq-signals` and 12-lead 500 Hz input.
  LoRA variants for either foundation model were also not run.
* **Hospital-wise generalisation (blocked: ids not released).** Only the unverified lead-signature proxy for devices is
  possible, and even that leave-one-device-out run has not been executed.
* **Prospective validation.** None; US ICU data only; not a medical device.
* Not blocked but not done: long pre-alarm context model (5.4), alarm-type conditioning / mixture of experts (5.5),
  full runs of the reproduced published methods and the protocol audit on them (5.7, 6.4), ECG-only encoder
  pretraining corpora (4.3), wearable motion datasets (4.4), and the diffusion-repair and label-free-consistency stretch goals.

## Late updates (after the ledger was first written)

* **Deep recipe search (Phase 5, "highest accuracy")**: DONE. `src/train_deep_plus.py`, `docs/deep_recipe.md`; hybrid
  deep recipe F1 0.828 (Challenge 2015) / 0.858 (VTaC); best single model. Selection used inner-validation scores
  only; small possible optimistic bias documented.
* **Explanation faithfulness (6.3)**: DONE. See `docs/analysis_findings.md` section 3 (occlusion most faithful;
  attention weights barely above random).
* **Time-to-verdict (6.6)**: PARTIAL. Fixed 10 s windows ending 0/5/10/30 s after onset; performance falls. A window
  that keeps the pre-alarm 10 s and appends post-alarm seconds was not run (`docs/analysis_findings.md` section 5).
* **Label reliability / soft labels (6.11)**: DONE. Soft labels +0.038 F1 for cross-attention (5 seeds).
* **Counterfactual tests (6.1)** now cover ResNet1D, TCN, InceptionTime, BiGRU and the fusion rivals.
* **Cross-dataset (6.5)**: Challenge 2015 -> VTaC and VTaC -> Challenge 2015 VT alarms DONE;
  leave-one-device-out (lead-signature proxy) code exists but was NOT run.
* **Final leaderboard**: `docs/LEADERBOARD.md`; consolidated analyses: `docs/analysis_findings.md`.
