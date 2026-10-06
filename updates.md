# Project Updates: Scope Expansion Plan

Multimodal ECG + PPG fusion for arrhythmia alarm verification
Status: planning document, October 2026

---

## 1. Summary

The current project shows that cross-attention fusion of ECG and PPG outperforms ECG-only, PPG-only and concatenation baselines on the PhysioNet/CinC Challenge 2015 dataset. Faculty feedback asks for more datasets, more algorithms and a broader scope.

A literature and dataset review changes the plan in three ways:

1. A much larger open labeled dataset exists (VTaC), so the project no longer has to live inside 592 labeled records.
2. Attention-based and contrastive multimodal false-alarm reduction has already been published, so "cross-attention fusion" alone is not a novel contribution.
3. Open ECG and PPG foundation models now exist and must be compared against.

The project is therefore reframed from "does cross-attention help" to a broader study:

**Proposed title:** Multimodal Cardiac Signal Verification: a controlled, leakage-audited, two-dataset study of how ECG and pulse signals should be fused for ICU alarm verification.

### Research questions

1. **Alignment:** Do fusion models actually use ECG-to-pulse temporal alignment, or only use the pulse signal as a quality cue?
2. **Label efficiency:** Does large-scale unlabeled paired data, or an open foundation model, reduce the labels needed?
3. **Modalities:** Which signals (ECG leads, PPG, ABP) add value, and how gracefully does performance degrade when sensors drop out?
4. **Generalization:** Do models transfer across datasets, hospitals and monitor manufacturers?
5. **Safety:** Can the verifier suppress false alarms while keeping sensitivity near 100%, and how fast can it reach a verdict?

---

## 2. Findings from the literature review

### 2.1 VTaC: a large labeled alarm dataset

- Open access on PhysioNet, published at NeurIPS 2023 (Datasets and Benchmarks).
- 5,037 annotated ventricular tachycardia (VT) alarm events, 1,441 (about 28.6%) true alarms.
- Each alarm labeled true or false by at least two independent expert annotators.
- Data from ICUs in two US hospitals and three bedside monitor manufacturers.
- Each recording has at least two ECG leads plus one or more pulsatile signals (PPG and/or ABP).
- Each recording covers 5 minutes before and 5 minutes after the alarm.
- For comparison, CinC 2015 contains 562 VT alarms in total, only 341 of them in the public training set.
- VT is the alarm type for which false alarms are hardest to detect reliably.

**Implication:** roughly fifteen times more labeled VT alarms than currently used, plus natural cross-site and cross-device splits.

**Leakage warning:** a real monitor cannot see data after the alarm. Inputs must be restricted to data available at decision time (or a stated short delay), otherwise the 5 minutes after the alarm leak the answer.

### 2.2 Prior work on multimodal false-alarm reduction

- **Mousavi et al. (2019):** attention-based CNN and RNN models on single and multimodal signals (ECG, ABP, PPG), reporting about 93.9% sensitivity and 92.1% specificity with three signals.
- **Contrastive learning approach (Scientific Reports, 2022):** trains across all alarm types at once using learned alarm-type embeddings.
- **Feature-based approaches:** for example, a 114-feature signal-quality and physiological model reporting per-type false alarm suppression rates with 0% true alarm suppression.
- **Challenge 2015 entries:** rule-based signal processing across ECG, ABP and PPG.
- **Multimodal explainability on VTaC (2025):** ECG, PPG and ABP for true versus false VT alarms.

**Implication:** the current results (sensitivity 0.797, specificity 0.831) sit below some published numbers. Protocols differ, but the report must explain the gap, and the novelty must come from evaluation rigor, generalization and analysis rather than the attention layer itself.

### 2.3 Open foundation models

- **PaPaGei (ICLR 2025):** open PPG foundation model pretrained on over 57,000 hours (20 million segments) of public data; usable as a feature extractor or as an encoder in multimodal models.
- **ECG-FM (JAMIA Open, 2025):** open-weight transformer ECG model pretrained on about 1.5 million 12-lead ECGs with masked reconstruction plus contrastive learning.
- **Caution:** a 2025 benchmark of eight ECG foundation models found that a compact state-space model performed best and many foundation models failed to beat supervised learning outside adult ECG interpretation. Outcome is uncertain, which makes this a genuine experiment.

### 2.4 Reporting standard

- **TRIPOD+AI (BMJ, 2024):** 27-item reporting checklist for prediction models built with regression or machine learning, plus an abstracts checklist. It supersedes TRIPOD 2015. The final report will follow it.

---

## 3. Phase 0: mandatory fixes to current results

1. **Statistics.** Replace the paired t-test on 15 fold/seed pairs (not independent, multiple comparisons) with a corrected resampled t-test or record-level bootstrap. Move to 10 seeds. Fix the set of comparisons before running.
2. **Headline metric.** Report false alarms suppressed at fixed sensitivity (95%, 99%, 100%), per-type suppression rates, and the official Challenge 2015 score (false negatives weighted five times). Current results show sensitivity falling as fusion improves, which must be addressed.
3. **Recover dropped records.** Audit CinC 2015 for the second ECG lead and ABP. 158 of 750 records were dropped for lacking usable PPG; many may have ABP.
4. **Baseline zoo.** Hand-crafted agreement features (beat-detector agreement, PTT statistics, signal quality indices) with gradient boosting; ResNet1D; InceptionTime; BiGRU/TCN; plain transformer; compact state-space model.
5. **Stronger fusion rivals.** Bidirectional cross-attention, early-fusion joint transformer, bottleneck-token fusion, signal-quality-gated fusion.
6. **External validation.** MIMIC PERform AF is a different task (AF vs non-AF). Stop treating it as a zero-shot generalization test; make it a properly fine-tuned AF task instead.

---

## 4. Data expansion

### 4.1 Labeled alarm data (primary)

| Dataset | Role | Notes |
|---|---|---|
| PhysioNet/CinC Challenge 2015 | Five alarm types, small scale | Current dataset; recover records using ABP |
| VTaC | VT alarms, large scale | Cross-hospital and cross-manufacturer evaluation; decision-time windowing required |

### 4.2 Unlabeled paired data (pretraining)

| Dataset | Role | Notes |
|---|---|---|
| PulseDB | Millions of paired ECG/PPG/ABP segments | Derived from MIMIC-III and VitalDB; use subject IDs |
| VitalDB | Large surgical multichannel dataset | Free registration; surgical noise differs from ICU |
| MIMIC-III waveform | ICU waveforms and 1 Hz numerics | Check overlap with CinC 2015 and VTaC patients |

### 4.3 ECG-only labeled data (encoder pretraining, optional)

PTB-XL, CPSC 2018, Chapman-Shaoxing.

### 4.4 Robustness and auxiliary data

| Dataset | Role |
|---|---|
| MIT-BIH Noise Stress Test | Real ECG noise for synthetic artifact injection |
| PPG-DaLiA, WESAD, IEEE SPC 2015 | Wearable ECG/PPG under motion, accelerometer as artifact ground truth |
| BIDMC, CapnoBase | Clean annotated paired signals |
| CinC 2014 | Multimodal beat annotations for beat and PTT detection |
| MIMIC PERform AF | Fine-tuned AF screening task |

**Before using any dataset:** verify license, access level (some PhysioNet sets are credentialed) and patient overlap with the labeled evaluation sets.

---

## 5. Model and algorithm expansion

1. **Multi-modality architecture.** Token-per-modality design with modality embeddings and modality dropout. Inputs: ECG leads, PPG, ABP. One model handles any available combination.
2. **Self-supervised pretraining.** On PulseDB, VitalDB and MIMIC-III. Compare contrastive ECG/PPG matching, masked reconstruction and ECG-to-PPG cross-modal prediction.
3. **Foundation-model track.** PaPaGei (PPG) and ECG-FM (ECG) as encoders feeding the fusion layer: frozen, linear probe and fine-tuned (LoRA) variants, versus from-scratch encoders and a compact state-space baseline.
4. **Long pre-alarm context.** Use the minutes before the alarm as a patient-specific baseline (normal rhythm and ECG-to-pulse delay) with a long-sequence model.
5. **Alarm-type conditioning.** Type-conditioned heads or mixture of experts, since asystole, bradycardia, tachycardia, VT and VF/flutter differ physiologically.
6. **Multi-task heads.** Auxiliary prediction of alarm type, signal quality and beat positions.
7. **Reproduced published methods.** A rule or feature-based method, an attention CNN-RNN in the Mousavi style and the alarm-type-embedding contrastive approach.

---

## 6. Analysis experiments

1. **Counterfactual alignment tests.** Swap PPG from another patient with the same label, time-shift it, reverse it, drop segments; same for ECG. Shows whether the model uses alignment or just pulse quality.
2. **Attention validated against measured PTT.** Compute pulse transit time from detected R-peaks and PPG feet; check whether attention offsets track it.
3. **Explanation faithfulness.** Compare attention with occlusion and integrated gradients using deletion and insertion tests.
4. **Evaluation-protocol audit.** Run reproduced methods under their original protocol and under strict record-wise splits. Quantify how much published performance survives.
5. **Cross-dataset and cross-site generalization.** CinC 2015 to VTaC and back; leave-one-hospital-out and leave-one-manufacturer-out on VTaC (if site and device fields are released).
6. **Time-to-verdict curve.** Performance versus seconds of post-alarm data allowed (0, 5, 10, 30 s).
7. **Label-efficiency curves.** Fine-tune with 10%, 25%, 50%, 100% of labels: pretrained versus scratch, fusion versus single signal.
8. **Modality-subset study.** Every subset of ECG leads, PPG and ABP; Shapley-style marginal contribution of each modality.
9. **Missing-modality robustness.** Performance as sensors drop out at inference.
10. **Synthetic stress benchmark.** Clean segments plus MIT-BIH noise at controlled levels, with PTT variation and sensor dropout. Clearly labeled synthetic; check the model is not learning the injection signature.
11. **Label reliability.** If VTaC releases individual annotator labels, test whether model uncertainty tracks annotator disagreement and try soft-label training.
12. **Error taxonomy.** Manual categorization of failures by alarm type and signal condition, with case studies.

---

## 7. New functionalities (heads on the shared backbone)

1. **Signal quality estimator** per channel, used for fusion gating.
2. **Beat detection and PTT module**, also the ground truth for analysis experiment 2.
3. **AF screening head**, with a PPG-only student distilled from the fusion model.
4. **Safety layer:** calibration, conformal or risk-controlled thresholding, a deferral band for human review, triage priority.
5. **Clinical utility analysis:** false alarms removed per bed per day, expected missed true alarms, decision-curve analysis against the alarm-on-everything policy.
6. **Streaming replay demo:** extend the existing results dashboard to replay records in real time with verdict, quality scores and attention; report int8 latency.

---

## 8. Stretch goals

- Signal repair with a conditional diffusion or flow-matching model (reconstruct corrupted ECG from PPG and vice versa), judged by downstream detection gain.
- Label-free consistency scoring: learn normal ECG-PPG coupling from unlabeled data and evaluate zero-shot on alarm verification.
- Cuffless blood pressure estimation on PulseDB with subject-disjoint splits.
- Deterioration forecasting on VitalDB (keep the label-defining signal out of the input window).
- Leave-one-site-out evaluation across all data sources.

## 9. Explicitly cut

LLM report generation, echocardiography, cardiac MRI, fetal ECG, PCG fusion, federated learning, and any architecture without a hypothesis behind it.

---

## 10. Final report structure

1. **Abstract** following the TRIPOD+AI abstracts checklist.
2. **Introduction:** alarm fatigue, why VT is hardest, research questions.
3. **Related work:** rule-based, feature-based, deep multimodal (including prior attention and contrastive work), foundation models, and a clear statement of what is new.
4. **Data:** CinC 2015, VTaC, pretraining corpora, cohort flow (exclusions and reasons), overlap checks, decision-time windowing, datasheet.
5. **Methods:** preprocessing, encoders, fusion variants, pretraining, safety layer, baselines and reproductions.
6. **Evaluation protocol:** splits, statistics, metrics and leakage controls as a standalone chapter.
7. **Results:** within-dataset, cross-dataset and cross-site, reproduction audit, label efficiency, foundation models, time-to-verdict, modality study.
8. **Analysis:** counterfactual alignment tests, PTT-validated attention, faithfulness, error taxonomy.
9. **Clinical utility:** suppression rates, workload estimates, decision curves.
10. **Limitations and ethics:** US-only sources, demographic gaps (including skin tone effects on optical PPG), no prospective validation, not a medical device.
11. **Conclusion.**
12. **Appendices:** TRIPOD+AI checklist, model card, hyperparameters, compute budget, reproduction instructions.

---

## 11. Contribution claim

Not: "cross-attention fusion for alarm verification" (prior work exists).

Instead: a controlled, leakage-audited, two-dataset evaluation of ECG and pulse fusion for ICU alarm verification, with causal tests of whether models use cross-signal alignment, cross-hospital and cross-device generalization, foundation-model and pretraining comparisons, and safety-oriented metrics. Results are reported whichever way they come out.

---

## 12. Order of work

1. Phase 0 fixes.
2. Add VTaC with decision-time windowing.
3. Baseline zoo and reproduced published methods; protocol audit.
4. Pretraining, foundation-model track and modality study.
5. Analysis experiments.
6. Supporting heads, safety layer and demo.
7. Stretch goals if time remains.

## 13. Guardrails

- Every task head gets its own record-wise or subject-wise split and its own baselines.
- Report alarm-verification results with and without other heads trained jointly, to detect task interference.
- Pretraining and foundation models may not help; report that honestly if so.
- Keep a small configuration runnable on free Colab/Kaggle tiers so the reproducibility objective still holds. Heavy experiments run on the 24 GB college workstation.

## 14. Open checks before committing

- [ ] VTaC: are hospital and monitor manufacturer identifiable per record?
- [ ] VTaC: are individual annotator labels released, or only consensus?
- [ ] Patient overlap between CinC 2015, VTaC and MIMIC-derived corpora.
- [ ] Which CinC 2015 records have ABP and a second ECG lead.
- [ ] License and access level for every dataset and model checkpoint.
- [ ] Current window length used in preprocessing (needed for the long-context experiment).

---

## References

- VTaC dataset (PhysioNet): https://www.physionet.org/content/vtac/
- Lehman et al., VTaC benchmark, NeurIPS 2023: https://proceedings.neurips.cc/paper_files/paper/2023/hash/7a53bf4e02022aad32a4019d41b3b476-Abstract.html
- Mousavi et al., attention-based single and multimodal false alarm reduction: https://arxiv.org/abs/1909.11791v1
- Contrastive learning approach for ICU false arrhythmia alarm reduction (Scientific Reports, 2022): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8933571/
- Optimized feature-based false alarm reduction (per-type suppression rates): https://pmc.ncbi.nlm.nih.gov/articles/PMC6728371/
- Multimodal explainability for ICU signals on VTaC (2025): https://m.math-net.ru/eng/vtpmk757
- PaPaGei, open PPG foundation model (ICLR 2025): https://arxiv.org/abs/2410.20542v1 and https://github.com/nokia-bell-labs/papagei-foundation-model
- ECG-FM, open ECG foundation model: https://arxiv.org/abs/2408.05178v2 and https://github.com/bowang-lab/ECG-FM/
- Benchmark of ECG foundation models (2025): https://arxiv.org/pdf/2509.25095v1
- TRIPOD+AI statement (BMJ 2024): https://www.bmj.com/node/1095803
