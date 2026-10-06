# TRIPOD+AI checklist (self-assessment)

Status key: **Done** = reported in the cited file; **Partial** = reported with stated gaps; **N/A** = not applicable;
**Not done** = honest gap. This is a research prototype and a self-assessment, not a journal submission.

| # | Item | Status | Where / note |
|---|---|---|---|
| 1 | Title identifies a prediction model study, target population, outcome | Partial | README title; population = ICU arrhythmia alarms |
| 2 | Structured abstract (TRIPOD+AI for Abstracts) | Not done | Written when the report is assembled |
| 3 | Background, rationale, existing models | Done | `proposal.md`, `updates.md` section 2 |
| 4 | Objectives (develop / validate) | Done | README, `updates.md` research questions |
| 5 | Patient / public involvement | N/A | none |
| 6 | Source of data, dates | Partial | `docs/datasheet.md`; collection dates not released |
| 7 | Participants: setting, eligibility | Done | `docs/datasheet.md` |
| 8 | Data preparation (preprocessing, quality checks) | Done | `src/data/`, `docs/datasheet.md` |
| 9 | Outcome definition | Done | expert true/false alarm label; VTaC >= 2 annotators, votes in v1.1 |
| 10 | Predictors, timing of measurement | Done | 10 s ending at onset; decision-time rule in `docs/evaluation_protocol.md` |
| 11 | Sample size justification | Not done | fixed by public data; low power stated |
| 12 | Missing data handling | Done | cohort flow; missing-modality study (`modality_subsets`) |
| 13 | Analytical methods (models, tuning, selection) | Done | `docs/evaluation_protocol.md`, `docs/classical_models.md` |
| 14 | Class imbalance | Done | focal loss / class weights; metrics beyond accuracy |
| 15 | Fairness approaches | Not done | no demographics released; limitation stated |
| 16 | Model output (probability, threshold) | Done | sigmoid probability; 0.5 and validation-chosen thresholds |
| 17 | Training vs evaluation differences | Done | inner/outer splits; cross-dataset tests |
| 18 | Ethical approval | Done | public de-identified data; sources' IRB statements |
| 19 | Open science: funding | N/A | |
| 20 | Conflicts of interest | N/A | |
| 21 | Protocol / registration | Partial | comparison family fixed in `configs/config.yaml` before final runs |
| 22 | Data availability | Done | PhysioNet / Zenodo links in `docs/datasheet.md` |
| 23 | Code availability | Done | this repository |
| 24 | Participant flow | Done | cohort flow tables |
| 25 | Participant characteristics | Partial | alarm-type and class balance; demographics not released |
| 26 | Model performance with CIs | Done | `docs/LEADERBOARD.md` (bootstrap CIs) |
| 27 | Model updating / calibration | Partial | temperature scaling in `src/safety.py`; no recalibration study |

Also relevant: limitations (`docs/model_card.md`), error analysis (`docs/error_taxonomy.md`),
robustness (`runs/*/stress`), generalisation (`runs/cross_dataset`).
