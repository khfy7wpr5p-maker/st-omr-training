# st-omr-training

Safe training and synthetic-data laboratory for ST-OMR: canonical music generation, MusicXML serialization and validation, deterministic notation rendering, controlled degradation, dataset validation/construction, training, and evaluation.

This repository is isolated from the ScoreMosaic production runtime. Its purpose is to build traceable synthetic OMR training data, train candidate ST-OMR models, and evaluate them before any later integration decision.

## Project documents

- [Architecture](ARCHITECTURE.md)
- [Canonical data contract](DATA_CONTRACT.md)
- [MusicXML contract](MUSICXML_CONTRACT.md)
- [Renderer contract](RENDERER_CONTRACT.md)
- [Controlled degradation contract](DEGRADATION_CONTRACT.md)
- [Synthetic dataset manifest contract](DATASET_CONTRACT.md)
- [Synthetic Dataset v1 construction contract](DATASET_BUILD_CONTRACT.md)
- [Baseline ST-OMR training contract](TRAINING_CONTRACT.md)
- [Stage 7-B training implementation profile](TRAINING_IMPLEMENTATION.md)
- [Stage 7-C bounded run profile](STAGE7C_RUNBOOK.md)
- [Verovio runtime evidence](VEROVIO_RUNTIME_EVIDENCE.md)
- [Safety and verification rules](SAFETY.md)
- [Current project status](STATUS.md)

## Current phase

The repository is public and GitHub Actions CI is active.

Completed and merged:

- Stage 0 — safety and architecture baseline;
- Stage 1 — deterministic ST Music Generator V1;
- Stage 2 — deterministic MusicXML 4.0 pipeline, offline XSD/semantic validation, and supported-V1 semantic round trip;
- Stage 3 — deterministic Verovio 6.2.1 SVG renderer adapter;
- Stage 4 — deterministic, bounded Controlled Degradation V1;
- Stage 5 — synthetic dataset contract and independent manifest validation;
- Stage 6 — deterministic Synthetic Dataset v1 construction and hash-addressed local persistence;
- Stage 7-A — Baseline ST-OMR Training Contract Freeze;
- Stage 7-B — deterministic tokenizer/data/model/trainer smoke implementation.

Stage 7-B merged through PR #21 at exact `main` commit `d02dce4ee17dfccf6f05519ab0970fdc188d0147`. Its closure documentation then squash-merged through PR #22 at exact `main` commit `1befaf260023852ef3bee5c8abab016f464557bb`; post-merge GitHub Actions run #33 (`31681668145`) succeeded on that exact SHA. The integrated repository through Stage 7-B is therefore **main CI verified**.

Stage 7-C — bounded real baseline training run + evidence — is the **active bounded package** on branch `stage-7c-baseline-run` / draft PR #23. The run orchestrator, incremental validation decoder, progress/heartbeat surface, authoritative evidence gate, and bounded CI regressions are implemented. A guarded GitHub-hosted benchmark must prove the exact frozen workload fits the conservative runtime budget before the one authorized synthetic-only execution can start. No accepted real-run evidence exists until that job finishes and its artifacts are reviewed.

Stage 8 real-data fine-tuning, Stage 9 sealed benchmark/candidate work, and Stage 10 ScoreMosaic integration remain locked.

## Core development rule

Each stage stays isolated behind explicit contracts and validation gates. Symbolic musical ground truth remains authoritative; rendering and visual degradation are derived artifacts and must never silently modify target notation semantics.

All derivatives of one symbolic source remain in one dataset family and one train/validation/test split. Builders never validate themselves by assertion; independent validation remains a veto gate.

Training may use only Stage 5/6 validated synthetic artifacts. The Stage 6 test split remains sealed throughout Stage 7 and is reserved for the later Stage 9 benchmark decision.

Stage 7-B uses only the compact frozen semantic target surface defined by Stage 7-A. The data path re-checks persisted Stage 6 hashes and token semantic round trips before a train/validation sample becomes eligible. `DatasetSplit.TEST` is rejected by the Stage 7-B adapter and batch boundary. Accepted semantic token sequences must consume a real `EOS`; EOF without `EOS` fails closed.

Stage 7-C reuses the same from-scratch CNN/GRU baseline, tokenizer, trusted data adapter, deterministic preprocessing, optimizer/loss policy, and exact `torch==2.13.0+cpu` runtime. It adds only the bounded real-run orchestration, validation metrics, selected-checkpoint handling, and auditable provenance/evidence surface defined in `STAGE7C_RUNBOOK.md`.

The selected baseline uses no pretrained model, external OCR/OMR teacher, LLM, or network-dependent label source. Ordinary GitHub-hosted CI remains limited to bounded CPU smoke/contract evidence. Draft PR #23 has one narrow exact-head exception: a conservative benchmark may admit one synthetic-only authoritative run; unsafe estimates fail closed before training.

Large datasets, model checkpoints, real user documents, private material, and rights-unclear score collections are not normal Git repository content.
