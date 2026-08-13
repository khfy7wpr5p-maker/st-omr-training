# st-omr-training

Safe training and synthetic-data laboratory for ST-OMR: canonical music generation, MusicXML serialization and validation, deterministic notation rendering, controlled degradation, dataset validation/construction, training, and evaluation.

This repository is isolated from the ScoreMosaic production runtime. Its purpose is to build traceable OMR training data, train candidate ST-OMR models, and evaluate them before any later integration decision.

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
- [Stage 7-C accepted evidence](STAGE7C_EVIDENCE.md)
- [Stage 8-0 real-data and fine-tuning contract](STAGE8_REAL_DATA_CONTRACT.md)
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
- Stage 7-B — deterministic tokenizer/data/model/trainer smoke implementation;
- Stage 7-C — bounded synthetic-only baseline training run and accepted evidence;
- Stage 8-0 — Real Data & Fine-Tuning Contract Freeze.

Stage 7-C merged through PR #23 from exact source head `7a993304218fa19609ea512665148dac3eea503a` to exact `main` commit `2c2c478eb361fa90a3bccd819b623680eb12de0b`. Its closure synchronization merged through PR #24 to `main` `e56fad04e43bc6302a8cda60c6f382e83a23d734`; post-merge run #70 (`31693998756`) succeeded. Stage 7 is therefore **closed and main CI verified**.

Stage 8-0 merged through PR #25 to exact `main` commit `86487a4c3c41264b02bd159cd647a1318d9b9b88`. Post-merge GitHub Actions run #75 (`31698691405`) succeeded on that exact SHA with pinned-runtime verification, `pip check`, the complete repository test suite, and `compileall`. Stage 8-0 is therefore **closed and main CI verified**.

Stage 8-1+ real-data intake/execution, Stage 9 sealed benchmark/candidate work, and Stage 10 ScoreMosaic integration remain locked. Stage 8-0 itself did not ingest real data, load a checkpoint, run fine-tuning, or open either held-out test partition.

The Stage 7-C checkpoint artifact `9177923796` is temporary and is scheduled to expire on 2026-09-12. The exact checkpoint identity is the recorded SHA-256, not the artifact location. Stage 8-0 did not move or republish it.

The Stage 7-C model remains a trainability baseline rather than a production candidate: exact sequence accuracy is 0% and token error rate is approximately 80.5% on its validation evidence.

## Core development rule

Each stage stays isolated behind explicit contracts and validation gates. Symbolic musical ground truth remains authoritative; rendering and visual degradation are derived artifacts and must never silently modify target notation semantics.

All derivatives of one symbolic or real-source family remain in one dataset family and one train/validation/test split. Builders never validate themselves by assertion; independent validation remains a veto gate.

Stage 7 training used only Stage 5/6 validated synthetic artifacts. Any later Stage 8 real data must pass the closed Stage 8-0 contract before a train/validation loader may see it. Both synthetic and real held-out test partitions remain sealed until the Stage 9 benchmark gate.

ScoreMosaic user uploads and teacher corrections are not automatic training data. User-derived material requires separate explicit training permission and the full quarantine/admission process. There is no online or automatic learning path, and no Stage 8 output may enter ScoreMosaic before Stage 9 candidate-quality evidence and the later Stage 10 integration gate.

Large datasets, model checkpoints, real user documents, private material, and rights-unclear score collections are not normal Git repository content.
