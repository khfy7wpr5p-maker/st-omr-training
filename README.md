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
- Stage 6 — deterministic Synthetic Dataset v1 construction and hash-addressed local persistence.

Stage 6 merged through PR #17 at `main` commit `7c3c736e6d3755d1bd098e2874d73ce5ed41e39f`, and its post-merge GitHub Actions run #18 (`31674014666`) succeeded. Stage 6 closure documentation then merged through PR #18 at `main` commit `046c9a4e7e41e94b0b4465a2610f30361055a3ed`; post-merge run #20 (`31674836433`) also succeeded. The integrated pipeline through Stage 6 is therefore **CI verified**.

Stage 7-A — Baseline Training Contract Freeze — is the active bounded package. It freezes the only allowed Stage 7 data inputs, strict family/split isolation, compact semantic target-token representation, tokenizer round-trip requirements, the bounded from-scratch baseline model interface, reproducibility/provenance requirements, resource ceilings, validation metrics, and the verification gate before implementation.

Stage 7-A adds no model or tokenizer code and runs no training. Stage 7-B implementation and Stage 7-C baseline training remain locked until separately approved. Stage 8 real-data fine-tuning, Stage 9 sealed benchmark/candidate work, and Stage 10 ScoreMosaic integration also remain locked.

## Core development rule

Each stage stays isolated behind explicit contracts and validation gates. Symbolic musical ground truth remains authoritative; rendering and visual degradation are derived artifacts and must never silently modify target notation semantics.

All derivatives of one symbolic source remain in one dataset family and one train/validation/test split. Builders never validate themselves by assertion; independent validation remains a veto gate.

Training may use only Stage 5/6 validated synthetic artifacts. The Stage 6 test split remains sealed throughout Stage 7 and is reserved for the later Stage 9 benchmark decision.

Large datasets, model checkpoints, real user documents, private material, and rights-unclear score collections are not normal Git repository content.