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
- [Stage 8-1 quarantine/intake byte-validation contract](STAGE8_1_INTAKE_CONTRACT.md)
- [Stage 8-2 paired experiment run profile](STAGE8_2_RUN_PROFILE.md)
- [Stage 8-3A pilot preparation/admission contract](STAGE8_3A_PREPARATION_CONTRACT.md)
- [Future real-test sealing boundary](STAGE8_TEST_SEALING_BOUNDARY.md)
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
- Stage 8-0 — Real Data & Fine-Tuning Contract Freeze;
- Stage 8-1 — Quarantine / Intake + Byte-Level Validation;
- Stage 8-2 — Paired Experiment Run Profile Freeze.

Stage 7-C merged through PR #23 from exact source head `7a993304218fa19609ea512665148dac3eea503a` to exact `main` commit `2c2c478eb361fa90a3bccd819b623680eb12de0b`. Its closure synchronization merged through PR #24 to `main` `e56fad04e43bc6302a8cda60c6f382e83a23d734`; post-merge run #70 (`31693998756`) succeeded. Stage 7 is therefore **closed and main CI verified**.

Stage 8-0 merged through PR #25 to exact `main` commit `86487a4c3c41264b02bd159cd647a1318d9b9b88`. Post-merge GitHub Actions run #75 (`31698691405`) succeeded on that exact SHA. Its closure synchronization merged through PR #26 to exact `main` `99ffaf41f9c8919827ca97edc1bc3900db29eea2`; post-merge run #77 (`31699497562`) succeeded. Stage 8-0 is therefore **closed and main CI verified**.

Stage 8-1 implementation merged through PR #29 from final source head `ed4113a25f6e12055b9277f959a4580259d37d40` to exact `main` commit `d551cb27e0244477379100c45e06193ea7ca0cf8`; post-merge run #92 (`31704137450`) succeeded with **394/394 tests**. Closure synchronization PR #30 then merged to exact `main` `de5d66d30c66c97c03bc1dc60fce094a9f0d64e7`; post-merge run #94 (`31705382162`) succeeded. Stage 8-1 is therefore **closed and main CI verified**.

Stage 8-2 implementation merged through PR #31 from exact source head `f0ae7ff6d7206789cbe543b75664e557368233b7` to exact `main` commit `ba93736efa1411bf49020e52ddb688d63f7876c1`. Closure synchronization PR #32 merged to exact `main` `99a32ef917ba4ba5c72ef6a537c24c90a1b0c47f`; exact-main GitHub Actions run #104 (`31712294207`) succeeded. Stage 8-2 is therefore **closed and main CI verified**.

Stage 8-3A — Pilot Data Preparation + Admission — is now the active package. Its first implementation freezes deterministic PNG-source→8-bit-grayscale-training-PNG preparation and fail-closed auxiliary-package triage. Real files remain external; no real bytes are committed. MEI/semantic/agnostic annotations remain auxiliary evidence and cannot bypass the existing supported-V1 MusicXML, rights, pairing, duplicate, leakage, or Stage 8-1 receipt gates.

The closed Stage 8-2 profile still freezes the first real-data pilot as **50 admitted development pairs: 40 train + 10 validation**, with Candidate A (exact Stage 7-C checkpoint fine-tuning) and Candidate B (same architecture from scratch) bound to the same admitted manifest, Stage 8-1 receipt set, sealed-test commitment, model/tokenizer/preprocess/trainer fingerprints, deterministic data order, metric family, and bounded CPU/time budget.

The first 50 raw files are not automatically 50 training samples. A pair counts only after Stage 8-0 rights/pairing admission and Stage 8-1 byte/semantic/leakage validation. Rejected pairs must be replaced before the exact 40/10 pilot manifest can be frozen.

For the first Stage 8-3A pilot, source images are intentionally restricted to existing PNG bytes. The deterministic derivative policy performs no crop, resize, rotation, or geometry normalization; it converts accepted single-frame non-transparent source PNG modes to mode `L`, strips source metadata, writes a fixed PNG encoding, and re-verifies the exact Stage 8-1 training-image structure. PDF/JPEG preparation remains outside this pilot rather than being improvised.

Both held-out test partitions remain sealed. Stage 8-3B execution, Stage 9 benchmark/candidate work, and Stage 10 ScoreMosaic integration remain locked.

The Stage 7-C checkpoint artifact `9177923796` remains temporary and scheduled to expire on 2026-09-12. Stage 8-3A does not load, move, copy, publish, or preserve it; if exact Candidate A bytes are unavailable or hash/state mismatched at execution time, Candidate A is blocked rather than substituted.

The Stage 7-C model remains a trainability baseline rather than a production candidate: exact sequence accuracy is 0% and token error rate is approximately 80.5% on its validation evidence.

## Core development rule

Each stage stays isolated behind explicit contracts and validation gates. Symbolic musical ground truth remains authoritative; rendering and visual degradation are derived artifacts and must never silently modify target notation semantics.

All derivatives of one symbolic or real-source family remain in one dataset family and one train/validation/test split. Builders never validate themselves by assertion; independent validation remains a veto gate.

Stage 7 training used only Stage 5/6 validated synthetic artifacts. Any later Stage 8 real data must pass the closed Stage 8-0 metadata contract and the closed Stage 8-1 byte-level handoff before a later train/validation loader may see it. The closed Stage 8-2 profile additionally requires both paired candidates to use the exact same 40/10 admitted development manifest and receipt-set identity. Stage 8-3A may prepare and triage external bytes but cannot weaken those gates. Both synthetic and real held-out test partitions remain sealed until the Stage 9 benchmark gate.

ScoreMosaic user uploads and teacher corrections are not automatic training data. User-derived material requires separate explicit training permission and the full quarantine/admission process. There is no online or automatic learning path, and no Stage 8 output may enter ScoreMosaic before Stage 9 candidate-quality evidence and the later Stage 10 integration gate.

Large datasets, model checkpoints, real user documents, private material, and rights-unclear score collections are not normal Git repository content.
