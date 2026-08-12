# st-omr-training

Safe training and synthetic-data laboratory for ST-OMR: canonical music generation, MusicXML serialization and validation, deterministic notation rendering, controlled degradation, dataset construction, training, and evaluation.

This repository is isolated from the ScoreMosaic production runtime. Its purpose is to build traceable synthetic OMR training data, train candidate ST-OMR models, and evaluate them before any later integration decision.

## Project documents

- [Architecture](ARCHITECTURE.md)
- [Canonical data contract](DATA_CONTRACT.md)
- [MusicXML contract](MUSICXML_CONTRACT.md)
- [Renderer contract](RENDERER_CONTRACT.md)
- [Controlled degradation contract](DEGRADATION_CONTRACT.md)
- [Verovio runtime evidence](VEROVIO_RUNTIME_EVIDENCE.md)
- [Safety and verification rules](SAFETY.md)
- [Current project status](STATUS.md)

## Current phase

The repository is public and GitHub Actions CI is active.

Completed and merged:

- Stage 0 — safety and architecture baseline;
- Stage 1 — deterministic ST Music Generator V1;
- Stage 2 — deterministic MusicXML 4.0 pipeline, offline XSD/semantic validation, and supported-V1 semantic round trip;
- Stage 3 — deterministic Verovio 6.2.1 SVG renderer adapter.

The exact current `main` commit `23739ddfab618a0406836e94bb0ced1a124f8886` passed GitHub-hosted post-merge CI in run `31648164533`, including pinned dependency verification, the full unittest suite with real Verovio runtime tests, and Python compile validation. The integrated state through Stage 3 is therefore CI verified.

Stage 4 — Controlled Degradation — is the active bounded development package on `stage-4-controlled-degradation`. V1 adds deterministic clean SVG-to-grayscale-PNG rasterization plus conservative, replayable appearance degradation behind a separate contract. Stage 5 Dataset Validation remains locked until Stage 4 passes its own PR CI, merge approval, and post-merge `main` CI.

## Core development rule

Each stage stays isolated behind explicit contracts and validation gates. Symbolic musical ground truth remains authoritative; rendering and visual degradation are derived artifacts and must never silently modify the target notation semantics.

Large datasets, model checkpoints, real user documents, private material, and rights-unclear score collections are not normal Git repository content.
