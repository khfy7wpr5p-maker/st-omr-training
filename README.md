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
- Stage 5 — synthetic dataset contract and independent manifest validation.

Stage 5-A merged through PR #15 at `main` commit `d677f3d27ac710c56c5ce677a46dc62bcf77bd84`. GitHub Actions run `31671919885` completed successfully on that exact merged commit. Stage 5 closure documentation merged through PR #16 at `main` commit `ed343d4984aae4507a2dd3238cfd1a98fb25b4b7`, and post-merge run `31672540732` also succeeded. The integrated pipeline through Stage 5 is therefore **CI verified**.

Stage 6 — Synthetic Dataset v1 — is the active bounded package on `stage-6-synthetic-dataset-v1`. It deterministically plans synthetic score families, assigns complete families to 80/10/10 train/validation/test splits, composes the existing Generator → MusicXML → Verovio → Controlled Degradation pipeline, requires the independent Stage 5 manifest veto gate, and can persist validated artifacts in a no-overwrite hash-addressed local directory layout. Bulk generated datasets remain outside normal Git content.

Model architecture and training remain locked for Stage 7.

## Core development rule

Each stage stays isolated behind explicit contracts and validation gates. Symbolic musical ground truth remains authoritative; rendering and visual degradation are derived artifacts and must never silently modify the target notation semantics.

All derivatives of one symbolic source must remain in one dataset family and one train/validation/test split. Builders never validate themselves by assertion; independent validation remains a veto gate.

Large datasets, model checkpoints, real user documents, private material, and rights-unclear score collections are not normal Git repository content.
