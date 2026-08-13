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
- Stage 4 — deterministic, bounded Controlled Degradation V1.

Stage 4 merged through PR #14 at `main` commit `f0fd8a732b51b4aa95a66c3a780d0cefa6661361`. GitHub Actions run `31660215130` completed successfully on that exact merged commit, so the integrated pipeline through Stage 4 is CI verified.

Stage 5-A — Dataset Contract + Independent Manifest Validator — is the active bounded package on `stage-5a-dataset-manifest-validator`. It defines immutable synthetic sample/manifest metadata, independently recomputes Stage 4 lineage identities, prevents family/target/SVG leakage across splits, rejects duplicates, and provides deterministic canonical manifest serialization. It does **not** build a bulk dataset or train a model.

## Core development rule

Each stage stays isolated behind explicit contracts and validation gates. Symbolic musical ground truth remains authoritative; rendering and visual degradation are derived artifacts and must never silently modify the target notation semantics.

All derivatives of one symbolic source must remain in one dataset family and one train/validation/test split. Builders never validate themselves by assertion; independent validation remains a veto gate.

Large datasets, model checkpoints, real user documents, private material, and rights-unclear score collections are not normal Git repository content.
