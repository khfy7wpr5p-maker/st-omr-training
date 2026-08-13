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

Stage 7-B merged through PR #21 at exact `main` commit `d02dce4ee17dfccf6f05519ab0970fdc188d0147`. Post-merge GitHub Actions run #31 (`31679810478`) succeeded on that exact commit with **336/336 tests**, exact pinned runtime checks, `pip check`, the missing-`EOS` regression, deterministic CPU smoke evidence, and `compileall`. The integrated repository through the Stage 7-B boundary is therefore **CI verified**.

Stage 7-C — bounded real baseline training run + evidence — is **next but has not started**. Stage 8 real-data fine-tuning, Stage 9 sealed benchmark/candidate work, and Stage 10 ScoreMosaic integration remain locked.

## Core development rule

Each stage stays isolated behind explicit contracts and validation gates. Symbolic musical ground truth remains authoritative; rendering and visual degradation are derived artifacts and must never silently modify target notation semantics.

All derivatives of one symbolic source remain in one dataset family and one train/validation/test split. Builders never validate themselves by assertion; independent validation remains a veto gate.

Training may use only Stage 5/6 validated synthetic artifacts. The Stage 6 test split remains sealed throughout Stage 7 and is reserved for the later Stage 9 benchmark decision.

Stage 7-B uses only the compact frozen semantic target surface defined by Stage 7-A. The data path re-checks persisted Stage 6 hashes and token semantic round trips before a train/validation sample becomes eligible. `DatasetSplit.TEST` is rejected by the Stage 7-B adapter and batch boundary. Accepted semantic token sequences must consume a real `EOS`; EOF without `EOS` fails closed.

The selected baseline is initialized from scratch and uses no pretrained model, external OCR/OMR teacher, LLM, or network-dependent label source. GitHub-hosted CI is limited to bounded CPU smoke evidence; the real Stage 7-C baseline run remains a separate approval gate and is not started by this documentation sync.

Large datasets, model checkpoints, real user documents, private material, and rights-unclear score collections are not normal Git repository content.
