# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

Stage 0 baseline is complete, with GitHub CI explicitly deferred because it is currently unavailable. Stage 1 ST Music Generator is complete through the merged deterministic generator package. Stage 2-A is the active documentation/architecture package and freezes the MusicXML boundary before any MusicXML implementation begins. No MusicXML writer, renderer, dataset generation, model training, or ScoreMosaic integration has started.

## Stage status

| Stage | Description | Status |
|---|---|---|
| 0-A | Repository baseline | ✅ Complete |
| 0-B | Architecture and data boundaries | ✅ Complete |
| 0-C | Canonical data contract | ✅ Complete |
| 0-D | Local verification strategy | ✅ Complete |
| 0-E | GitHub CI | ⏸ Deferred / unavailable |
| 0-F | Architecture consistency audit | ✅ Complete |
| 0-G | Baseline documentation | ✅ Complete |
| 0 | Safety and architecture baseline | ✅ Closed with CI deferred |
| 1-A | Canonical core model | ✅ Complete |
| 1-B | Independent V1 validator | ✅ Complete |
| 1-C | Score structure model and validator | ✅ Complete |
| 1-D | Deterministic ST Music Generator v1 | ✅ Complete |
| 1 | ST Music Generator | ✅ Closed — local verification only |
| 2-A | MusicXML contract freeze | 🔄 PR package in progress |
| 2-B | Deterministic MusicXML 4.0 writer | 🔒 Not started |
| 2-C | Offline XSD + independent MusicXML validator | 🔒 Not started |
| 2-D | Supported-V1 semantic round-trip verifier | 🔒 Not started |
| 3 | Renderer integration | 🔒 Not started |
| 4 | Controlled degradation | 🔒 Not started |
| 5 | Dataset validation | 🔒 Not started |
| 6 | Synthetic Dataset v1 | 🔒 Not started |
| 7 | Baseline ST-OMR training | 🔒 Not started |
| 8 | Real-data fine-tuning | 🔒 Not started |
| 9 | Benchmark and candidate decision | 🔒 Not started |
| 10 | ScoreMosaic candidate integration | 🔒 Not started |

## Completed Stage 1 evidence

Stage 1-D merged through PR #6 at main commit `4d10aac736ddb41191f1ed55a868932ea479fd9d`.

Recorded local evidence before merge:

- 122 unit/regression tests passed;
- Python compile validation passed;
- 2,000 generated scores across mixed and constrained configurations passed independent `validate_score()` validation;
- cross-process checks with different `PYTHONHASHSEED` values produced the same deterministic score representation digest for the same generator version, config, and seed.

GitHub CI was unavailable, so Stage 1 closure means **local verification only**. It must never be reported as CI-verified.

## Current branch package

Branch: `stage-2a-musicxml-contract`

Scope:

- close Stage 1 status after the merged Stage 1-D generator;
- add the frozen V1 MusicXML contract;
- pin MusicXML 4.0 `score-partwise` for V1;
- freeze exact rational divisions conversion;
- freeze note/rest/chord/accidental mapping;
- freeze deterministic XML construction rules;
- define offline official W3C MusicXML 4.0 XSD validation strategy;
- define independent MusicXML semantic validation requirements;
- define golden fixture requirements;
- define supported-V1 semantic round-trip boundary;
- split Stage 2 into 2-A / 2-B / 2-C / 2-D before implementation.

Explicitly out of scope:

- MusicXML writer implementation;
- MusicXML parser implementation;
- schema files or schema-validation dependency;
- Verovio or renderer integration;
- image augmentation;
- dataset creation or storage;
- model files or AI training;
- real-score ingestion;
- ScoreMosaic integration;
- Guitar TAB work;
- GitHub Actions / CI changes.

## Verification status

Stage 2-A is documentation and architecture only. The branch must be reviewed against `main` and confirmed to contain only the approved contract/status/navigation changes.

The MusicXML version decision is pinned to MusicXML 4.0 for V1. A future MusicXML 4.1 or later release must not be adopted automatically; it requires a separate compatibility decision.

GitHub CI remains unavailable. No CI verification claim is permitted.

## Next gate

After Stage 2-A is reviewed and explicitly accepted for merge, the next implementation package is Stage 2-B: deterministic MusicXML 4.0 writer plus small golden fixtures. Stage 2-C XSD/semantic validation and Stage 2-D round-trip verification remain separate packages. Renderer, dataset, model-training, and ScoreMosaic work remain locked.
