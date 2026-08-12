# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

Stage 0 baseline is complete, with GitHub CI explicitly deferred because it is currently unavailable. No ST Music Generator implementation, dataset generation, model training, or ScoreMosaic integration has started.

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
| 1 | ST Music Generator | 🔒 Not started |
| 2 | Canonical / MusicXML validation | 🔒 Not started |
| 3 | Renderer integration | 🔒 Not started |
| 4 | Controlled degradation | 🔒 Not started |
| 5 | Dataset validation | 🔒 Not started |
| 6 | Synthetic Dataset v1 | 🔒 Not started |
| 7 | Baseline ST-OMR training | 🔒 Not started |
| 8 | Real-data fine-tuning | 🔒 Not started |
| 9 | Benchmark and candidate decision | 🔒 Not started |
| 10 | ScoreMosaic candidate integration | 🔒 Not started |

## Current branch package

Branch: `stage-0-closure-status`

Scope:

- update `STATUS.md` only
- record the successful completion of Stage 0-G
- close Stage 0 while preserving the explicit CI-deferred status
- keep Stage 1 locked and not started

Out of scope for this package:

- application code
- generator code
- dependencies
- workflows or CI
- datasets
- model files
- ScoreMosaic changes
- Guitar TAB work

## Verification status

This is a documentation-only closure package. Its required verification is a branch diff confirming that only the approved `STATUS.md` state correction is present.

GitHub CI is currently unavailable. Local verification evidence, when later produced, must not be reported as GitHub CI evidence.

## Next gate

The next planned work is Stage 1-A: define and implement the smallest canonical core model package for the ST Music Generator. Stage 1 must begin only as a separate approved package and must not include MusicXML rendering, datasets, model training, or ScoreMosaic integration.
