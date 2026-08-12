# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

The project is in Stage 0 baseline setup. No ST Music Generator implementation, dataset generation, model training, or ScoreMosaic integration has started.

## Stage status

| Stage | Description | Status |
|---|---|---|
| 0-A | Repository baseline | ✅ Complete |
| 0-B | Architecture and data boundaries | ✅ Complete |
| 0-C | Canonical data contract | ✅ Complete |
| 0-D | Local verification strategy | ✅ Complete |
| 0-E | GitHub CI | ⏸ Deferred / unavailable |
| 0-F | Architecture consistency audit | ✅ Complete |
| 0-G | Baseline documentation | 🔄 In progress on `stage-0g-baseline-docs` |
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

Branch: `stage-0g-baseline-docs`

Scope:

- `ARCHITECTURE.md`
- `DATA_CONTRACT.md`
- `SAFETY.md`
- `STATUS.md`
- limited `README.md` navigation update

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

Stage 0-G documentation is not complete until the branch diff is reviewed and confirmed to contain only the approved documentation scope.

GitHub CI is currently unavailable. Local verification evidence, when later produced, must not be reported as GitHub CI evidence.

## Next gate

After Stage 0-G documentation is reviewed and explicitly accepted, the next planned work is to define the first small Stage 1 ST Music Generator implementation package. Stage 1 must not begin implicitly as part of Stage 0-G.
