# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

Stage 0 baseline is complete, with GitHub CI explicitly deferred because it is currently unavailable. Stage 1-A is the active package and introduces only the immutable canonical core model primitives for the ST Music Generator. No MusicXML writer, renderer, dataset generation, model training, or ScoreMosaic integration has started.

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
| 1 | ST Music Generator | 🔄 In progress |
| 1-A | Canonical core model | 🔄 PR package in progress |
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

Branch: `stage-1a-canonical-core-model`

Scope:

- immutable `RationalDuration`
- structured `Pitch`
- `DisplayAccidental` and `NotationIntent`
- immutable `NoteEvent`
- immutable `RestEvent`
- immutable `ChordEvent`
- Stage 1-A unit, negative, immutability, equality, and hash tests
- update this status file to reflect the active package

Explicitly out of scope:

- Score / Measure generation
- random or procedural music generation
- MusicXML writer or parser
- Verovio or any renderer
- image augmentation
- datasets or model files
- AI/model training
- ScoreMosaic integration
- Guitar TAB work
- dependencies or GitHub Actions

## Verification status

The Stage 1-A implementation has been exercised locally with the standard-library unittest suite: 28 tests passed. Python compile validation also passed. GitHub CI is currently unavailable, so this evidence must be reported as `LOCAL VERIFIED — CI NOT AVAILABLE`, never as CI evidence.

The final package still requires branch diff review and PR review before merge.

## Next gate

After Stage 1-A is reviewed and explicitly accepted for merge, the next package is Stage 1-B: independent validation of canonical core objects and higher-level invariants. MusicXML rendering, datasets, model training, and ScoreMosaic integration remain out of scope.
