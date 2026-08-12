# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

Stage 0 baseline is complete, with GitHub CI explicitly deferred because it is currently unavailable. Stage 1-A canonical core model and Stage 1-B independent V1 validator are merged. Stage 1-C is the active package and adds the immutable Score → Part → Measure → Voice structure plus independent V1 structure validation. No procedural generator, MusicXML writer, renderer, dataset generation, model training, or ScoreMosaic integration has started.

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
| 1-A | Canonical core model | ✅ Complete |
| 1-B | Independent V1 validator | ✅ Complete |
| 1-C | Score structure model and validator | 🔄 PR package in progress |
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

Branch: `stage-1c-score-structure`

Scope:

- immutable `TimeSignature` with V1 2/4, 3/4, and 4/4 capacities
- immutable `Voice`, `Measure`, `Part`, and `Score`
- explicit V1 treble clef, key signature 0, one voice, one staff, and one part policy
- exact rational `expected_duration`
- independent score-structure validation
- event ordering, overlap, gap, measure underflow, and measure overflow checks
- V1 note/chord/rest duration-set checks
- explicit-rest requirement for silence inside a complete measure
- sequential measure-number validation
- score reproduction metadata fields for later deterministic generation
- Stage 1-C positive, negative, immutability, timing, chord-timeline, and deterministic issue-order tests

Explicitly out of scope:

- random or procedural music generation
- pickup/anacrusis measures
- full-measure-rest notation semantics
- multiple voices or staffs
- non-zero key signatures
- clefs other than treble
- MusicXML writer or parser
- Verovio or any renderer
- image augmentation
- datasets or model files
- AI/model training
- ScoreMosaic integration
- Guitar TAB work
- dependencies or GitHub Actions

## Verification status

The Stage 1-C candidate was exercised locally with the standard-library unittest suite together with the existing Stage 1-A and Stage 1-B regression coverage: 101 tests passed. Python compile validation also passed. GitHub CI remains unavailable, so this evidence is `LOCAL VERIFIED — CI NOT AVAILABLE`, not CI evidence.

The final package still requires branch diff review and PR review before merge.

## Next gate

After Stage 1-C is reviewed and explicitly accepted for merge, the next planned package is Stage 1-D: deterministic ST Music Generator v1. MusicXML rendering, datasets, model training, and ScoreMosaic integration must not begin implicitly.
