# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

Stage 0 baseline is complete, with GitHub CI explicitly deferred because it is currently unavailable. Stage 1-A canonical core model is merged. Stage 1-B is the active package and adds an independent V1 validator for the canonical core objects. No MusicXML writer, renderer, dataset generation, model training, or ScoreMosaic integration has started.

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
| 1-B | Independent V1 validator | 🔄 PR package in progress |
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

Branch: `stage-1b-independent-validator`

Scope:

- immutable `ValidationIssue`
- immutable `ValidationResult`
- independent validation for `NoteEvent`
- independent validation for `RestEvent`
- independent validation for `ChordEvent`
- generic `validate_v1_event()` dispatch
- V1 voice/staff policy enforcement
- exact/canonical duration and onset checks
- V1 pitch checks
- chord size, duplicate-pitch, and member consistency checks
- visible accidental / pitch-alter coherence checks
- negative tests that deliberately corrupt otherwise immutable core objects to prove the validator does not trust constructor success

Explicitly out of scope:

- Score / Part / Measure / Voice generation
- random or procedural music generation
- measure-level accidental-state interpretation
- MusicXML writer or parser
- Verovio or any renderer
- image augmentation
- datasets or model files
- AI/model training
- ScoreMosaic integration
- Guitar TAB work
- dependencies or GitHub Actions

## Verification status

The exact Stage 1-B code and tests were mirrored locally and exercised with the standard-library unittest suite: 62 tests passed in total, including the existing Stage 1-A regression suite. Python compile validation also passed. GitHub CI remains unavailable, so this evidence is `LOCAL VERIFIED — CI NOT AVAILABLE`, not CI evidence.

The final package still requires branch diff review and PR review before merge.

## Next gate

After Stage 1-B is reviewed and explicitly accepted for merge, the next Stage 1 package will be scoped separately. MusicXML rendering, datasets, model training, and ScoreMosaic integration must not begin implicitly.
