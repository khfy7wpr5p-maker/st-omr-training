# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

Stage 0 baseline is complete, with GitHub CI explicitly deferred because it is currently unavailable. Stage 1-A canonical core model, Stage 1-B independent V1 validator, and Stage 1-C score-structure model/validator are merged. Stage 1-D is the active package and adds the deterministic ST Music Generator v1. No MusicXML writer, renderer, dataset generation, model training, or ScoreMosaic integration has started.

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
| 1-C | Score structure model and validator | ✅ Complete |
| 1-D | Deterministic ST Music Generator v1 | 🔄 PR package in progress |
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

Branch: `stage-1d-deterministic-generator`

Scope:

- immutable `GeneratorConfig`
- deterministic SHA-256 counter-based pseudo-random selection independent of Python `random` implementation changes
- same generator version + normalized config + seed produces the same canonical `Score`
- deterministic score identity and stable provenance without runtime timestamps
- V1 generation for 2/4, 3/4, and 4/4
- exact measure filling with rational eighth, quarter, half, and where valid whole-note durations
- eighth, quarter, and half rhythmic rests only; full-measure-rest notation remains deferred
- single notes and 2–4 note chords
- default staff-oriented generator pitch policy with configurable V1-safe steps/octaves inside a restricted octave range
- controlled sharp, flat, and context-dependent natural display intent within each measure
- one part, one staff, one voice, treble clef, key signature 0
- mandatory independent `validate_score()` gate before generated output is accepted
- configuration, determinism, provenance, accidental-state, chord, rest, whole-note, exact-fill, and hard-validator-gate tests

Explicitly out of scope:

- MusicXML writer or parser
- canonical MusicXML schema validation
- Verovio or any renderer
- image augmentation
- dataset creation or storage
- model files or AI training
- real-score ingestion
- ScoreMosaic integration
- Guitar TAB work
- dependencies or GitHub Actions

## Verification status

The Stage 1-D candidate and the existing Stage 1-A, Stage 1-B, and Stage 1-C regression suite were exercised locally with the standard-library unittest runner: 122 tests passed. Python compile validation also passed. All current Python implementation and test files in the local verification mirror matched their GitHub branch blob identities before the final run.

As supplemental stress evidence, 2,000 generated scores across mixed, note-only, rest-only, chord-only, each supported time signature, and accidentals-disabled configurations passed independent `validate_score()` validation. A cross-process check with different `PYTHONHASHSEED` values also produced the same deterministic score representation digest for the same generator version, config, and seed.

GitHub CI remains unavailable, so this evidence is `LOCAL VERIFIED — CI NOT AVAILABLE`, not CI evidence.

The final package still requires branch diff review and PR review before merge.

## Next gate

After Stage 1-D is reviewed and explicitly accepted for merge, Stage 1 can be closed and Stage 2 can be scoped. MusicXML serialization/validation must not begin implicitly, and renderer, dataset, model-training, and ScoreMosaic work remain locked.
