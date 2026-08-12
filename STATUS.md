# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

Stage 0 baseline is complete, with GitHub CI explicitly deferred because it is currently unavailable. Stage 1 ST Music Generator is closed with local verification only. Stage 2-A MusicXML contract freeze is merged. Stage 2-B is the active implementation package and adds the deterministic MusicXML 4.0 writer plus small synthetic golden fixtures. XSD validation, round-trip parsing, rendering, dataset generation, model training, and ScoreMosaic integration have not started.

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
| 2-A | MusicXML contract freeze | ✅ Complete |
| 2-B | Deterministic MusicXML 4.0 writer | 🔄 PR package in progress |
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

Branch: `stage-2b-musicxml-writer`

Scope:

- deterministic MusicXML 4.0 `score-partwise` writer using Python `xml.etree.ElementTree`;
- mandatory independent canonical `validate_score()` gate before serialization;
- fixed V1 part identity `P1` and part name `ST-OMR Synthetic`;
- exact score-wide MusicXML `divisions` computation with rational arithmetic only;
- deterministic NoteEvent, RestEvent, and ChordEvent mapping;
- pitch spelling and explicit sharp/flat/natural display-intent preservation;
- first-measure key/time/treble-clef attributes and later time-signature-change attributes;
- deterministic UTF-8 bytes and SHA-256 helper;
- six small synthetic golden MusicXML fixtures covering 2/4, 3/4, 4/4, half rest, 2/3/4-note chords, accidentals, and a time-signature change;
- focused writer tests, determinism tests, safety rejection tests, and existing regression coverage.

Explicitly out of scope:

- MusicXML XSD validation;
- MusicXML semantic parser/importer;
- schema files or schema-validation dependency;
- `.mxl` packaging;
- Verovio or renderer integration;
- image augmentation;
- dataset creation or storage;
- model files or AI training;
- real-score ingestion;
- ScoreMosaic integration;
- Guitar TAB work;
- GitHub Actions / CI changes.

## Verification status

The final Stage 2-B writer source, writer test file, and all six golden MusicXML fixtures were mirrored locally and their Git blob identities matched the GitHub branch blobs before the final run.

Local verification results:

- focused Stage 2-B writer suite: 24 tests passed;
- full available unit/regression suite: 146 tests passed;
- Python compile validation passed;
- supplemental serialization stress: 1,000 generated scores across mixed, note-only, rest-only, chord-only, 2/4, 3/4, 4/4, and accidentals-disabled configurations produced deterministic bytes/digests and parsed as well-formed `score-partwise` 4.0 XML.

This is `LOCAL VERIFIED — CI NOT AVAILABLE`. It is not XSD validation and must not be reported as MusicXML schema validation or GitHub CI evidence.

## Next gate

After Stage 2-B is reviewed and explicitly accepted for merge, the next separate package is Stage 2-C: offline official MusicXML 4.0 XSD validation plus an independent ST MusicXML semantic validator. Schema assets/dependencies require their own review. Stage 2-D round-trip verification and Stage 3 renderer integration remain locked.
