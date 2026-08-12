# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

Stage 0 baseline is complete, with GitHub CI explicitly deferred because it is currently unavailable. Stage 1 ST Music Generator is closed with local verification only. Stage 2-A MusicXML contract freeze, Stage 2-B deterministic MusicXML 4.0 writer, and Stage 2-C offline XSD plus independent semantic validation are merged. Stage 2-D is the active package: a supported-V1 semantic round-trip verifier. Rendering, dataset generation, model training, and ScoreMosaic integration have not started.

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
| 2-B | Deterministic MusicXML 4.0 writer | ✅ Complete |
| 2-C | Offline XSD + independent MusicXML validator | ✅ Complete |
| 2-D | Supported-V1 semantic round-trip verifier | 🔄 PR package in progress |
| 3 | Renderer integration | 🔒 Not started |
| 4 | Controlled degradation | 🔒 Not started |
| 5 | Dataset validation | 🔒 Not started |
| 6 | Synthetic Dataset v1 | 🔒 Not started |
| 7 | Baseline ST-OMR training | 🔒 Not started |
| 8 | Real-data fine-tuning | 🔒 Not started |
| 9 | Benchmark and candidate decision | 🔒 Not started |
| 10 | ScoreMosaic candidate integration | 🔒 Not started |

## Completed evidence

Stage 1-D merged through PR #6 at main commit `4d10aac736ddb41191f1ed55a868932ea479fd9d` with recorded local verification only.

Stage 2-B merged through PR #8 at main commit `1940d43b3986e5fe359aa79b86cc2af26e96fe98`. Its pre-merge evidence was 24 focused writer tests, 146 full available regression tests, Python compile validation, and 1,000 deterministic serialization stress cases.

Stage 2-C merged through PR #9 at main commit `81cbfdb2958b8b8b8f4ee5cbd50960a7a75049f0`. Its pre-merge evidence was 40 focused validation tests, 186 full available regression tests, Python compile validation, 300 real official-XSD plus semantic stress cases, and 7 fail-closed security-negative cases. The official MusicXML 4.0 schema assets are pinned and integrity-checked offline.

GitHub CI remains unavailable. No stage in this repository may be reported as CI-verified unless GitHub-hosted evidence actually exists.

## Current branch package

Branch: `stage-2d-semantic-roundtrip`

Scope:

- immutable semantic projection types for the frozen supported-V1 comparison surface;
- canonical `Score` semantic projection that excludes generator-only provenance such as score ID, seed, generator version, and config/source metadata;
- limited MusicXML parser that accepts only documents which already pass the independent Stage 2-C XSD and semantic gates;
- exact whole-note `Fraction` reconstruction from MusicXML divisions and duration values;
- effective per-measure time-signature projection, including time changes;
- note, rest, and chord reconstruction with chord-member order preserved and chord continuations not advancing musical position;
- pitch step/alter/octave and visible sharp/flat/natural intent projection;
- independent field-by-field semantic comparator for part, measure, voice, staff, event timing/type, chord membership, pitch, and accidental intent;
- end-to-end verifier: canonical Score → Stage 2-B writer → Stage 2-C validation → limited parser → semantic comparison;
- fail-closed rejection of unsupported/noncanonical MusicXML rather than silent normalization.

Explicitly out of scope:

- general-purpose MusicXML import;
- reconstruction of generator-only provenance or score identity;
- `.mxl` packaging;
- MusicXML rewriting or normalization;
- Verovio or renderer integration;
- image augmentation;
- dataset creation or storage;
- model files or AI training;
- real-score ingestion;
- ScoreMosaic integration;
- Guitar TAB work;
- GitHub Actions / CI changes.

## Verification status

Local Stage 2-D verification results:

- focused Stage 2-D round-trip suite: 21 tests passed;
- full available unit/regression suite: 207 tests passed;
- Python compile validation passed;
- supplemental semantic round-trip stress: 600 generated scores across mixed, note-only, rest-only, chord-only, fixed 2/4, and accidentals-disabled configurations passed writer determinism, Stage 2-C validation, limited parsing, and exact semantic projection comparison.

This is `LOCAL VERIFIED — CI NOT AVAILABLE`.

## Next gate

Stage 2-D must be reviewed as one bounded PR package. Merge requires separate explicit approval. Stage 3 renderer integration remains locked until Stage 2-D is merged and post-merge state is verified.
