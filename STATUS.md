# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

Stage 0 baseline is complete, with GitHub CI explicitly deferred because it is currently unavailable. Stage 1 ST Music Generator is closed with local verification only. Stage 2-A MusicXML contract freeze and Stage 2-B deterministic MusicXML 4.0 writer are merged. Stage 2-C is the active implementation package: pinned offline official MusicXML 4.0 XSD assets plus an independent fail-closed ST-OMR V1 semantic validator. Stage 2-D round-trip parsing, rendering, dataset generation, model training, and ScoreMosaic integration have not started.

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
| 2-C | Offline XSD + independent MusicXML validator | 🔄 PR package in progress |
| 2-D | Supported-V1 semantic round-trip verifier | 🔒 Not started |
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

Stage 2-B merged through PR #8 at main commit `1940d43b3986e5fe359aa79b86cc2af26e96fe98`. Its pre-merge evidence was 24 focused writer tests, 146 full available regression tests, Python compile validation, and 1,000 deterministic serialization stress cases. That Stage 2-B evidence was not XSD validation.

GitHub CI remains unavailable. No stage in this repository may be reported as CI-verified unless GitHub-hosted evidence actually exists.

## Current branch package

Branch: `stage-2c-musicxml-validation`

Scope:

- official MusicXML 4.0 `musicxml.xsd`, `xlink.xsd`, `xml.xsd`, and `catalog.xml` vendored from the pinned W3C MusicXML source commit;
- explicit SHA-256 integrity manifest and runtime integrity gate for the exact schema bytes;
- exact `lxml==6.1.1` dependency pin;
- fail-closed XML parser configuration with entity resolution, DTD loading/validation, network access, recovery mode, and huge-tree mode disabled;
- offline XSD compilation with an allowlisted resolver restricted to the two required MusicXML schema imports;
- independent ST-OMR V1 semantic validation for root/version, P1 identity, measure numbering, attributes, time/key/clef rules, durations/types, voice/staff, pitch/accidental coherence, chord rules, supported elements, and exact measure fill;
- explicit rejection of DOCTYPE/XXE surfaces, malformed/oversized input, schema tampering, unknown external schema imports, unsupported namespaces/elements, and non-V1 musical structure;
- real-schema integration checks against all six Stage 2-B golden fixtures and deterministic generated writer output.

Explicitly out of scope:

- general MusicXML import or normalization;
- Stage 2-D semantic round-trip verification;
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

The final Stage 2-C validator source, committed focused validator test file, real-schema integration test file, and the four vendored schema assets were mirrored locally. Their Git blob identities were checked against the GitHub branch; the four schema blobs also match the pinned official W3C MusicXML source blobs.

Local verification results:

- focused Stage 2-C validator suites: 40 tests passed;
- full available unit/regression suite: 186 tests passed;
- Python compile validation passed;
- real official XSD + semantic stress: 300 generated scores across mixed, note-only, rest-only, chord-only, fixed 2/4, and accidentals-disabled configurations passed all gates;
- supplemental security-negative harness: 7 attack/fail-closed cases passed, including DOCTYPE/XXE rejection, malformed/oversized input, wrong namespace/root, unknown external schema import refusal, and schema tamper detection.

This is `LOCAL VERIFIED — CI NOT AVAILABLE`.

## Next gate

Stage 2-C must be reviewed as one bounded PR package. Merge requires separate explicit approval. Stage 2-D remains locked until Stage 2-C is merged and post-merge state is verified.
