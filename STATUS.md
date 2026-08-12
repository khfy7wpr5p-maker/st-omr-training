# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

Stage 0 baseline is complete, with GitHub CI explicitly deferred because it is currently unavailable. Stage 1 ST Music Generator is closed with local verification only. Stage 2-A through Stage 2-D are merged, so the bounded MusicXML pipeline is closed. Stage 3 Renderer Integration is the active package. Controlled degradation, dataset generation, model training, and ScoreMosaic integration have not started.

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
| 2-D | Supported-V1 semantic round-trip verifier | ✅ Complete |
| 2 | MusicXML pipeline | ✅ Closed — local verification only |
| 3 | Renderer integration | 🔄 Active package — real runtime evidence pending |
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

Stage 2-C merged through PR #9 at main commit `81cbfdb2958b8b8b8f4ee5cbd50960a7a75049f0`. Its pre-merge evidence was 40 focused validation tests, 186 full available regression tests, Python compile validation, 300 real official-XSD plus semantic stress cases, and 7 fail-closed security-negative cases.

Stage 2-D merged through PR #10 at main commit `4f36da277540d5a1b7a074215f2def968db73739`. Its pre-merge evidence was 21 focused round-trip tests, 207 full available regression tests, Python compile validation, and 600 generated-score semantic round-trip stress cases.

GitHub CI remains unavailable. No stage in this repository may be reported as CI-verified unless GitHub-hosted evidence actually exists.

## Current branch package

Branch: `stage-3-renderer-integration`

Scope:

- renderer boundary frozen in `RENDERER_CONTRACT.md`;
- Verovio Python toolkit pinned to `verovio==6.2.1`;
- dedicated `st_omr_training.renderer` adapter so generator/MusicXML layers never call Verovio directly;
- Stage 2-C validation required before the renderer runtime is imported or invoked;
- explicit direct MusicXML input mode (`xml`);
- frozen V1 layout/font/SVG options, including Leipzig font and checksum-derived XML IDs;
- deterministic renderer-config fingerprint;
- per-page SVG bytes and SHA-256 plus source/renderer provenance in immutable render results;
- fail-closed page-count/resource limits;
- SVG output checks that reject active elements and external references;
- mocked/fake-runtime tests for adapter boundaries and failure modes;
- separate real-runtime tests that must run against the exact pinned Verovio package before Stage 3 may close.

Explicitly out of scope:

- PNG/JPEG rasterization;
- blur, noise, skew, perspective, compression, or other degradation;
- dataset construction/storage/splitting;
- model files or AI training;
- real-score/user-file ingestion;
- ScoreMosaic integration;
- Guitar TAB work;
- GitHub Actions / CI changes.

## Verification status

Current local evidence while the real Verovio binary dependency is unavailable in the execution environment:

- focused Stage 3 adapter suite using an injected fake renderer runtime: 26 tests passed;
- full available unit/regression suite excluding the real-runtime-required Stage 3 file: 233 tests passed;
- Python compile validation passed;
- generated MusicXML crossed the adapter boundary in 30 fake-runtime cases;
- fail-closed tests cover invalid MusicXML, version drift, renderer setup/load failures, invalid page counts, malformed/non-SVG output, active SVG content, and external SVG references.

The real-runtime test file is intentionally not skipped. It currently fails closed because `verovio==6.2.1` cannot be installed from the network in this local execution environment.

Stage 3 therefore remains **UNVERIFIED — REAL VEROVIO RUNTIME EVIDENCE PENDING**. Mock-backed adapter tests are not sufficient to close the renderer stage.

## Next gate

Obtain the exact Verovio 6.2.1 Python wheel for the local Python/platform, install it only into the local verification environment, then run the real-runtime renderer suite, full regression, compile validation, deterministic repeated rendering, and bounded generated-score render stress. Only after those pass should Stage 3 be opened as a merge-ready PR. Stage 4 remains locked.
