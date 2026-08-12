# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

Stage 0 safety/architecture baseline is closed with GitHub CI deferred. Stage 1 ST Music Generator is closed. Stage 2-A through Stage 2-D are merged, so the bounded MusicXML pipeline is closed. Stage 3 Renderer Integration is implemented and locally verified with the exact pinned Verovio runtime. Stage 4 Controlled Degradation has not started.

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
| 3 | Renderer integration | ✅ PR package ready — local verification only |
| 4 | Controlled degradation | 🔒 Not started |
| 5 | Dataset validation | 🔒 Not started |
| 6 | Synthetic Dataset v1 | 🔒 Not started |
| 7 | Baseline ST-OMR training | 🔒 Not started |
| 8 | Real-data fine-tuning | 🔒 Not started |
| 9 | Benchmark and candidate decision | 🔒 Not started |
| 10 | ScoreMosaic candidate integration | 🔒 Not started |

## Completed Stage 2 evidence

Stage 2-B merged through PR #8 at main commit `1940d43b3986e5fe359aa79b86cc2af26e96fe98`.

Stage 2-C merged through PR #9 at main commit `81cbfdb2958b8b8f4ee5cbd50960a7a75049f0`.

Stage 2-D merged through PR #10 at main commit `4f36da277540d5a1b7a074215f2def968db73739`.

Stage 2 remains `LOCAL VERIFIED — CI NOT AVAILABLE`.

## Current branch package

Branch: `stage-3-renderer-integration`

Scope:

- renderer boundary frozen in `RENDERER_CONTRACT.md`;
- Verovio Python toolkit pinned to `verovio==6.2.1`;
- isolated `st_omr_training.renderer` adapter;
- Stage 2-C validation before renderer import/invocation;
- explicit MusicXML input mode;
- frozen page/layout/font/SVG options and deterministic renderer-config fingerprint;
- Leipzig music font pinned for V1;
- immutable render result with source MusicXML hash, renderer/runtime provenance, per-page SVG bytes and SHA-256;
- fail-closed page-count limit and renderer setup/load errors;
- SVG output checks rejecting non-SVG/malformed output, active elements, external references, and external stylesheet URL surfaces;
- exact tested runtime identity recorded in `VEROVIO_RUNTIME_EVIDENCE.md`.

Explicitly out of scope:

- PNG/JPEG rasterization;
- blur, noise, skew, perspective, compression, or other degradation;
- dataset construction/storage/splitting;
- model files or AI training;
- real-score/user-file ingestion;
- ScoreMosaic integration;
- Guitar TAB work;
- GitHub Actions / CI changes.

## Stage 3 local verification

Exact runtime used:

- `verovio==6.2.1`;
- CPython 3.13 manylinux x86-64 wheel;
- wheel SHA-256 `00b9ab551de859fa61ac67d6a5f4f3d97a7f9389197644d9493b5e9c0b7b69ab`;
- locally computed wheel hash matched the published PyPI hash for the exact wheel.

Verification results on the committed Stage 3 implementation state:

- focused Stage 3 adapter + real-runtime suite: 29 tests passed;
- full available unit/regression suite: 236 tests passed;
- all six Stage 2 golden MusicXML fixtures rendered successfully with the exact pinned runtime;
- committed live-runtime generated-score coverage: 50 renders passed;
- same-input real rendering produced byte-identical SVG and identical page hashes;
- supplemental real-render stress: 120 generated scores passed across mixed, note-only, rest-only, chord-only, fixed 2/4, fixed 3/4, fixed 4/4, and accidentals-disabled configurations;
- supplemental deterministic repeats: 20/20 byte-identical;
- Python compile validation passed.

This is `LOCAL VERIFIED — CI NOT AVAILABLE`. GitHub-hosted CI did not run and is not claimed.

## Next gate

Stage 3 is ready for bounded PR review. Merge requires separate explicit approval. Stage 4 remains locked until Stage 3 is merged and post-merge state is verified.
