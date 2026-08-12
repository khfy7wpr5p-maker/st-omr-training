# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

The repository is public. Stage 0 safety/architecture baseline, Stage 1 ST Music Generator, Stage 2 MusicXML Pipeline, and Stage 3 Renderer Integration are complete on `main`.

The GitHub Actions baseline was merged through PR #12. The exact merged `main` commit `5abbc9859a4a69bf9a17936bc41e722256f87472` completed GitHub-hosted CI successfully in run `31647615123`, including Python 3.13 setup, pinned dependency checks, the complete unittest suite with real Verovio runtime tests, and Python compile validation.

The current integrated state through Stage 3 is therefore `CI VERIFIED`.

Stage 4 Controlled Degradation is the next approved development package. Its implementation has not started in this documentation-sync branch.

## Stage status

| Stage | Description | Status |
|---|---|---|
| 0-A | Repository baseline | ✅ Complete |
| 0-B | Architecture and data boundaries | ✅ Complete |
| 0-C | Canonical data contract | ✅ Complete |
| 0-D | Local verification strategy | ✅ Complete |
| 0-E | GitHub CI | ✅ Complete — main CI green |
| 0-F | Architecture consistency audit | ✅ Complete |
| 0-G | Baseline documentation | ✅ Complete |
| 0 | Safety and architecture baseline | ✅ Closed |
| 1-A | Canonical core model | ✅ Complete |
| 1-B | Independent V1 validator | ✅ Complete |
| 1-C | Score structure model and validator | ✅ Complete |
| 1-D | Deterministic ST Music Generator v1 | ✅ Complete |
| 1 | ST Music Generator | ✅ Closed — current main CI verified |
| 2-A | MusicXML contract freeze | ✅ Complete |
| 2-B | Deterministic MusicXML 4.0 writer | ✅ Complete |
| 2-C | Offline XSD + independent MusicXML validator | ✅ Complete |
| 2-D | Supported-V1 semantic round-trip verifier | ✅ Complete |
| 2 | MusicXML pipeline | ✅ Closed — current main CI verified |
| 3 | Renderer integration | ✅ Closed — current main CI verified |
| 4 | Controlled degradation | ⏭ Approved next package — implementation not started |
| 5 | Dataset validation | 🔒 Not started |
| 6 | Synthetic Dataset v1 | 🔒 Not started |
| 7 | Baseline ST-OMR training | 🔒 Not started |
| 8 | Real-data fine-tuning | 🔒 Not started |
| 9 | Benchmark and candidate decision | 🔒 Not started |
| 10 | ScoreMosaic candidate integration | 🔒 Not started |

## Completed evidence

- Stage 2-B merged through PR #8 at main commit `1940d43b3986e5fe359aa79b86cc2af26e96fe98`.
- Stage 2-C merged through PR #9 at main commit `81cbfdb2958b8b8b8f4ee5cbd50960a7a75049f0`.
- Stage 2-D merged through PR #10 at main commit `4f36da277540d5a1b7a074215f2def968db73739`.
- Stage 3 merged through PR #11 at main commit `3b1e94cea8ac3a26a5df1e2038acc4331a24d371`. Its pre-merge local evidence included 29 focused adapter/real-runtime tests, 236 full regression tests, all six Stage 2 goldens rendered with pinned Verovio 6.2.1, 120 generated real-render stress cases, 20/20 deterministic repeats, and Python compile validation.
- CI baseline merged through PR #12 at main commit `5abbc9859a4a69bf9a17936bc41e722256f87472`.
- Post-merge GitHub Actions run `31647615123` on that exact main commit completed successfully with every job step green.

Historical stage-specific local evidence remains recorded as local evidence. The `CI VERIFIED` statement applies to the current integrated `main` content that GitHub-hosted CI actually exercised.

## CI baseline

The active workflow is `.github/workflows/ci.yml` and currently:

- runs on pull requests, pushes to `main`, and manual dispatch;
- uses GitHub-hosted `ubuntu-24.04` and Python 3.13;
- pins `actions/checkout` and `actions/setup-python` to exact commit SHAs;
- uses read-only repository contents permission;
- installs pinned dependencies from `requirements.txt`;
- independently verifies `lxml==6.1.1`, `verovio==6.2.1`, and `pip check`;
- runs the complete unittest discovery suite, including real Verovio runtime tests;
- runs Python compile validation.

## Current documentation package

Branch: `docs-current-state-after-ci`

Purpose: synchronize repository documents with the verified post-PR-#12 state. This package does not implement Stage 4 and does not change runtime architecture.

## Next gate

Merge this documentation-only synchronization after its pull-request CI is green and separate merge approval is given. Then Stage 4 Controlled Degradation may begin as its own bounded implementation package.
