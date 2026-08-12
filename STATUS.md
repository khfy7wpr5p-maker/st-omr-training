# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

Stage 0 safety/architecture baseline is closed. The repository is public and the previously deferred GitHub CI capability is now implemented on a bounded follow-up PR. Stage 1 ST Music Generator, Stage 2 MusicXML Pipeline, and Stage 3 Renderer Integration are merged. Stage 4 Controlled Degradation has not started.

## Stage status

| Stage | Description | Status |
|---|---|---|
| 0-A | Repository baseline | ✅ Complete |
| 0-B | Architecture and data boundaries | ✅ Complete |
| 0-C | Canonical data contract | ✅ Complete |
| 0-D | Local verification strategy | ✅ Complete |
| 0-E | GitHub CI | ✅ PR CI green — merge pending |
| 0-F | Architecture consistency audit | ✅ Complete |
| 0-G | Baseline documentation | ✅ Complete |
| 0 | Safety and architecture baseline | ✅ Closed — CI follow-up merge pending |
| 1-A | Canonical core model | ✅ Complete |
| 1-B | Independent V1 validator | ✅ Complete |
| 1-C | Score structure model and validator | ✅ Complete |
| 1-D | Deterministic ST Music Generator v1 | ✅ Complete |
| 1 | ST Music Generator | ✅ Closed — exercised by PR CI |
| 2-A | MusicXML contract freeze | ✅ Complete |
| 2-B | Deterministic MusicXML 4.0 writer | ✅ Complete |
| 2-C | Offline XSD + independent MusicXML validator | ✅ Complete |
| 2-D | Supported-V1 semantic round-trip verifier | ✅ Complete |
| 2 | MusicXML pipeline | ✅ Closed — exercised by PR CI |
| 3 | Renderer integration | ✅ Merged — exercised by PR CI |
| 4 | Controlled degradation | 🔒 Not started |
| 5 | Dataset validation | 🔒 Not started |
| 6 | Synthetic Dataset v1 | 🔒 Not started |
| 7 | Baseline ST-OMR training | 🔒 Not started |
| 8 | Real-data fine-tuning | 🔒 Not started |
| 9 | Benchmark and candidate decision | 🔒 Not started |
| 10 | ScoreMosaic candidate integration | 🔒 Not started |

## Completed evidence

Stage 2-B merged through PR #8 at main commit `1940d43b3986e5fe359aa79b86cc2af26e96fe98`.

Stage 2-C merged through PR #9 at main commit `81cbfdb2958b8b8b8f4ee5cbd50960a7a75049f0`.

Stage 2-D merged through PR #10 at main commit `4f36da277540d5a1b7a074215f2def968db73739`.

Stage 3 merged through PR #11 at main commit `3b1e94cea8ac3a26a5df1e2038acc4331a24d371`. Pre-merge local evidence included 29 focused adapter/real-runtime tests, 236 full regression tests, all six Stage 2 goldens rendered with pinned Verovio 6.2.1, 120 generated real-render stress cases, 20/20 deterministic repeats, and Python compile validation.

The first GitHub-hosted CI run on PR #12 completed successfully on the repository content derived from main `3b1e94cea8ac3a26a5df1e2038acc4331a24d371`. It verified pinned dependency installation, `lxml==6.1.1`, `verovio==6.2.1`, `pip check`, the full unittest suite including real Verovio runtime tests, and Python compile validation. This establishes CI evidence for the PR candidate; exact merged-main CI evidence still requires the post-merge push run.

## Current branch package

Branch: `ci-baseline`

Scope:

- add one GitHub Actions workflow at `.github/workflows/ci.yml`;
- run on pull requests, pushes to `main`, and manual dispatch;
- use GitHub-hosted `ubuntu-24.04` with Python 3.13;
- pin `actions/checkout` and `actions/setup-python` to exact commit SHAs;
- install only the repository's pinned `requirements.txt` dependencies;
- independently assert `lxml==6.1.1` and `verovio==6.2.1` and run `pip check`;
- run the complete unittest discovery suite, including the real Verovio runtime tests;
- run Python compile validation;
- use read-only repository permissions and a 15-minute job timeout.

Explicitly out of scope:

- Stage 4 implementation;
- dependency upgrades;
- application architecture changes;
- branch protection/ruleset changes;
- deployment, release, model training, datasets, or ScoreMosaic integration.

## Verification status

PR #12 has GitHub-hosted CI evidence. The workflow run completed successfully with every job step green. Because this STATUS update creates a new PR head commit, the final merge gate still requires the automatically triggered CI run for this exact head to succeed.

## Next gate

Require the GitHub-hosted CI run for the final PR head to pass. Merge then requires separate explicit approval. After merge, verify the automatic `push` run on exact `main`. Stage 4 remains locked until that post-merge main CI is green.
