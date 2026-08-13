# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

The repository is public and GitHub Actions CI is active.

Stage 0 safety/architecture baseline, Stage 1 ST Music Generator, Stage 2 MusicXML Pipeline, Stage 3 Renderer Integration, and Stage 4 Controlled Degradation are complete on `main`.

Stage 4 merged through PR #14 at exact `main` commit `f0fd8a732b51b4aa95a66c3a780d0cefa6661361`. GitHub-hosted post-merge CI run `31660215130` completed successfully on that exact commit. The integrated implementation through Stage 4 is therefore `CI VERIFIED`.

Stage 5-A — Dataset Contract + Independent Manifest Validator — is the active bounded implementation package on branch `stage-5a-dataset-manifest-validator` and PR #15.

Stage 6 Synthetic Dataset v1, model training, real-data work, and ScoreMosaic integration remain locked.

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
| 1 | ST Music Generator | ✅ Closed — main CI verified |
| 2-A | MusicXML contract freeze | ✅ Complete |
| 2-B | Deterministic MusicXML 4.0 writer | ✅ Complete |
| 2-C | Offline XSD + independent MusicXML validator | ✅ Complete |
| 2-D | Supported-V1 semantic round-trip verifier | ✅ Complete |
| 2 | MusicXML pipeline | ✅ Closed — main CI verified |
| 3 | Renderer integration | ✅ Closed — main CI verified |
| 4 | Controlled degradation | ✅ Closed — PR #14 merged + main CI verified |
| 5-A | Dataset contract + independent manifest validator | ✅ PR package ready — exact-current-head CI must be green for merge |
| 5 | Dataset validation | 🔄 In progress through Stage 5-A |
| 6 | Synthetic Dataset v1 | 🔒 Not started |
| 7 | Baseline ST-OMR training | 🔒 Not started |
| 8 | Real-data fine-tuning | 🔒 Not started |
| 9 | Benchmark and candidate decision | 🔒 Not started |
| 10 | ScoreMosaic candidate integration | 🔒 Not started |

## Completed evidence through Stage 4

- Stage 2-B merged through PR #8 at main commit `1940d43b3986e5fe359aa79b86cc2af26e96fe98`.
- Stage 2-C merged through PR #9 at main commit `81cbfdb2958b8b8f4ee5cbd50960a7a75049f0`.
- Stage 2-D merged through PR #10 at main commit `4f36da277540d5a1b7a074215f2def968db73739`.
- Stage 3 merged through PR #11 at main commit `3b1e94cea8ac3a26a5df1e2038acc4331a24d371`.
- CI baseline merged through PR #12 at main commit `5abbc9859a4a69bf9a17936bc41e722256f87472`; post-merge run `31647615123` succeeded.
- Current-state documentation synchronized through PR #13 at main commit `23739ddfab618a0406836e94bb0ced1a124f8886`; post-merge run `31648164533` succeeded.
- Stage 4 merged through PR #14 at main commit `f0fd8a732b51b4aa95a66c3a780d0cefa6661361`.
- Stage 4 final PR head passed GitHub Actions run `31649798684` with 264/264 tests and compile validation.
- Exact post-merge Stage 4 `main` commit passed GitHub Actions run `31660215130`; all job steps completed successfully.

The `CI VERIFIED` statement applies only to exact GitHub commits that GitHub-hosted CI exercised.

## Stage 5-A package

PR: #15 — `Stage 5-A: synthetic dataset manifest validator`

Branch: `stage-5a-dataset-manifest-validator`

Base: `f0fd8a732b51b4aa95a66c3a780d0cefa6661361`

Contract: [DATASET_CONTRACT.md](DATASET_CONTRACT.md)

Stage 5-A responsibility is limited to defining and independently validating synthetic dataset manifest metadata. It does not bulk-generate dataset files.

Implemented package scope:

- immutable `DatasetSplit`, `DatasetDegradationConfig`, `DatasetSample`, and `DatasetManifest` models;
- exact V1 `train`, `validation`, and `test` split vocabulary;
- synthetic-only source-class gate;
- family-exclusive split policy;
- independent Stage 4 replay-config fingerprint recomputation;
- independent Stage 4 derivative-ID recomputation;
- deterministic Stage 5-A sample identity independent of split assignment;
- duplicate `sample_id`, `derivative_id`, and PNG hash vetoes;
- `family_id` split-leakage veto;
- identical MusicXML target alias leakage veto across families/splits;
- identical clean SVG alias leakage veto across families/splits;
- bounded PNG/dimension/mode/format metadata checks;
- narrow Stage 4 `DegradedPage` → Stage 5-A bridge that independently checks actual PNG signature/IHDR/CRC/hash/dimensions before creating sample metadata;
- deterministic canonical JSON manifest serialization and manifest SHA-256;
- focused corruption, negative, determinism, leakage, and duplicate tests;
- real Generator → MusicXML → Verovio → Stage 4 → Stage 5-A integration tests.

Explicitly out of scope:

- bulk synthetic dataset creation;
- final split ratios, balancing, curriculum, or sampling strategy;
- dataset image storage/filesystem/cloud provider design;
- model loaders/training/checkpoints;
- real/user score ingestion;
- teacher-correction learning;
- ScoreMosaic integration;
- Guitar TAB training.

## Stage 5-A verification evidence

GitHub-hosted PR verification has already passed on two successive package heads:

- implementation/documentation head `9c290ee2bac5c45e357a98072ec7e2c05cf9b33b` — run `31671458220` — SUCCESS;
- evidence-update head `c2dc88072b4fba24ab170088f33e550a2644d1ad` — run `31671544975` — SUCCESS.

The fresh Ubuntu 24.04 / Python 3.13 runs verified:

- dependency installation and existing exact runtime pins;
- `pip check` with no broken requirements;
- 28 focused Stage 5-A unit/negative/leakage/determinism tests;
- 3 real Generator → MusicXML → Verovio → Stage 4 → Stage 5-A integration tests;
- complete repository regression: **295/295 tests passed**;
- Python `compileall` validation.

This final status wording creates one last PR head. The authoritative merge evidence is therefore the GitHub-hosted CI check attached to the exact current PR head, not either older run listed above. No further source/document change should be made after that current-head check is green unless the check is repeated again.

A complete independent local repository regression is not claimed for Stage 5-A; GitHub-hosted fresh-environment CI is the authoritative full-suite evidence for this package.

## Required merge gate

```text
Focused Stage 5-A tests
        ↓
Real Stage 4 → Stage 5-A integration tests
        ↓
Full repository regression
        ↓
Python compile validation
        ↓
Diff/scope review
        ↓
GitHub-hosted CI on exact current PR head
        ↓
CI VERIFIED — PR HEAD
        ↓
Separate merge approval
        ↓
Post-merge GitHub-hosted CI on exact main
```

No CI result from an older commit may substitute for current-head verification.

## CI baseline

The active `.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual dispatch using GitHub-hosted Ubuntu 24.04 / Python 3.13, read-only repository contents permission, pinned GitHub action commits, dependency installation from `requirements.txt`, pinned runtime checks, `pip check`, complete unittest discovery, and Python compile validation.

Stage 5-A adds no new runtime dependency and does not change the CI workflow.

## Next gate

Confirm GitHub-hosted CI is green on the exact current PR #15 head. If green, Stage 5-A is `CI VERIFIED — PR HEAD`, but merge still requires separate explicit approval. Stage 6 must remain locked until Stage 5-A is merged, post-merge `main` CI passes, and Stage 5 closure is explicitly assessed.
