# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

The repository is public and GitHub Actions CI is active.

Stages 0 through 5 are complete on `main`.

Stage 5-A — Dataset Contract + Independent Manifest Validator — merged through PR #15 at exact `main` commit `d677f3d27ac710c56c5ce677a46dc62bcf77bd84`. GitHub-hosted post-merge CI run `31671919885` completed successfully on that exact commit. The integrated implementation through Stage 5 is therefore `CI VERIFIED`.

Stage 6 — Synthetic Dataset v1 — is the next architectural stage but has **not started**. Model training, real-data work, and ScoreMosaic integration remain locked.

## Stage status

| Stage | Description | Status |
|---|---|---|
| 0 | Safety and architecture baseline | ✅ Closed |
| 1 | ST Music Generator | ✅ Closed — main CI verified |
| 2 | MusicXML pipeline | ✅ Closed — main CI verified |
| 3 | Renderer integration | ✅ Closed — main CI verified |
| 4 | Controlled degradation | ✅ Closed — main CI verified |
| 5-A | Dataset contract + independent manifest validator | ✅ Closed — PR #15 merged + main CI verified |
| 5 | Dataset validation | ✅ Closed — CI VERIFIED |
| 6 | Synthetic Dataset v1 | ⏭ Next — not started |
| 7 | Baseline ST-OMR training | 🔒 Not started |
| 8 | Real-data fine-tuning | 🔒 Not started |
| 9 | Benchmark and candidate decision | 🔒 Not started |
| 10 | ScoreMosaic candidate integration | 🔒 Not started |

## Completed evidence through Stage 5

- Stage 2-B merged through PR #8 at main commit `1940d43b3986e5fe359aa79b86cc2af26e96fe98`.
- Stage 2-C merged through PR #9 at main commit `81cbfdb2958b8b8b8f4ee5cbd50960a7a75049f0`.
- Stage 2-D merged through PR #10 at main commit `4f36da277540d5a1b7a074215f2def968db73739`.
- Stage 3 merged through PR #11 at main commit `3b1e94cea8ac3a26a5df1e2038acc4331a24d371`.
- CI baseline merged through PR #12 at main commit `5abbc9859a4a69bf9a17936bc41e722256f87472`; post-merge run `31647615123` succeeded.
- Current-state documentation synchronized through PR #13 at main commit `23739ddfab618a0406836e94bb0ced1a124f8886`; post-merge run `31648164533` succeeded.
- Stage 4 merged through PR #14 at main commit `f0fd8a732b51b4aa95a66c3a780d0cefa6661361`; post-merge run `31660215130` succeeded.
- Stage 5-A final PR head `5165fe6669bce582ccc16d64695d4a7730e29660` passed GitHub Actions run `31671655623` with **295/295 tests**, pinned runtime checks, `pip check`, and `compileall`.
- Stage 5-A merged through PR #15 at main commit `d677f3d27ac710c56c5ce677a46dc62bcf77bd84`.
- Exact post-merge Stage 5-A `main` commit passed GitHub Actions run `31671919885`; dependency installation, pinned runtime verification, full test suite, and compile validation all succeeded.

The `CI VERIFIED` statement applies only to exact GitHub commits that GitHub-hosted CI exercised.

## Stage 5 closed boundary

Stage 5 provides an independent acceptance gate for synthetic dataset manifest metadata before bulk dataset creation is allowed.

Closed Stage 5 capabilities include:

- immutable `DatasetSplit`, `DatasetDegradationConfig`, `DatasetSample`, and `DatasetManifest` models;
- exact V1 `train`, `validation`, and `test` split vocabulary;
- synthetic-only source-class gate;
- family-exclusive split policy;
- independent Stage 4 replay-config fingerprint and derivative-ID recomputation;
- deterministic sample identity independent of split assignment;
- duplicate `sample_id`, `derivative_id`, and PNG hash vetoes;
- `family_id` split-leakage veto;
- identical MusicXML target and clean-SVG alias leakage vetoes;
- bounded PNG/dimension/mode/format checks;
- Stage 4 `DegradedPage` → Stage 5 bridge with independent PNG signature/IHDR/CRC/hash/dimension checks;
- deterministic canonical JSON manifest serialization and manifest SHA-256;
- focused negative, corruption, determinism, leakage, and duplicate tests;
- real Generator → MusicXML → Verovio → Stage 4 → Stage 5 integration tests.

Contract: [DATASET_CONTRACT.md](DATASET_CONTRACT.md)

## Stage 6 boundary

Stage 6 is **not started**. Its future responsibility is bulk Synthetic Dataset v1 construction behind the already-merged Stage 5 validator. Starting Stage 6 requires a separately bounded package and explicit approval.

Stage 6 must not silently expand into model training, real/user score ingestion, teacher-correction learning, ScoreMosaic integration, or Guitar TAB training.

## CI baseline

The active `.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual dispatch using GitHub-hosted Ubuntu 24.04 / Python 3.13, read-only repository contents permission, pinned GitHub action commits, dependency installation from `requirements.txt`, pinned runtime checks, `pip check`, complete unittest discovery, and Python compile validation.

## Next gate

Stage 5 is closed. The next implementation gate is a separately scoped Stage 6 — Synthetic Dataset v1 package. Stage 6 remains unstarted until that package is explicitly approved.
