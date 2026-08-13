# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

The repository is public and GitHub Actions CI is active.

Stages 0 through 6 are complete on `main`.

Stage 6 — Synthetic Dataset v1 — merged through PR #17 at exact `main` commit `7c3c736e6d3755d1bd098e2874d73ce5ed41e39f`. GitHub-hosted post-merge CI run #18 (`31674014666`) completed successfully on that exact commit. The integrated repository through Stage 6 is therefore `CI VERIFIED`.

Stage 7 — Baseline ST-OMR Training — is the next architectural stage but has **not started**. Real-data fine-tuning, benchmark/candidate work, and ScoreMosaic integration remain locked.

## Stage status

| Stage | Description | Status |
|---|---|---|
| 0 | Safety and architecture baseline | ✅ Closed |
| 1 | ST Music Generator | ✅ Closed — main CI verified |
| 2 | MusicXML pipeline | ✅ Closed — main CI verified |
| 3 | Renderer integration | ✅ Closed — main CI verified |
| 4 | Controlled degradation | ✅ Closed — main CI verified |
| 5-A | Dataset contract + independent manifest validator | ✅ Closed — main CI verified |
| 5 | Dataset validation | ✅ Closed — main CI verified |
| 6 | Synthetic Dataset v1 | ✅ Closed — PR #17 merged + main CI verified |
| 7 | Baseline ST-OMR training | ⏭ Next — not started |
| 8 | Real-data fine-tuning | 🔒 Not started |
| 9 | Benchmark and candidate decision | 🔒 Not started |
| 10 | ScoreMosaic candidate integration | 🔒 Not started |

## Verified merge evidence

- Stage 3 merged through PR #11 at main commit `3b1e94cea8ac3a26a5df1e2038acc4331a24d371`.
- CI baseline merged through PR #12 at main commit `5abbc9859a4a69bf9a17936bc41e722256f87472`; post-merge run `31647615123` succeeded.
- Current-state documentation synchronized through PR #13 at main commit `23739ddfab618a0406836e94bb0ced1a124f8886`; post-merge run `31648164533` succeeded.
- Stage 4 merged through PR #14 at main commit `f0fd8a732b51b4aa95a66c3a780d0cefa6661361`; post-merge run `31660215130` succeeded.
- Stage 5-A merged through PR #15 at main commit `d677f3d27ac710c56c5ce677a46dc62bcf77bd84`; post-merge run `31671919885` succeeded.
- Stage 5 closure documentation merged through PR #16 at main commit `ed343d4984aae4507a2dd3238cfd1a98fb25b4b7`; post-merge run `31672540732` succeeded.
- Stage 6 final PR head `cfd0ac780595a38e0fe041d2d70293b39f96fcf3` passed GitHub Actions run #17 (`31673631608`) with **309/309 tests**, pinned runtime checks, `pip check`, real Stage 1→6 integration/rebuild/persistence tests, and `compileall`.
- Stage 6 merged through PR #17 at exact main commit `7c3c736e6d3755d1bd098e2874d73ce5ed41e39f`; post-merge GitHub Actions run #18 (`31674014666`) succeeded on that exact commit with **309/309 tests** and compile validation.

The `CI VERIFIED` statement applies only to exact GitHub commits that GitHub-hosted CI exercised.

## Stage 6 closed capability boundary

Stage 6 is governed by [DATASET_BUILD_CONTRACT.md](DATASET_BUILD_CONTRACT.md) and provides deterministic construction and safe local persistence of Synthetic Dataset v1 behind the independent Stage 5 veto gate.

Closed Stage 6 capability includes:

- immutable bounded `SyntheticDatasetConfig`;
- deterministic symbolic family planning;
- family-level 80/10/10 train/validation/test allocation;
- split-independent Stage 4 degradation seeds;
- complete Generator → MusicXML → Verovio → Controlled Degradation → Stage 5 composition;
- exact MusicXML and PNG hash-addressed artifacts;
- Stage 2-C validation of persisted MusicXML targets;
- mandatory independent Stage 5 manifest validation;
- deterministic config/manifest/build identities;
- duplicate target/final-PNG vetoes;
- no-overwrite hash-addressed local persistence with staging and persisted-byte SHA-256 verification;
- bulk generated datasets remaining outside normal Git content.

Stage 6 does **not** include model training, real/user score ingestion, teacher-correction learning, rights-unclear web corpora, Guitar TAB training, cloud storage credentials/providers, or ScoreMosaic integration.

## CI baseline

The active `.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual dispatch using GitHub-hosted Ubuntu 24.04 / Python 3.13, read-only repository contents permission, pinned GitHub action commits, dependency installation from `requirements.txt`, pinned runtime checks, `pip check`, complete unittest discovery, and Python compile validation.

The current pinned checkout/setup-python action commits emit a non-failing GitHub-hosted runner warning because they target deprecated Node.js 20 and are being forced onto Node.js 24. That is a separate CI-maintenance concern and did not invalidate Stage 6 verification.

## Next gate

Stage 7 must not start implicitly. Before any Baseline ST-OMR Training implementation, freeze a bounded Stage 7 package with explicit model/input/target contracts, training/evaluation separation, reproducibility controls, resource limits, acceptance metrics, and a new verification gate. Merge remains separately approved for every PR.
