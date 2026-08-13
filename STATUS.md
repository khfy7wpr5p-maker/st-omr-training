# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

The repository is public and GitHub Actions CI is active.

Stages 0 through 5 are complete on `main`.

Stage 5-A — Dataset Contract + Independent Manifest Validator — merged through PR #15 at exact `main` commit `d677f3d27ac710c56c5ce677a46dc62bcf77bd84`. GitHub-hosted post-merge CI run `31671919885` completed successfully on that exact commit.

Stage 5 closure documentation merged through PR #16 at exact `main` commit `ed343d4984aae4507a2dd3238cfd1a98fb25b4b7`; its post-merge GitHub Actions run `31672540732` completed successfully. The integrated repository through Stage 5 is therefore `CI VERIFIED`.

Stage 6 — Synthetic Dataset v1 — is now the active bounded implementation package on branch `stage-6-synthetic-dataset-v1`.

Stage 7 model training, real-data work, and ScoreMosaic integration remain locked.

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
| 6 | Synthetic Dataset v1 | 🔄 Active package — verification pending |
| 7 | Baseline ST-OMR training | 🔒 Not started |
| 8 | Real-data fine-tuning | 🔒 Not started |
| 9 | Benchmark and candidate decision | 🔒 Not started |
| 10 | ScoreMosaic candidate integration | 🔒 Not started |

## Completed evidence through Stage 5

- Stage 2-B merged through PR #8 at main commit `1940d43b3986e5fe359aa79b86cc2af26e96fe98`.
- Stage 2-C merged through PR #9 at main commit `81cbfdb2958b8b8f4ee5cbd50960a7a75049f0`.
- Stage 2-D merged through PR #10 at main commit `4f36da277540d5a1b7a074215f2def968db73739`.
- Stage 3 merged through PR #11 at main commit `3b1e94cea8ac3a26a5df1e2038acc4331a24d371`.
- CI baseline merged through PR #12 at main commit `5abbc9859a4a69bf9a17936bc41e722256f87472`; post-merge run `31647615123` succeeded.
- Current-state documentation synchronized through PR #13 at main commit `23739ddfab618a0406836e94bb0ced1a124f8886`; post-merge run `31648164533` succeeded.
- Stage 4 merged through PR #14 at main commit `f0fd8a732b51b4aa95a66c3a780d0cefa6661361`; post-merge run `31660215130` succeeded.
- Stage 5-A final PR head `5165fe6669bce582ccc16d64695d4a7730e29660` passed GitHub Actions run `31671655623` with **295/295 tests**, pinned runtime checks, `pip check`, and `compileall`.
- Stage 5-A merged through PR #15 at main commit `d677f3d27ac710c56c5ce677a46dc62bcf77bd84`; post-merge run `31671919885` succeeded.
- Stage 5 closure docs merged through PR #16 at main commit `ed343d4984aae4507a2dd3238cfd1a98fb25b4b7`; post-merge run `31672540732` succeeded.

The `CI VERIFIED` statement applies only to exact GitHub commits that GitHub-hosted CI exercised.

## Stage 6 active package

Base `main`: `ed343d4984aae4507a2dd3238cfd1a98fb25b4b7`

Contract: [DATASET_BUILD_CONTRACT.md](DATASET_BUILD_CONTRACT.md)

Stage 6 responsibility is limited to deterministic construction and safe local persistence of Synthetic Dataset v1 behind the already-merged Stage 5 veto gate.

Current package scope:

- immutable bounded `SyntheticDatasetConfig`;
- deterministic family planning across mixed, note-only, rest-only, chord-only, fixed-meter, and no-accidental symbolic profiles;
- deterministic family-level 80/10/10 split allocation with non-empty train/validation/test splits;
- split-independent deterministic Stage 4 degradation seeds;
- full Generator → MusicXML → Verovio → Stage 4 → Stage 5 composition;
- exact MusicXML and PNG hash-addressed artifact sets;
- independent Stage 5 manifest validation before a build can be accepted;
- deterministic manifest/config/build identities;
- no-overwrite atomic local directory persistence with persisted-byte SHA-256 verification;
- small temporary integration fixtures only; bulk dataset artifacts remain outside Git.

Explicitly out of scope:

- model loaders, architecture, loss, optimizer, checkpoints, or training;
- real/user scores or teacher-correction learning;
- rights-unclear/copyrighted web corpora;
- ScoreMosaic integration;
- Guitar TAB training;
- cloud buckets, credentials, signed URLs, or production storage providers.

## Required Stage 6 verification gate

```text
Focused Stage 6 configuration/split/artifact tests
        ↓
Real Stage 1 → 6 integration tests
        ↓
Same-config rebuild determinism
        ↓
Hash-addressed filesystem persistence/no-overwrite test
        ↓
Full repository regression
        ↓
Python compile validation
        ↓
Diff/scope review
        ↓
GitHub-hosted CI on exact final PR head
        ↓
CI VERIFIED — PR HEAD
        ↓
Separate merge approval
        ↓
Post-merge GitHub-hosted CI on exact main
```

No older CI run may substitute for the exact final Stage 6 PR head.

## CI baseline

The active `.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual dispatch using GitHub-hosted Ubuntu 24.04 / Python 3.13, read-only repository contents permission, pinned GitHub action commits, dependency installation from `requirements.txt`, pinned runtime checks, `pip check`, complete unittest discovery, and Python compile validation.

Stage 6 introduces no new runtime dependency and does not modify the CI workflow.

## Next gate

Complete Stage 6 verification and scope review. If the exact final PR head is green, merge still requires separate explicit approval. Stage 7 must remain locked until Stage 6 is merged and the exact resulting `main` commit passes GitHub-hosted CI.
