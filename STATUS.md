# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

The repository is public and GitHub Actions CI is active.

Stages 0 through 6, all Stage 7 packages, and Stage 8-0 are complete on `main`.

Stage 7-A — Baseline Training Contract Freeze — merged through PR #19 at exact `main` commit `0f04b0182b6753cfb8816d1287adb5ee973e0c28`; post-merge GitHub Actions run #22 (`31675913632`) succeeded. Stage 7-A closure/status synchronization then merged through PR #20 at exact `main` commit `6a13760d9d17130ea86636f4828ff1bff035f30d`; post-merge run #24 (`31676871798`) also succeeded.

Stage 7-B — Tokenizer/Data/Model/Trainer Implementation — merged through PR #21 at exact `main` commit `d02dce4ee17dfccf6f05519ab0970fdc188d0147`. Post-merge GitHub Actions run #31 (`31679810478`) succeeded on that exact commit with **336/336 tests**, pinned runtime verification, `pip check`, the missing-`EOS` regression, deterministic CPU smoke evidence, and `compileall`. Stage 7-B closure documentation then squash-merged through PR #22 at exact `main` commit `1befaf260023852ef3bee5c8abab016f464557bb`; post-merge GitHub Actions run #33 (`31681668145`) succeeded on that exact SHA. The integrated repository through the Stage 7-B boundary is therefore `CI VERIFIED`.

Stage 7-C — Bounded Baseline Training Run + Evidence — merged through PR #23 from exact source head `7a993304218fa19609ea512665148dac3eea503a` to exact `main` commit `2c2c478eb361fa90a3bccd819b623680eb12de0b`. Exact-head run #67 (`31691794239`) passed **361/361 tests**, the guarded benchmark, and the authoritative synthetic-only run. Post-merge exact-main run #68 (`31692849892`) passed **361/361 tests**, exact runtime checks, `pip check`, and `compileall`. Stage 7-C closure synchronization then merged through PR #24 to exact `main` commit `e56fad04e43bc6302a8cda60c6f382e83a23d734`; post-merge run #70 (`31693998756`) succeeded. The accepted evidence is recorded in `STAGE7C_EVIDENCE.md`; Stage 7-C is `CLOSED — MAIN CI VERIFIED`.

Stage 8-0 — Real Data & Fine-Tuning Contract Freeze — merged through PR #25 to exact `main` commit `86487a4c3c41264b02bd159cd647a1318d9b9b88`. Post-merge GitHub Actions run #75 (`31698691405`) succeeded on that exact commit with pinned dependency checks, `pip check`, the complete repository test suite, and `compileall`. Stage 8-0 is `CLOSED — MAIN CI VERIFIED`.

Stage 8-0 added only contract metadata, independent metadata validators, negative tests, and documentation. It did not ingest real data, load a checkpoint, run training/fine-tuning, open a sealed test split, start Stage 9/10, or integrate with ScoreMosaic.

The Stage 7-C checkpoint artifact `9177923796` remains temporary and scheduled to expire on 2026-09-12. Stage 8-0 records its identity but did not move or republish the artifact. The accepted Stage 7-C model remains non-production: exact sequence accuracy is 0% and token error rate is approximately 80.5%.

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
| 6 | Synthetic Dataset v1 | ✅ Closed — main CI verified |
| 7-A | Baseline training contract freeze | ✅ Closed — main CI verified |
| 7-B | Tokenizer/data/model/trainer smoke implementation | ✅ Closed — main CI verified |
| 7-C | Bounded baseline training run + evidence | ✅ Closed — main CI verified |
| 8-0 | Real data & fine-tuning contract freeze | ✅ Closed — main CI verified |
| 8-1+ | Real-data intake / experiment execution | 🔒 Not started |
| 9 | Benchmark and candidate decision | 🔒 Not started — test sealed |
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
- Stage 6 closure documentation merged through PR #18 at exact main commit `046c9a4e7e41e94b0b4465a2610f30361055a3ed`; post-merge GitHub Actions run #20 (`31674836433`) succeeded.
- Stage 7-A PR merge candidate from source head `0eca1d881c494c763692aa72b1092d741c32be83` against exact base `046c9a4e7e41e94b0b4465a2610f30361055a3ed` passed GitHub Actions run #21 (`31675330236`) with **309/309 tests**, pinned runtime checks, `pip check`, and `compileall`.
- Stage 7-A merged through PR #19 at exact main commit `0f04b0182b6753cfb8816d1287adb5ee973e0c28`; post-merge GitHub Actions run #22 (`31675913632`) succeeded.
- Stage 7-A closure synchronization merged through PR #20 at exact main commit `6a13760d9d17130ea86636f4828ff1bff035f30d`; post-merge run #24 (`31676871798`) succeeded.
- Stage 7-B final source head `a8ad8bc9f14953f0ed35ef5a5a8275be69af5ebd` against exact base `6a13760d9d17130ea86636f4828ff1bff035f30d` passed GitHub Actions run #30 (`31679413312`) on GitHub-generated PR merge candidate `3d5ee4e2ca479614e2a3322e0339ca21f25e5cae` with **336/336 tests**, exact runtime pins, `pip check`, the missing-`EOS` regression, deterministic CPU smoke evidence, and `compileall`.
- Stage 7-B merged through PR #21 at exact main commit `d02dce4ee17dfccf6f05519ab0970fdc188d0147`; post-merge run #31 (`31679810478`) succeeded.
- Stage 7-B closure documentation squash-merged through PR #22 at exact main commit `1befaf260023852ef3bee5c8abab016f464557bb`; post-merge run #33 (`31681668145`) succeeded.
- Stage 7-C exact source head `7a993304218fa19609ea512665148dac3eea503a` passed run #67 (`31691794239`) with **361/361 tests**, a safe guarded benchmark, and accepted authoritative evidence; PR #23 then squash-merged to exact main commit `2c2c478eb361fa90a3bccd819b623680eb12de0b`.
- Stage 7-C post-merge run #68 (`31692849892`) succeeded on exact main commit `2c2c478eb361fa90a3bccd819b623680eb12de0b` with **361/361 tests**, exact dependency checks, `pip check`, and `compileall`.
- Stage 7-C closure synchronization merged through PR #24 at exact main commit `e56fad04e43bc6302a8cda60c6f382e83a23d734`; post-merge run #70 (`31693998756`) succeeded.
- Stage 8-0 PR #25 merged to exact main commit `86487a4c3c41264b02bd159cd647a1318d9b9b88`; post-merge run #75 (`31698691405`) succeeded with pinned runtime verification, `pip check`, the complete repository test suite, and `compileall`.

The `CI VERIFIED` statement applies only to exact GitHub commits or GitHub-generated PR merge candidates that GitHub-hosted CI exercised.

## Stage 7 closed capability boundary

Stage 7 froze and verified:

- Stage 5/6 validated synthetic artifacts as the only Stage 7 input;
- strict train/validation/test isolation with test sealed until Stage 9;
- deterministic grayscale preprocessing and compact supported-V1 semantic token targets;
- one from-scratch CNN/GRU baseline with explicit runtime/configuration fingerprints;
- train-only optimization, validation-only checkpoint selection, NaN/Infinity vetoes, and exact checkpoint/evidence hashes;
- a real bounded Stage 7-C run with deterministic evidence and no test access.

The accepted Stage 7-C run completed 40 epochs/1560 steps, improved validation loss from `3.560342441426505` to `0.9992435761603257`, and produced 21/21 semantically valid and MusicXML-regenerable validation predictions. Exact sequence accuracy remained 0% and token error rate remained approximately 80.5%, so no production-candidate claim is made.

## Stage 8-0 closed capability boundary

Contract: [STAGE8_REAL_DATA_CONTRACT.md](STAGE8_REAL_DATA_CONTRACT.md)

Stage 8-0 froze and enforced metadata-level rules for:

- source provenance and training-use rights evidence;
- explicit training permission and privacy review for user-derived material;
- quarantine versus admitted state;
- image/MusicXML pairing evidence and supported-V1 target requirements;
- immutable content identity, duplicate rejection, and family/source/target/semantic split-leakage vetoes;
- train/validation-only development access with real test metadata excluded from the development manifest and test sealed until Stage 9;
- two later experiment candidates: exact Stage 7-C checkpoint fine-tuning versus the same architecture from scratch;
- ScoreMosaic upload/teacher-correction isolation and prohibition of online/automatic learning.

Stage 8-0 did not ingest data, inspect real file bytes, load a checkpoint, or run training. Byte-level intake/admission and experiment execution require later separately authorized packages.

## CI baseline

The active `.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual dispatch using GitHub-hosted Ubuntu 24.04 / Python 3.13, read-only repository contents permission, pinned GitHub action commits, exact dependency/runtime checks, `pip check`, complete unittest discovery, and Python compile validation.

The isolated official-PyTorch-CPU-index install verifies exact `torch==2.13.0+cpu`. The one-shot PR #23 benchmark/training exception is retired; the active workflow no longer contains a full Stage 7-C execution path.

## Required Stage 8-0 gate

```text
exact verified Stage 7-C main baseline
        ↓
real-data rights/provenance/pairing contract
        ↓
quarantine/admission metadata validator
        ↓
duplicate + split-leakage negative tests
        ↓
Stage 8 test-access veto
        ↓
Candidate A / Candidate B experiment contract
        ↓
ScoreMosaic/teacher-correction isolation contract
        ↓
full repository regression + pip check + compileall
        ↓
exact PR-head GitHub CI
        ↓
separate explicit merge approval
        ↓
post-merge exact-main CI
        ↓
Stage 8-0 CLOSED
```

The Stage 8-0 gate above is complete through exact-main post-merge CI on `86487a4c3c41264b02bd159cd647a1318d9b9b88`.

## Next gate

Stage 8-0 is closed and main CI verified. The next planned small gate is **Stage 8-1 — quarantine/intake + byte-level validation**, still without training and without opening either sealed test split. Stage 8-1 has not started. Stage 9 and Stage 10 remain locked.
