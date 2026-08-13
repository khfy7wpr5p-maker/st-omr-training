# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

The repository is public and GitHub Actions CI is active.

Stages 0 through 6, all Stage 7 packages, and Stage 8-0 are complete on `main`.

Stage 7-A — Baseline Training Contract Freeze — merged through PR #19 at exact `main` commit `0f04b0182b6753cfb8816d1287adb5ee973e0c28`; post-merge GitHub Actions run #22 (`31675913632`) succeeded. Stage 7-A closure/status synchronization then merged through PR #20 at exact `main` commit `6a13760d9d17130ea86636f4828ff1bff035f30d`; post-merge run #24 (`31676871798`) also succeeded.

Stage 7-B — Tokenizer/Data/Model/Trainer Implementation — merged through PR #21 at exact `main` commit `d02dce4ee17dfccf6f05519ab0970fdc188d0147`. Post-merge GitHub Actions run #31 (`31679810478`) succeeded on that exact commit with **336/336 tests**, pinned runtime verification, `pip check`, the missing-`EOS` regression, deterministic CPU smoke evidence, and `compileall`. Stage 7-B closure documentation then squash-merged through PR #22 at exact `main` commit `1befaf260023852ef3bee5c8abab016f464557bb`; post-merge GitHub Actions run #33 (`31681668145`) succeeded on that exact SHA.

Stage 7-C — Bounded Baseline Training Run + Evidence — merged through PR #23 from exact source head `7a993304218fa19609ea512665148dac3eea503a` to exact `main` commit `2c2c478eb361fa90a3bccd819b623680eb12de0b`. Exact-head run #67 (`31691794239`) passed **361/361 tests**, the guarded benchmark, and the authoritative synthetic-only run. Post-merge exact-main run #68 (`31692849892`) passed **361/361 tests**, exact runtime checks, `pip check`, and `compileall`. Stage 7-C closure synchronization then merged through PR #24 to exact `main` commit `e56fad04e43bc6302a8cda60c6f382e83a23d734`; post-merge run #70 (`31693998756`) succeeded. The accepted evidence is recorded in `STAGE7C_EVIDENCE.md`; Stage 7-C is `CLOSED — MAIN CI VERIFIED`.

Stage 8-0 — Real Data & Fine-Tuning Contract Freeze — merged through PR #25 to exact `main` commit `86487a4c3c41264b02bd159cd647a1318d9b9b88`; post-merge run #75 (`31698691405`) succeeded. Closure synchronization merged through PR #26 to exact `main` commit `99ffaf41f9c8919827ca97edc1bc3900db29eea2`; post-merge run #77 (`31699497562`) succeeded. Stage 8-0 is `CLOSED — MAIN CI VERIFIED`.

Stage 8-1 — Quarantine / Intake + Byte-Level Validation — is the only active package. It is limited to a bytes-only quarantine validator, exact source/image/MusicXML hash binding, strict grayscale-PNG verification, the existing supported-V1 MusicXML/token semantic gate, deterministic hash-only byte-validation receipts, perceptual near-duplicate review, synthetic-fixture tests, and documentation. It does not authorize real-data ingestion/persistence, checkpoint access, training/fine-tuning, test access, Stage 8-2/8-3, Stage 9/10, or ScoreMosaic integration.

The Stage 7-C checkpoint artifact `9177923796` remains temporary and scheduled to expire on 2026-09-12. Stage 8-1 does not load, move, copy, publish, or preserve it. The accepted Stage 7-C model remains non-production: exact sequence accuracy is 0% and token error rate is approximately 80.5%.

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
| 8-1 | Quarantine/intake + byte-level validation | 🔄 Active package — synthetic fixtures only |
| 8-2+ | Experiment profile / train-validation execution | 🔒 Not started |
| 9 | Benchmark and candidate decision | 🔒 Not started — test sealed |
| 10 | ScoreMosaic candidate integration | 🔒 Not started |

## Verified merge evidence

- Stage 3 merged through PR #11 at main commit `3b1e94cea8ac3a26a5df1e2038acc4331a24d371`.
- CI baseline merged through PR #12 at main commit `5abbc9859a4a69bf9a17936bc41e722256f87472`; post-merge run `31647615123` succeeded.
- Current-state documentation synchronized through PR #13 at main commit `23739ddfab618a0406836e94bb0ced1a124f8886`; post-merge run `31648164533` succeeded.
- Stage 4 merged through PR #14 at main commit `f0fd8a732b51b4aa95a66c3a780d0cefa6661361`; post-merge run `31660215130` succeeded.
- Stage 5-A merged through PR #15 at main commit `d677f3d27ac710c56c5ce677a46dc62bcf77bd84`; post-merge run `31671919885` succeeded.
- Stage 5 closure documentation merged through PR #16 at main commit `ed343d4984aae4507a2dd3238cfd1a98fb25b4b7`; post-merge run `31672540732` succeeded.
- Stage 6 final PR head `cfd0ac780595a38e0fe041d2d70293b39f96fcf3` passed run #17 (`31673631608`) with **309/309 tests**, pinned runtime checks, `pip check`, real Stage 1→6 integration/rebuild/persistence tests, and `compileall`; PR #17 merged to `7c3c736e6d3755d1bd098e2874d73ce5ed41e39f`, and run #18 (`31674014666`) succeeded.
- Stage 6 closure documentation merged through PR #18 to `046c9a4e7e41e94b0b4465a2610f30361055a3ed`; run #20 (`31674836433`) succeeded.
- Stage 7-A PR merge candidate from source `0eca1d881c494c763692aa72b1092d741c32be83` passed run #21 (`31675330236`) with **309/309 tests**, pinned runtime checks, `pip check`, and `compileall`; PR #19 merged to `0f04b0182b6753cfb8816d1287adb5ee973e0c28`, and run #22 (`31675913632`) succeeded.
- Stage 7-A closure synchronization merged through PR #20 to `6a13760d9d17130ea86636f4828ff1bff035f30d`; run #24 (`31676871798`) succeeded.
- Stage 7-B final source `a8ad8bc9f14953f0ed35ef5a5a8275be69af5ebd` passed run #30 (`31679413312`) with **336/336 tests**, exact runtime pins, `pip check`, missing-`EOS` regression, deterministic CPU smoke evidence, and `compileall`; PR #21 merged to `d02dce4ee17dfccf6f05519ab0970fdc188d0147`, and run #31 (`31679810478`) succeeded.
- Stage 7-B closure documentation merged through PR #22 to `1befaf260023852ef3bee5c8abab016f464557bb`; run #33 (`31681668145`) succeeded.
- Stage 7-C source `7a993304218fa19609ea512665148dac3eea503a` passed run #67 (`31691794239`) with **361/361 tests**, a guarded benchmark, and accepted authoritative evidence; PR #23 merged to `2c2c478eb361fa90a3bccd819b623680eb12de0b`, and run #68 (`31692849892`) succeeded.
- Stage 7-C closure synchronization merged through PR #24 to `e56fad04e43bc6302a8cda60c6f382e83a23d734`; run #70 (`31693998756`) succeeded.
- Stage 8-0 PR #25 merged to `86487a4c3c41264b02bd159cd647a1318d9b9b88`; run #75 (`31698691405`) succeeded with pinned runtime verification, `pip check`, the complete repository test suite, and `compileall`.
- Stage 8-0 closure synchronization PR #26 merged to `99ffaf41f9c8919827ca97edc1bc3900db29eea2`; run #77 (`31699497562`) succeeded.

The `CI VERIFIED` statement applies only to exact GitHub commits or GitHub-generated PR merge candidates that GitHub-hosted CI exercised.

## Stage 7 closed capability boundary

Stage 7 froze and verified Stage 5/6 validated synthetic-only input; strict train/validation/test isolation; deterministic grayscale preprocessing and supported-V1 token targets; the bounded CNN/GRU baseline; train-only optimization and validation-only checkpoint selection; exact evidence hashes; and one real bounded synthetic-only Stage 7-C run with no test access.

The accepted Stage 7-C run completed 40 epochs/1560 steps, improved validation loss from `3.560342441426505` to `0.9992435761603257`, and produced 21/21 semantically valid and MusicXML-regenerable validation predictions. Exact sequence accuracy remained 0% and token error rate remained approximately 80.5%, so no production-candidate claim is made.

## Stage 8-0 closed capability boundary

Contract: [STAGE8_REAL_DATA_CONTRACT.md](STAGE8_REAL_DATA_CONTRACT.md)

Stage 8-0 froze metadata-level rights/provenance/permission/privacy, quarantine/admission, image–MusicXML pairing, immutable identity, exact duplicate/leakage vetoes, train/validation-only development access, sealed-test isolation, Candidate A/B experiment definitions, and ScoreMosaic/teacher-correction isolation. It did not ingest files, load a checkpoint, or run training.

## Stage 8-1 active boundary

Contract: [STAGE8_1_INTAKE_CONTRACT.md](STAGE8_1_INTAKE_CONTRACT.md)

Stage 8-1 may implement only:

- in-memory byte validation of quarantined train/validation candidates;
- exact source-document, training-image, and MusicXML SHA-256 binding;
- bounded full-decode verification of 8-bit grayscale non-interlaced single-frame PNG training images;
- existing offline MusicXML 4.0 XSD, ST semantic, supported-V1 projection, and tokenizer/detokenizer gates;
- deterministic semantic and policy fingerprints;
- hash-only byte-validation receipts;
- deterministic perceptual near-duplicate candidate detection as an additional family/leakage veto;
- a handoff validator requiring both the admitted Stage 8-0 manifest and one matching Stage 8-1 receipt per sample.

A test record is rejected before supplied bytes are inspected. The development manifest remains train/validation-only plus the opaque sealed-test-manifest commitment. Stage 8-1 has no persistence writer, network fetcher, upload path, real corpus, checkpoint access, training code, or ScoreMosaic connector.

## CI baseline

The active `.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual dispatch using GitHub-hosted Ubuntu 24.04 / Python 3.13, read-only repository contents permission, pinned GitHub action commits, exact dependency/runtime checks, `pip check`, complete unittest discovery, and Python compile validation.

The isolated official-PyTorch-CPU-index install verifies exact `torch==2.13.0+cpu`. The one-shot PR #23 benchmark/training exception is retired; ordinary CI contains no full training path.

## Required Stage 8-1 gate

```text
exact verified Stage 8-0 main baseline
        ↓
bytes-only quarantine validator
        ↓
exact source/image/MusicXML hash binding
        ↓
full grayscale-PNG structural/decode verification
        ↓
MusicXML supported-V1/token semantic gate
        ↓
deterministic semantic + policy fingerprints
        ↓
hash-only byte-validation receipt
        ↓
perceptual near-duplicate review veto
        ↓
sealed-test early veto
        ↓
focused positive/negative tests
        ↓
full regression + pip check + compileall
        ↓
exact PR-head / merge-candidate GitHub CI
        ↓
separate explicit merge approval
        ↓
post-merge exact-main CI
        ↓
Stage 8-1 CLOSED
```

## Next gate

Stage 8-1 must close through merge and exact-main CI before any later Stage 8 work. The next planned small gate after closure is **Stage 8-2 — Paired Experiment Run Profile Freeze**: configuration/acceptance contract only, without running fine-tuning and without opening either sealed test split. Stage 8-2, Stage 8-3, Stage 9, and Stage 10 remain locked.
