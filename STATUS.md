# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

The repository is public and GitHub Actions CI is active.

Stages 0 through 6 and Stage 7-A are complete on `main`.

Stage 7-A — Baseline Training Contract Freeze — merged through PR #19 at exact `main` commit `0f04b0182b6753cfb8816d1287adb5ee973e0c28`; post-merge GitHub Actions run #22 (`31675913632`) succeeded. Stage 7-A closure/status synchronization then merged through PR #20 at exact `main` commit `6a13760d9d17130ea86636f4828ff1bff035f30d`; post-merge run #24 (`31676871798`) also succeeded. The integrated repository through the Stage 7-A boundary is therefore `CI VERIFIED`.

Stage 7-B — Tokenizer/Data/Model/Trainer Implementation — is the active bounded package on branch `stage-7b-training-smoke`.

Stage 7-B is limited to the deterministic smoke path defined by `TRAINING_CONTRACT.md`: exact framework pin, frozen tokenizer/detokenizer, persisted Stage 5/6 artifact revalidation, deterministic no-crop preprocessing, one bounded from-scratch baseline model, train-only CPU smoke update, validation-only metric path, fail-closed numeric checks, and deterministic CPU replay evidence.

Stage 7-C real baseline training has **not started**. Stage 8 real-data fine-tuning, Stage 9 sealed benchmark/candidate work, and Stage 10 ScoreMosaic integration remain locked.

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
| 7-B | Tokenizer/data/model/trainer smoke implementation | 🔄 Active bounded package — PR CI required |
| 7-C | Bounded baseline training run + evidence | 🔒 Not started |
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
- Stage 6 closure documentation merged through PR #18 at exact main commit `046c9a4e7e41e94b0b4465a2610f30361055a3ed`; post-merge GitHub Actions run #20 (`31674836433`) succeeded.
- Stage 7-A PR merge candidate from source head `0eca1d881c494c763692aa72b1092d741c32be83` against exact base `046c9a4e7e41e94b0b4465a2610f30361055a3ed` passed GitHub Actions run #21 (`31675330236`) with **309/309 tests**, pinned runtime checks, `pip check`, and `compileall`.
- Stage 7-A merged through PR #19 at exact main commit `0f04b0182b6753cfb8816d1287adb5ee973e0c28`; post-merge GitHub Actions run #22 (`31675913632`) succeeded.
- Stage 7-A closure synchronization merged through PR #20 at exact main commit `6a13760d9d17130ea86636f4828ff1bff035f30d`; post-merge GitHub Actions run #24 (`31676871798`) succeeded.

The `CI VERIFIED` statement applies only to exact GitHub commits or GitHub-generated PR merge candidates that GitHub-hosted CI exercised.

## Stage 7-A closed capability boundary

Contract: [TRAINING_CONTRACT.md](TRAINING_CONTRACT.md)

Stage 7-A froze:

- Stage 5/6 validated synthetic artifacts as the only Stage 7 input;
- strict train/validation/test isolation with test sealed until Stage 9;
- deterministic grayscale input preprocessing requirements;
- a compact semantic `ST-OMR V1 token sequence` rather than raw XML text;
- a finite explicit vocabulary and tokenizer/detokenizer semantic round-trip gate;
- one from-scratch visual-encoder/sequence-decoder baseline with a 25M parameter ceiling;
- no pretrained weights or external recognition teachers;
- run/config/model/tokenizer fingerprints and provenance;
- bounded resources, validation-only Stage 7 metrics, and separate Stage 7-B/7-C gates.

## Stage 7-B active capability boundary

Concrete implementation profile: [TRAINING_IMPLEMENTATION.md](TRAINING_IMPLEMENTATION.md)

Current Stage 7-B scope includes:

- `torch==2.13.0+cpu` pinned separately in `requirements-training.txt` and installed from the official PyTorch CPU index;
- exact 35-token frozen vocabulary with deterministic ids and fingerprint;
- Stage 2-D semantic MusicXML → token → semantic-projection exact round trip;
- persisted Stage 6 manifest/build/target/image revalidation before train/validation admission;
- hard rejection of the sealed `test` split at the Stage 7-B data and batch boundary;
- fingerprinted 1×64×512 grayscale fit/no-upscale/no-crop/center-white-pad preprocessing;
- one small from-scratch CNN + context-conditioned GRU baseline, runtime parameter-ceiling enforcement, and no external model calls;
- cross-entropy with PAD masking, AdamW, no scheduler, max gradient norm 1.0, and a bounded CPU smoke-step count;
- NaN/Infinity vetoes across inputs, logits, loss, gradients, model state, optimizer state, gradient norm, and reported metrics;
- validation-loss path that is forbidden to mutate model state;
- exact same-seed CPU smoke replay requirement based on model-state SHA-256.

Stage 7-B does **not** run the real baseline training job, open test data, retain production checkpoints, ingest real/user material, or integrate with ScoreMosaic.

## CI baseline

The active `.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual dispatch using GitHub-hosted Ubuntu 24.04 / Python 3.13, read-only repository contents permission, pinned GitHub action commits, exact dependency/runtime checks, `pip check`, complete unittest discovery, and Python compile validation.

Stage 7-B adds an isolated official-PyTorch-CPU-index install for `requirements-training.txt` and verifies the exact PyTorch runtime pin before tests. Full Stage 7-C training remains prohibited in ordinary GitHub-hosted CI.

The existing pinned checkout/setup-python action commits may emit a non-failing Node.js runtime maintenance warning. That is a separate CI-maintenance concern and is not mixed into Stage 7-B unless it becomes a blocking failure.

## Required Stage 7-B gate

```text
framework/runtime compatibility + exact pin
        ↓
frozen tokenizer vocabulary
        ↓
MusicXML → tokens → semantic projection exact round trip
        ↓
Stage 5/6 artifact + split revalidation
        ↓
deterministic input preprocessing
        ↓
bounded baseline model construction
        ↓
parameter ceiling check
        ↓
CPU smoke forward/backward/update
        ↓
NaN/Infinity fail-closed tests
        ↓
train-only update / validation-only metric / test-sealed tests
        ↓
same-seed deterministic CPU smoke replay
        ↓
real persisted Stage 6 → Stage 7-B bridge
        ↓
full repository regression
        ↓
compile validation
        ↓
GitHub-hosted PR CI
        ↓
separate merge approval
        ↓
post-merge exact-main CI
```

## Next gate

Stage 7-B must not merge until its final GitHub-hosted PR CI is green and merge receives separate explicit approval. Stage 7-C must not start implicitly after Stage 7-B. Stage 7-C, Stage 8, Stage 9, and Stage 10 remain locked until their own bounded packages are separately approved.
