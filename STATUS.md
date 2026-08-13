# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

The repository is public and GitHub Actions CI is active.

Stages 0 through 6, all Stage 7 packages, Stage 8-0, and Stage 8-1 are complete on `main`.

The exact verified Stage 8-2 starting main is `de5d66d30c66c97c03bc1dc60fce094a9f0d64e7`. That commit is the Stage 8-1 closure synchronization from PR #30; post-merge GitHub Actions run #94 (`31705382162`) succeeded with pinned dependency verification, the complete repository test suite, `pip check`, and `compileall`.

Stage 8-2 — Paired Experiment Run Profile Freeze — is the only active package. It is contract/configuration/evidence-binding work only. No real data is ingested, no checkpoint is loaded or moved, no training/fine-tuning is executed, and neither sealed test split is opened.

The Stage 7-C checkpoint artifact `9177923796` remains temporary and scheduled to expire on 2026-09-12. The accepted Stage 7-C model remains non-production: exact sequence accuracy is 0% and token error rate is approximately 80.5%.

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
| 8-1 | Quarantine/intake + byte-level validation | ✅ Closed — main CI verified |
| 8-2 | Paired experiment run profile freeze | 🔄 Active package — no training |
| 8-3A | Pilot source→training-PNG preparation + admission | 🔒 Not started |
| 8-3B | Paired real train/validation execution | 🔒 Not started |
| 9 | Benchmark and candidate decision | 🔒 Not started — test sealed |
| 10 | ScoreMosaic candidate integration | 🔒 Not started |

## Recent verified merge evidence

- Stage 6 closure documentation merged through PR #18 to `046c9a4e7e41e94b0b4465a2610f30361055a3ed`; run #20 (`31674836433`) succeeded.
- Stage 7-A merged through PR #19 to `0f04b0182b6753cfb8816d1287adb5ee973e0c28`; post-merge run #22 (`31675913632`) succeeded. Closure synchronization PR #20 merged to `6a13760d9d17130ea86636f4828ff1bff035f30d`; run #24 (`31676871798`) succeeded.
- Stage 7-B final source `a8ad8bc9f14953f0ed35ef5a5a8275be69af5ebd` passed run #30 (`31679413312`) with **336/336 tests**; PR #21 merged to `d02dce4ee17dfccf6f05519ab0970fdc188d0147`, and run #31 (`31679810478`) succeeded. Closure synchronization PR #22 merged to `1befaf260023852ef3bee5c8abab016f464557bb`; run #33 (`31681668145`) succeeded.
- Stage 7-C source `7a993304218fa19609ea512665148dac3eea503a` passed run #67 (`31691794239`) with **361/361 tests**, guarded benchmark, and accepted authoritative evidence. PR #23 merged to `2c2c478eb361fa90a3bccd819b623680eb12de0b`; run #68 (`31692849892`) succeeded. Closure synchronization PR #24 merged to `e56fad04e43bc6302a8cda60c6f382e83a23d734`; run #70 (`31693998756`) succeeded.
- Stage 8-0 PR #25 merged to `86487a4c3c41264b02bd159cd647a1318d9b9b88`; run #75 (`31698691405`) succeeded. Closure synchronization PR #26 merged to `99ffaf41f9c8919827ca97edc1bc3900db29eea2`; run #77 (`31699497562`) succeeded.
- Stage 8-1 PR #29 merged from exact source `ed4113a25f6e12055b9277f959a4580259d37d40` to `d551cb27e0244477379100c45e06193ea7ca0cf8`; post-merge run #92 (`31704137450`) succeeded with **394/394 tests**, `pip check`, pinned runtime checks, and `compileall`.
- Stage 8-1 closure synchronization PR #30 merged to exact `main` `de5d66d30c66c97c03bc1dc60fce094a9f0d64e7`; post-merge run #94 (`31705382162`) succeeded.

The `CI VERIFIED` statement applies only to exact GitHub commits or GitHub-generated PR merge candidates exercised by GitHub-hosted CI.

## Stage 7 closed capability boundary

Stage 7 proved that the validated Stage 5/6 synthetic pipeline can train the frozen CNN/GRU baseline and produce reproducible evidence without test access. The accepted Stage 7-C run completed 40 epochs/1560 steps, improved validation loss from `3.560342441426505` to `0.9992435761603257`, and produced 21/21 semantically valid and MusicXML-regenerable validation predictions. Exact sequence accuracy remained 0% and token error rate remained approximately 80.5%, so no production-candidate claim is made.

## Stage 8-0 closed capability boundary

Contract: [STAGE8_REAL_DATA_CONTRACT.md](STAGE8_REAL_DATA_CONTRACT.md)

Stage 8-0 froze rights/provenance/permission/privacy, quarantine/admission, image–MusicXML pairing, immutable identity, exact duplicate/leakage vetoes, train/validation-only development access, sealed-test isolation, Candidate A/B definitions, and ScoreMosaic/teacher-correction isolation.

## Stage 8-1 closed capability boundary

Contract: [STAGE8_1_INTAKE_CONTRACT.md](STAGE8_1_INTAKE_CONTRACT.md)

Stage 8-1 froze and verified in-memory byte validation, pre-hash size guards, exact source/image/MusicXML hashes, grayscale-PNG validation, supported-V1 MusicXML/token semantics, deterministic hash-only receipts, bounded dHash near-duplicate review, a one-receipt-per-sample development handoff, and a contract-only real-test sealing boundary. No real files were ingested or persisted.

## Stage 8-2 active boundary

Contract: [STAGE8_2_RUN_PROFILE.md](STAGE8_2_RUN_PROFILE.md)

Stage 8-2 may implement only:

- the exact first real-data pilot profile: **50 admitted development pairs = 40 train + 10 validation**;
- a shared deterministic model/tokenizer/preprocess/trainer/data-order/metric/resource profile for Candidate A and Candidate B;
- Candidate A binding to the exact accepted Stage 7-C checkpoint/model-state hashes with no fallback;
- Candidate B binding to deterministic from-scratch initialization with no checkpoint;
- deterministic Stage 8-2 profile, admitted-manifest, Stage 8-1 receipt-set, and sealed-test commitment bindings;
- synthetic/hash-only focused tests and architecture/status documentation.

Stage 8-2 does not authorize real-data IO, source conversion, dataset persistence, checkpoint loading, optimizer steps, sealed-test access, Stage 9/10, or ScoreMosaic integration.

### First-pilot data rule

The pilot requires exactly 50 **admitted** pairs, not merely 50 raw JPEG/PDF + MusicXML files. A pair that fails rights, pairing, semantic, duplicate, leakage, near-duplicate, or byte validation does not count and must be replaced.

Source JPEG/PDF bytes remain unchanged outside Git. The model input remains Stage 8-1-valid grayscale PNG. Stage 8-1 deliberately did not define source-to-PNG normalization, so Stage 8-2 makes ad-hoc conversion a hard blocker. The next Stage 8 execution package must begin with Stage 8-3A to freeze and verify source→training-PNG derivation before any real training.

### Paired execution profile

Both candidates are frozen to the same 40/10 admitted manifest and Stage 8-1 receipt set, 40 epochs, batch size 4, canonical sample-id order, exact Stage 7-C model/trainer/preprocessing surface, AdamW at learning rate 0.001, no scheduler, one retained min-validation-loss checkpoint, 8-measure/1536-token validation decode, CPU/one thread, at most 1800 seconds per candidate and 3600 seconds for the pair.

The validation metric family remains validation loss, token error rate, exact sequence accuracy, detokenization success, semantic validity, and MusicXML regeneration validity. Stage 9 owns production-quality thresholds.

## CI baseline

The active `.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual dispatch using GitHub-hosted Ubuntu 24.04 / Python 3.13, read-only repository contents permission, pinned GitHub action commits, exact dependency/runtime checks, `pip check`, complete unittest discovery, and Python compile validation. Ordinary CI contains no full training path.

## Stage 8-2 closure gate

```text
exact Stage 8-1 closed main baseline
        ↓
50 admitted pair / exact 40+10 profile
        ↓
A/B same manifest + same receipt set + same run budget
        ↓
Candidate A exact Stage 7-C identity / no fallback
        ↓
Candidate B deterministic from scratch / no checkpoint
        ↓
sealed-test and online-learning vetoes
        ↓
focused tests + full regression + pip check + compileall
        ↓
exact PR-head GitHub CI
        ↓
separate merge approval
        ↓
post-merge exact-main CI
        ↓
Stage 8-2 CLOSED
```

## Next gate

Stage 8-2 must close through merge and exact-main CI before any real data preparation or training. The next small gate is **Stage 8-3A — Pilot Data Preparation + Admission**: freeze and verify source-document→training-PNG derivation, run Stage 8-0/8-1 admission on the external pilot material, and produce the exact 40/10 hash-bound handoff. It will still perform no model optimization until that handoff is accepted. Stage 8-3B training, Stage 9, and Stage 10 remain locked.
