# Stage 7-C Bounded Baseline Run Profile

Status: active bounded Stage 7-C package on `stage-7c-baseline-run`.

This document defines how the first real synthetic-only ST-OMR baseline run is executed and evidenced. It does not open the Stage 6 test split, ingest real/user material, or authorize Stage 8–10.

## Verified starting boundary

Stage 7-C starts from exact verified `main` commit `1befaf260023852ef3bee5c8abab016f464557bb`, after Stage 7-B closure documentation merged through PR #22 and post-merge GitHub Actions run #33 (`31681668145`) succeeded on that exact SHA.

Stage 7-C reuses the already-frozen Stage 7-A contract and the closed Stage 7-B implementation profile. It does not introduce a second model family, tokenizer, data adapter, optimizer policy, or hidden augmentation path.

## Execution surface

Implementation: `st_omr_training/training_run.py`

The run consumes only a validated Stage 6 `SyntheticDatasetBuild` and its exact persisted dataset directory. The existing Stage 7-B adapter revalidates manifest/build bytes, MusicXML target hashes, PNG hashes/dimensions, and token semantic round trips before samples are admitted.

Only `train` and `validation` are requested. `DatasetSplit.TEST` remains sealed until Stage 9.

Default bounded run profile:

- CPU only;
- exact `torch==2.13.0+cpu` runtime inherited from Stage 7-B;
- 40 epochs;
- batch size 4;
- at most 1024 train samples;
- at most 256 validation samples;
- at most 512 greedy decode tokens per validation sample;
- exactly one retained selected checkpoint;
- fixed sample-id ascending data order for every epoch;
- frozen PAD-masked cross-entropy + AdamW + gradient clipping from Stage 7-B;
- minimum validation loss selects the checkpoint.

The Stage 7-A outer ceilings remain authoritative: no more than 100 epochs, 25M trainable parameters, or 10 retained checkpoints. This first implementation deliberately freezes a tighter one-checkpoint retention rule.

## Acceptance gate

A Stage 7-C run is accepted only when all of the following are true:

1. the pinned PyTorch runtime is exact;
2. train/validation artifacts cross the existing Stage 5/6 and Stage 7-B gates;
3. the model is created through the Stage 7-B bounded model builder;
4. all numeric training state remains finite;
5. the best trained validation loss is strictly lower than the deterministic untrained validation loss;
6. validation predictions report token error rate, exact sequence accuracy, detokenization success, semantic validity, and MusicXML regeneration validity;
7. at least one validation prediction detokenizes and reconstructs a canonically valid supported-V1 score;
8. the selected checkpoint and metrics/provenance evidence are SHA-256 addressed;
9. the run records repository SHA, dataset build/manifest identities, run/tokenizer/preprocess/model/trainer fingerprints, runtime/dependency identity, seeds, parameter count, epochs/steps, checkpoint hash, and metrics hash;
10. the sealed test split is not opened.

Stage 7-C is still a trainability/baseline gate. These metrics do not declare the model production-quality. Production candidacy remains Stage 9 after sealed-test evaluation.

## Artifact safety

Each run receives a deterministic `run_id` from the repository SHA, Stage 6 build identity, manifest SHA, and run fingerprint.

The run directory must not already exist. A new run starts with an `INCOMPLETE` marker. Silent checkpoint resume is not implemented or permitted. Only after the selected checkpoint and canonical metrics file are successfully written and hash-verified is `COMPLETE` written and `INCOMPLETE` removed.

Checkpoint files use `checkpoint-<sha256>.pt`. Metrics/provenance use `metrics-<sha256>.json`. These are derived run artifacts and remain outside normal Git content under the existing `.gitignore` rules. A later small evidence summary may be committed only after the real run is independently verified.

## CI boundary

GitHub-hosted CI does not execute the real Stage 7-C baseline run.

PR CI may run only bounded contract/component smoke tests: configuration bounds, semantic reconstruction, metric accounting, hash-addressed artifact handling, no-resume behavior, full repository regression, and compile validation.

The real baseline run must execute separately on the exact pinned CPU training runtime. Its evidence must then be reviewed before this package can be considered merge-ready.

## Explicitly locked

Stage 7-C does not:

- inspect or score the Stage 6 test split;
- tune thresholds against test data;
- ingest real/user/rights-unclear scores;
- fine-tune on teacher corrections;
- use pretrained weights, Audiveris, Scan2Notes, LLMs, or other recognition teachers;
- use CUDA, ROCm, distributed, cloud, or network-dependent training;
- declare an ST-OMR production candidate;
- integrate with ScoreMosaic;
- start Stage 8, Stage 9, or Stage 10.

Merge remains a separate explicit approval gate.