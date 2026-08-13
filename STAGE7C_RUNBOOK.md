# Stage 7-C Bounded Baseline Run Profile

Status: active bounded Stage 7-C package on `stage-7c-baseline-run`.

This document defines how the first real synthetic-only ST-OMR baseline run is executed and evidenced. It does not open the Stage 6 test split, ingest real/user material, or authorize Stage 8–10.

## Verified starting boundary

Stage 7-C starts from exact verified `main` commit `1befaf260023852ef3bee5c8abab016f464557bb`, after Stage 7-B closure documentation merged through PR #22 and post-merge GitHub Actions run #33 (`31681668145`) succeeded on that exact SHA.

Stage 7-C reuses the already-frozen Stage 7-A contract and the closed Stage 7-B implementation profile. It does not introduce a second model family, tokenizer, data adapter, optimizer policy, or hidden augmentation path.

## Execution surface

Low-level bounded run implementation: `st_omr_training/training_run.py`.

Authoritative provenance gate: `st_omr_training/stage7c_execution.py`.

Frozen baseline dataset profile: `st_omr_training/stage7c_dataset.py`.

Authoritative knobs-free entrypoint: `python -m st_omr_training.stage7c_cli --workspace <fresh-directory-outside-repository>`.

The authoritative entrypoint is the only normal Stage 7-C execution path. It verifies the clean repository and exact runtime, builds and persists the frozen Stage 6 dataset profile, then executes the authoritative training/provenance gate. The workspace must be fresh and outside the Git repository so generated artifacts cannot alter source provenance.

Only `train` and `validation` are requested. `DatasetSplit.TEST` remains sealed until Stage 9.

## Frozen Stage 7-C dataset

The real baseline run is not allowed to accept an arbitrary caller-selected Stage 6 build. Stage 7-C freezes one exact synthetic build profile:

- dataset name: `st-omr-stage7c-baseline-v1`;
- dataset version: `v1`;
- 64 symbolic families;
- seed start: `70000`;
- split seed: `7001`;
- 8 measures per generated score;
- raster width: 1000;
- all frozen Stage 6 V1 family profiles: mixed, note-only, rest-only, chord-only, 2/4, 3/4, 4/4, and no-accidentals;
- all frozen Stage 4 dataset degradation profiles: clean, light, and medium;
- exact Stage 6 80/10/10 family-level split policy.

The profile has its own deterministic Stage 6 configuration fingerprint. The authoritative entrypoint refuses any build whose config fingerprint differs from the frozen value. This prevents a tiny or selectively simplified synthetic dataset from being substituted for Stage 7-C evidence.

## Bounded training profile

- CPU only;
- exact `torch==2.13.0+cpu` runtime inherited from Stage 7-B;
- exact pinned `lxml`, `verovio`, `CairoSVG`, and `Pillow` runtime identities;
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

## Authoritative source/runtime gate

A low-level `COMPLETE` marker proves only that the bounded training function reached its normal artifact-writing end state. It is **not sufficient** for Stage 7-C acceptance.

An authoritative run must execute through `run_verified_baseline_training(...)`. That gate:

1. requires an explicit Git repository root;
2. resolves and records the exact 40-character Git `HEAD` itself rather than trusting caller-supplied provenance;
3. rejects a nested path masquerading as the repository root;
4. rejects modified or untracked files before training;
5. verifies the exact pinned runtime dependency versions before training;
6. executes the bounded baseline run using the verified Git HEAD;
7. verifies the repository remains on the same clean HEAD after training;
8. verifies the runtime identities remain unchanged after training;
9. independently re-hashes the selected checkpoint and metrics file;
10. requires the `COMPLETE` marker to bind exactly to the metrics artifact;
11. requires the metrics artifact to be canonical JSON;
12. cross-checks repository SHA, Stage 6 build/manifest identity, runtime provenance, and sealed-test status inside the metrics evidence;
13. safely reloads the selected checkpoint with `weights_only=True`, strictly loads it into the frozen bounded model, rechecks finite model state, and compares the reloaded state SHA-256 with metrics evidence;
14. verifies the clean Git HEAD once more before final acceptance;
15. writes a hash-addressed `VERIFIED-<sha256>.json` marker only after every check passes.

Only a run containing a valid `VERIFIED-<sha256>.json` from this authoritative gate is eligible to become Stage 7-C evidence.

## Acceptance gate

A Stage 7-C run is accepted only when all of the following are true:

1. the clean repository checkout and exact Git HEAD are independently verified before and after the run;
2. every pinned Stage 7-C runtime dependency is exact before and after the run;
3. the exact frozen Stage 7-C Stage 6 dataset profile and fingerprint are used;
4. train/validation artifacts cross the existing Stage 5/6 and Stage 7-B gates;
5. the model is created through the Stage 7-B bounded model builder;
6. all numeric training state remains finite;
7. the best trained validation loss is strictly lower than the deterministic untrained validation loss;
8. validation predictions report token error rate, exact sequence accuracy, detokenization success, semantic validity, and MusicXML regeneration validity;
9. at least one validation prediction detokenizes and reconstructs a canonically valid supported-V1 score;
10. the selected checkpoint and metrics/provenance evidence are SHA-256 addressed;
11. the run records repository SHA, dataset build/manifest identities, run/tokenizer/preprocess/model/trainer fingerprints, runtime/dependency identity, seeds, parameter count, epochs/steps, checkpoint hash, and metrics hash;
12. the selected checkpoint safely reloads and reproduces the recorded model-state hash;
13. the sealed test split is not opened;
14. the authoritative `VERIFIED-<sha256>.json` marker is present and independently hash-valid.

Stage 7-C is still a trainability/baseline gate. These metrics do not declare the model production-quality. Production candidacy remains Stage 9 after sealed-test evaluation.

## Artifact safety

Each run receives a deterministic `run_id` from the verified repository SHA, Stage 6 build identity, manifest SHA, and run fingerprint.

The run directory must not already exist. A new run starts with an `INCOMPLETE` marker. Silent checkpoint resume is not implemented or permitted. Only after the selected checkpoint and canonical metrics file are successfully written and hash-verified is `COMPLETE` written and `INCOMPLETE` removed.

Checkpoint files use `checkpoint-<sha256>.pt`. Metrics/provenance use `metrics-<sha256>.json`. The authoritative gate adds `VERIFIED-<sha256>.json`. These are derived run artifacts and remain outside normal Git content under the existing `.gitignore` rules. A later small evidence summary may be committed only after the real run is independently verified.

## CI boundary

GitHub-hosted CI does not execute the real Stage 7-C baseline run.

PR CI may run only bounded contract/component smoke tests: configuration bounds, frozen dataset-profile identity, workspace safety, semantic reconstruction, metric accounting, hash-addressed artifact handling, no-resume behavior, source/runtime provenance checks, strict checkpoint reload verification, full repository regression, and compile validation.

The real baseline run must execute separately on the exact pinned CPU training runtime through the authoritative entrypoint and provenance gate. Its evidence must then be reviewed before this package can be considered merge-ready.

## Explicitly locked

Stage 7-C does not:

- inspect or score the Stage 6 test split;
- tune thresholds against test data;
- accept an alternate or caller-customized dataset profile as Stage 7-C evidence;
- ingest real/user/rights-unclear scores;
- fine-tune on teacher corrections;
- use pretrained weights, Audiveris, Scan2Notes, LLMs, or other recognition teachers;
- use CUDA, ROCm, distributed, cloud, or network-dependent training;
- declare an ST-OMR production candidate;
- integrate with ScoreMosaic;
- start Stage 8, Stage 9, or Stage 10.

Merge remains a separate explicit approval gate.