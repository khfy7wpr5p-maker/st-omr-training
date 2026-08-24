# Meter V5-3H — Exact-SHA Authoritative Rescue TRAIN Colab Wrapper V1

## Decision

V5-3H is the execution wrapper for the already-authorized single V5-3G TRAIN run.
It is pinned to:

- V5-3G HEAD `b36a9d2f5daade2c3568cac8cbc736ca75ca435f`
- CI run `32769348282` (CI #609, SUCCESS)

The notebook contains exactly one code cell and exactly one call to
`run_authoritative_rescue_training_v1`.

## Runtime binding

Before importing ST-OMR code, the wrapper:

1. mounts Google Drive;
2. verifies the existing V5, specialist-checkpoint, M4A and D10 roots;
3. fetches the exact V5-3G commit;
4. checks out that commit detached;
5. requires a clean repository worktree;
6. installs the repository's pinned runtime requirements;
7. verifies the exact package versions and `pip check`.

The training execution therefore cannot silently run against a different Git
HEAD or a drifted training runtime.

## One-shot output boundary

Before any training call, the wrapper refuses to run when any of these already
exists:

- V5-3G authoritative report;
- V5-3G rescue artifact directory;
- V5-3G temporary artifact directory;
- V5-3H execution envelope.

No overwrite or automatic rerun path exists.

## Single authorized execution

The wrapper locates the exact frozen 2-AI and 3-AI checkpoint hashes and calls
only the V5-3G authoritative entry with its exact approval token.

V5-3G itself:

- materializes only V5 TRAIN and historical TRAIN;
- reuses V5-2N frozen 64D features;
- selects same-specialist frozen-negative rows;
- requires exact preregistered group counts;
- delegates to the fixed V5-3F 110-step rescue harness;
- writes only separate rescue artifacts;
- verifies frozen specialist state bit-identity.

4-AI is not loaded.

## Post-run verification

The wrapper accepts the run only if:

- exactly one candidate configuration is recorded;
- numerical integrity = PASS;
- frozen-state isolation = PASS;
- both specialists have exact group counts;
- both specialists completed exactly 110 steps;
- all V5-3F finite-value guards are true;
- frozen state before and after is identical;
- rescue artifacts exist, were reload-verified and match their recorded SHA256;
- repository HEAD remains exactly the pinned V5-3G SHA;
- repository worktree remains clean.

It then writes one hash-bound execution envelope containing the report SHA,
rescue artifact SHAs and four-group feature fingerprints.

## Explicit stop

V5-3H does not execute:

- TRAIN performance acceptance;
- historical validation retention;
- immutable First-30;
- V5 reserve;
- V5 validation;
- FINAL_HOLDOUT;
- threshold tuning;
- hyperparameter sweep;
- second configuration;
- runtime/resolver wiring;
- production promotion.

The next stage may only read the completed V5-3G report/artifacts and evaluate
the preregistered TRAIN acceptance gate. A HOLD does not authorize retraining.
