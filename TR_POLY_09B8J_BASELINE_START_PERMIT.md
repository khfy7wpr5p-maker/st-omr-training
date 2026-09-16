# TR-POLY-09B8J — First Real Baseline Start Permit

Status: implementation package; real corpus remains absent.

B8J is the final fail-closed authorization boundary before the first real Native Polyphonic V2 quality-training run may call B6.

## Purpose

B8J does not admit data and does not train by itself. It requires the exact identity chain already established by:

1. **B8A** — persisted Native V2 dataset reload/preflight;
2. **B8I** — admitted real-data ↔ Native V2 lineage receipt;
3. **B8B** — frozen first-baseline experiment recipe.

A start permit is emitted only when all three surfaces bind the same dataset manifest, build, B8A preflight receipt, TRAIN population, VALIDATION population and repository identity.

## Authorized execution path

```text
rights/provenance-admitted real corpus
        ↓
B8A exact persisted Native V2 preflight
        ↓
B8I exact real-data ↔ Native V2 lineage admission
        ↓
B8B exact first-baseline experiment recipe
        ↓
B8J exact baseline start permit
        ↓
B8J authorized B6 wrapper (full population only)
        ↓
B6/B4 training + B5 checkpoint persistence
```

The B8J wrapper does not expose B6 prefix/subsample controls. Both `max_train_samples` and `max_validation_samples` are forced to `None` for the first real baseline.

## Permit identity

The permit hash-binds:

- repository SHA;
- dataset manifest SHA-256;
- dataset build ID;
- B8A preflight receipt fingerprint;
- B8I admission receipt fingerprint;
- B8B recipe fingerprint;
- exact TRAIN and VALIDATION sample IDs;
- exact deterministic batch sizes;
- batch size, epoch count and optimizer-step budget;
- model profile;
- quality trainer profile;
- Native V2 materialization fingerprint;
- B6 quality execution profile.

## Runtime revalidation

Immediately before B6 execution, B8J independently rechecks:

- current B8A/B8I/B8B evidence still reproduces the supplied permit;
- runtime model config equals the B8B model fingerprint;
- runtime quality-training config equals the B8B trainer fingerprint;
- runtime materialization and B6 execution profiles equal the B8B recipe;
- epoch count and optimizer-step ceiling equal the frozen recipe.

After B6 returns, B8J verifies that the result used the exact permitted dataset/build, full TRAIN/VALIDATION populations, deterministic batch plan, execution/materialization/trainer identities and exact required optimizer-step count.

## Safety boundary

B8J:

- never opens TEST artifact bytes;
- never grants production authority;
- never grants commercial-use authority;
- never converts teacher corrections or ScoreMosaic uploads into training data;
- never bypasses B8I rights/provenance/permission/privacy/pairing admission;
- never allows prefix/subsample execution for the first measured baseline.

A green B8J CI result proves only that this authorization machinery works on test fixtures. It is not real model-quality evidence.

## Current blocker

The connected real-corpus storage remains empty. Therefore no real B8A preflight, B8I admission receipt, B8B real recipe fingerprint, B8J permit, B6 real checkpoint or B7 measured VALIDATION result can currently be produced.
