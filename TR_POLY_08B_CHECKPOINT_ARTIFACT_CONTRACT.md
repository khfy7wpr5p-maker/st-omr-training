# TR-POLY-08B — Exact Checkpoint Artifact Contract

## Purpose

TR-POLY-08A added a bounded in-memory training/provenance harness for the tiny Polyphonic V2 2D Transformer. It intentionally wrote no checkpoint and kept the model registry checkpoint gate closed.

TR-POLY-08B adds one narrow capability: persist that already-bounded research state as an auditable, one-shot checkpoint artifact and prove that it reloads exactly.

It does not run repository data, open TEST, benchmark a model, or promote a model.

## Artifact layout

One artifact directory contains exactly the required named surfaces:

- `model.pt` — model state plus duplicated exact metadata;
- `metadata.json` — canonical ASCII JSON metadata;
- `receipt.json` — canonical ASCII JSON hashes for `model.pt` and `metadata.json`.

The write path is one-shot. Existing final or temporary artifact directories are rejected rather than overwritten.

## Checkpoint schema

Checkpoint schema:

`st-omr-poly-2d-checkpoint-v1`

Receipt schema:

`st-omr-poly-2d-checkpoint-receipt-v1`

The model file contains only:

- schema version;
- metadata object;
- CPU tensor `state_dict`.

No optimizer object, arbitrary Python object, executable callback, dataset object, or runtime authority is persisted.

## Reload safety

Acceptance requires:

1. the receipt to be canonical JSON with the exact key set;
2. `model.pt` SHA-256 to match the receipt;
3. `metadata.json` SHA-256 to match the receipt;
4. metadata to be canonical and schema-valid;
5. `torch.load(..., map_location="cpu", weights_only=True)`;
6. exact checkpoint top-level keys;
7. embedded metadata to equal the canonical sidecar;
8. `state_dict` to contain only string→tensor entries;
9. strict state loading into the exact model config;
10. finite reloaded model state;
11. exact parameter count;
12. exact reloaded `model_state_sha256` equality.

Byte tampering is therefore rejected before deserialization when the receipt hash no longer matches.

## Exact provenance binding

Metadata binds:

- repository SHA-40;
- current registry-row fingerprint;
- dataset-manifest SHA-256;
- preprocess fingerprint SHA-256;
- model-profile SHA-256;
- trainer-profile SHA-256;
- provenance SHA-256;
- V2 tokenizer fingerprint;
- final model-state SHA-256;
- parameter count;
- optimizer-step count;
- complete bounded model config;
- complete bounded trainer config;
- complete training provenance payload;
- pinned PyTorch version;
- frozen V2 representation/tokenizer versions.

The embedded configs and provenance are reconstructed and independently fingerprinted during verification.

## Claim boundary

Every TR-POLY-08B checkpoint is fixed to:

- `checkpoint_role = bounded_research_artifact_only`;
- `authoritative_dataset_execution = false`;
- `test_split_accessed = false`;
- `benchmark_evidence = false`;
- `production_authority = false`.

A checkpoint cannot opt into any of those claims by metadata mutation.

## Registry boundary remains closed

`candidate.poly-2d-transformer.v1` deliberately remains:

- lifecycle: `architecture_only`;
- authority: `none`.

TR-POLY-08B verifies that state while constructing and loading its metadata.

Reason: the CI checkpoint created by this contract is a bounded synthetic-tensor research witness, not an authoritative repository-data training artifact. Merely proving safe persistence must not open the model-registry evidence gate.

A later execution stage may introduce a new exact implemented-artifact registry identity only after a real authorized TRAIN execution produces an artifact with the same provenance/reload guarantees.

## Non-overwrite / temporary path

The writer rejects:

- an existing final artifact directory;
- a symlink final path;
- an existing temporary artifact directory;
- a symlink temporary path;
- a missing or symlink parent directory.

It writes and fully reload-verifies the temporary artifact before moving it to the final location, then reload-verifies the final location again.

## Resource bounds

- checkpoint file: maximum 128 MiB;
- metadata sidecar: maximum 256 KiB;
- receipt: maximum 16 KiB.

All required files must be regular non-symlink files.

## Explicit non-goals

TR-POLY-08B does not:

- load the repository TRAIN set;
- execute an authoritative training experiment;
- open VALIDATION beyond caller-provided bounded batches;
- open TEST;
- retain optimizer state;
- select a checkpoint by metric;
- tune a hyperparameter;
- run TR-POLY-09 benchmark comparison;
- claim an accuracy improvement;
- update ScoreMosaic;
- grant shadow or production authority.

## Next gate

The next safe stage is an **authorized exact TRAIN artifact execution contract** that materializes prevalidated V2 TRAIN/VALIDATION batches from a hash-bound dataset without opening TEST.

Only after such an exact artifact exists should the model registry gain an implemented-artifact identity eligible for common benchmark evidence.
