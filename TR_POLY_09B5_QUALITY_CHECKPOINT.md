# TR-POLY-09B5 — Quality checkpoint artifact and B1 bridge

Updated: 2026-09-14

## Purpose

TR-POLY-09B5 closes the persistence gap between the multi-epoch TRAIN-only quality regime introduced in TR-POLY-09B4 and the already-merged TR-POLY-09B1 free-running inference path.

The existing TR-POLY-08B checkpoint format remains frozen for its original bounded smoke-training purpose. B5 introduces a separate quality-checkpoint schema instead of weakening the old `optimizer_steps <= 2` contract.

## Artifact set

A B5 directory contains exactly four regular files:

```text
model.pt
metadata.json
training_result.json
receipt.json
```

No additional file is admitted.

`model.pt` contains only the selected B4 model `state_dict`. `metadata.json` binds the exact model/training/provenance identities. `training_result.json` preserves the complete B4 epoch-selection evidence. `receipt.json` binds file SHA-256 values and the model-registry artifact binding.

## Selection boundary

B5 does not choose a new epoch. It persists only the state already selected by B4 using the frozen `min-mean-validation-loss-earliest-epoch-v1` policy.

Before persistence B5 verifies:

- returned model state equals the B4 selected-state SHA;
- exact B4 training-result provenance;
- repository identity;
- dataset manifest identity;
- model profile identity;
- quality-trainer profile identity;
- tokenizer and representation identities.

## Reload verification

Reload is fail-closed. Before accepting the model B5 checks:

- exact directory file set;
- size bounds and regular/non-symlink files;
- canonical JSON bytes;
- receipt file hashes;
- metadata fingerprint;
- training-result fingerprint and selected-epoch fields;
- `torch.load(..., weights_only=True)` state dictionary;
- strict state-dict compatibility;
- finite model parameters;
- exact model-state SHA;
- exact parameter count;
- registry artifact binding.

## Research registry identity

The quality artifact uses the explicit research identity:

```text
candidate.poly-2d-transformer.quality-v1
```

It is TRAINING_IMPLEMENTED / EXPERIMENTAL research evidence only. It does not grant production authority and is not a benchmark result by itself.

## B1 inference bridge

`run_verified_poly_2d_quality_checkpoint_inference(...)` first reloads and verifies the complete B5 artifact, then calls the frozen B1 greedy decoder.

The resulting inference identity additionally binds:

- checkpoint file SHA-256;
- metadata file SHA-256;
- receipt SHA-256;
- metadata fingerprint;
- dataset manifest;
- preprocessing fingerprint;
- quality trainer profile;
- training provenance;
- research registry record fingerprint;
- repository SHA.

This makes downstream B2/B3 VALIDATION evidence checkpoint-bound without changing B1 decoding semantics.

## Claim boundary

A green B5 merge means the project can persist, reload, and identify a B4-selected quality candidate safely. It does **not** mean that a real quality-training run has occurred or that any accuracy claim is available.

The next evidence path is:

```text
admitted native V2 TRAIN artifacts
→ B4 quality training
→ B5 selected quality checkpoint
→ B1 free-running VALIDATION inference
→ B2 per-sample metrics
→ B3 aggregate VALIDATION report
```

TEST remains sealed through this path.
