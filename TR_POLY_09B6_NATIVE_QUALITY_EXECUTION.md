# TR-POLY-09B6 — Native V2 quality execution

Updated: 2026-09-14

## Purpose

TR-POLY-09B6 connects the admitted native Polyphonic V2 dataset to the quality-training/checkpoint path without changing the older smoke execution contract.

The previous native execution path is intentionally bounded to one training batch and at most eight samples. B6 adds a separate deterministic multi-batch path suitable for quality training.

## Pipeline

```text
Native V2 manifest + persisted artifacts
        ↓
sorted TRAIN / VALIDATION sample selection
        ↓
hash + canonical V2 + profile + PNG verification
        ↓
deterministic preprocessing
        ↓
contiguous batches, each <= 8 samples
        ↓
TR-POLY-09B4 quality training
        ↓
TR-POLY-09B5 verified selected checkpoint
```

## Frozen v1 ordering

- sample order: ascending `sample_id`;
- optional bounded selection: sorted prefix;
- batch order: contiguous chunks in that exact order;
- no shuffle or random sampling;
- batch size is part of the execution profile fingerprint.

## Safety

B6 rejects TEST before dataset-root inspection. It also rejects split identity drift, family leakage, sample leakage, hash mismatch, canonical V2 drift, polyphony-profile drift, target truncation, preprocessing failure, and checkpoint identity drift.

Existing TR-POLY-09A smoke materialization/execution remains unchanged.

## Claim boundary

A successful B6 run produces a quality-trained research checkpoint. It does not itself produce benchmark evidence or production authority.

The next measurement path is:

```text
B6 quality checkpoint
→ B1 checkpoint-bound free-running VALIDATION inference
→ B2 per-sample metric reports
→ B3 aggregate VALIDATION evidence
```

TEST remains sealed until the final Stage 9 decision.
