# Meter V5-2C — Historical 2/3 Retention Audit V2

## Why V2 exists

V1 failed closed before candidate interpretation because its hard-coded historical confusion-count oracle did not match the exact frozen checkpoint + exact original M4A validation replay. The observed 2-AI replay was TP=185, FP=30, FN=1, TN=3156 at threshold 0.48.

That observed result is not a new regression: it is exactly the recorded result of the frozen `m4c5-2ai-train-hard-none-v1` step-0768 checkpoint that is bound by SHA-256 `92b985d989e4338e3ae39b0a984879f4188be32c0d281390839117e1e9a715fa`.

The same audit of the original `m4c2-none75-negative-sampling-v1` evidence binds the frozen 3-AI and 4-AI checkpoints to their original validation results at the unchanged frozen thresholds:

- 3-AI @ 0.60: TP=203, FP=1, FN=1, TN=3167;
- 4-AI @ 0.47: TP=788, FP=46, FN=4, TN=2534.

Therefore V1 mixed later summary metrics with the original M4A replay surface. V2 corrects only that oracle mismatch. It does not change pixels, crop geometry, thresholds, model weights, data roles, or the retention acceptance rule.

## Frozen evidence identities

- M4A dataset manifest SHA-256: `ebda40dae10f0d6490df2c7728dab5cc2cc6f58b5420b198dfbb441a99ecebb9`
- D10 authoritative manifest SHA-256: `6927e1bcc5251257a983a306e2f1875c9515f97c6724a8fe9f24382c6ff30db4`
- 2-AI frozen checkpoint: `92b985d989e4338e3ae39b0a984879f4188be32c0d281390839117e1e9a715fa`
- 3-AI frozen checkpoint: `5ee45faf2efe0e2c83dbad716736d7ae16ad7251730431d368c10c4574836485`
- 4-AI frozen checkpoint: `dcd582b60b39e65798aa77aacea3cc797cd7513b7925151f0573be4aec6af43f`
- V5-2B 2-AI candidate: `61e4ed5c595d66214ab863f53094998e5cc5167094dc8a9b5934470e3188d4f2`
- V5-2B 3-AI candidate: `5d8dd8ea3aed5c2aaa383d2a494e762276afa952f5da6d37fc1dc214900f1c62`

## Historical replay surface

The replay remains exactly the same 3,372 M4A validation records:

- `2`: 186
- `3`: 204
- `4`: 792
- `NONE`: 2,190

Each input remains the original D10 meter ROI plus the stored M4A bbox using the historical preprocessing contract: floor/ceil/clip, grayscale L, aspect-preserving LANCZOS thumbnail up to 64x64 with no upscaling, white-centered 64x64 canvas, and uint8/255 tensor semantics.

No new spatial rule is introduced.

## Corrected pixel-path self-check

Before any V5 candidate may be interpreted, the frozen checkpoints must reproduce these exact original experiment results:

- 2-AI @ 0.48: TP=185, FP=30, FN=1, TN=3156;
- 3-AI @ 0.60: TP=203, FP=1, FN=1, TN=3167;
- 4-AI @ 0.47: TP=788, FP=46, FN=4, TN=2534.

A mismatch still fails closed.

## Candidate retention gate

The retention gate itself is unchanged from V1. For each candidate 2-AI and 3-AI:

- F1 drop versus the exactly replayed frozen checkpoint <= 0.005 absolute;
- recall drop <= 0.005 absolute;
- candidate precision >= 0.98;
- candidate recall >= 0.98;
- probabilities finite and in [0,1].

Thresholds remain frozen at 2=0.48, 3=0.60, 4=0.47. No threshold tuning is authorized.

## Safety boundary

V2 is inference-only. V5 VAL stays closed, FINAL_HOLDOUT stays locked, 4-AI remains frozen, optimizer steps remain zero, and Resolver/production authority stays closed. PASS authorizes only preparation of the isolated V5 validation-BBox stage.