# Meter V5-2C — Historical 2/3 Retention Audit V1

## Purpose

Before any V5 validation-BBox work, compare the new V5-2B 2-AI / 3-AI candidates against the exact frozen historical 2-AI / 3-AI checkpoints on the original M4A/D10 validation-domain digit crops.

This is an inference-only retention gate. It does not train, tune, open V5 VAL, access FINAL_HOLDOUT, change 4-AI, wire Resolver authority, or promote production behavior.

## Frozen evidence identities

- M4A dataset manifest SHA-256: `ebda40dae10f0d6490df2c7728dab5cc2cc6f58b5420b198dfbb441a99ecebb9`
- D10 authoritative manifest SHA-256: `6927e1bcc5251257a983a306e2f1875c9515f97c6724a8fe9f24382c6ff30db4`
- 2-AI frozen checkpoint: `92b985d989e4338e3ae39b0a984879f4188be32c0d281390839117e1e9a715fa`
- 3-AI frozen checkpoint: `5ee45faf2efe0e2c83dbad716736d7ae16ad7251730431d368c10c4574836485`
- 4-AI frozen checkpoint: `dcd582b60b39e65798aa77aacea3cc797cd7513b7925151f0573be4aec6af43f`
- V5-2B 2-AI candidate: `61e4ed5c595d66214ab863f53094998e5cc5167094dc8a9b5934470e3188d4f2`
- V5-2B 3-AI candidate: `5d8dd8ea3aed5c2aaa383d2a494e762276afa952f5da6d37fc1dc214900f1c62`

## Exact historical validation surface

M4A validation contains exactly 3,372 records:

- digit `2`: 186
- digit `3`: 204
- digit `4`: 792
- `NONE`: 2,190

Every input is recreated from the bound D10 meter ROI image plus the M4A bbox using the historical transform:

1. floor/ceil/clip bbox to source pixels;
2. grayscale `L` crop;
3. aspect-preserving LANCZOS thumbnail to at most 64x64, no upscaling;
4. center on white 64x64 canvas;
5. tensor `uint8 / 255.0`.

No new crop geometry, nearest-staff rule, tolerance, midpoint, or V5 annotation is used.

## Pixel-path self-check

The retention result is invalid unless the frozen historical checkpoints reproduce the already-recorded D10/M4A validation confusion counts at unchanged thresholds:

- 2-AI @ 0.48: TP=185, FP=4, FN=1, TN=3182, F1=0.9866666666666667
- 3-AI @ 0.60: TP=203, FP=0, FN=1, TN=3168, F1=0.9975429975429976
- 4-AI @ 0.47: TP=788, FP=23, FN=4, TN=2557, F1=0.9831578947368421

A mismatch means the historical pixel/replay contract was not reconstructed faithfully and the audit must HOLD before candidate interpretation.

## Candidate retention gate

Thresholds remain fixed at 2=0.48 and 3=0.60. Candidate retention PASS requires, for each of 2-AI and 3-AI:

- F1 drop from the exactly reproduced frozen baseline <= 0.005 absolute;
- recall drop from the exactly reproduced frozen baseline <= 0.005 absolute;
- candidate precision >= 0.98;
- candidate recall >= 0.98;
- all probabilities finite and in [0,1].

The 0.005 paired-drop rule is preregistered before running the V5 candidates. It is intentionally strict because the historical specialists were already near saturation on this domain. A HOLD does not authorize threshold tuning; it means V5 adaptation needs a separate retention-preserving training design.

## Safety boundary

This audit:

- reads M4A/D10 development validation only;
- does not access V5 VAL or FINAL_HOLDOUT;
- performs zero optimizer steps;
- never writes model checkpoints;
- verifies frozen and candidate checkpoint SHA identities before inference;
- keeps 4-AI read-only and uses it only as a pixel-path control;
- writes one JSON retention report under the existing V5-2B annotations directory;
- does not merge PR #98 or open Resolver/production authority.

PASS authorizes only preparation of the isolated V5 validation-BBox stage. It does not open V5 VAL automatically.
