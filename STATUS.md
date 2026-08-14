# ST-OMR Training Lab Status

This file is the current stage-status source for this repository. Detailed closed-stage history remains in `ARCHITECTURE.md`; the current active-lane overlay is in `ARCHITECTURE_CURRENT.md`.

## Current repository phase

Verified baseline before Stage 7-D10 work:

- `main`: `d3021a6cb64d35a5a216101b84f9aa3545527535`
- PR #46 — Stage 7-D9 Structure refinement contract: MERGED
- D9 post-merge main CI: run #181 (`31839302798`) — SUCCESS
- Stage 7-D0 through Stage 7-D9: CLOSED

The current active lane is **Stage 7-D10 — deterministic local barline/meter ROI derivatives**. D10 is a data-derivative gate only. It performs no model training, preserves the accepted D7 Structure core unchanged, and keeps TEST sealed.

## Stage status

| Stage | Description | Status |
|---|---|---|
| 0–6 | Deterministic music → validated synthetic dataset pipeline | ✅ Closed / main CI verified |
| 7-A | Training contract freeze | ✅ Closed / main CI verified |
| 7-B | Tokenizer/data/model/trainer implementation | ✅ Closed / main CI verified |
| 7-C | Bounded baseline training + evidence | ✅ Closed / non-production baseline |
| 7-D0 | Synthetic Curriculum v1 export-evidence identity gate | ✅ Closed |
| 7-D1 | Synthetic corpus transport/byte/manifest acceptance | ✅ Closed |
| 7-D2 | Synthetic V1 monolithic train/validation execution | ✅ Closed / non-production baseline |
| 7-D3 | Validation-only semantic error diagnostics | ✅ Closed / specialist decomposition selected |
| 7-D4 | Specialist OMR architecture + GT/fusion contract | ✅ Closed |
| 7-D5 | StaffSet + StructureSet deterministic geometry | ✅ Closed |
| 7-D6 | TRAIN/VALIDATION StaffSet + StructureSet derivatives | ✅ Closed / PR #43 / main CI #171 PASS |
| 7-D7 | StaffSet + StructureSet specialist training | ✅ Closed / PR #44 / main CI #175 PASS |
| 7-D8 | Structure validation-only diagnostics | ✅ Closed / PR #45 / main CI #177 PASS |
| 7-D9 | Structure refinement architecture/contract | ✅ Closed / PR #46 / main CI #181 PASS |
| 7-D10 | Deterministic barline/meter local ROI derivatives | 🔄 Active — no training / TEST sealed |
| 8-0 | Real-data rights/provenance/fine-tuning contract | ✅ Closed / preserved |
| 8-1 | Real-data quarantine/intake + byte validation | ✅ Closed / preserved |
| 8-2 | Paired experiment profile | ✅ Closed / preserved |
| 8-3A | Real pilot preparation/admission components | ⏸ Parked during specialist synthetic work |
| 8-3B | Paired real train/validation execution | 🔒 Not started |
| 9 | Sealed benchmark and candidate decision | 🔒 Not started — TEST sealed |
| 10 | ScoreMosaic candidate integration | 🔒 Not started |

## Frozen Synthetic Curriculum v1

```text
source commit       adc8139539d3c8cd6a2e3ee4ce4de6db4dcfeb90
config fingerprint  154bf1c3e6dfe4e6db096f8b668f29df0623cfd38352b89a04d295764c7458cb
build id            d9320e362f162cd2ace2a830a7b93e0c21ceba2d51a4e95ef1c7a9b11a108352
manifest SHA-256     44a963cd7dbc612fa29c2953ea8b2c8776d89ce470074e8f8b3fe25c6e165f34
transport SHA-256    4a9f3bb337ef99386081dff29c4c1fc3047dc3ada4db13c93b6254e680918e2b
families             512 = 410 train + 51 validation + 51 test
images               1536 = 1230 train + 153 validation + 153 test
targets              512 MusicXML
```

## Accepted D6 derivative identity

```text
derivative build ID      0faafe229f3497b1147cf0f0ac0ce4b7efe6fa31f360a6a33a3b82c986c8c519
manifest SHA-256          e8e415eb6ba9d91a1a880709c3f31d559aa20bf5149734f45b5f84ced16afee9
artifact binding SHA-256  3b7558f0f927ad47a61ed5afb5faa8584dca8647cf8683d4043686eb7b077ea1
receipt SHA-256           8fe85747b77f2282be3662f0c3d180a440c88028638bf1df7ddadfbb7650fff2
TRAIN labels              1,230 / 410 families
VALIDATION labels           153 /  51 families
TEST specialist records       0
```

## Accepted D7 specialist result

```text
run ID                 4ce2903206c7965471bb9569d379d8d9d1022d9248d80886638acfe0bd822598
checkpoint SHA-256     5f009ca8ba68d38497a7dd25590d4dd98c537f20c5d5525bf66e288afbf417dc
metrics SHA-256        43cd98a75c2db740b4af6ee3c8826122fa387347820d2e7d2c639ac2fe30f792
verification SHA-256   cdc0733af1bd6c7336f5bd2a0cb12fcae269120d8b5a9a564f08db860ee21a0a
TEST opened            false
```

Staff learned strongly (`staff_lines` Dice `0.9217`, `staff_region` Dice `0.9127`). Structure learned system/measure/clef well but left barline and meter weak.

## Accepted D8 diagnostic finding

D8 used VALIDATION only, optimizer `0`, TEST `0`.

```text
D8 repository head       e0e721bf5a6d13025546fdf5eeb755647eef383f
report SHA-256            46de5f6766f78bb567f70794a364ccd44835d09af94ef29c3f1eab5cd13ce968
barline Dice@0.50         0.2736204205
barline tolerant F1@2px   0.3670878904
meter 2/4 Dice            0.3481060606
meter 3/4 Dice            0.3528485803
meter 4/4 Dice            0.3103351169
```

Threshold sweeps and 1–2 pixel tolerance did not repair the weak channels. D8 therefore selected local higher-resolution specialists rather than more whole-page epochs.

## Closed D9 contract

D9 preserves the strong accepted D7 Structure core and freezes only local refinement for the weak sparse targets:

```text
structure_core   -> system_region / measure_region / clef_g2 (frozen)
barline_refiner  -> measure-end 192x128 ROI
meter_refiner    -> measure-start 192x256 ROI, none|2/4|3/4|4/4
structure_fusion -> deterministic fail-closed fusion to unchanged StructureSet
```

Frozen future validation gates:

```text
TEST records                                  0
accepted D7 core mutation                     forbidden
new trainable parameters                      <= 1,250,000
barline strict Dice                           >= 0.500
barline tolerant F1 @2px                      >= 0.700
meter none|2/4|3/4|4/4 macro F1              >= 0.800
meter positive localization tolerant F1 @2px  >= 0.600
```

D9 itself trained nothing.

## Active D10 boundary

D10 materializes the D9 local data surface from accepted D6 final-PNG geometry. The authoritative runner must consume exactly:

```text
TRAIN        1,230 source images / 410 families
VALIDATION     153 source images /  51 families
TOTAL        1,383 source images / 461 families
TEST             0
optimizer         0
```

For every source measure D10 emits exactly two records:

```text
measure-end   -> barline ROI -> barline_segment
measure-start -> meter ROI   -> none|2/4|3/4|4/4 + meter_bbox
```

If the active meter continues but no current-measure meter glyph is visible, the local meter class is `none`. This preserves the accepted D6 courtesy-meter semantics.

D10 binds each artifact to source PNG SHA, accepted D6 label SHA, split, measure, D9 contract fingerprint and ROI policy. Output stays outside normal Git.

Before `COMPLETE`, an independent persisted-output verifier reopens every ROI PNG/label and rederives hashes, identities, split/family cardinality, barline/meter pair completeness and receipt evidence. Only then is `COMPLETE` written and verified again.

## Safety boundaries

- No direct commits to `main`; changes use branch/PR packages.
- Large datasets/checkpoints stay outside normal Git.
- D10 contains no model/optimizer/backward/checkpoint/DataLoader path.
- D7 accepted Structure weights are untouched.
- VALIDATION is data-only/read-only in D10.
- TEST remains sealed until Stage 9.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- Real geometry labels require human-verified annotation and explicit admission.
- No online or automatic learning path is allowed.
- Deterministic validators retain veto authority over learned candidates.

## Next gate

Close D10 code/review/CI on one exact branch head, then run the authoritative external D10 build against the accepted frozen corpus + D6 derivatives. Record manifest/artifact-binding/receipt evidence and confirm TEST=0 / optimizer=0 before requesting merge approval.

See `STAGE7D10_LOCAL_ROI_DERIVATIVES.md` for the active contract.