# Meter V2 — Real-Model Shadow Evidence

Status: **CHECKPOINT-BOUND / SHADOW-ONLY / REAL-IMAGE E2E NOT YET CLOSED**

This record advances the Meter V2 shadow design without changing the existing runtime architecture and without opening sealed TEST.

## Safety boundary

- TRAIN/VALIDATION evidence only;
- no new Meter training;
- no optimizer steps added;
- no checkpoint mutation;
- no checkpoint binary committed to GitHub;
- no sealed TEST access;
- no runtime Deterministic Resolver wiring;
- no production promotion;
- no merge authorization.

## TRAIN / VALIDATION audit

Authoritative Meter source surface:

- total Meter records: `11064`;
- visible Meter: `5346`;
- none: `5718`;
- validation Meter records: `1224`;
- validation classes: `none=633`, `2/4=186`, `3/4=204`, `4/4=201`;
- invalid positive digit-zone geometry: `0`;
- TRAIN/VALIDATION family overlap: `0`.

Frozen digit-specialist dataset v2:

- records: `30336`;
- TRAIN: `26964`;
- VALIDATION: `3372`;
- TEST: `0`;
- TRAIN labels: `2=1527`, `3=1587`, `4=6396`, `NONE=17454`;
- VALIDATION labels: `2=186`, `3=204`, `4=792`, `NONE=2190`.

The larger count for digit `4` is expected in the current 2/4, 3/4, 4/4 product baseline because every visible meter has denominator `4`.

## Frozen real checkpoint identities

The following private Drive checkpoint files were downloaded read-only on 2026-08-18 and independently SHA-256 hashed. The observed bytes matched the previously frozen pre-closure identities exactly.

| Model | Frozen checkpoint SHA-256 | Frozen threshold |
|---|---|---:|
| D11 Meter technical baseline / temporary Presence bridge | `cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3` | n/a — no new Presence threshold invented |
| 2-AI | `92b985d989e4338e3ae39b0a984879f4188be32c0d281390839117e1e9a715fa` | `0.48` |
| 3-AI | `5ee45faf2efe0e2c83dbad716736d7ae16ad7251730431d368c10c4574836485` | `0.60` |
| 4-AI | `dcd582b60b39e65798aa77aacea3cc797cd7513b7925151f0573be4aec6af43f` | `0.47` |

No private checkpoint bytes are included in this PR or loaded by CI.

## Frozen digit validation evidence

The pre-closure deterministic geometry gate preserved all positives (`positive_geometry_loss=0`).

| Specialist | TP | FP | FN | TN | Recall | F1 | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| 2-AI | 185 | 4 | 1 | 3182 | `0.994624` | `0.986667` | PASS |
| 3-AI | 203 | 0 | 1 | 3168 | `0.995098` | `0.997543` | PASS |
| 4-AI | 788 | 23 | 4 | 2557 | `0.994949` | `0.983157` | PASS |

Therefore a new 2/3/4 training run is **not justified by current VALIDATION evidence**.

## Temporary D11 → Presence bridge

The accepted D11 four-class baseline is not relabeled as a product-quality Presence specialist. For V2 shadow work only, an externally accepted D11 class is collapsed as:

```text
none                -> ABSENT
2/4 | 3/4 | 4/4     -> PRESENT
```

No new binary threshold is tuned.

A read-only real-checkpoint spot check was run on nine real D10 VALIDATION meter ROIs already available to the audit environment:

- 7 visible-meter examples;
- 2 `none` examples;
- presence collapse correct: `9/9`;
- one visible `4/4` example was semantically classified by D11 as `3/4`, but still collapsed correctly to `PRESENT`.

This spot check demonstrates why the V2 decomposition is useful, but **9/9 is not a full Presence acceptance gate**. A full binary Presence metric over all 1224 VALIDATION records remains a separate evidence task.

## Shadow evidence bridge frozen in PR #65

`meter_v2_real_model_shadow_v1.py` binds only the evidence boundary:

```text
accepted D11 class
       ↓ collapse only
PRESENT / ABSENT
       ↓
deterministic candidate digit slots
       ↓
2-AI / 3-AI / 4-AI scores
       ↓
per-slot deterministic arbitration
       ↓
DigitObservation(2|3|4+bbox)
       ↓
Deterministic Meter Composer v1
       ↓
none | 2/4 | 3/4 | 4/4
or AMBIGUOUS / REJECTED
```

Per-slot arbitration is fail closed:

- exactly one specialist above its frozen threshold -> accepted visual digit;
- no specialist above threshold -> no digit observation for that slot;
- more than one specialist above threshold -> `AMBIGUOUS`;
- malformed slot/bbox/score -> `REJECTED`;
- confidence cannot resolve a multi-specialist conflict.

## Important remaining gap

The real 2/3/4 checkpoints are validated against the frozen 64x64 digit-specialist dataset, but this PR does **not** silently invent the runtime pixel adapter that turns a new measure-start ROI into those exact 64x64 candidate-slot tensors.

Historical M3-C2 deterministic anchor work showed high VALIDATION coverage (`591/591` geometry coverage; end-to-end presence+geometry `590/591`) but explicitly did not freeze all rare anchor modes as broadly generalized. Therefore the production/runtime digit-slot crop adapter must be frozen and proven separately rather than reconstructed by guesswork.

Accordingly:

- checkpoint identity: **PASS**;
- digit specialist validation: **PASS**;
- deterministic score/slot/composer bridge: **SHADOW PASS**;
- temporary D11 Presence collapse: **SPOT-CHECK PASS / FULL GATE PENDING**;
- exact runtime 64x64 digit crop adapter: **PENDING**;
- full real-image Meter V2 end-to-end validation: **PENDING**;
- Resolver wiring: **NOT AUTHORIZED**.

## Next safe gate

1. compute full VALIDATION binary Presence metrics from the frozen D11 checkpoint or freeze a dedicated Presence specialist only if those metrics justify it;
2. freeze an exact deterministic runtime digit-slot/crop adapter with provenance and 10/10 repeatability;
3. run all 1224 VALIDATION meter ROIs through Presence -> real 2/3/4 checkpoints -> deterministic composer;
4. compare exact Meter class, ambiguity/rejection rate, and localization against D11 baseline;
5. only then consider a runtime evidence adapter; Resolver wiring and merge remain separate approval gates.
