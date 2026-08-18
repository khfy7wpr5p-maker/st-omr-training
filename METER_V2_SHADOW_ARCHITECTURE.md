# Meter V2 — Shadow Presence/Digit/Composer Architecture

Status: **DESIGN-FROZEN CANDIDATE / SHADOW-ONLY**

## Why V2 exists

Stage 7-D11 Meter is retained as a valid technical baseline. Its authoritative VALIDATION result passed the frozen D11 gates:

- four-class Macro F1: `0.908946` >= `0.800`;
- positive localization F1@2px: `0.710525` >= `0.600`;
- meter optimizer steps: `2464/2464`;
- independent persisted-run verification: PASS;
- sealed TEST: closed.

V2 does not rewrite or invalidate D11. It narrows the learned questions so that product-quality meter recognition can be debugged separately from measure-start geometry and deterministic musical composition.

## Architecture

```text
accepted Measure Geometry / measure-start ROI
        ↓
Meter Presence specialist
        ↓
PRESENT | ABSENT | AMBIGUOUS | REJECTED
        ↓
if PRESENT:
Digit visual evidence
        ↓
2 | 3 | 4 + bbox + confidence
        ↓
Deterministic Meter Composer v1
        ↓
none | 2/4 | 3/4 | 4/4
or AMBIGUOUS / REJECTED
```

The digit specialist does **not** decide numerator versus denominator. Upper/lower role is derived deterministically from geometry.

## Learned boundary

The learned layer may report only visual evidence.

### Presence observation

- `status = accepted | ambiguous | rejected`
- when accepted: `present = true | false`
- confidence

### Digit observation

- stable observation id
- `status = accepted | ambiguous | rejected`
- accepted digit vocabulary: `2 | 3 | 4`
- bbox
- confidence

No learned component may directly invent `2/4`, `3/4`, or `4/4` from musical context in this package.

## Deterministic composer rules

Priority is fixed:

1. malformed, non-finite, invalid-bbox, unsupported-digit, or explicitly rejected evidence -> `REJECTED`;
2. ambiguous presence/digit evidence -> `AMBIGUOUS`;
3. explicit meter absence with no accepted digit evidence -> accepted `none`;
4. absence plus digit evidence -> `AMBIGUOUS`;
5. explicit presence requires exactly two accepted digit observations;
6. digit geometry must form one plausible horizontally grouped vertical stack;
7. upper digit becomes numerator, lower digit becomes denominator;
8. v1 admits only `2/4`, `3/4`, `4/4`;
9. any otherwise valid but unsupported composition fails closed as `AMBIGUOUS`;
10. confidence ranking never breaks a structural conflict.

## Acceptance cases

| ID | Evidence | Expected |
|---|---|---|
| M01 | explicit ABSENT, no digits | accepted `none` |
| M02 | PRESENT + upper 2 + lower 4 | accepted `2/4` |
| M03 | PRESENT + upper 3 + lower 4 | accepted `3/4` |
| M04 | PRESENT + upper 4 + lower 4 | accepted `4/4` |
| M05 | PRESENT + no digits | `AMBIGUOUS` |
| M06 | PRESENT + one digit | `AMBIGUOUS` |
| M07 | PRESENT + three accepted digits | `AMBIGUOUS` |
| M08 | two digits with equal vertical centers | `AMBIGUOUS` |
| M09 | two digits not horizontally groupable | `AMBIGUOUS` |
| M10 | ABSENT + accepted digit evidence | `AMBIGUOUS` |
| M11 | unsupported visual digit, e.g. 6 | `REJECTED` |
| M12 | invalid bbox | `REJECTED` |
| M13 | non-finite bbox | `REJECTED` |
| M14 | invalid confidence/range | `REJECTED` |
| M15 | ambiguous presence | `AMBIGUOUS` |
| M16 | rejected presence | `REJECTED` |
| M17 | ambiguous digit observation | `AMBIGUOUS` |
| M18 | valid digits but unsupported composition, e.g. 4/2 | `AMBIGUOUS` |

All frozen representative cases must reproduce identically `10/10`.

## Runtime compatibility

The current runtime specialist evidence boundary already accepts meter classes:

`none | 2/4 | 3/4 | 4/4`

Therefore a future admitted Meter V2 adapter can translate an accepted composer result into the existing meter `SpecialistObservation` without changing the Deterministic Resolver class vocabulary.

This document does **not** authorize that adapter or wiring.

## What this package does not do

- no Meter model training;
- no D11 checkpoint modification;
- no threshold tuning from TEST;
- no sealed TEST access;
- no change to ST Page Normalizer, Multi-Staff Geometry, Measure Geometry, Runtime Local ROI, or Deterministic Resolver;
- no runtime model adapter;
- no production promotion;
- no merge authorization;
- no claim of support for 6/8, 9/8, 12/8, common-time glyphs, cut-time glyphs, additive meters, or arbitrary numerator/denominator digits.

Those require later explicit evidence/contracts rather than silent extrapolation.

## Next gates

After this deterministic shadow package is green:

1. validate Meter Presence on TRAIN/VALIDATION only;
2. create a digit-level GT/ROI audit for `2|3|4` without changing D11 evidence;
3. if data are sufficient, train small digit specialist(s) under a separately approved training package;
4. run real-image shadow composition on the same VALIDATION meter surface;
5. compare against D11 four-class baseline by per-class recall/F1, localization, ambiguity rate, and exact meter composition;
6. only after those gates, consider a real Meter V2 adapter to the existing evidence boundary;
7. Resolver wiring/promotion/merge remain separately approved gates.
