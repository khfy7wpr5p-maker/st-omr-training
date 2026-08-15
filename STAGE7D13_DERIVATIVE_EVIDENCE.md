# Stage 7-D13 — authoritative measure-derivative evidence

This file freezes the externally produced and independently verified Stage 7-D13 measure-derivative surface used by the later NoteHeadSet, RestSet, and AccidentalSet training implementation.

## Authoritative executable identity

The bundle was built on exact repository head:

```text
d5fe4d2c120202ec7f962ef6d849b6e36af224ef
```

That exact head passed CI #234 before the authoritative external run.

## Persisted derivative identity

```text
derivative build id:
44f1932532fb511dfa59a164f94be6b899f3aa0594c0ac0a6f499a38e5fb5697

manifest SHA-256:
8cfb87b5c6135be14b4c9ad488868c0edb0d37bb3bb18ad1b5e79d04fdf24f7b

artifact binding SHA-256:
c42c1f69e21d61d3eefdcafc40dabf2f0fcd6ac2ceb4d5cf88d8e158246dd33e

external receipt SHA-256:
4e644c5a110c738fd99b905f093a9acb0ca07cd6bd1b7b52c4904aba7964466b
```

## Independent verification result

```text
D13 DERIVATIVE BUNDLE : VERIFIED
INDEPENDENT VERIFY    : True
TEST RECORDS          : 0
OPTIMIZER EXECUTED    : 0
COMPLETE PRESENT      : False
```

`COMPLETE=False` is expected at the derivative gate. The measure-derivative builder cannot authorize training completion and no optimizer ran during derivative production or verification.

## Exact record cardinality

```text
TRAIN records       9840
VALIDATION records  1224
TOTAL records      11064

persisted images   11062
persisted labels   11064
```

There are 11,064 distinct measure records and labels. The persisted image count is 11,062 because image storage is content-addressed: when two independently identified measure records render to exactly identical PNG bytes, the builder stores one shared image hash while keeping separate record/label identities. The builder and independent verifier both accepted and recomputed this deduplication.

## Target-instance preservation

```text
TRAIN
  NoteHeadSet     38334
  RestSet         10602
  AccidentalSet   22392

VALIDATION
  NoteHeadSet      5232
  RestSet           969
  AccidentalSet    3330
```

These totals preserve the accepted D12 class inventory after the deterministic measure crop + 512x128 letterbox transform.

## Exact optimizer-step expectation

The frozen D13 profile is batch size 16 and 10 epochs. Independently verified TRAIN record cardinality is 9,840, therefore:

```text
batches per epoch = ceil(9840 / 16) = 615

NoteHeadSet     6150 optimizer steps
RestSet         6150 optimizer steps
AccidentalSet   6150 optimizer steps

TOTAL          18450 optimizer steps
```

The three specialists remain independent: each has its own model, optimizer, validation history, best epoch and accepted state. No optimizer state is shared.

## Safety boundary carried forward to training

Any later D13 training runner must fail closed unless all of the following match this evidence exactly:

- derivative build id;
- manifest SHA-256;
- artifact binding SHA-256;
- 9,840 TRAIN and 1,224 VALIDATION records;
- 11,064 labels and 11,062 content-addressed images;
- TEST specialist records exactly 0;
- expected optimizer steps exactly 6,150 per specialist / 18,450 total;
- accepted D12 lineage and class inventory remain unchanged;
- independent derivative verification passes before model creation;
- no validation-dependent decoder/threshold tuning occurs.

If the D13-1/D13-2 derivative executable or the frozen pre-training contract changes, this evidence does not authorize reuse of the existing bundle; the derivative surface must be regenerated and independently verified.