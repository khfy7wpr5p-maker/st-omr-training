# Stage 7-D12 — authoritative closure evidence

This file freezes the externally produced Stage 7-D12 authoritative evidence for the NoteHeadSet, RestSet, and AccidentalSet deterministic ground-truth gate.

D12 remains a **data / ground-truth stage only**. No learned model was trained, fine-tuned, loaded, or selected by this stage.

## Authoritative execution identity

The authoritative D12 bundle was produced from the exact executable repository head:

```text
repository HEAD:
e2de6f64c27be2dd6d706a700553ef4f5c236e25
```

That head already passed CI #224 before the external authoritative run.

The independently verified persisted bundle identity is:

```text
derivative build id:
35323e831c5c693bf607808c5f846624445bf537f30e1d93db9ca949a7eed106

manifest SHA-256:
a372eba640b38704020922ad4eb102738fc4492d278a38e4b51b8ad0b78d4ea1

artifact binding SHA-256:
14c64e16ca2f993bf94f8009bf0bcd974b7ddee87c19bb748219ba3f774b229d
```

The canonical authoritative bundle root is the successful `stage7d12-symbol-derivatives-e2de6f64c27be2dd6d706a700553ef4f5c236e25` Drive artifact. An interrupted earlier attempt was preserved under a `.partial-*` quarantine name and is explicitly **not** accepted as D12 evidence.

## Cardinality and split evidence

```text
samples:      1383
families:      461
labels:       1383

TRAIN:        1230 samples / 410 families
VALIDATION:    153 samples /  51 families
TEST:            0 specialist records
```

The persisted bundle contains one D12 symbol label per accepted development image.

TEST remained sealed for D12. The D12 development-only path is required to skip TEST after reading only the source `split` field; no D12 TEST derivative, label, geometry target, model input, or training record was produced.

## Training boundary evidence

```text
optimizer steps: 0
model loading:   none authorized
checkpoint:      none
training:        none
```

D12 does not authorize a joint or specialist optimizer. Future NoteHead, Rest, and Accidental training remains a separate stage with separate acceptance criteria.

## Independent persisted-bundle verification

The D12 independent verifier returned:

```text
independent verification: True
COMPLETE present:         False
```

`COMPLETE=False` is the expected D12-6 state. The D12 builder cannot write `COMPLETE`, and the verifier rejects a premature `COMPLETE` marker.

The accepted uncompleted bundle top-level surface is exactly:

```text
manifest.json
manifest.sha256
build.json
labels/
```

The independent verifier reopens the frozen source corpus, accepted D6 artifacts, and persisted D12 artifacts and recomputes the required identities, hashes, cardinalities, geometry checks, class inventory, target-instance counts, total label bytes, and artifact binding rather than trusting the builder's in-memory receipt.

## Verified class inventory

### TRAIN

```text
NoteHeadSet
  open    7935
  filled 30399

RestSet
  half     1998
  quarter  3417
  eighth   5187

AccidentalSet
  sharp   10665
  flat    10596
  natural  1131
```

### VALIDATION

```text
NoteHeadSet
  open    1128
  filled  4104

RestSet
  half      162
  quarter   327
  eighth    480

AccidentalSet
  sharp    1566
  flat     1575
  natural   189
```

Every supported V1 class is present in both TRAIN and VALIDATION. D12 intentionally does not convert this observed inventory into post-hoc training thresholds. Class weighting, sampling, minimum-readiness rules, model architecture, optimizer profile, metrics, and acceptance thresholds must be frozen by the later specialist-training stage before any optimizer step.

## Source and accepted-D6 binding

D12 consumes the accepted development surface only after rechecking the source and D6 identities. The accepted D6 dependency remains:

```text
D6 derivative build id:
0faafe229f3497b1147cf0f0ac0ce4b7efe6fa31f360a6a33a3b82c986c8c519

D6 manifest SHA-256:
e8e415eb6ba9d91a1a880709c3f31d559aa20bf5149734f45b5f84ced16afee9

D6 artifact binding SHA-256:
3b7558f0f927ad47a61ed5afb5faa8584dca8647cf8683d4043686eb7b077ea1
```

Every referenced source PNG and accepted D6 label is re-hashed before the D12 derivative is admitted.

## Closure invariants

For D12 technical acceptance, all of the following remain required:

- authoritative executable head is exactly bound to the persisted bundle identity above;
- the closure commit changes documentation/evidence only relative to the authoritative executable head;
- final exact-head focused/full regression passes;
- TEST specialist records remain exactly `0`;
- optimizer steps remain exactly `0`;
- no model/checkpoint loading is introduced;
- canonical/renderer linkage remains fail-closed;
- source/D6/D12 hash and artifact binding checks remain fail-closed;
- premature `COMPLETE` remains rejected;
- final fresh-read P1/P2 review has zero unresolved blockers;
- PR remains mergeable;
- merge occurs only after explicit user merge approval.

If executable D12 code changes after the authoritative head above, this closure evidence is invalid for that changed executable and the authoritative D12 bundle must be regenerated and independently verified on the new executable head.

## Merge status

The presence of this file does **not** authorize merge by itself. D12 becomes merge-ready only after the closure-only final head passes final CI/regression and final fresh-read P1/P2 review with no blocker. Explicit user merge approval remains a separate final gate.
