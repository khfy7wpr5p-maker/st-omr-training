# ST-OMR Real Specialist Shadow Acceptance v1

Status: **shadow-only; no real specialist is wired to Deterministic Resolver**.

Checkpoint binaries remain external to GitHub. Sealed TEST is closed. This package freezes real-artifact evidence and deterministic acceptance decisions only.

## NoteHead

- real artifact: D13 epoch-10
- classes: `open|filled`
- Center F1: `0.9882845985`
- BBox F1: `0.9882845985`
- Macro F1: `0.9855884316`
- epochs: `10/10`
- real smoke: `10/10` deterministic, `4/4` malformed/non-finite fail closed
- decision: **PASS**

## Accidental

Real artifact: authoritative D13 Accidental epoch-10 checkpoint from repository provenance:

`cf82ecbc0ef8df3d635e6e1923b4c4000c40da5b`

Frozen quality gates:

- Center F1 `>= 0.80`
- BBox F1 `>= 0.70`
- Macro F1 `>= 0.85`

Verified epoch-10 VALIDATION evidence:

- Center F1: `0.973754` — PASS
- BBox F1: `0.973754` — PASS
- Macro F1: `0.956454` — PASS
- completed epochs: `10/10`
- training step boundary: `6150/6150`
- best epoch: `10`
- TEST: closed

Real-checkpoint shadow smoke:

- checkpoint SHA-256: `dd207a460cea4d826eba742aeb31fccac6f65c31aaac867d472182bceca0a171`
- fixed-input output digest: `f5b0385afa7d78026b0f61092914c746bee216caccdb7239d4b404063effd56c`
- deterministic output: `10/10` identical
- malformed/non-finite inputs: `4/4` fail closed
- smoke execution runtime used for this evidence refresh: CPU Torch `2.10.0+cpu`

The current real D13 vocabulary remains only `sharp|flat|natural`. `double_sharp` and `double_flat` remain Accidental R2 design targets and are **not** claimed as real-model classes by this PASS.

Decision: **PASS for shadow acceptance**. This is not Resolver wiring, end-to-end key/local image validation, sealed TEST acceptance, or production promotion.

## Rest R4 / value-specific Rest path

Architecture:

`high-recall Rest proposals -> half/quarter/eighth verifier -> deterministic arbitration -> SpecialistObservation`

Frozen class evidence:

| Class | Proposal threshold | Verifier threshold | Recall | FP reduction | Gate | TEST |
|---|---:|---:|---:|---:|---|---|
| half | 0.50 | 0.0706893578 | 1.0000 | 0.8125 | PASS | closed |
| quarter | 0.20 | 0.3782260716 | 0.989071 | 0.752369 | PASS | closed |
| eighth | 0.50 | 0.5620679259 | 0.985612 | 0.792714 | PASS | closed |

Deterministic arbitration v1 is frozen and CI-tested:

- exactly one accepted class -> accepted Rest value;
- none accepted -> `AMBIGUOUS`;
- two/three accepted -> `AMBIGUOUS`;
- NaN/Inf or invalid bbox/schema -> `REJECTED`;
- threshold boundary deterministic;
- 10/10 repeatability.

Decision: **PASS for shadow acceptance**.

## Shadow acceptance matrix

| Specialist | Real/model evidence | Integrated deterministic gate | Shadow decision | Resolver connected |
|---|---|---|---|---|
| NoteHead | PASS | n/a | **PASS** | no |
| Accidental | epoch-10 PASS | key/local deterministic contract PASS 22/22 | **PASS** | no |
| Rest R4 | Half/Quarter/Eighth PASS | arbitration PASS | **PASS** | no |

## Safety boundary

`resolver_connection_allowed()` remains `false`.

Shadow PASS does **not** authorize:

- runtime Resolver wiring;
- sealed TEST access;
- production promotion;
- PR merge;
- end-to-end OMR acceptance.
