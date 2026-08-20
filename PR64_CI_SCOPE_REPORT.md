# PR #64 — CI Scope Report

Status: **SHADOW ACCEPTANCE GREEN / REAL SPECIALIST EVIDENCE ACCEPTED / RESOLVER NOT WIRED**

This report separates deterministic/shadow acceptance from real runtime Resolver integration.

## Verified in PR #64

### Accidental deterministic key/local contract

- `K01-K06`: 6/6 PASS
- `L01-L10`: 10/10 PASS
- `A01-A06`: 6/6 PASS
- total: **22/22 PASS**

This proves the frozen deterministic rules for explicit evidence inputs. It does not prove image-derived key-signature/local-accidental separation.

### Real Accidental epoch-10 evidence

Authoritative resume completed with sealed TEST closed.

- repository provenance: `cf82ecbc0ef8df3d635e6e1923b4c4000c40da5b`
- epoch: `10/10`
- step boundary: `6150/6150`
- best epoch: `10`
- Center F1: `0.973754` >= `0.80` — PASS
- BBox F1: `0.973754` >= `0.70` — PASS
- Macro F1: `0.956454` >= `0.85` — PASS
- checkpoint SHA-256: `dd207a460cea4d826eba742aeb31fccac6f65c31aaac867d472182bceca0a171`
- real checkpoint smoke: `10/10` deterministic
- malformed/non-finite smoke: `4/4` fail closed
- frozen output digest: `f5b0385afa7d78026b0f61092914c746bee216caccdb7239d4b404063effd56c`
- TEST: closed

Real Accidental D13 shadow decision: **PASS**.

Limitation: the real D13 model still recognizes only `sharp|flat|natural`. `double_sharp` and `double_flat` remain R2 design targets, not validated real-model classes.

### Rest R4 shadow path

- Half specialist: PASS
- Quarter specialist: PASS
- Eighth specialist: PASS
- deterministic Half/Quarter/Eighth arbitration: PASS
- conflict/no-class inputs fail closed to ambiguity
- malformed/non-finite/invalid geometry inputs reject
- TEST: closed

Rest shadow decision: **PASS**.

### NoteHead

Real epoch-10 evidence remains **PASS**.

## What is still NOT validated

PR #64 does not validate or authorize:

1. a real specialist adapter connected to `runtime_deterministic_resolver`;
2. end-to-end raster -> specialist -> context router -> Resolver behavior;
3. real-image key-signature versus local-accidental separation;
4. real `double_sharp` / `double_flat` recognition;
5. production promotion;
6. sealed TEST acceptance;
7. merge authorization.

## Current acceptance matrix

| Surface | Status |
|---|---|
| NoteHead real shadow evidence | **PASS** |
| Accidental epoch-10 real shadow evidence | **PASS** |
| Accidental deterministic key/local 22-case contract | **PASS 22/22** |
| Rest Half/Quarter/Eighth specialists | **PASS** |
| Rest deterministic arbitration | **PASS** |
| Resolver wiring | **NOT WIRED** |
| End-to-end real Resolver validation | **NOT DONE** |
| Sealed TEST | **CLOSED** |
| Production promotion | **NOT AUTHORIZED** |
| PR merge | **NOT AUTHORIZED** |

## Merge-before controls

Before any merge decision:

1. require CI green on the exact final PR head;
2. fresh-read the complete diff and confirm no Resolver wiring, training mutation, checkpoint binary, TEST access, or production promotion;
3. keep PR draft until explicit merge approval;
4. preserve the distinction between shadow PASS and runtime/end-to-end PASS.

**Bottom line:** the real NoteHead, real Accidental epoch-10, Rest value-specific specialists, and their deterministic shadow contracts are now accepted at the shadow layer. Runtime Resolver integration remains a separate unopened gate.
