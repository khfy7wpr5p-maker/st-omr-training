# Stage 7-D12 — NoteHead + Rest + Accidental deterministic GT gate

## Purpose

D12 prepares the next specialist layer after the accepted Structure / barline / meter work. It is a **data and ground-truth gate only**. It does not train or fine-tune a model.

The D12 target surface is exactly:

```text
NoteHeadSet   -> note-head bbox + center + open|filled + canonical event audit id
RestSet       -> rest bbox + half|quarter|eighth + canonical event audit id
AccidentalSet -> accidental bbox + sharp|flat|natural + canonical event audit id
```

The three target families may share one deterministic derivative bundle, but future training remains three separate specialist models/optimizers. D12 does not authorize a joint learned model.

## Accepted dependency surface

D12 starts only from the already accepted development surface used by the specialist pipeline:

```text
TRAIN        1,230 source images / 410 families
VALIDATION     153 source images /  51 families
TEST             0 specialist records exposed
```

Every accepted source PNG and accepted D6 sidecar must be re-hashed before use. D12 inherits the existing family-exclusive split. TEST must be skipped after reading only the `split` field; no TEST path, hash, label, geometry, or symbolic field may be touched.

## Ground-truth authority

D12 preserves the Stage 7-D4 authority split.

**Symbolic authority**

```text
canonical ST music
    -> deterministic MusicXML
    -> supported-V1 semantic projection
```

This determines event type, duration, displayed accidental intent and stable audit ordering.

**Spatial authority**

```text
pinned Verovio geometry render
    -> explicit visible renderer object
    -> accepted final-PNG geometry transform
```

This determines only spatial bounds/centers. AI predictions never create or repair ground truth.

If canonical and renderer evidence cannot be linked uniquely, the affected specialist sample is rejected. D12 may not invent geometry from MusicXML or infer a missing class from appearance.

## Symbol classes

### NoteHeadSet

Supported visual head classes:

```text
open   <- whole, half
filled <- quarter, eighth
```

A chord contributes one note-head target for each chord member. NoteHeadSet does not assign authoritative pitch.

### RestSet

Supported V1 rest classes/durations:

```text
half
quarter
eighth
```

Whole/full-measure rest semantics remain outside V1.

### AccidentalSet

Supported visible accidental classes:

```text
sharp
flat
natural
```

The specialist detects a visible glyph only. Its musical effect remains a later deterministic association/scope decision.

## Canonical event audit identity

D12 creates deterministic audit IDs from canonical measure/event ordering:

```text
m<measure>-e<event>
m<measure>-e<event>-n<chord-member>
```

These IDs are not renderer IDs and are not learned outputs. They exist only to prove that persisted spatial labels remain bound to the authoritative canonical event ordering.

## Fail-closed linkage

For each measure and target family, canonical and renderer object cardinality must agree exactly before linkage is admitted.

The concrete extractor additionally proves:

- visible renderer object class is explicit and supported;
- every bbox is finite and positive;
- every bbox lies inside the bound source measure/accepted crop lineage;
- note-head centers are inside their bboxes;
- canonical event IDs are unique within a source sample;
- no renderer object is silently dropped or duplicated;
- no canonical target is silently left unlinked;
- ambiguity rejects the affected specialist sample rather than guessing.

Pinned Verovio 6.2.1 exposes an anonymous `g.notehead` wrapper without a stable object bbox. D12 therefore binds NoteHead identity to the owning renderer `g.note`, verifies the expected SMuFL notehead glyph family explicitly, and uses the renderer-authored owning-note bounding box as NoteHeadSet spatial authority. Empty `g.accid` containers are structural placeholders and are not counted as visible accidentals.

## Deterministic derivative builder

The D12 builder consumes the exact accepted D6 TRAIN/VALIDATION surface and:

- re-verifies accepted D6 identity;
- re-hashes every referenced source PNG and accepted D6 label;
- replays pinned Verovio symbol geometry per family;
- maps symbol geometry through the accepted final-PNG transform;
- requires the resulting transform fingerprint to equal the accepted D6 fingerprint;
- persists one canonical hash-addressed symbol label per development image;
- emits manifest/build evidence and observed TRAIN/VALIDATION class inventory;
- writes no model, checkpoint, optimizer state, or `COMPLETE` marker.

## Independent persisted-bundle verifier

D12-5 is implemented as a separate verifier that does not trust the builder's in-memory receipt. It independently reopens the persisted bundle, frozen source corpus, and accepted D6 sidecars and recomputes:

- exact source/D6/sample identity;
- every referenced source PNG, D6 label, and D12 label SHA-256;
- manifest checksum and canonical JSON shape;
- exact `1230 TRAIN + 153 VALIDATION` sample counts;
- exact `410 + 51` family counts and family-exclusive split isolation;
- label-file/manifest one-to-one cardinality;
- source, renderer, D6 geometry, and final-PNG transform lineage;
- bbox containment and notehead-center containment;
- canonical-event and renderer-object uniqueness per target family;
- per-class TRAIN/VALIDATION inventory;
- target-instance counts, total label bytes, and artifact binding SHA-256;
- `build.json` from independently recomputed evidence.

At the D12-5 gate, the persisted root must contain exactly `manifest.json`, `manifest.sha256`, `build.json`, and `labels/`. A premature `COMPLETE` marker or unexpected file fails verification. The verifier itself does not write completion evidence.

## D12 acceptance gates

D12 may close only after all of the following are true:

1. exact `1230 TRAIN + 153 VALIDATION` source image cardinality is consumed from exactly `410 + 51` families;
2. TEST specialist records remain exactly `0`;
3. optimizer steps remain exactly `0` and no model/checkpoint is loaded;
4. source PNG hashes and accepted D6 label hashes are rechecked;
5. pinned renderer geometry and accepted final-PNG transform fingerprints are bound into every record;
6. canonical/renderer target cardinality agrees exactly for each admitted measure/kind;
7. ambiguous/unlinked/out-of-bounds geometry is rejected fail-closed;
8. emitted class inventory is reported separately for TRAIN and VALIDATION for every target class;
9. persisted artifacts are independently reopened and all hashes/identities/cardinalities are recomputed;
10. `COMPLETE` is impossible until the independent persisted-output verifier passes;
11. exact-head focused tests, full regression and independent P1/P2 review have no blocker;
12. explicit user merge approval is obtained.

D12 intentionally does **not** invent minimum class-balance counts before the deterministic inventory exists. After the verified D12 inventory is known, the next training package must freeze class-readiness/balance gates before any optimizer step. This prevents post-hoc threshold selection while avoiding an arbitrary pre-inventory number.

## Why the next training remains separate

The three tasks have different visual priors and output semantics:

```text
NoteHead -> dense small-object localization + open/filled classification
Rest     -> sparse glyph detection + 3-class duration/glyph classification
Accident -> sparse contextual glyph detection + 3-class classification
```

Therefore D12 shares deterministic data infrastructure but does not collapse them into one optimizer. A later approved training package may use common utility code, while checkpoint identity, metrics and acceptance remain specialist-specific.

## Safety boundary

- no TEST derivation;
- no optimizer/backward/training;
- no checkpoint/model loading;
- no real-data ingestion;
- no ScoreMosaic/teacher correction auto-learning;
- no learned ground truth;
- no absolute-pitch prediction authority;
- no unsupported V1 symbol class may be approximated or silently mapped.

## Current implementation sequence

```text
contract + invariant tests                         PASS
        ↓
pinned-Verovio symbol geometry pilot              PASS
        ↓
canonical-event linkage proof                      PASS
        ↓
TRAIN/VALIDATION deterministic derivative builder PASS
        ↓
independent persisted-bundle verifier              PASS
        ↓
authoritative bundle + verified class inventory   PASS
        ↓
closure evidence                                   PASS
        ↓
final exact-head regression / review               IN PROGRESS
        ↓
explicit merge approval                            PENDING
```

The authoritative bundle identity, verified class inventory, and closure invariants are frozen in `STAGE7D12_CLOSURE_EVIDENCE.md`. No D12 training is authorized by this document; the later NoteHead, Rest, and Accidental training package must be scoped and approved separately.
