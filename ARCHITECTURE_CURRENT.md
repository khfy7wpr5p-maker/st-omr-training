# ST-OMR Training — Current Architecture Overlay

Updated: 2026-09-14

This file records the current active lane. `ARCHITECTURE.md` remains the long-form historical architecture record. The detailed Polyphonic V2 roadmap is in `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md`.

## Current repository head

- protected branch: `main`
- current main head reviewed for this overlay: `6a6bf1e1faf0149ebd097a310536076b77feac65`
- latest merged package: PR #153 — `TR-POLY-09A: native Polyphonic V2 dataset materialization`
- latest architecture gate completed on main: native explicit Polyphonic Representation V2 TRAIN/VALIDATION materialization into the existing bounded 2D Transformer training/checkpoint chain
- sealed TEST: remains closed
- ScoreMosaic / production authority: not granted

## Active pipeline

```text
Canonical / explicit notation targets
        ↓
Deterministic symbolic validation + rendering
        ↓
Hash-bound image/target artifacts
        ↓
Polyphonic Representation V2                    ✅ FROZEN
        ↓
V2 parser + tokenizer + lossless roundtrip       ✅ CLOSED
        ↓
Research model registry                           ✅ CLOSED
        ↓
Tiny 2D Transformer architecture                  ✅ IMPLEMENTED / RESEARCH
        ↓
Bounded TRAIN-only trainer + VALIDATION read-only ✅ IMPLEMENTED
        ↓
Exact checkpoint persistence/reload               ✅ IMPLEMENTED
        ↓
Exact Stage 6 V1→V2 execution bridge              ✅ CLOSED / single-voice evidence only
        ↓
Native explicit Polyphonic V2 dataset path        ✅ CLOSED — TR-POLY-09A
        ↓
Free-running V2 inference / semantic decoding     🔄 NEXT REQUIRED CAPABILITY
        ↓
TR-POLY-09B common benchmark execution            🔒 NEXT COMPARATIVE GATE
        ↓
Polyphonic error-strata diagnosis/refinement       🔒
        ↓
Frozen candidate + sealed TEST decision            🔒 TEST SEALED
        ↓
Separate ScoreMosaic shadow/integration gate       🔒
```

## What TR-POLY-09A actually established

TR-POLY-09A added a separate native Polyphonic V2 dataset boundary without relabeling the frozen V1 corpus.

The admitted path is:

```text
explicit canonical V2 target + grayscale PNG
        ↓
native V2 manifest/build SHA-256 identity
        ↓
TRAIN/VALIDATION artifact hash verification
        ↓
lossless V2 JSON/token roundtrip
        ↓
deterministic image preprocessing
        ↓
Poly2DTrainingBatch
        ↓
TR-POLY-08A bounded trainer
        ↓
TR-POLY-08B checkpoint/reload
```

The native dataset contract requires explicit polyphonic evidence including voice 2 and corpus-level coverage of voice 3, voice 4+, note/rest/chord, simultaneous independent voices, and chord-vs-independent-voice same-onset cases. Missing voice/onset/duration/staff information is not inferred.

TR-POLY-09A does **not** establish benchmark success, quality improvement, production readiness, ScoreMosaic readiness, or a winner over existing baselines.

## Current architectural gap

The 2D Transformer currently has a teacher-forced forward/training surface. The repository does not yet expose a Polyphonic V2 free-running inference path equivalent to the V1 constrained greedy decode surface.

That gap must be closed before a common benchmark can measure real recognition output. A benchmark that scores only teacher-forced logits would not be a valid end-to-end OMR comparison.

The next capability therefore is a bounded V2 inference contract with:

- deterministic greedy decoding first;
- exact BOS/EOS/PAD handling;
- maximum decode-step/resource bounds;
- fail-closed invalid-token/state handling;
- canonical V2 parse/detokenize validation;
- explicit semantic-invalid / abstain outcome rather than silent repair;
- no TEST access;
- no hidden beam-search advantage in the first common baseline comparison.

Beam or more advanced search may be evaluated later as a separately fingerprinted experiment after the greedy baseline is frozen.

## Recommended development order

The following is the current recommended architecture order. The sub-gate names below are roadmap labels, not claims that packages already exist.

```text
P09B-0  Freeze benchmark candidate identities
        - exact checkpoint / registry / tokenizer / dataset / runtime fingerprints
        - no hyperparameter changes after benchmark start

P09B-1  Implement bounded Polyphonic V2 inference
        - free-running greedy decode
        - semantic parse/roundtrip validation
        - deterministic resource limits
        - VALIDATION only

P09B-2  Complete benchmark metric implementations/adapters
        - TR-POLY-02 required metric families
        - implement/admit missing structural metric code such as TEDn only after license/algorithm review
        - no invented metrics for unsupported outputs

P09B-3  Execute common VALIDATION benchmark
        - identical benchmark identity/splits for compared candidates
        - 1 / 2 / 3 / 4+ voice strata
        - clean + robustness buckets where admitted
        - compare sequence, structure, semantic and relation metrics

P09C    Validation-only error decomposition and bounded refinement
        - pitch / duration / onset / voice / staff
        - chord grouping / tie / beam / accidental association
        - focus model/data changes only where evidence shows a failure mode
        - TRAIN/VALIDATION only; TEST remains sealed

P09D    Freeze final candidate and evaluation recipe
        - architecture + preprocessing + decoder + checkpoint + metric set fixed
        - no further tuning from sealed-test outcomes

Stage 9 sealed TEST candidate decision
        - one-shot held-out evidence
        - accept / reject / abstain on candidate promotion

Stage 10 separate shadow integration
        - no automatic production authority
        - deterministic validators retain veto authority
```

## Why this order is preferred

The repository already has a strong provenance/checkpoint/data-contract surface. The biggest immediate risk is not missing another model architecture; it is attempting comparison before real free-running Polyphonic V2 inference exists.

Therefore the next work should **not** be a larger Transformer, wider training sweep, ScoreMosaic integration, or TEST opening. First make the current candidate measurable end-to-end under the already-frozen representation and benchmark identity.

## Existing specialist lane

The Stage 7-D specialist work, including D10/D11/D12/D13-era contracts, remains valid historical evidence and may later contribute to hybrid/fusion designs. It is no longer the current top-level active lane in this repository overlay.

Do not delete or rewrite that evidence. If a later hybrid architecture is compared against the Polyphonic V2 model, it must enter through the same frozen benchmark identity and metric surface.

## Safety boundaries that remain unchanged

- TEST remains sealed until the final Stage 9 decision gate.
- TRAIN alone may update model parameters.
- VALIDATION is read-only evaluation/tuning evidence.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- External datasets require the existing rights/license/install-pin contract.
- Large datasets and checkpoints remain outside ordinary Git content.
- Every candidate artifact remains hash/provenance bound.
- Deterministic musical validators retain veto authority over learned outputs.
- No benchmark result may be presented as production accuracy unless its exact dataset/split/metric identity is stated.

## Next gate

Implement the **bounded Polyphonic V2 free-running inference/semantic-decoding contract**, then enter **TR-POLY-09B common benchmark execution** on VALIDATION under the frozen TR-POLY-02 identity. Do not open sealed TEST and do not redesign the model before this measurement surface exists.
