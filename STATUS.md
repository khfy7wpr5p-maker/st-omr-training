# ST-OMR Training Lab Status

Updated: 2026-09-16

This is the current stage-status source. `ARCHITECTURE.md` preserves historical detail; `ARCHITECTURE_CURRENT.md` and `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` describe the active lane.

## Current repository phase

- active package: TR-POLY-09B8P exact OSSQ system-image ↔ systemwise-MusicXML materialization
- B8N canonical source-byte receipt: `9f9b678e2365ec849cc19424b28d8dbdb435af3a5a9ef5b47cf7a460e72a801c`
- B8O canonical pairing-preflight receipt: `b77718f90f5865082a36f18da8701457baa5014d278e4ad33e49e727abbab65a`
- B8P canonical materialization receipt: `3f9f2df43b287dd95e03eaf1ffc4c5a3c9e25bdced8f09675918bc9b1f99e0cf`
- B8P materialized population: 412 hash-bound candidate pairs across seven READY score records
- READY scores: `7070781`, `7075297`, `7078259`, `7093885`, `7103818`, `7108150`, `8071278`
- blocked score: `7397765` — upstream opaque `a` alignment marker remains uninterpreted and fail-closed
- independent image ↔ MusicXML pair review: not yet completed
- real Stage 8 TRAIN/VALIDATION corpus: not yet completed
- real model training: not started
- real validation metrics: UNKNOWN / NOT MEASURED
- TEST: sealed
- ScoreMosaic / production authority: not granted
- commercial-use / redistribution authority: not granted by B8P

## Current stage status

| Stage / package | Description | Status |
|---|---|---|
| 0–6 | Deterministic symbolic → rendered → validated synthetic dataset | Closed / preserved |
| 7-A/B/C | Baseline model/training evidence | Historical baseline |
| 7-D | Specialist architecture/evidence | Historical evidence |
| TR-POLY-02 | Evaluation taxonomy + benchmark identity | Frozen |
| TR-POLY-03/04 | External-data registry + benchmark harness | Closed |
| TR-POLY-05/06 | V2 representation + tokenizer/roundtrip | Frozen |
| TR-POLY-07/08 | Registry + tiny 2D Transformer | Research implementation |
| TR-POLY-08A | ≤2-step deterministic smoke trainer | Frozen / preserved |
| TR-POLY-08B | ≤2-step exact research checkpoint | Frozen / preserved |
| TR-POLY-08C | Stage 6 V1→V2 execution | Single-voice evidence only |
| TR-POLY-09A | Native explicit V2 dataset/materialization | Merged |
| TR-POLY-09B1 | Free-running greedy inference | Merged |
| TR-POLY-09B2 | Deterministic V2 metric/adaptor layer | Merged |
| TR-POLY-09B3 | VALIDATION aggregation by voice/robustness | Merged |
| TR-POLY-09B4 | Multi-epoch TRAIN-only quality-training regime | Merged |
| TR-POLY-09B5 | Separate verified quality-checkpoint artifact + B1 bridge | Merged |
| TR-POLY-09B6 | Full native TRAIN/VALIDATION multi-batch quality execution | Merged |
| TR-POLY-09B7 | Exact descriptor-bound quality VALIDATION B1→B2→B3 execution | Merged / CI green |
| TR-POLY-09B8A | Persisted Native V2 reload + baseline artifact preflight | Merged / CI green |
| TR-POLY-09B8B | Dataset-specific experiment recipe + descriptor/decode/step freeze | Merged / CI green |
| TR-POLY-09B8I | Stage 8 admitted real data ↔ Native V2 lineage admission | Merged / CI green |
| TR-POLY-09B8J | Exact B8A+B8I+B8B start permit + authorized full-population B6 wrapper | Merged / CI green |
| TR-POLY-09B8K | OSSQ-OMR source selection + scanned-rights boundary | Merged / CI green |
| TR-POLY-09B8L | OSSQ per-score source inventory + independent rights/provenance review gate | Merged / CI green |
| TR-POLY-09B8M | First 5 IMSLP sources / 8 OSSQ scores independently reviewed for research training | Merged / CI green |
| TR-POLY-09B8N | Exact PDF source-byte SHA-256 pin + live revalidation | Merged / CI + live byte green |
| TR-POLY-09B8O | Exact scanned-alignment + cleaned-MusicXML pairing-source preflight | Receipt pinned / predecessor complete |
| TR-POLY-09B8P | Exact real system-image + systemwise-MusicXML materialization | 412 candidates / receipt pinned / final exact-head VERIFIED + CI pending |
| First measured quality baseline | Real admitted B6 checkpoint + B7 full VALIDATION run | Pair review/admission still required |
| Missing metric admission | Five frozen metrics remain unsupported | Separate work |
| P09C | Evidence-driven refinement | After measured quality evidence |
| P09D | Final candidate/evaluation freeze | Locked |
| Stage 9 | One-shot sealed TEST decision | TEST sealed |
| Stage 10 | ScoreMosaic shadow/integration | Not started |

## Executable quality path

```text
Independently reviewed OSSQ source identity / rights evidence
        ↓
Exact source-PDF SHA-256 byte pin (B8N)
        ↓
Exact pairing-source preflight (B8O)
        ↓
Deterministic system-image + systemwise MusicXML materialization (B8P)
        ↓
Independent image/MusicXML pair review
        ↓
Stage 8 real-data intake of VERIFIED pairs only
        ↓
Leakage-safe TRAIN + VALIDATION family split
        ↓
Persisted Native V2 TRAIN + VALIDATION root
        ↓
B8A exact artifact reload/preflight
        ↓
B8I exact real-data ↔ Native V2 lineage admission
        ↓
B8B exact experiment-recipe freeze
        ↓
B8J exact baseline start permit
        ↓
B6 full-population TRAIN-only optimization
        ↓
B5 verified selected-state checkpoint
        ↓
B7 full frozen VALIDATION execution
        ↓
B1 inference → B2 per-sample metrics → B3 aggregation
        ↓
P09C evidence-driven refinement
        ↓
P09D frozen candidate/evaluation identity
        ↓
Stage 9 one-shot sealed TEST
```

Repository regression fixtures and smoke artifacts are test evidence only. They must never be reported as real OMR model-quality evidence.

## B8A / B8I / B8B / B8J baseline gates

B8A reloads and independently validates a persisted Native V2 root containing `manifest.json`, `manifest.sha256`, `build.json`, TRAIN/VALIDATION image records and target records while keeping TEST artifact bytes absent/unread.

B8I reuses the frozen Stage 8 real-data safety path. Successful B8I lineage admission requires exact one-to-one real-data ↔ Native V2 bindings, byte identities, split/family equality, independent target-review evidence and no near-duplicate leakage veto. B8I grants quality-training eligibility only, not production or commercial authority.

B8B freezes the first real experiment recipe before quality evidence is observed. The frozen research recipe uses the existing model/trainer contracts, 8 epochs, AdamW, no scheduler, validation every epoch, minimum mean VALIDATION-loss checkpoint selection with earliest exact tie, bounded batch size and complete TRAIN/VALIDATION populations.

B8J requires B8A, B8I and B8B to bind exactly the same real dataset identity and population before the first B6 quality-training run. The authorized wrapper disallows prefix/subsample execution for the first measured baseline. B8J does not open TEST.

## B8K–B8M external-source and rights boundary

B8K pins the reviewed OSSQ-OMR camera-ready source snapshot:

`MALerLab/ossq-omr@7a17e45cddc0b7064fc3a179b62caeb57595e993`

B8L binds the camera-ready scanned-score inventory to exact upstream metadata and requires separate provenance evidence and separate independent rights evidence for each real-image score record. Upstream copyright labels are not independent rights approval.

B8M independently reviewed the first five IMSLP source documents covering eight OSSQ score records. Those records are research-training candidates only; commercial use and redistribution remain false. Every other scanned-track OSSQ source remains pending and blocked.

## B8N exact source-byte pin

B8N validates the live bytes of the five reviewed source PDFs and commits hash-only evidence.

Canonical receipt:

`9f9b678e2365ec849cc19424b28d8dbdb435af3a5a9ef5b47cf7a460e72a801c`

Raw PDF bytes are neither committed nor uploaded as evidence. B8N grants no Stage 8 admission, TEST, production or commercial authority.

## B8O pairing-source preflight

B8O binds B8N source-document identities to exact camera-ready OSSQ alignment metadata and cleaned MusicXML annotation bytes.

Canonical receipt:

`b77718f90f5865082a36f18da8701457baa5014d278e4ad33e49e727abbab65a`

Ready for deterministic materialization: `7070781`, `7075297`, `7078259`, `7093885`, `7103818`, `7108150`, `8071278`.

Blocked: `7397765`. Its upstream alignment payload contains the literal `a` marker; the pinned preprocessor does not establish its semantic meaning, so it remains fail-closed rather than guessed.

B8O does not independently approve image/MusicXML pairs or grant Stage 8 admission.

## B8P exact system-pair materialization

B8P consumes only the seven B8O READY score records and deterministically reproduces final systemwise scanned PNG ↔ systemwise MusicXML candidate pairs from exact upstream identities.

Frozen source identities:

- OSSQ commit: `7a17e45cddc0b7064fc3a179b62caeb57595e993`
- preprocessor commit: `bdea0d1829c9db84480ebd2e0385f6f5fe324274`
- B8N receipt: `9f9b678e2365ec849cc19424b28d8dbdb435af3a5a9ef5b47cf7a460e72a801c`
- B8O receipt: `b77718f90f5865082a36f18da8701457baa5014d278e4ad33e49e727abbab65a`

Canonical B8P receipt:

`3f9f2df43b287dd95e03eaf1ffc4c5a3c9e25bdced8f09675918bc9b1f99e0cf`

Committed evidence:

`evidence/ossq_b8p_system_pair_materialization.json`

Materialized candidate population:

- `7070781`: 9
- `7075297`: 28
- `7078259`: 26
- `7093885`: 26
- `7103818`: 87
- `7108150`: 117
- `8071278`: 119
- total: 412

The receipt preserves source-document SHA-256, YOLO-info SHA-256, PNG SHA-256/byte count/dimensions, MusicXML SHA-256/byte count, segment identity and upstream exclusion evidence for every candidate pair.

Raw source-PDF, PNG and MusicXML bytes are not persisted as repository evidence.

B8P explicitly does **not** grant:

- independent image ↔ MusicXML pairing approval;
- Stage 8 admission;
- TRAIN/VALIDATION assignment;
- model-training authority;
- TEST access;
- production authority;
- commercial-use authority.

Therefore 412 is the number of real materialized **candidates**, not the number of approved training examples. Real OMR accuracy remains **UNKNOWN / NOT MEASURED**.

## Current metric surface

Supported numeric B2/B3 metrics:

```text
parse_success
ter
normalized_edit_distance
exact_sequence_accuracy
pitch_accuracy
duration_accuracy
onset_accuracy
voice_accuracy
staff_accuracy
accidental_note_f1
note_staff_f1
```

Explicitly unsupported until real implementations exist:

```text
musicxml_validity
tedn
notehead_stem_f1
beam_relation_f1
tie_relation_f1
```

No unsupported metric receives a proxy number.

## Required order from here

1. finish B8P exact-head standard CI plus dedicated live `VERIFIED` receipt reproduction and merge PR #171 only if both are green;
2. independently review every B8P image/MusicXML candidate and classify it `VERIFIED`, `REVIEW_REQUIRED` or `BLOCKED`;
3. admit only `VERIFIED` pairs through Stage 8 real-data intake with exact byte/provenance/training-permission evidence;
4. group admitted samples by musical work/source-document family and split whole families into exactly one of TRAIN or VALIDATION;
5. audit duplicate/near-duplicate leakage across TRAIN/VALIDATION;
6. materialize the admitted real TRAIN/VALIDATION corpus to Native V2;
7. run B8A on the persisted real root;
8. instantiate B8I real-corpus lineage admission;
9. freeze B8B against the exact real manifest and populations;
10. issue B8J only if B8A+B8I+B8B identities agree exactly;
11. execute first real B6 training on full admitted TRAIN;
12. select checkpoint only by the frozen VALIDATION-loss rule;
13. execute B7 on the complete frozen VALIDATION population and report supported real metrics;
14. expand the real corpus before judging broad model quality;
15. freeze the final model/recipe/checkpoint before Stage 9;
16. open sealed TEST once according to Stage 9 protocol;
17. only then consider Stage 10 ScoreMosaic shadow integration.

## Safety invariants

- TEST remains sealed until Stage 9.
- TRAIN alone changes model parameters.
- VALIDATION is read-only.
- Smoke contracts remain immutable.
- Synthetic/regression fixtures are never reported as real OMR quality.
- Invalid/abstain predictions remain in metric denominators.
- Unsupported metrics remain unsupported.
- No teacher correction, ScoreMosaic upload, public score or external corpus silently becomes training data.
- Public availability does not equal training permission.
- Research-training permission does not imply commercial or redistribution authority.
- Same musical work/source-document family may not cross TRAIN/VALIDATION boundaries.
- Checkpoint, manifest, receipt and artifact identities may not be mixed across commits or runs.
- B8N source-byte pins are hash-only evidence, not Stage 8 admission.
- B8O `READY` means ready to attempt deterministic materialization only.
- B8O score `7397765` remains blocked until the `a` marker semantics are independently resolved or a separate verified pairing path is supplied.
- B8P materialization is not independent pairing approval, Stage 8 admission, TRAIN/VALIDATION assignment or training authority.
- B8P raw third-party PDF/PNG/MusicXML bytes are not persisted as evidence.
- Real OMR accuracy remains UNKNOWN / NOT MEASURED until real B6 training and full frozen B7 VALIDATION complete.
- No B8x receipt grants production authority unless explicitly designed to do so.
- Every merge requires exact-head green CI.
- A changed live receipt is investigated; it is never re-blessed merely to make CI green.
