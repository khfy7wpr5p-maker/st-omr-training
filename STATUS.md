# ST-OMR Training Lab Status

Updated: 2026-09-16

This is the current stage-status source. `ARCHITECTURE.md` preserves historical detail; `ARCHITECTURE_CURRENT.md` and `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` describe the active lane.

## Current repository phase

- latest merged package: PR #169 — TR-POLY-09B8N exact OSSQ source-PDF byte pinning
- latest verified `main`: `c72a2696ccdc664de63253811068fbd3ad758f56`
- B8N exact PR head `e4040c18eb4f03d4861618c1e704fd3340dd7aef`: CI run #724 success plus OSSQ B8N Source Byte Pin run #6 success before merge
- B8N canonical source-byte receipt: `9f9b678e2365ec849cc19424b28d8dbdb435af3a5a9ef5b47cf7a460e72a801c`
- active package: TR-POLY-09B8O exact OSSQ pairing-source preflight
- B8O discovery receipt: `b77718f90f5865082a36f18da8701457baa5014d278e4ad33e49e727abbab65a`
- B8O current readiness: 7 score records ready for materialization; `7397765` blocked on an uninterpreted upstream `a` alignment marker
- frozen ≤2-step smoke trainer/checkpoint remain unchanged
- TEST: sealed
- ScoreMosaic / production authority: not granted

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
| TR-POLY-09B8O | Exact scanned-alignment + cleaned-MusicXML pairing-source preflight | Active package |
| First measured quality baseline | Real admitted B6 checkpoint + B7 full VALIDATION run | Pair materialization/review/admission blocked |
| Missing metric admission | Five frozen metrics remain unsupported | Separate work |
| P09C | Evidence-driven refinement | After measured quality evidence |
| P09D | Final candidate/evaluation freeze | Locked |
| Stage 9 | One-shot sealed TEST decision | TEST sealed |
| Stage 10 | ScoreMosaic shadow/integration | Not started |

## Executable quality path

```text
Independently reviewed OSSQ source identity / rights evidence
        ↓
Exact source-PDF SHA-256 byte pin
        ↓
Exact pairing-source preflight (B8O)
        ↓
Deterministic system-image + systemwise MusicXML materialization
        ↓
Independent image/MusicXML pair review
        ↓
Leakage-safe Stage 8 TRAIN + VALIDATION quarantine/intake
        ↓
Admitted Stage 8 real TRAIN + VALIDATION metadata/byte receipts
        ↓
Persisted Native V2 TRAIN + VALIDATION root
        ↓
B8A fail-closed reload / identity preflight
        ↓
B8I exact real-data ↔ Native V2 lineage admission
        ↓
complete explicit B7 descriptor metadata
        ↓
B8B exact experiment-recipe freeze
        ↓
B8J exact baseline start permit
        ↓
B8J authorized full-population B6 wrapper
        ↓
B6 deterministic full-population batching
        ↓
B4 multi-epoch TRAIN-only optimization
        ↓
B5 verified selected-state checkpoint
        ↓
B7 exact VALIDATION descriptor/artifact binding
        ↓
B1 free-running inference
        ↓
B2 per-sample metrics
        ↓
B3 overall + voice + robustness aggregation
```

The remaining blocker is no longer an execution-code gap in B1–B8J. The first OSSQ source batch has independent rights evidence, exact source-PDF identities, and a reproducible B8O pairing-source preflight. Seven score records are currently ready to attempt deterministic materialization. Schubert `7397765` remains excluded because the exact camera-ready `sq7397765_scanned.csv` payload is `:\na`; the pinned preprocessor does not establish the meaning of the literal `a` marker, so B8O fails closed instead of inventing an interpretation. No image/MusicXML pair has yet been independently approved, and no admitted Native V2 real corpus exists yet.

Repository regression fixtures are test evidence only. They must not be reported as real model-quality evidence.

## B8A artifact gate

TR-POLY-09A persists:

```text
manifest.json
manifest.sha256
build.json
targets/<sha256>.json       # TRAIN/VALIDATION only
images/<sha256>.png          # TRAIN/VALIDATION only
```

B8A independently reconstructs and verifies the persisted build while keeping TEST artifact bytes absent/unread.

A real baseline may continue only after preflight yields an exact dataset manifest SHA-256, deterministic build ID, full TRAIN/VALIDATION population, sealed TEST metadata population without TEST bytes, and accepted provenance/license evidence where external data is involved.

## B8I real-corpus admission gate

B8I reuses the frozen Stage 8-0/8-1 real-data safety path rather than inventing a second rights/provenance system. A successful admission requires:

- an admitted `RealDataManifest` with no TEST records;
- exactly one Stage 8-1 byte receipt per admitted real sample;
- exact one-to-one real-data ↔ Native V2 development bindings;
- split, family, image SHA-256 and image-dimension equality;
- one V2 conversion-profile SHA-256 per binding;
- one independent V2 target-review evidence SHA-256 per binding;
- no Stage 8 near-duplicate leakage veto;
- exact B8A TRAIN/VALIDATION population coverage.

A successful B8I receipt grants quality-training eligibility only. It does not grant production authority or commercial-use authority and does not open TEST.

## B8B experiment gate

B8B freezes the first real baseline before quality evidence is observed. The default recipe reuses the merged contracts:

- `FROZEN_POLY_2D_CONFIG`;
- `FROZEN_POLY_2D_QUALITY_CONFIG`;
- 8 epochs;
- AdamW, no scheduler;
- max 8192 optimizer steps;
- validation every epoch;
- minimum mean VALIDATION-loss checkpoint selection, earliest exact tie;
- B6 batch size ≤8;
- full TRAIN and full VALIDATION populations only;
- B7 decode bound at least the longest admitted VALIDATION target and within the model target boundary.

The recipe hash-binds repository SHA, B8A receipt, dataset manifest/build identity, exact split populations/families, batch plan, model/trainer/materialization/B6 profiles, descriptor manifest, B7 split manifest, benchmark identity and decode bound.

Freeze fails if the optimizer-step ceiling cannot cover every full TRAIN batch across every frozen epoch. This prevents silent partial training on a corpus that is larger than the accepted recipe can execute.

## B8J baseline-start gate

B8J requires B8A, B8I and B8B to bind the same exact real baseline before B6 can start through the first-baseline path. The permit binds:

- repository SHA;
- dataset manifest/build identity;
- B8A preflight fingerprint;
- B8I admission fingerprint;
- B8B recipe fingerprint;
- exact TRAIN/VALIDATION sample populations;
- deterministic batch plan;
- model/trainer/materialization/B6 execution profiles;
- epoch and optimizer-step budget.

The authorized B8J execution wrapper forces `max_train_samples=None` and `max_validation_samples=None`. Prefix/subsample execution therefore cannot be used for the first measured baseline. Runtime configs are revalidated before B6 and B6 evidence is revalidated after execution.

B8J does not open TEST and grants neither production nor commercial-use authority.

## B8K external-source boundary

B8K pins the reviewed OSSQ-OMR camera-ready source snapshot:

`MALerLab/ossq-omr@7a17e45cddc0b7064fc3a179b62caeb57595e993`

The source family is deliberately split into two registry components:

- annotation sources plus publisher-created synthetic/derived artifacts: CC0, `LICENSE_VERIFIED`, not install-pinned;
- IMSLP-derived scanned-image track: `LICENSE_REVIEW_REQUIRED`, no training/evaluation/commercial permission asserted until per-score upstream rights are independently reviewed.

B8K does not download or admit corpus bytes and does not change TEST, production, or commercial authority.

## B8L per-score rights/provenance gate

B8L binds the camera-ready OSSQ source inventory to the exact upstream metadata objects:

- `data/scanned_score_types.tsv` Git blob `35ebe84f0c9f03a41231c92fa21183242bf12774`;
- `data/scores_w_pub.yaml` Git blob `7a72b220f4faa897900f51bcb85e49b845cb1304`.

The pinned scanned-score table has 122 score entries across 116 unique work paths. B8L recomputes the Git blob SHA from caller-supplied TSV bytes before parsing them.

Every real-image score type (`0`, `1`, `1\``, `1\`\``, `3`, `4`) requires an exact score-ID/IMSLP-ID/source-type review record. Approved records require separate provenance evidence and separate independent rights evidence plus explicit training/commercial/redistribution permission booleans. The upstream OSSQ copyright label itself cannot serve as the independent rights evidence.

Scores sharing one IMSLP source file may not carry contradictory rights/provenance decisions. Pending or rejected records remain blocked. B8L only identifies later Stage 8 candidates; it does not download source bytes, grant production authority, grant final commercial-use authority, or open TEST.

## B8M first independent rights-evidence batch

B8M independently reviewed five exact IMSLP sources covering eight OSSQ score records:

- `#04047` → `7397765`;
- `#04755` → `8071278`;
- `#64136` → `7103818`;
- `#64141` → `7070781`, `7075297`, `7078259`, `7093885`;
- `#242305` → `7108150`.

For these records only, B8M records `PUBLIC_DOMAIN`, independent rights review and research-training candidacy. It deliberately keeps commercial use and redistribution false. Every other scanned-track OSSQ record remains pending and blocked.

B8M grants neither Stage 8 admission nor production/commercial authority and opens no TEST bytes.

## B8N exact source-byte pin

B8N fetches only the five B8M-reviewed source-PDF URLs and validates actual source bytes before hashing. Non-PDF responses, population drift and sources over the 64 MiB source-document limit fail closed.

Canonical receipt:

`9f9b678e2365ec849cc19424b28d8dbdb435af3a5a9ef5b47cf7a460e72a801c`

The hash-only receipt is committed as `evidence/ossq_b8n_source_byte_pins.json`. On the final PR head, standard CI #724 and dedicated OSSQ B8N Source Byte Pin run #6 both succeeded; the latter re-fetched the five exact source URLs and proved the live bytes still matched the committed hashes and byte counts.

Raw PDF bytes are neither committed nor uploaded as workflow artifacts. B8N grants no Stage 8 admission, TEST, production or commercial authority.

## B8O pairing-source preflight

B8O binds the B8N source-document identities to exact camera-ready OSSQ alignment metadata and cleaned MusicXML annotation bytes for all eight score records. It pins:

- B8N receipt `9f9b678e2365ec849cc19424b28d8dbdb435af3a5a9ef5b47cf7a460e72a801c`;
- OSSQ camera-ready commit `7a17e45cddc0b7064fc3a179b62caeb57595e993`;
- preprocessor snapshot `bdea0d1829c9db84480ebd2e0385f6f5fe324274`;
- exact `sq<id>_scanned.csv` Git blob identity and SHA-256 per score;
- exact `sq<id>_cleaned.musicxml` Git blob identity and SHA-256 per score.

Canonical preflight receipt:

`b77718f90f5865082a36f18da8701457baa5014d278e4ad33e49e727abbab65a`

Ready for later deterministic materialization: `7070781`, `7075297`, `7078259`, `7093885`, `7103818`, `7108150`, `8071278`.

Blocked: `7397765`. Its exact alignment payload contains the literal `a` marker. The pinned preprocessor establishes the first line as a scanned-PDF page-range selector but does not establish the semantic meaning of `a`; B8O therefore records `blocked-uninterpreted-alignment-marker` rather than guessing. The literal `x` marker found in several K.464 files is retained in evidence and is never converted into a numeric value.

B8O does not materialize system-image bytes, does not independently approve image/MusicXML pairs, does not assign TRAIN/VALIDATION, does not grant Stage 8 admission, and does not open TEST or grant production/commercial authority.

## Current metric surface

Numeric B2/B3 metrics:

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

Explicitly unsupported:

```text
musicxml_validity
tedn
notehead_stem_f1
beam_relation_f1
tie_relation_f1
```

No unsupported metric receives a proxy number.

## Required order from here

1. finish B8O exact-head standard CI plus live upstream receipt verification and merge only if both are green;
2. deterministically materialize system images and corresponding systemwise MusicXML for the seven B8O-ready score records using exact B8N source bytes and pinned OSSQ/preprocessor/YOLO identities;
3. hash every materialized image/MusicXML pair and independently review pair correctness;
4. assign leakage-safe TRAIN/VALIDATION families and create Stage 8 quarantined records only from approved pairs;
5. validate exact source/image/MusicXML bytes and emit Stage 8-1 receipts, then admit only reviewed development records;
6. expand the OSSQ rights/byte-pinned batch before recipe freeze if the first batch is insufficient for voice/robustness coverage;
7. materialize one Native V2 persisted root and verify it through B8A;
8. emit the exact B8I lineage-admission receipt;
9. complete explicit B7 descriptors and emit the real B8B recipe fingerprint;
10. emit the exact B8J baseline start permit;
11. execute the authorized B8J→B6 full-population path to create the first real quality checkpoint;
12. independently verify the B5 checkpoint round trip;
13. execute B7 over the complete admitted VALIDATION population;
14. inspect B2/B3 failures by voice/robustness and choose P09C from evidence;
15. admit the five missing metric implementations without proxies;
16. freeze P09D candidate/evaluation identity;
17. open sealed TEST once at Stage 9;
18. only then consider Stage 10 ScoreMosaic shadow integration.

## Safety invariants

- TEST remains sealed until Stage 9.
- TRAIN alone changes model parameters.
- VALIDATION is read-only.
- Smoke contracts remain immutable.
- Quality work uses separate versioned provenance.
- No semantic truncation is allowed.
- Invalid/abstain predictions stay visible in denominators.
- Unsupported metrics stay unsupported.
- Benchmark identity and candidate identity must match exactly.
- External data requires rights/license/install-pin admission.
- Teacher corrections and ScoreMosaic uploads are not automatic training data.
- B8I admission is not commercial-use authorization.
- B8J permit authorizes the exact full first-baseline run only.
- B8K scanned-source selection is not permission to use IMSLP-derived bytes.
- B8L upstream copyright labels are not independent rights approval.
- B8M grants research-training candidacy only to the explicitly reviewed first batch; no commercial/redistribution authority.
- B8N source-byte pins are hash-only evidence, not Stage 8 sample admission.
- B8O `READY` means ready to attempt deterministic materialization only; it is not pairing approval or Stage 8 admission.
- B8O score `7397765` remains blocked until the `a` alignment marker semantics are independently resolved or a separate verified pairing path is supplied.
- No training or VALIDATION benchmark package grants production authority.
- Every merge requires exact-head green CI.
