# TR-POLY-02 — Polyphonic Evaluation Taxonomy + Benchmark Contract

Status: **implementation package; no dataset access, no inference, no training, no sealed-TEST access**.

## Purpose

TR-POLY-02 freezes the common evaluation vocabulary that later ST-OMR polyphonic
research must use before Model A, Model B, or Model C can be compared.

This package is additive. It does not replace or reinterpret the frozen Stage 7-D3
supported-V1 diagnostics. Historical metrics remain historical evidence.

## Starting authority

- repository: `khfy7wpr5p-maker/st-omr-training`
- base protected `main`: `a6a40b218a95c72349984ee2aee7262f467021fc`
- current Stage 7-D2 monolithic baseline remains non-production evidence
- current specialist/shadow work remains independent and is not promoted by this package
- TEST remains sealed

## Evaluation hierarchy

Future comparable polyphonic runs must expose the following metric families.

### Serialization

- `parse_success`
- `musicxml_validity`

### Sequence

- `ter`
- `normalized_edit_distance`
- `exact_sequence_accuracy`

### Structural

- `tedn`

TR-POLY-02 reserves TEDn in the evaluation contract but does **not** implement or
vendor a TEDn algorithm. Metric implementation and license review belong to a
later benchmark package.

### Musical semantic

- `pitch_accuracy`
- `duration_accuracy`
- `onset_accuracy`
- `voice_accuracy`
- `staff_accuracy`

### Relations

- `notehead_stem_f1`
- `beam_relation_f1`
- `tie_relation_f1`
- `accidental_note_f1`
- `note_staff_f1`

Confidence calibration metrics are intentionally not required here. ECE, Brier,
temperature scaling, conformal methods and selective-prediction research remain
TR-POLY-20 scope.

## Versioned error taxonomy

The frozen taxonomy is:

- `PITCH`
- `DURATION`
- `ONSET`
- `VOICE`
- `STAFF`
- `REST`
- `ACCIDENTAL`
- `TIE`
- `SLUR`
- `TUPLET`
- `BEAM`
- `STEM`
- `CHORD_GROUPING`
- `CROSS_STAFF`
- `METER`
- `MEASURE_BOUNDARY`
- `GRACE`
- `ORNAMENT`
- `OTHER`
- `AMBIGUOUS`

Taxonomy version:

`st-omr-poly-error-taxonomy-v1`

A later taxonomy change requires a new version. Historical reports must not be
silently relabeled under a changed taxonomy.

## Polyphonic complexity profile

Every future benchmark sample should be able to carry:

- `voice_count`
- `staff_count`
- `simultaneous_note_density`
- `chord_density`
- `overlap_density`
- `tie_density`
- `beam_complexity`
- `rhythmic_complexity`
- `tuplet_present`
- `grace_present`
- `cross_staff_present`

Density/complexity values are normalized to `[0, 1]`. The precise extraction
algorithm is not defined by this package; when implemented it must itself be
versioned and deterministic.

Voice strata are frozen as:

- `1_voice`
- `2_voice`
- `3_voice`
- `4_plus_voice`

The existing V1 corpus is expected to populate only the first stratum. That is
not an error; it is a limitation that must remain visible rather than being
reported as broad polyphonic coverage.

## Robustness buckets

The minimum common robustness surface is:

- `clean`
- `scan`
- `phone`
- `blur`
- `perspective`
- `low_contrast`

This package does not synthesize those conditions and does not change the
existing degradation pipeline. TR-POLY-17 owns degradation/domain-randomization
v2.

## Benchmark identity

Any Model A/B/C comparison must be bound to the same exact:

- `benchmark_id`
- `benchmark_version`
- `dataset_manifest_sha256`
- `split_manifest_sha256`
- taxonomy version
- evaluation-contract version

The contract emits a deterministic canonical SHA-256 over this identity.

If one candidate uses a different benchmark or split identity, the comparison
fails closed instead of reporting a winner.

## Sample identity

A benchmark sample descriptor binds:

- sample SHA-256 identity
- family identity
- split
- polyphonic complexity profile
- robustness bucket

This does not replace the existing dataset manifest, rights, provenance,
leakage, or sealed-test controls. It is an evaluation overlay.

## Relationship to Stage 7-D3

Stage 7-D3 already measures useful V1 fields such as token error rate, exact
sequence accuracy, onset, duration, pitch, rest and chord-size diagnostics.

TR-POLY-02 does not mutate that code because doing so would change historical
baseline semantics. Future adapters may map compatible D3 outputs into this
contract while marking unavailable polyphonic fields as unsupported at the
adapter boundary; they must not invent voice/staff/relation scores.

## Fail-closed rules

A future model-family comparison must fail if:

- required metrics are missing;
- unknown metrics are silently injected into the frozen required set;
- accuracy/F1/probability-like values are outside `[0, 1]`;
- TER/NED/TEDn-like distances are negative;
- benchmark identities differ;
- benchmark hashes are malformed;
- duplicate error classes appear in a canonical error-count set;
- invalid split or complexity metadata is supplied.

## Safety boundaries

TR-POLY-02:

- does not open TRAIN, VALIDATION, or TEST bytes;
- does not load a checkpoint;
- does not run model inference;
- does not run an optimizer;
- adds no dependency;
- adds no external dataset;
- changes no ScoreMosaic behavior;
- changes no Correction Engine behavior;
- changes no current specialist thresholds;
- changes no existing Stage 7 tokenizer;
- changes no current data split;
- does not implement TEDn yet.

## Next package boundary

After this package passes exact-head CI, the next independent package may be
TR-POLY-03 — external dataset/license registry.

TR-POLY-03 must not download or commit external dataset bytes. It should first
freeze legal/usage metadata and checksum/install-manifest rules.
