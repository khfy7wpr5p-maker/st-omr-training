# Canonical ST Music Data Contract

## Purpose

This document defines the conceptual data contract shared by the ST Music Generator, independent validator, MusicXML writer, renderer boundary, degradation layer, and dataset validation/construction pipeline.

The implementation may evolve, but it must preserve these semantic boundaries unless the contract is explicitly revised.

## Root object

```text
Score
├── score_id
├── schema_version
├── generator_version
├── seed
├── provenance
└── parts[]
```

A generated score must carry enough information to reproduce and trace its origin.

## Part

V1 uses one part and one staff.

Conceptual fields:

```text
Part
├── part_id
├── instrument_class
├── staff_count
└── measures[]
```

For V1, `staff_count` is 1 and the instrument class is a generic treble-staff score rather than a Guitar TAB or piano-specific representation.

## Measure

```text
Measure
├── number
├── time_signature
├── key_signature
├── clef
├── voices[]
└── expected_duration
```

V1 constraints:

- time signature: 2/4, 3/4, or 4/4
- key signature: 0
- clef: treble
- one voice

The independent validator must compute actual occupied duration and verify it against the expected measure duration.

## Voice

```text
Voice
├── voice_id
└── events[]
```

V1 uses `voice_id = 1`, while the contract keeps voice as an explicit concept so later polyphony can be added without replacing the data model.

## Music events

Every event is one explicit semantic type:

```text
MusicEvent
├── NoteEvent
├── ChordEvent
└── RestEvent
```

A chord is a first-class event. The internal model must not represent a chord merely as an XML serialization trick.

## NoteEvent

Conceptual fields:

```text
NoteEvent
├── onset
├── duration
├── voice
├── staff
├── pitch
└── notation_intent
```

Pitch is structured:

```text
Pitch
├── step       # A through G
├── alter      # e.g. -1, 0, +1 in V1
└── octave
```

Do not use an opaque string such as `C#4` as the primary source of truth.

## ChordEvent

```text
ChordEvent
├── onset
├── duration
├── voice
├── staff
└── notes[]
```

Chord invariants include:

- at least two notes
- a common onset
- compatible common duration for V1
- same voice and staff for V1
- no duplicate pitch entries inside the same chord event

V1 supports 2-note, 3-note, and 4-note chords.

## RestEvent

```text
RestEvent
├── onset
├── duration
├── voice
└── staff
```

A rest has no pitch.

V1 supports eighth, quarter, and half rests. Full-measure rests are intentionally deferred because notation semantics differ from a simple whole-duration rest.

## Time representation

Musical time must use exact rational values, not floating-point duration values.

Examples:

```text
whole   = 1/1
half    = 1/2
quarter = 1/4
eighth  = 1/8
```

Onsets are also exact rational values within the measure.

Negative, zero where forbidden, non-finite, or otherwise invalid durations must be rejected.

## Pitch and notation semantics

Pitch semantics and visible notation semantics are distinct.

For example, an altered pitch and whether an accidental glyph is explicitly displayed are separate concepts. Canonicalization must preserve visually meaningful notation intent.

Conceptually:

```text
notation_intent
└── display_accidental = none | sharp | flat | natural
```

The implementation representation may evolve, but the semantic distinction is mandatory.

## Provenance

Each generated score must record provenance sufficient for later audit and reproduction.

Conceptual fields:

```text
provenance
├── source_type
├── source_id
├── generator_version
├── seed
├── config_version
└── pipeline_identity
```

Initial source types:

- `reference`
- `procedural`
- `targeted`

A future `real_verified` class, if introduced, must remain distinct from synthetic classes.

## Family identity

All rendered, degraded, and dataset-manifest derivatives of the same symbolic source must share one `family_id`.

Stage 4 V1 example:

```text
family: score-1842
├── clean-svg-page-1
├── raster-clean-page-1
├── light-seed-17-page-1
└── medium-seed-42-page-1
```

Dataset splitting must operate on source families, not individual images, so derivatives of one symbolic score cannot leak across train, validation, and test partitions.

Stage 5-A strengthens this rule by independently checking identical MusicXML target hashes and identical clean SVG hashes. A builder may not bypass the family rule simply by assigning different `family_id` values to identical underlying content.

## Derived artifact provenance

Stage 3 clean renderer output and Stage 4 raster/degraded derivatives remain derived artifacts; they do not become symbolic ground truth.

A derived artifact record should preserve, as applicable:

```text
DerivedArtifact
├── family_id
├── source_artifact_sha256
├── artifact_sha256
├── artifact_type
├── producer_name
├── producer_version
├── config_fingerprint
├── seed_or_replay_parameters
└── transformation_metadata
```

For Stage 3, producer identity includes the pinned renderer/runtime and renderer configuration.

Stage 4 V1 additionally preserves the original MusicXML hash, renderer configuration fingerprint, source SVG hash, clean raster hash, exact `DegradationConfig`, degradation configuration fingerprint, final PNG hash, Stage 4 version, direct image-library versions, Cairo runtime version, Python version, and platform system/machine. The exact implemented representation is defined in [DEGRADATION_CONTRACT.md](DEGRADATION_CONTRACT.md).

## Dataset sample metadata

Stage 5-A is governed by [DATASET_CONTRACT.md](DATASET_CONTRACT.md).

One manifest sample preserves the Stage 4 lineage required for independent validation:

```text
DatasetSample
├── sample_id
├── family_id
├── split
├── page_number
├── source_musicxml_sha256
├── renderer_config_fingerprint
├── source_svg_sha256
├── clean_raster_sha256
├── degradation_config_fingerprint
├── degradation_config
├── derivative_id
├── png_sha256
├── degradation/runtime provenance
├── clean dimensions
├── final dimensions
├── mode
└── image_format
```

Stage 5-A V1 samples are synthetic only. The manifest source class is explicit rather than inferred.

The sample identity does not contain train/validation/test assignment. A split move therefore cannot manufacture a new sample identity. Split assignment remains manifest semantics and participates in the canonical manifest hash.

Stage 5-A independently recomputes the Stage 4 replay-configuration fingerprint and derivative identity before accepting the sample lineage.

## Dataset manifest metadata

The V1 manifest records:

```text
DatasetManifest
├── schema_version = st-dataset-manifest-v1
├── source_class = synthetic
├── split_policy = family-exclusive-v1
├── dataset_name
├── dataset_version
└── samples[]
```

The in-memory sample collection is immutable. Canonical serialization sorts logically unordered samples by stable identity fields and uses canonical JSON, so tuple insertion order does not affect the manifest SHA-256.

A training-eligible V1 manifest requires `train`, `validation`, and `test` to be present. Final split percentages/balancing policy are intentionally not defined by Stage 5-A; those belong to the later dataset builder.

Duplicate sample IDs, Stage 4 derivative IDs, and final PNG hashes are rejected rather than counted as extra examples.

## Hash lineage

Pipeline stages must preserve a traceable hash chain where applicable:

```text
Canonical model
    ↓ hash
MusicXML
    ↓ hash
Clean rendered SVG page
    ↓ hash
Clean grayscale raster
    ↓ hash
Controlled degradation derivative
    ↓ hash
Stage 5-A DatasetSample identity
    ↓
Canonical DatasetManifest
    ↓ hash
Dataset manifest SHA-256
```

Hashes support traceability; they do not replace semantic, renderer-safety, degradation-safety, dataset leakage, or artifact validation.

## Canonicalization constraints

Canonicalization may normalize ordering or serialization choices that do not alter musical or visual notation meaning.

It must not silently collapse notation-distinct forms such as different enharmonic spellings or explicit accidental-display intent.

Visual degradation is never canonicalization. A degraded image is a derivative of the same symbolic source and must retain lineage back to that source.

Manifest sample ordering is non-semantic and may be canonicalized. Split assignment, family identity, target hashes, replay parameters, and artifact identities are semantic metadata and must not be discarded during canonicalization.
