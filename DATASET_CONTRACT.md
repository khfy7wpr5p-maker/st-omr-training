# ST-OMR Synthetic Dataset Manifest Contract

## Status

This document defines the bounded Stage 5-A V1 contract after the completed Stage 4 Controlled Degradation boundary.

Stage 5-A defines when a collection of synthetic Stage 4 derivatives is structurally eligible to become a dataset manifest. It does **not** generate a large dataset, choose final split ratios, write image artifacts to storage, train a model, ingest real/user scores, or integrate anything into ScoreMosaic.

The manifest validator is an independent acceptance layer. It must not trust a Stage 4 object merely because that object was produced by the degradation module.

## 1. Input boundary

One Stage 5-A sample records one Stage 4 PNG derivative and the symbolic/renderer/degradation lineage required to audit it:

```text
DatasetSample
├── sample_id
├── family_id
├── split = train | validation | test
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
├── mode = L
└── image_format = png
```

The symbolic target remains the MusicXML/canonical score lineage. The PNG is a derived visual input, never a replacement source of musical truth.

## 2. Synthetic-only V1 boundary

The V1 manifest source class is exactly:

```text
synthetic
```

Real scores, user documents, teacher corrections, private data, copyrighted/rights-unclear score collections, and scraped MusicXML are not valid Stage 5-A V1 manifest content.

Real-data training requires a separate later rights/provenance/privacy/quality contract.

## 3. Immutable model

The public Stage 5-A objects are frozen/immutable dataclasses.

`DatasetManifest.samples` must be an immutable tuple. A mutable list is rejected at the boundary.

The manifest does not contain a runtime creation timestamp, host path, temporary directory, random UUID, or machine-local artifact path. Those fields would make identical logical manifests serialize differently and are not part of training semantics.

## 4. Split contract

Supported splits are exactly:

```text
train
validation
test
```

The V1 split policy is:

```text
family-exclusive-v1
```

All derivatives of one symbolic source family must remain in one split.

Forbidden example:

```text
train: score-A / clean
validation: score-A / blur
```

The independent validator maintains a `family_id → split` map and vetoes any family that appears in more than one split.

A training-eligible V1 manifest must contain at least one sample in every split. Stage 5-A does not define the final train/validation/test percentage ratios; ratio policy belongs to the later dataset construction package.

## 5. Leakage-defense aliases

`family_id` alone is not trusted as sufficient evidence against leakage because a faulty builder could assign two family IDs to identical underlying content.

The independent validator therefore also enforces:

- one identical `source_musicxml_sha256` may not be represented under multiple family IDs;
- one identical `source_musicxml_sha256` may not cross splits;
- one identical clean `source_svg_sha256` may not be represented under multiple family IDs;
- one identical clean `source_svg_sha256` may not cross splits.

These rules deliberately prefer rejecting ambiguous duplicate targets over allowing exact symbolic/visual duplicates to contaminate held-out evaluation.

## 6. Duplicate and collision policy

The V1 manifest rejects duplicate:

- `sample_id`;
- Stage 4 `derivative_id`;
- final `png_sha256`.

A duplicate image is not accepted as an additional training sample merely because metadata or ordering differs. This prevents accidental weighting inflation and exact-image leakage.

## 7. Independent Stage 4 lineage recomputation

Stage 5-A independently mirrors the frozen Stage 4 V1 replay fields and producer versions:

```text
degradation_version = st-controlled-degradation-v1
CairoSVG            = 2.8.2
Pillow               = 12.3.0
```

For every sample the validator independently recomputes:

1. the Stage 4 degradation configuration fingerprint from the exact replay integers;
2. the Stage 4 derivative identity from family/page/source/config/final hashes;
3. the Stage 5-A sample identity from the immutable artifact/target lineage.

A mismatch fails closed.

This is intentionally separate from Stage 4 implementation logic so a corrupted provenance field cannot be accepted merely because a precomputed ID is present.

## 8. Sample identity

`sample_id` identifies the artifact/target lineage and intentionally does **not** include the train/validation/test split.

Moving an otherwise identical sample to another split therefore does not manufacture a new sample identity. Split assignment remains manifest semantics and changes the manifest hash, while the sample itself retains the same identity.

## 9. PNG bridge verification

`sample_from_degraded_page(...)` is a narrow Stage 4 → Stage 5-A adapter. It does not write files.

Before accepting a Stage 4 PNG it independently verifies, using the Python standard library:

- non-empty bounded bytes;
- PNG signature;
- canonical first `IHDR` chunk;
- IHDR CRC;
- positive bounded dimensions;
- 8-bit grayscale color type;
- non-interlaced V1 representation;
- supplied PNG SHA-256 against the actual bytes;
- supplied dimensions against the PNG header;
- `mode = L`;
- exact Stage 4 direct producer versions;
- replay configuration fingerprint and derivative identity.

Full persisted-artifact discovery/storage verification remains outside Stage 5-A because Stage 5-A does not create or own dataset storage.

## 10. Image bounds

Stage 5-A preserves the Stage 4 V1 resource ceiling:

```text
maximum pixels per clean/final image = 16,000,000
maximum Stage 3 page number          = 64
```

The manifest records clean and final dimensions. Invalid, zero, boolean-as-integer, or over-budget dimensions fail closed.

V1 accepts only:

```text
mode         = L
image_format = png
```

## 11. Runtime provenance

Each sample preserves Stage 4 runtime evidence required to audit raster determinism:

```text
degradation_version
CairoSVG version
Pillow version
Cairo runtime version
Python version
platform system
platform machine
```

Cairo is a system runtime and is not fully pinned by the Python package requirements. Therefore cross-platform/cross-Cairo byte identity is not inferred automatically.

## 12. Canonical manifest serialization

The V1 schema version is:

```text
st-dataset-manifest-v1
```

Canonical manifest serialization is UTF-8/ASCII-compatible JSON with:

- sorted object keys;
- fixed compact separators;
- no whitespace-dependent semantics;
- deterministic sample ordering independent of input tuple order.

Canonical sample ordering is based on stable identity fields rather than insertion order.

`dataset_manifest_sha256(...)` is SHA-256 over these canonical bytes.

Two logically identical valid manifests containing the same samples in different tuple orders must produce identical canonical bytes and identical manifest SHA-256.

Split assignment is semantic and is serialized; changing a sample from train to validation therefore changes the manifest hash.

## 13. Validator failure policy

The independent validator fails closed for, among other cases:

- unsupported schema/source/split policy;
- malformed identifiers or hashes;
- unsupported Stage 4 producer versions;
- invalid replay configuration;
- config fingerprint mismatch;
- derivative identity mismatch;
- sample identity mismatch;
- invalid dimensions/mode/format;
- duplicate sample, derivative, or PNG identity;
- family leakage across splits;
- MusicXML-target alias leakage;
- clean-SVG alias leakage;
- missing train, validation, or test split.

Validation issues are returned in deterministic order so repeated validation of the same invalid manifest produces stable evidence.

## 14. Dataset construction boundary

Stage 5-A does not choose which generated scores should exist and does not create thousands of PNG files.

The later dataset construction stage may:

```text
validated symbolic families
        ↓
validated Stage 4 derivatives
        ↓
family-level split assignment
        ↓
Stage 5-A DatasetSample metadata
        ↓
independent manifest validation
        ↓
canonical manifest + external artifact set
```

A builder is never allowed to mark its own output valid by assertion. The independent Stage 5-A validator remains a veto gate.

## 15. Storage boundary

Large PNG datasets are not normal Git repository content.

Stage 5-A intentionally does not define local absolute paths, cloud bucket names, credentials, signed URLs, or production storage providers. Storage design belongs to the dataset build/storage package and must retain artifact SHA-256 verification.

## 16. Verification gate

Stage 5-A may be accepted only after all applicable evidence passes:

1. focused model/config/identity tests;
2. negative and corruption tests;
3. family/target/SVG leakage tests;
4. duplicate/collision tests;
5. canonical serialization and manifest-hash determinism tests;
6. real Generator → MusicXML → Verovio → Stage 4 → Stage 5-A bridge tests;
7. full repository regression;
8. Python compile validation;
9. GitHub-hosted CI on the exact final pull-request head;
10. separate merge approval;
11. post-merge GitHub-hosted CI on the exact resulting `main` commit.

## 17. Explicitly out of scope

Stage 5-A does not add:

- bulk dataset generation;
- final split ratios or sampling/balancing policy;
- filesystem/cloud artifact storage;
- training loaders;
- model training or checkpoints;
- real/user score ingestion;
- teacher-correction learning;
- ScoreMosaic integration;
- Guitar TAB training;
- automatic deployment.
