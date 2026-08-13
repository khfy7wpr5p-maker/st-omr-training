# ST-OMR Synthetic Dataset v1 Construction Contract

## Status

This document defines the bounded Stage 6 V1 dataset-construction layer after the completed Stage 5 independent manifest validator.

Stage 6 may construct synthetic training artifacts only. It does not train a model, ingest real/user scores, learn from teacher corrections, integrate with ScoreMosaic, or create Guitar TAB data.

## 1. Required upstream gates

Stage 6 may use only the already-validated pipeline:

```text
Stage 1 deterministic generator
        ↓
Stage 2 MusicXML writer + validators + semantic round trip
        ↓
Stage 3 pinned Verovio renderer
        ↓
Stage 4 controlled degradation
        ↓
Stage 5 independent dataset manifest validator
```

Stage 6 is a builder, never its own validator. A produced dataset is rejected unless the independent Stage 5 manifest gate passes.

## 2. Builder identity

The V1 builder version is:

```text
st-synthetic-dataset-builder-v1
```

A build records a deterministic configuration fingerprint, canonical Stage 5 manifest SHA-256, and a `build_id` derived from those identities and the builder version.

## 3. Bounded build configuration

The V1 builder accepts immutable configuration only.

Current ceilings:

```text
families                  3 .. 5,000
measures per family       1 .. 64
raster width              512 .. 2,400
maximum emitted samples   50,000
seed                       0 .. 2^63-1
```

The lower family bound of three exists because the Stage 5 contract requires non-empty train, validation, and test splits.

## 4. Family construction profiles

The V1 family planner may cycle through these deterministic symbolic profiles:

```text
mixed
note-only
rest-only
chord-only
time-2-4
time-3-4
time-4-4
no-accidentals
```

These profiles are not a claim of statistical completeness. They are a bounded V1 coverage policy over the symbolic scope already supported by Stages 1 and 2.

Every family uses a unique sequential generator seed from the configured seed range. The canonical generator `score_id` becomes the Stage 4/5 `family_id`.

## 5. Split policy

Stage 6 freezes the V1 family-level target weights to:

```text
train       80
validation  10
test        10
```

Split planning occurs before rendering/degradation. Families are deterministically ranked from builder version, split seed, family index, family seed, and family profile. Integer allocation uses the 80/10/10 weights while guaranteeing at least one family in every split.

All pages and all degradation derivatives of one family inherit the same split. A split move never creates a new sample identity.

The Stage 5 validator remains authoritative for family leakage, identical-target alias leakage, identical-clean-render alias leakage, and duplicate/collision vetoes.

## 6. Degradation profiles

The Stage 6 V1 builder may request only the already-frozen Stage 4 profiles:

```text
clean
light
medium
```

Derivative seeds are deterministically derived from builder version, `family_id`, page number, and degradation profile. They do not depend on the train/validation/test split.

No new augmentation primitive is introduced by Stage 6.

## 7. Symbolic target rule

MusicXML produced from the canonical generated score remains the symbolic target.

Before a target artifact is admitted to a Stage 6 build it must:

- be non-empty bytes;
- match its SHA-256 identity;
- pass the independent Stage 2-C MusicXML validation gate.

Rendered/degraded PNG content never replaces or rewrites musical truth.

## 8. Image artifact rule

A final image artifact must:

- be non-empty PNG bytes;
- match its Stage 4/5 PNG SHA-256 identity;
- already have passed the Stage 5 PNG/hash/lineage bridge;
- be unique inside the final build.

If two families produce identical MusicXML targets or two samples produce identical final PNG bytes, Stage 6 fails closed rather than silently weighting the duplicate.

## 9. Independent final manifest gate

After all families are constructed, Stage 6 creates an immutable Stage 5 `DatasetManifest` and calls the existing independent validator.

A build object is created only if that validator returns valid.

The build object additionally requires its target-artifact set and image-artifact set to match the manifest exactly: no missing artifacts and no unreferenced extras.

## 10. Determinism scope

For the same Stage 6 configuration and the same verified runtime environment, the builder must reproduce:

- the same family plan;
- the same split assignments;
- the same generator inputs;
- the same Stage 4 derivative seeds;
- the same canonical manifest bytes and SHA-256;
- the same target/image artifact hashes;
- the same `build_id`.

Cross-platform raster byte identity is not inferred automatically because Stage 4 already records Cairo/platform runtime provenance. Mixed-runtime artifacts require separate verification.

## 11. Hash-addressed storage layout

Stage 6 may persist a validated build only outside normal Git-tracked dataset content, using the deterministic layout:

```text
<dataset-root>/
├── manifest.json
├── manifest.sha256
├── build.json
├── targets/
│   └── <source_musicxml_sha256>.musicxml
└── images/
    └── <png_sha256>.png
```

Paths are derived from validated lowercase SHA-256 identities, not user filenames.

The writer:

- refuses to overwrite an existing dataset directory;
- writes into a deterministic temporary sibling directory first;
- re-hashes persisted manifest/target/image bytes;
- renames the completed staging directory into place only after all writes succeed;
- removes its own temporary directory on failure.

Cloud buckets, credentials, signed URLs, production storage providers, and remote upload policy remain outside Stage 6 V1.

## 12. Git boundary

Large generated PNG datasets are not normal repository content.

Tests may create small temporary datasets and must remove them after verification. Stage 6 source code, contracts, and deterministic recipes may live in Git; bulk artifacts do not.

## 13. Required verification gate

Stage 6 may close only after all applicable evidence passes:

1. configuration and bound tests;
2. deterministic split-planning tests;
3. family-profile planning tests;
4. artifact hash/type failure tests;
5. real Generator → MusicXML → Verovio → degradation → Stage 5 → Stage 6 integration;
6. same-config rebuild determinism test;
7. verified hash-addressed filesystem writer test with no-overwrite behavior;
8. complete repository regression;
9. Python compile validation;
10. GitHub-hosted CI on the exact final PR head;
11. separate explicit merge approval;
12. post-merge GitHub-hosted CI on the exact resulting `main` commit.

## 14. Explicitly out of scope

Stage 6 does not add:

- model architecture or training loaders;
- optimizer/loss/checkpoint logic;
- baseline training runs;
- real or user score ingestion;
- copyrighted/rights-unclear web corpora;
- teacher-correction learning;
- ScoreMosaic runtime integration;
- Guitar TAB training;
- cloud dataset upload;
- automatic deployment.

Those responsibilities remain locked behind later stages.
