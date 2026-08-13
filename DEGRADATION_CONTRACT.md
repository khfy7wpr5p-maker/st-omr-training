# ST-OMR Controlled Degradation Contract

## Status

This document defines the bounded Stage 4 V1 contract after the completed Stage 3 renderer boundary.

Stage 4 converts a validated, self-contained Stage 3 SVG page into a clean grayscale PNG raster and, optionally, a deterministic appearance-degraded derivative. It must not change the canonical symbolic target, MusicXML semantics, renderer identity, or source-family identity.

Stage 4 is a training-data preparation layer. It is not a renderer replacement, dataset splitter, model-training stage, or production ScoreMosaic image processor.

## 1. Input boundary

Stage 4 consumes one Stage 3 SVG page plus explicit lineage:

```text
family_id
page_number
source_musicxml_sha256
renderer_config_fingerprint
svg bytes
svg_sha256
```

The supplied SVG hash is recomputed before rasterization. A mismatch fails closed.

The SVG boundary independently re-checks the safety properties needed by the rasterizer rather than trusting Stage 3 merely because the input was produced there.

V1 rejects:

- empty or oversized SVG input;
- malformed XML;
- a non-SVG root;
- a DTD/DOCTYPE;
- missing, non-finite, non-positive, extreme, or implausibly shaped `viewBox` dimensions;
- active/external content such as script, iframe, object, embed, image, audio, or video elements;
- external `href` values;
- external CSS imports or external `url(...)` references;
- an excessive SVG element count.

Internal fragment references such as `href="#glyph-id"` remain permitted because the pinned Stage 3 Verovio output uses them for embedded glyph reuse.

## 2. Pinned V1 image runtime

Direct Stage 4 Python dependencies are pinned:

```text
CairoSVG==2.8.2
Pillow==12.3.0
```

CairoSVG performs SVG-to-PNG rasterization through Cairo. Pillow performs the bounded grayscale image transformations and deterministic PNG/JPEG encoding steps.

The application checks the exact CairoSVG and Pillow package versions before processing. The Cairo runtime version and execution environment are recorded in every output result because system Cairo is not represented by the Python package pins alone.

Cross-platform or cross-Cairo byte identity is therefore not assumed automatically. Artifacts from a new operating system, architecture, or Cairo runtime require separate equivalence evidence before they are mixed into one dataset build.

Dependency upgrades require a separate review and regression package.

## 3. Clean raster boundary

Rasterization uses the validated SVG bytes directly with:

```text
output width = explicit bounded config value
background   = white
unsafe       = false
```

The resulting image is converted to grayscale mode `L` and encoded as a canonical V1 PNG using fixed PNG options.

V1 raster width is bounded to `512..2400` pixels. Total output pixels are capped at `16,000,000`.

The clean raster is retained as a separately hashed lineage point. Its SHA-256 is recorded even when the final derivative contains no additional degradation.

A clean raster that is effectively blank, has invalid dimensions, exceeds resource limits, or is implausibly dark fails closed.

## 4. V1 transformation set

The V1 degradation set is intentionally conservative. The allowed appearance operations are:

- rotation up to ±3.000 degrees;
- Gaussian blur up to 2.000 pixels;
- bounded deterministic grayscale noise, level 0..20;
- brightness multiplier 0.800..1.200;
- contrast multiplier 0.750..1.250;
- optional JPEG round-trip quality 65..95, with final storage returned as PNG.

All public configuration values use integers. Fractional rotation/blur/brightness/contrast values are encoded in milli-units so the replay contract does not depend on floating-point configuration serialization.

V1 applies operations in this fixed order:

```text
validated SVG
    ↓
clean grayscale raster
    ↓
rotation (optional, expand=true, white fill)
    ↓
blur
    ↓
contrast
    ↓
brightness
    ↓
deterministic noise
    ↓
optional JPEG round-trip
    ↓
final grayscale PNG
```

Rotation expands the canvas instead of cropping. A geometric ink-retention gate rejects an implausible loss or gain in dark score content before photometric transformations continue.

## 5. Explicitly deferred transformations

The following transformations are not part of Stage 4 V1:

- arbitrary cropping;
- perspective warping;
- shear;
- elastic deformation;
- synthetic shadows/occlusion;
- staff-line deletion;
- symbol deletion;
- content-aware erasing;
- page tearing or partial-page removal.

These operations can destroy or relocate musical content in ways that require stronger score-region/content-preservation validation. They may be considered only in a later, separately reviewed augmentation package.

## 6. Determinism and seeded profiles

Stage 4 never uses mutable global random state.

`sample_degradation_config(seed, profile, raster_width=...)` derives all V1 random-looking parameters deterministically from a cryptographic digest of the degradation version, profile name, and explicit seed.

Supported V1 profiles:

- `clean` — rasterization only;
- `light` — small bounded rotation/blur/noise/photometric/compression effects;
- `medium` — wider but still conservative bounds.

There is intentionally no `heavy` V1 profile.

For identical SVG bytes, exact configuration, Stage 4 version, pinned direct Python dependencies, Cairo runtime, Python/runtime platform, and relevant image libraries/resources, repeated processing is expected to produce byte-identical PNG output in the verified environment.

A configuration fingerprint is SHA-256 over a canonical JSON representation containing the Stage 4 version, direct dependency pins, and exact configuration.

## 7. Provenance and family identity

Every derivative preserves:

```text
family_id
page_number
source MusicXML SHA-256
renderer configuration fingerprint
source SVG SHA-256
clean raster SHA-256
degradation configuration fingerprint
exact DegradationConfig
final PNG SHA-256
Stage 4 version
CairoSVG version
Pillow version
Cairo runtime version
Python version
platform system / machine
```

A deterministic `derivative_id` is derived from canonical lineage data and the final PNG hash.

The symbolic score remains the source of musical truth. Degraded images never become independent symbolic targets.

All derivatives of one symbolic source must retain the same `family_id`. Stage 5 must split datasets by family, not by individual image, so clean and degraded forms of one score cannot leak across train/validation/test partitions.

## 8. Content-preservation rule

Stage 4 may alter appearance only. It must never knowingly delete or invent score semantics.

Current V1 safeguards include:

- no crop operation;
- expanded rotation canvas;
- white fill outside rotated content;
- bounded transformation strengths;
- blank/dark-output rejection;
- geometric ink-retention rejection;
- source/clean/final hashes and exact replay parameters.

These safeguards are necessary but do not themselves prove semantic equivalence from pixels. Stage 5 Dataset Validation remains a separate gate and may impose stronger acceptance rules before a derivative becomes training-eligible.

## 9. Network and external-resource policy

Stage 4 must not download fonts, images, stylesheets, score content, or other resources during rasterization/degradation.

Only the already self-contained Stage 3 SVG may enter the rasterizer. External SVG resource surfaces are rejected before CairoSVG is invoked.

Large raster datasets are not normal Git repository content.

## 10. Failure policy

Stage 4 fails closed when:

- lineage is invalid or hashes do not match;
- the SVG safety boundary fails;
- the exact pinned direct Python runtime dependencies are unavailable or drifted;
- rasterization fails;
- image dimensions/resource limits fail;
- the output is effectively blank or implausibly dark;
- geometric ink retention is outside its bounded range;
- PNG/JPEG encode/decode steps fail.

No failed derivative is silently repaired, substituted, or accepted.

## 11. Verification gate

Stage 4 may close only after all applicable evidence passes:

1. focused configuration, lineage, safety, determinism, and negative tests;
2. real Stage 3 Verovio SVG → Stage 4 clean-raster integration tests;
3. real generated-score → MusicXML → Verovio → degradation integration tests;
4. deterministic repeated-output checks;
5. full repository regression suite;
6. Python compile validation;
7. GitHub-hosted pull-request CI on the exact final PR head;
8. separate merge approval;
9. post-merge GitHub-hosted CI on the exact resulting `main` commit.

Stage 5 remains locked until this gate is complete.

## 12. Explicitly out of scope

Stage 4 does not add:

- dataset manifests, train/validation/test splitting, or dataset storage;
- model training or model artifacts;
- real/user score ingestion;
- teacher-correction learning;
- ScoreMosaic production integration;
- Guitar TAB work;
- automatic model deployment.
