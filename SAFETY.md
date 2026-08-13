# ST-OMR Training Safety Baseline

## Purpose

This repository develops training data and candidate models for ST-OMR. Safety means preserving data provenance, musical correctness, reproducibility, evaluation integrity, and separation from production systems.

## Repository data boundary

The Git repository is for source code, configuration, small test fixtures, documentation, and lightweight metadata.

Do not commit:

- real user PDFs or photographs
- personal or private documents
- copyrighted or rights-unclear score collections
- unverified MusicXML scraped from the internet
- API keys, tokens, credentials, or secrets
- large generated image datasets
- large model checkpoints or caches
- temporary renderer/degradation output
- binary dependency wheels or other large third-party runtime bundles

Large datasets and model artifacts must use separate approved storage when those stages are introduced.

## Synthetic versus real data

Synthetic and real data must remain explicitly distinguishable.

Synthetic data may enter training only after generator, musical, MusicXML, renderer/degradation provenance, and dataset validation gates pass.

Real data, if introduced later, requires a separate gate covering rights or permission, provenance, quality, privacy where relevant, and approved training use.

Teacher corrections or user submissions must never become training data automatically.

## Ground-truth rule

A generated target is not trusted merely because the generator created it.

Required conceptual sequence:

```text
Generator
   ↓
Canonical ST Music Model
   ↓
Independent Musical Validator
   ↓
MusicXML Writer
   ↓
MusicXML Validator
   ↓
Supported-V1 Semantic Round Trip
   ↓
Eligible for rendering
   ↓
Validated self-contained SVG
   ↓
Eligible for controlled degradation
```

A failure at any validation stage rejects the sample.

Rendering and visual degradation create derived artifacts. They never replace or silently alter the canonical symbolic ground truth.

## Independent validation

Validators must independently recompute invariants instead of trusting flags or precomputed claims from the generator or another stage.

Examples include:

- measure-duration consistency
- valid time signatures
- valid pitch structure
- valid rational durations
- chord membership and shared onset rules
- voice and staff constraints
- duplicate-note rejection where prohibited
- V1 scope enforcement
- MusicXML schema and semantic validity
- safe renderer output structure
- Stage 4 source hash, SVG safety, resource, parameter, and output-integrity checks

## Negative testing

The project must prove that invalid input is rejected, not only that valid input is accepted.

Negative tests should include, as applicable:

- invalid pitch step
- invalid octave type or bounds
- negative duration
- zero duration where forbidden
- non-finite numeric values where numeric inputs exist
- measure overflow
- measure underflow when a complete measure is required
- empty chord
- duplicate chord pitch
- invalid staff
- invalid voice
- unsupported notation outside the active stage contract
- malformed or unsafe XML/SVG surfaces
- source/hash lineage mismatch
- out-of-range degradation parameters
- effectively blank/corrupt raster output
- transformations that fail content-retention/resource gates

## Determinism

For a fixed generator version, generator configuration, and seed, canonical symbolic output must be reproducible.

MusicXML determinism is independently verified at the serialization boundary.

Stage 3 renderer determinism is scoped to the same pinned Verovio package/runtime, renderer configuration, platform/resource bundle, adapter version, and input bytes. The verified environment produced byte-identical SVG output for deterministic repeats, but cross-platform binary identity is not assumed without separate evidence.

Stage 4 uses explicit replay parameters. Seeded V1 profiles derive parameters without mutable global RNG state. Every derivative records the exact configuration and source/clean/final hashes. Byte identity is scoped to the same Stage 4 version, source SVG, direct dependency versions, Cairo runtime, Python/runtime platform, and configuration; cross-platform equality must be separately demonstrated.

## Stage 4 augmentation safety

Stage 4 may alter visual quality but must not change symbolic ground truth.

The frozen V1 operation set is intentionally conservative:

- clean SVG-to-grayscale-PNG rasterization;
- rotation limited to ±3 degrees with `expand=true` and white fill, so rotation does not intentionally crop the page;
- Gaussian blur limited to 2 pixels;
- deterministic grayscale noise limited to level 20;
- brightness limited to 0.80..1.20;
- contrast limited to 0.75..1.25;
- optional JPEG round-trip limited to quality 65..95, with final output returned as PNG;
- raster width limited to 512..2400 and output capped at 16,000,000 pixels.

Every derivative retains lineage to the self-contained Stage 3 SVG, clean raster, original MusicXML, renderer configuration, and symbolic `family_id`.

Stage 4 rejects unsafe/external SVG resources before CairoSVG is invoked, validates source hashes, rejects effectively blank or implausibly dark images, and applies an ink-retention gate after rotation.

V1 deliberately does **not** permit arbitrary cropping, perspective warp, shear, elastic deformation, synthetic shadows/occlusion, staff-line deletion, symbol deletion, or content-aware erasing. These operations can remove or relocate musical content and require stronger score-region/content-preservation validation before they can be considered.

Stage 5 Dataset Validation remains an independent acceptance gate. Passing Stage 4 does not make a derivative automatically eligible for training.

## Dataset leakage prevention

All derivatives of one symbolic source belong to one source family.

Train, validation, and test splits must be performed at family level.

This is forbidden:

```text
train: score-A-clean
test:  score-A-blur
```

All score-A derivatives must remain in one split.

## Evaluation integrity

The held-out test or golden set must not be silently recycled into training after model errors are observed.

Training data may grow. Validation data may evolve under explicit versioning. Held-out evaluation data must remain controlled and versioned so benchmark comparisons stay meaningful.

## Verification gate

GitHub Actions CI is active for this public repository. Local evidence and GitHub-hosted evidence remain distinct and must be reported accurately.

Each implementation package must pass the applicable sequence:

```text
Focused tests
    ↓
Domain invariant tests
    ↓
Negative / fail-closed tests
    ↓
Determinism tests
    ↓
Golden / real-runtime tests where applicable
    ↓
Full local regression where a complete local runtime is available
    ↓
Generated-artifact checks
    ↓
Diff / scope review
    ↓
LOCAL VERIFIED when applicable
    ↓
Pull-request GitHub-hosted CI
    ↓
CI VERIFIED — PR HEAD
    ↓
Merge approval
    ↓
Post-merge GitHub-hosted CI on exact main
```

Status vocabulary:

- `UNVERIFIED` — required evidence has not passed
- `LOCAL VERIFIED` — all applicable local gates passed
- `CI VERIFIED — PR HEAD` — GitHub-hosted CI passed on the exact pull-request head
- `CI VERIFIED` — applicable GitHub-hosted CI passed on the exact integrated `main` state being reported

A successful CI run must never be inferred from a different commit.

## Current CI baseline

The merged baseline workflow uses GitHub-hosted Ubuntu, Python 3.13, pinned runtime dependencies, full unittest discovery including real-runtime tests, and Python compile validation.

The exact pre-Stage-4 `main` commit `23739ddfab618a0406836e94bb0ced1a124f8886` passed GitHub Actions run `31648164533`. The integrated state through Stage 3 is therefore CI verified.

Stage 4 adds exact direct pins for CairoSVG and Pillow to the branch CI runtime. The final Stage 4 PR head must pass the workflow with those pins and the real Verovio → raster/degradation integration tests before merge may be considered.

## Change-control rule

Development proceeds in small, reversible packages on non-main branches. Documentation, generator work, validation work, renderer integration, controlled degradation, dataset creation, model training, and ScoreMosaic integration must not be mixed into one uncontrolled change.

A later ST-OMR model candidate must pass held-out benchmark, real-score evaluation when available, chord/polyphony-focused evaluation, regression testing, and comparison against relevant existing OMR baselines before any ScoreMosaic integration decision.
