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
- temporary renderer output
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
- degradation parameter bounds and output integrity when Stage 4 is implemented

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
- out-of-range degradation parameters
- transformations that remove required musical content

## Determinism

For a fixed generator version, generator configuration, and seed, canonical symbolic output must be reproducible.

MusicXML determinism is independently verified at the serialization boundary.

Stage 3 renderer determinism is scoped to the same pinned Verovio package/runtime, renderer configuration, platform/resource bundle, adapter version, and input bytes. The verified environment produced byte-identical SVG output for deterministic repeats, but cross-platform binary identity is not assumed without separate evidence.

Stage 4 degradation must use explicit, replayable parameterization and seed/provenance recording. A derivative must be reproducible from its clean source artifact plus recorded degradation configuration where the chosen transform is defined as deterministic.

## Augmentation safety

Augmentation may alter visual quality but must not change symbolic ground truth.

Allowed categories may include controlled blur, skew, perspective, noise, shadows, compression, fading, rasterization, and similar document degradation only after the Stage 4 contract defines bounds and validation rules.

Every degraded derivative must retain lineage to its clean rendered source and canonical symbolic family.

An augmentation result must be rejected if it destroys or removes required musical content, for example by cropping away a measure or staff region needed by the target, producing an empty/corrupt image, or violating configured geometry/resource bounds.

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
Full local regression
    ↓
Generated-artifact checks
    ↓
Diff / scope review
    ↓
LOCAL VERIFIED
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

The merged baseline workflow uses GitHub-hosted Ubuntu, Python 3.13, pinned runtime dependencies, full unittest discovery including real Verovio runtime tests, and Python compile validation.

After PR #12 merged, the exact `main` commit `5abbc9859a4a69bf9a17936bc41e722256f87472` passed GitHub Actions run `31647615123`. The current integrated state through Stage 3 is therefore CI verified.

## Change-control rule

Development proceeds in small, reversible packages on non-main branches. Documentation, generator work, validation work, renderer integration, controlled degradation, dataset creation, model training, and ScoreMosaic integration must not be mixed into one uncontrolled change.

A later ST-OMR model candidate must pass held-out benchmark, real-score evaluation when available, chord/polyphony-focused evaluation, regression testing, and comparison against relevant existing OMR baselines before any ScoreMosaic integration decision.
