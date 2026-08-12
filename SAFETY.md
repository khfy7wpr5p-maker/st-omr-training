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

Large datasets and model artifacts must use separate approved storage when those stages are introduced.

## Synthetic versus real data

Synthetic and real data must remain explicitly distinguishable.

Synthetic data may enter training only after generator, musical, MusicXML, provenance, and dataset validation gates pass.

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
Eligible for rendering
```

A failure at any validation stage rejects the sample.

## Independent validation

The validator must independently recompute invariants instead of trusting flags or precomputed claims from the generator.

Examples include:

- measure-duration consistency
- valid time signatures
- valid pitch structure
- valid rational durations
- chord membership and shared onset rules
- voice and staff constraints
- duplicate-note rejection where prohibited
- V1 scope enforcement

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

## Determinism

For a fixed generator version, generator configuration, and seed, canonical symbolic output must be reproducible.

Required evidence later includes equality of the canonical representation and its hash across repeated generation.

Rendered binary identity is a separate concern and must not be claimed unless the rendering environment, renderer version, fonts, and settings are pinned and tested.

## Augmentation safety

Augmentation may alter visual quality but must not change symbolic ground truth.

Allowed categories may later include controlled blur, skew, perspective, noise, shadows, compression, fading, and similar document degradation.

An augmentation result must be rejected if it destroys or removes required musical content, for example by cropping away a measure or staff region needed by the target.

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

## Local verification gate

GitHub CI is currently unavailable for this project. Local evidence must never be described as CI evidence.

Each implementation package must eventually pass the applicable sequence:

```text
Focused tests
    ↓
Musical invariant tests
    ↓
Negative tests
    ↓
Determinism tests
    ↓
Golden/reference tests
    ↓
Full local regression
    ↓
Generated-artifact checks
    ↓
Git status / diff review
    ↓
LOCAL VERIFIED
```

Status vocabulary:

- `UNVERIFIED` — required evidence has not passed
- `LOCAL VERIFIED` — all applicable local gates passed
- `CI VERIFIED` — GitHub-hosted CI also passed

Until CI is available, reports must state `LOCAL VERIFIED — CI NOT AVAILABLE` when appropriate.

## Change-control rule

Development should proceed in small, reversible packages on non-main branches. Documentation, generator work, validation work, renderer integration, augmentation, dataset creation, model training, and ScoreMosaic integration should not be mixed into one uncontrolled change.

A later ST-OMR model candidate must pass held-out benchmark, real-score evaluation when available, chord/polyphony-focused evaluation, regression testing, and comparison against relevant existing OMR baselines before any ScoreMosaic integration decision.
