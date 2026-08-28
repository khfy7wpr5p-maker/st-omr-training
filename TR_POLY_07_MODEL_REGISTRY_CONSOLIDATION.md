# TR-POLY-07 — Model Registry Consolidation

## Purpose

TR-POLY-07 creates one deterministic registry for the research model families already present or explicitly planned in ST-OMR Training.

The registry is **descriptive, not authoritative**. A row does not prove that a checkpoint exists, does not prove performance, and never grants production authority.

This package changes no trainer, no dataset, no checkpoint, no runtime resolver, no ScoreMosaic integration, and does not open the sealed TEST split.

## Why this is required before a 2D Transformer

The repository previously exposed model identity in several independent modules:

- CNN-GRU sequence baseline;
- Staff and Structure specialists;
- Barline and Meter refiners;
- declarative specialist tasks;
- deterministic context fusion;
- future polyphonic candidate families.

Without a single registry, a later experiment could accidentally compare:

- different tokenizer versions;
- different representation versions;
- different datasets;
- different training profiles;
- different runtime environments;
- or an unbound checkpoint.

TR-POLY-07 makes those distinctions explicit before TR-POLY-08 creates a new model family.

## Registry version

`st-omr-model-registry-v1`

Model-card schema:

`st-omr-model-card-v1`

Exact checkpoint binding schema:

`st-omr-model-artifact-binding-v1`

## Lifecycle vocabulary

| Lifecycle | Meaning |
| --- | --- |
| `frozen_reference` | Existing reference baseline; not a new candidate winner |
| `training_implemented` | Training code exists, but each evidence claim still requires an exact checkpoint binding |
| `architecture_only` | Task/architecture is declared but this registry does not claim an implemented trained model |
| `planned` | Future candidate family only; no inference authority |
| `deterministic` | Non-learned fusion/validation component; no checkpoint |

## Seed registry

### Existing learned implementations

- `baseline.cnn-gru.v1`
  - source: `st-omr-cnn-gru-baseline-v1`
  - frozen reference only
  - V1 semantic tokenizer bound
  - not Polyphonic V2 capable

- `specialist.staff.d7.v1`
  - source: `stage7d7-staff-dense-segmentation-v1`
  - experimental visual evidence

- `specialist.structure.d7.v1`
  - source: `stage7d7-structure-dense-segmentation-v1`
  - experimental visual evidence

- `refiner.barline.d11.v1`
  - source: `stage7d11-barline-refiner-v1`
  - shadow-only local evidence

- `refiner.meter.d11.v1`
  - source: `stage7d11-meter-refiner-v1`
  - shadow-only local evidence

### Declared specialist tasks without a model claim

The Stage 7-D4 architecture defines NoteHead, Rest, Accidental, Rhythm, Staff Position and Chord Grouping tasks. TR-POLY-07 registers those task identities as `architecture_only` rather than pretending that a generic trained checkpoint is already admitted.

This distinction is intentional. Historical experiments or specialist-specific work can later add a new exact record only when their model/version/checkpoint evidence is explicit.

### Deterministic fusion

`fusion.context-validator.v1` is registered as deterministic, non-learned fusion. It cannot carry checkpoint evidence.

### Planned polyphonic candidates

- `candidate.poly-2d-transformer.v1`
- `candidate.relation-graph.v1`

They are `planned`, have no inference authority, and are bound to the frozen Polyphonic Representation V2 target surface. Registry presence is not implementation evidence.

## Exact artifact binding

Any learned model result used as comparable evidence must identify:

- registry record ID;
- exact repository SHA-40;
- checkpoint SHA-256;
- model fingerprint SHA-256;
- training-profile SHA-256;
- dataset-manifest SHA-256;
- runtime fingerprint SHA-256;
- tokenizer version/fingerprint when the model is tokenizer-bound;
- representation version when applicable.

An architecture-only, planned or deterministic registry record cannot accept a learned checkpoint binding.

## Evaluation evidence binding

Comparable benchmark evidence additionally binds:

- artifact-binding SHA-256;
- benchmark-identity SHA-256;
- metrics SHA-256;
- `st-omr-poly-evaluation-contract-v1`.

This prevents a score from being detached from the exact model artifact or benchmark surface that produced it.

## Model cards

Every registry row has a deterministic model-card payload and hash.

Each model card explicitly states:

- production authority is false;
- registry presence is not performance evidence;
- registry presence is not promotion authority;
- learned evidence requires an exact checkpoint where applicable.

## Safety boundary

TR-POLY-07 does **not**:

- train a model;
- create or load a checkpoint;
- open TEST;
- modify V1 tokenizer/core;
- modify Polyphonic Representation V2;
- modify the V2 tokenizer;
- change Meter/Rest specialist decisions;
- change deterministic resolver behavior;
- wire any candidate into ScoreMosaic;
- select a winning architecture.

## Next gate

TR-POLY-08 may implement a **tiny 2D Transformer prototype** only after this registry package is exact-head CI green and review-clean.

The TR-POLY-08 prototype must replace the planned registry identity with an explicit implemented model version and must bind the frozen V2 representation/tokenizer. It remains a research candidate until evaluated under the common benchmark contract.
