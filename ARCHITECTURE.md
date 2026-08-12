# ST-OMR Training Lab Architecture

## Purpose

This repository is the isolated training and synthetic-data laboratory for ST-OMR.
Its job is to create traceable OMR training data, train candidate ST-OMR models, and evaluate them before any later ScoreMosaic integration.

It is not the ScoreMosaic production runtime, a user-file store, a Guitar TAB training project, or an automatic deployment system.

## Core pipeline

```text
Generator Config + Seed
        ↓
ST Music Generator
        ↓
Canonical ST Music Model
        ↓
Independent Musical Validator
        ↓
MusicXML Writer
        ↓
MusicXML Validator
        ↓
Renderer Adapter
        ↓
Controlled Degradation
        ↓
Dataset Builder
        ↓
Dataset Validator
        ↓
Synthetic Dataset
        ↓
ST-OMR Training
        ↓
Evaluation / Error Analysis
        ↓
ST-OMR Candidate
        ↓
Separate integration gate
        ↓
ScoreMosaic
```

## Architectural boundaries

1. The generator must not emit unvalidated training targets directly.
2. Musical generation, validation, MusicXML serialization, rendering, augmentation, dataset construction, training, and evaluation remain separate layers.
3. The generator must first create a canonical internal music model. MusicXML is a deterministic serialization target, not the primary source of musical truth.
4. The musical validator must independently re-check generated content rather than trusting generator claims.
5. Rendering must be accessed through an adapter so the music generator does not depend on a renderer API.
6. Controlled degradation may change image appearance but must not change the symbolic musical ground truth.
7. Training datasets and large model artifacts are not normal Git repository content.
8. Real data, if introduced later, must remain distinct from synthetic data and pass separate rights, provenance, and quality gates.
9. ST-OMR candidates never enter ScoreMosaic automatically. Integration is a later, independent decision after held-out evaluation and regression evidence.

## Canonicalization rule

Canonicalization may normalize serialization differences only. It must not erase notation semantics that affect the rendered score.

Examples that must remain distinguishable when the notation differs include enharmonic spelling and accidental display intent. Pitch semantics and notation semantics are both preserved.

## V1 scope

The initial generator is intentionally small.

Supported in V1:

- one part
- one staff
- treble clef
- one voice
- key signature fixed to 0
- 2/4, 3/4, and 4/4
- whole, half, quarter, and eighth notes where metrically valid
- eighth, quarter, and half rests
- single notes
- 2-note, 3-note, and 4-note chords
- controlled sharp, flat, and natural notation intent

Explicitly deferred:

- full-measure rests
- multiple voices
- beams
- ties and slurs
- tuplets
- piano grand staff
- cross-staff notation
- Guitar TAB

## Determinism boundary

For a fixed generator version, configuration, and seed, the canonical model output must be reproducible and yield the same canonical-model hash.

Rendered binary identity is not assumed until renderer version, fonts, rendering configuration, and environment are also pinned and verified.

## Stage roadmap

```text
Stage 0  Safety and architecture baseline
Stage 1  ST Music Generator
Stage 2  Canonical / MusicXML validation
Stage 3  Renderer integration
Stage 4  Controlled degradation
Stage 5  Dataset validation
Stage 6  Synthetic Dataset v1
Stage 7  Baseline ST-OMR training
Stage 8  Real-data fine-tuning
Stage 9  Benchmark and candidate decision
Stage 10 ScoreMosaic candidate integration
```
