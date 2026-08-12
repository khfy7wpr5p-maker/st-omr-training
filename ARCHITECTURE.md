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
Supported-V1 Semantic Round Trip
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
3. The generator first creates a canonical internal music model. MusicXML is a deterministic serialization target, not the primary source of musical truth.
4. The musical validator independently re-checks generated content rather than trusting generator claims.
5. MusicXML must pass offline XSD validation, independent ST semantic validation, and the supported-V1 semantic round-trip gate before rendering.
6. Rendering is accessed only through a dedicated adapter so the music generator and symbolic layers do not depend on a renderer API.
7. Controlled degradation may change visual appearance but must not change symbolic musical ground truth.
8. Training datasets and large model artifacts are not normal Git repository content.
9. Real data, if introduced later, must remain distinct from synthetic data and pass separate rights, provenance, privacy where relevant, and quality gates.
10. ST-OMR candidates never enter ScoreMosaic automatically. Integration is a later, independent decision after held-out evaluation and regression evidence.

## Canonicalization rule

Canonicalization may normalize serialization differences only. It must not erase notation semantics that affect the rendered score.

Examples that must remain distinguishable when the notation differs include enharmonic spelling and accidental display intent. Pitch semantics and notation semantics are both preserved.

## V1 symbolic scope

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

For a fixed generator version, configuration, and seed, the canonical model output must be reproducible and yield the same canonical-model identity.

MusicXML is a derived deterministic serialization. For the same canonical `Score` and writer version, Stage 2 requires stable MusicXML bytes and a stable MusicXML SHA-256 digest under the supported runtime contract.

Stage 3 pins Verovio 6.2.1, renderer configuration, Leipzig font selection, adapter version, and deterministic XML-ID behavior. In the verified Linux/Python environment, repeated rendering of identical supported MusicXML produced byte-identical SVG output and identical page hashes.

Cross-platform SVG byte identity is not assumed automatically. A different operating system, architecture, renderer resource bundle, or runtime must be separately verified before artifacts from different environments are mixed.

## MusicXML serialization boundary

Stage 2 is governed by [MUSICXML_CONTRACT.md](MUSICXML_CONTRACT.md).

```text
Validated Canonical Score
        ↓
Stage 2-B Deterministic MusicXML 4.0 Writer
        ↓
MusicXML 4.0 score-partwise
        ↓
Stage 2-C Offline XSD Validation
        +
Independent ST MusicXML Semantic Validation
        ↓
Stage 2-D Supported-V1 Semantic Round Trip
        ↓
Stage 3 Renderer Gate
```

Key rules:

1. MusicXML 4.0 `score-partwise` is pinned for V1. Later MusicXML versions require a separate compatibility decision.
2. The writer uses an XML tree API rather than raw XML string concatenation.
3. Duration conversion uses exact rational arithmetic; floating point is prohibited for symbolic musical time.
4. One score-wide divisions value is computed deterministically from canonical durations.
5. MusicXML schema validation is offline and based on an exact pinned official W3C MusicXML 4.0 XSD asset set.
6. Schema validation and ST semantic validation are independent of the writer and fail closed.
7. The supported-V1 round-trip verifier rejects unsupported constructs rather than silently normalizing them.

## Renderer boundary

Stage 3 is governed by [RENDERER_CONTRACT.md](RENDERER_CONTRACT.md).

The V1 renderer path is:

```text
Stage-2-valid MusicXML bytes
        ↓
Verovio Adapter
        ↓
Pinned Verovio 6.2.1 runtime
        ↓
Validated self-contained SVG page(s)
        ↓
Per-page SHA-256 + renderer provenance
```

The renderer adapter validates MusicXML before invoking Verovio, uses explicit MusicXML input mode, limits page count, records runtime/configuration provenance, and rejects unsafe SVG surfaces such as scripts or external references. Rasterization is intentionally deferred to Stage 4 so the clean vector render remains independently auditable.

## Controlled degradation boundary

Stage 4 is the next development stage. It may transform clean rendered output into visually degraded derivatives for training, but it must preserve the symbolic target and source-family identity.

Stage 4 must remain separate from the renderer and dataset builder. It must define deterministic/replayable augmentation parameters, provenance for every derivative, explicit safety bounds, and rejection rules for transformations that remove or destroy required score content.

No Stage 4 implementation is part of the completed Stage 3 architecture.

## Verification boundary

GitHub Actions CI is active for the public repository. The current baseline uses GitHub-hosted Ubuntu with Python 3.13, pinned runtime dependencies, the complete unittest suite including real Verovio runtime tests, and Python compile validation.

The exact `main` commit `5abbc9859a4a69bf9a17936bc41e722256f87472` passed the post-merge CI run after PR #12. The current integrated implementation through Stage 3 is therefore CI verified.

Future implementation packages must still pass their own focused tests, full regression, relevant real-runtime/integration evidence, pull-request CI, and post-merge main CI before a stage is closed.

## Stage roadmap

```text
Stage 0   Safety and architecture baseline              ✅
Stage 1   ST Music Generator                            ✅
Stage 2-A MusicXML contract freeze                      ✅
Stage 2-B Deterministic MusicXML 4.0 writer            ✅
Stage 2-C Offline XSD + independent validator          ✅
Stage 2-D Supported-V1 semantic round-trip verifier    ✅
Stage 3   Renderer integration                          ✅
Stage 4   Controlled degradation                        ⏭ next
Stage 5   Dataset validation                            🔒
Stage 6   Synthetic Dataset v1                          🔒
Stage 7   Baseline ST-OMR training                      🔒
Stage 8   Real-data fine-tuning                         🔒
Stage 9   Benchmark and candidate decision              🔒
Stage 10  ScoreMosaic candidate integration             🔒
```
