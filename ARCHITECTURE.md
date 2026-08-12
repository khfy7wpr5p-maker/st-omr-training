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

Stage 4 derives its complete public degradation configuration from explicit integer parameters and, for sampled profiles, an explicit seed without mutable global RNG state. The clean raster, exact degradation configuration, source hashes, dependency/runtime provenance, and final PNG hash are recorded for replay and audit.

Cross-platform SVG or raster byte identity is not assumed automatically. A different operating system, architecture, renderer resource bundle, Cairo runtime, or relevant image runtime must be separately verified before artifacts from different environments are mixed.

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

The renderer adapter validates MusicXML before invoking Verovio, uses explicit MusicXML input mode, limits page count, records runtime/configuration provenance, and rejects unsafe SVG surfaces such as scripts or external references. Rasterization remains outside the renderer so the clean vector render stays independently auditable.

## Controlled degradation boundary

Stage 4 is governed by [DEGRADATION_CONTRACT.md](DEGRADATION_CONTRACT.md).

The bounded V1 path is:

```text
Validated self-contained Stage 3 SVG page
        + lineage / family_id
        ↓
Independent Stage 4 SVG/hash/resource preflight
        ↓
Pinned CairoSVG 2.8.2 clean rasterization
        ↓
Canonical grayscale PNG + clean hash
        ↓
Bounded deterministic transform pipeline
        ↓
Final grayscale PNG
        ↓
Exact replay config + source/clean/final hashes + runtime provenance
```

Stage 4 is separate from both the renderer and the dataset builder. It revalidates Stage 3 SVG/hash lineage, uses a bounded raster width/pixel budget, and performs only conservative V1 transformations: expanded-canvas rotation, Gaussian blur, brightness, contrast, deterministic noise, and optional JPEG round-trip compression. The final artifact remains PNG.

Arbitrary crop, perspective warp, shear, elastic deformation, synthetic occlusion/shadow, staff deletion, symbol deletion, and content-aware erasing are deliberately deferred. Those transforms require stronger score-region/content-preservation evidence before they can become training-data operations.

The original canonical symbolic score remains the musical target. Stage 4 creates derived appearance artifacts only and preserves `family_id` across all derivatives.

## Verification boundary

GitHub Actions CI is active for the public repository. The baseline uses GitHub-hosted Ubuntu with Python 3.13, pinned runtime dependencies, complete unittest discovery including real-runtime integration tests, and Python compile validation.

The exact pre-Stage-4 `main` commit `23739ddfab618a0406836e94bb0ced1a124f8886` passed post-merge CI run `31648164533`. The integrated implementation through Stage 3 is therefore CI verified.

Stage 4 is not complete merely because its implementation exists on a feature branch. It must pass its focused and real-pipeline tests, full regression, exact final PR-head GitHub-hosted CI, separate merge approval, and post-merge CI on the exact resulting `main` commit before Stage 5 may begin.

## Stage roadmap

```text
Stage 0   Safety and architecture baseline              ✅
Stage 1   ST Music Generator                            ✅
Stage 2-A MusicXML contract freeze                      ✅
Stage 2-B Deterministic MusicXML 4.0 writer            ✅
Stage 2-C Offline XSD + independent validator          ✅
Stage 2-D Supported-V1 semantic round-trip verifier    ✅
Stage 3   Renderer integration                          ✅
Stage 4   Controlled degradation                        🔄 active package
Stage 5   Dataset validation                            🔒
Stage 6   Synthetic Dataset v1                          🔒
Stage 7   Baseline ST-OMR training                      🔒
Stage 8   Real-data fine-tuning                         🔒
Stage 9   Benchmark and candidate decision              🔒
Stage 10  ScoreMosaic candidate integration             🔒
```
