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
Independent Dataset Validator
        ↓
Synthetic Dataset
        ↓
Stage 7 target tokenizer / semantic round-trip gate
        ↓
Baseline ST-OMR Training
        ↓
Validation / Error Analysis
        ↓
Stage 9 sealed benchmark
        ↓
ST-OMR Candidate
        ↓
Separate integration gate
        ↓
ScoreMosaic
```

## Architectural boundaries

1. The generator must not emit unvalidated training targets directly.
2. Musical generation, validation, MusicXML serialization, rendering, augmentation, dataset construction, dataset validation, training, and evaluation remain separate layers.
3. The generator first creates a canonical internal music model. MusicXML is a deterministic serialization target, not the primary source of musical truth.
4. The musical validator independently re-checks generated content rather than trusting generator claims.
5. MusicXML must pass offline XSD validation, independent ST semantic validation, and the supported-V1 semantic round-trip gate before rendering.
6. Rendering is accessed only through a dedicated adapter so the music generator and symbolic layers do not depend on a renderer API.
7. Controlled degradation may change visual appearance but must not change symbolic musical ground truth.
8. Dataset construction may propose samples and split assignments, but the independent dataset validator has veto authority over lineage, duplicates, leakage, and manifest integrity.
9. All derivatives of one symbolic family must remain in one train/validation/test split. Exact target or clean-render aliases may not be hidden behind different family IDs to bypass this rule.
10. Stage 7 may train only from Stage 5/6 validated synthetic artifacts and must re-check persisted hashes before use.
11. Stage 7 may update parameters only from the `train` split and may select checkpoints only from `validation`; the Stage 6 `test` split remains sealed until Stage 9.
12. Training datasets and large model artifacts are not normal Git repository content.
13. Real data, if introduced later, must remain distinct from synthetic data and pass separate rights, provenance, privacy where relevant, and quality gates.
14. ST-OMR candidates never enter ScoreMosaic automatically. Integration is a later, independent decision after sealed held-out evaluation and regression evidence.

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

Stage 5 canonical manifest serialization sorts samples by stable identity fields and uses canonical JSON. The same valid logical manifest therefore produces identical bytes and manifest SHA-256 regardless of input tuple order. Split assignment is semantic and remains part of the manifest hash.

Stage 6 adds a deterministic family plan, split seed, fixed 80/10/10 family-level allocation policy, symbolic coverage-profile cycle, split-independent degradation-seed derivation, build-configuration fingerprint, and build identity. The same Stage 6 configuration must reproduce the same family plan and artifact identities within the same verified runtime boundary.

Stage 7 adds a deterministic semantic token target, deterministic data ordering for a fixed run configuration, explicit model/training configuration fingerprints, explicit RNG seeds, and checkpoint/metrics SHA-256 provenance. Accelerator bit identity is not assumed automatically; any device/runtime reproducibility claim must be demonstrated for that exact environment, while deterministic CPU smoke evidence remains a separate verification surface.

Cross-platform SVG or raster byte identity is not assumed automatically. A different operating system, architecture, renderer resource bundle, Cairo runtime, image runtime, training framework, or accelerator stack must be separately verified before determinism claims are generalized.

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

Arbitrary crop, perspective warp, shear, elastic deformation, synthetic occlusion/shadow, staff deletion, symbol deletion, and content-aware erasing remain deferred. Those transforms require stronger score-region/content-preservation evidence before they can become training-data operations.

The original canonical symbolic score remains the musical target. Stage 4 creates derived appearance artifacts only and preserves `family_id` across all derivatives.

## Dataset validation boundary

Stage 5 is governed by [DATASET_CONTRACT.md](DATASET_CONTRACT.md). Its implemented V1 package is Stage 5-A — Dataset Contract + Independent Manifest Validator.

```text
Stage 4 DegradedPage + PNG bytes
        ↓
Independent Stage 5 PNG/hash/replay-lineage bridge
        ↓
Immutable DatasetSample
        ↓
family-level split assignment
        ↓
Immutable DatasetManifest
        ↓
Independent manifest validation
        ├── duplicate/collision veto
        ├── family split-leakage veto
        ├── identical MusicXML target alias veto
        ├── identical clean SVG alias veto
        └── lineage identity recomputation
        ↓
Canonical manifest JSON + manifest SHA-256
```

Stage 5 V1 is synthetic-only. It supports exactly `train`, `validation`, and `test` and requires every symbolic family to remain in one split. The validator does not trust `family_id` alone: identical MusicXML targets or identical clean SVGs cannot be hidden behind multiple families or allowed to cross splits.

The validator independently mirrors the frozen Stage 4 replay fields and recomputes the Stage 4 degradation-config fingerprint and derivative identity. It also computes a Stage 5 sample identity that is independent of split assignment, preventing a split move from manufacturing a new sample.

The narrow Stage 4 → Stage 5 bridge verifies actual PNG signature/IHDR/CRC/hash/dimensions before metadata is accepted. Stage 5 deliberately does not write bulk artifact files, choose final split ratios, define cloud/filesystem storage, or start training.

## Stage 6 construction boundary

Stage 6 — Synthetic Dataset v1 — is closed and governed by [DATASET_BUILD_CONTRACT.md](DATASET_BUILD_CONTRACT.md).

```text
SyntheticDatasetConfig
        ↓
deterministic family plan + 80/10/10 family split assignment
        ↓
Stage 1 generator profiles
        ↓
Stage 2 MusicXML target bytes
        ↓
Stage 3 rendered page(s)
        ↓
Stage 4 clean/light/medium derivatives
        ↓
Stage 5 DatasetSample bridge
        ↓
independent Stage 5 DatasetManifest validator
        ↓
SyntheticDatasetBuild
        ↓
optional hash-addressed local persistence
```

The builder uses the canonical generated `score_id` as `family_id`; all pages and derivatives of that family inherit one split. Split ranking is deterministic and occurs before rendering. Degradation seeds depend on family/page/profile and not on split assignment.

Stage 6 refuses identical MusicXML targets across distinct generated families, duplicate final PNG artifacts, artifact/hash mismatch, Stage 5 vetoes, and overwrite of an existing dataset directory. Valid local persistence uses `manifest.json`, `build.json`, SHA-256-named MusicXML targets, and SHA-256-named PNG images. Bulk artifacts remain outside normal Git content.

Stage 6 does not add training logic, real/user data ingestion, teacher-correction learning, cloud credentials/storage providers, Guitar TAB training, or ScoreMosaic integration.

## Stage 7 baseline-training boundary

Stage 7 is governed by [TRAINING_CONTRACT.md](TRAINING_CONTRACT.md). The concrete closed Stage 7-B implementation profile is recorded in [TRAINING_IMPLEMENTATION.md](TRAINING_IMPLEMENTATION.md).

Stage 7 is decomposed so the contract, implementation, and actual baseline run cannot be silently combined:

```text
Stage 7-A Training Contract Freeze
        ↓
Stage 7-B Tokenizer + Data + Model + Trainer smoke implementation
        ↓
Stage 7-C Bounded baseline training run + evidence
        ↓
Stage 9 sealed-test benchmark gate
```

Stage 7-A and Stage 7-B are closed and exact-main CI verified. Stage 7-C is the next package but has **not started**.

### Training input

Stage 7 accepts only exact Stage 6 persisted PNG/MusicXML artifacts whose manifest and hashes pass the existing Stage 5/6 gates. Loose files and bypass inputs are prohibited.

The implemented Stage 7-B adapter revalidates the persisted Stage 6 manifest/build metadata, target/image hashes, PNG structure/dimensions, and semantic target round trip before train/validation admission. The Stage 6 `test` split is rejected at the Stage 7-B data and batch boundaries.

The input tensor is the verified Stage 4 final grayscale PNG. Stage 7-B freezes deterministic 1×64×512 grayscale fit/no-upscale/no-crop/center-white-pad preprocessing. No hidden/random Stage 7 augmentation is used; appearance variability belongs to Stage 4.

### Training target

The baseline does not learn raw XML text as the primary target. Stage 7-B implements the finite deterministic 35-token `ST-OMR V1 token sequence` derived from the independently parsed supported-V1 projection. The token vocabulary covers measure boundaries, meter, note/rest/chord type, V1 duration, chord size/member order, step, alter, octave, and display-accidental intent.

Tokenizer output independently detokenizes back to an exactly equal supported-V1 semantic projection before a sample can become training-eligible. Accepted semantic sequences must consume a real `EOS`; EOF without `EOS` and tokens after `EOS` fail closed.

### Model boundary

Stage 7-B selects exact `torch==2.13.0+cpu` and implements one from-scratch CNN visual encoder plus context-conditioned GRU decoder. It uses no ensemble, external pretrained weights, hidden OCR/OMR teacher labels, network-dependent inference, Audiveris/Scan2Notes output, LLM, or another recognition engine as part of training.

The V1 outer model ceiling remains 25,000,000 trainable parameters and is enforced at construction. Stage 7-B also freezes PAD-masked token cross-entropy, AdamW, no scheduler, gradient clipping, train-only parameter updates, validation-only loss evaluation, deterministic CPU RNG/thread policy, and NaN/Infinity fail-closed checks.

### Split and evaluation boundary

Only the `train` split may update parameters. Only `validation` may select checkpoints and supply Stage 7 development metrics. The Stage 6 `test` split remains sealed until Stage 9 and cannot influence Stage 7 architecture, optimization, thresholds, or checkpoint choice.

Stage 7-C must record validation loss, token error rate, exact sequence accuracy, detokenization success, semantic-validity rate, and MusicXML regeneration validity. These metrics establish trainability and evidence only; Stage 9 owns production-candidate thresholds and sealed-test decisions.

### Reproducibility and artifact boundary

Every real Stage 7-C run must record repository SHA, dataset build identity, manifest SHA, configuration/tokenizer/model fingerprints, dependency/runtime/device identity, all seeds, parameter count, checkpoint SHA-256, and metrics SHA-256. Checkpoints are hash-addressed derived artifacts and remain outside normal Git content.

Stage 7-B demonstrated same-seed deterministic CPU smoke replay for the exact verified CPU runtime. GitHub-hosted CI remains limited to bounded smoke training. Full Stage 7-C training does not run in ordinary repository CI.

## Verification boundary

GitHub Actions CI is active for the public repository. The baseline uses GitHub-hosted Ubuntu with Python 3.13, pinned runtime dependencies, complete unittest discovery including real-runtime integration tests, and Python compile validation.

Stage 4 merged through PR #14 at exact `main` commit `f0fd8a732b51b4aa95a66c3a780d0cefa6661361`; post-merge GitHub Actions run `31660215130` passed on that exact commit.

Stage 5-A merged through PR #15 at exact `main` commit `d677f3d27ac710c56c5ce677a46dc62bcf77bd84`; post-merge GitHub Actions run `31671919885` passed on that exact commit. Stage 5 closure documentation merged through PR #16 at exact `main` commit `ed343d4984aae4507a2dd3238cfd1a98fb25b4b7`; post-merge run `31672540732` passed.

Stage 6 final PR head `cfd0ac780595a38e0fe041d2d70293b39f96fcf3` passed GitHub Actions run #17 (`31673631608`) with **309/309 tests**, pinned runtime verification, real Stage 1→6 integration/rebuild/persistence tests, `pip check`, and compile validation. PR #17 then merged at exact `main` commit `7c3c736e6d3755d1bd098e2874d73ce5ed41e39f`; post-merge run #18 (`31674014666`) passed on that exact commit. Stage 6 closure documentation merged through PR #18 at exact `main` commit `046c9a4e7e41e94b0b4465a2610f30361055a3ed`; post-merge run #20 (`31674836433`) also passed.

Stage 7-A PR merge candidate from source head `0eca1d881c494c763692aa72b1092d741c32be83` against exact base `046c9a4e7e41e94b0b4465a2610f30361055a3ed` passed GitHub Actions run #21 (`31675330236`) with **309/309 tests**, pinned runtime verification, `pip check`, and compile validation. PR #19 then merged at exact `main` commit `0f04b0182b6753cfb8816d1287adb5ee973e0c28`; post-merge GitHub Actions run #22 (`31675913632`) passed on that exact commit. Stage 7-A closure synchronization merged through PR #20 at exact `main` commit `6a13760d9d17130ea86636f4828ff1bff035f30d`; post-merge run #24 (`31676871798`) passed.

Stage 7-B final source head `a8ad8bc9f14953f0ed35ef5a5a8275be69af5ebd` against exact base `6a13760d9d17130ea86636f4828ff1bff035f30d` passed GitHub Actions run #30 (`31679413312`) on GitHub-generated PR merge candidate `3d5ee4e2ca479614e2a3322e0339ca21f25e5cae` with **336/336 tests**, pinned runtime verification including `torch==2.13.0+cpu`, `pip check`, missing-`EOS` regression coverage, deterministic CPU smoke evidence, and compile validation. PR #21 then merged at exact `main` commit `d02dce4ee17dfccf6f05519ab0970fdc188d0147`; post-merge GitHub Actions run #31 (`31679810478`) passed on that exact commit with **336/336 tests**.

The integrated repository through the Stage 7-B implementation boundary is therefore `CI VERIFIED`. Stage 7-C remains not started and requires its own bounded training/evidence package and explicit approval.

## Stage roadmap

```text
Stage 0   Safety and architecture baseline              ✅
Stage 1   ST Music Generator                            ✅
Stage 2-A MusicXML contract freeze                      ✅
Stage 2-B Deterministic MusicXML 4.0 writer            ✅
Stage 2-C Offline XSD + independent validator          ✅
Stage 2-D Supported-V1 semantic round-trip verifier    ✅
Stage 3   Renderer integration                          ✅
Stage 4   Controlled degradation                        ✅
Stage 5-A Dataset contract + manifest validator         ✅
Stage 5   Dataset validation                            ✅ CLOSED — CI VERIFIED
Stage 6   Synthetic Dataset v1                          ✅ CLOSED — CI VERIFIED
Stage 7-A Baseline training contract freeze             ✅ CLOSED — CI VERIFIED
Stage 7-B Tokenizer/data/model/trainer implementation   ✅ CLOSED — CI VERIFIED
Stage 7-C Bounded baseline training run + evidence      ⏭ NEXT — NOT STARTED
Stage 8   Real-data fine-tuning                         🔒
Stage 9   Benchmark and candidate decision              🔒
Stage 10  ScoreMosaic candidate integration             🔒
```
