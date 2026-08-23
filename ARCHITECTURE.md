# ST-OMR Training Lab Architecture

## Purpose

This repository is the isolated training and synthetic-data laboratory for ST-OMR.
Its job is to create traceable OMR training data, train candidate ST-OMR models, and evaluate them before any later ScoreMosaic integration.

It is not the ScoreMosaic production runtime, a user-file store, a Guitar TAB training project, or an automatic deployment system.

## Current-status authority

This long-form document preserves stable architecture boundaries and closed-stage history. Some stage-status labels below are historical snapshots and are not the current operational authority.

For current status, use the following order:

1. frozen machine-readable/stage contracts define invariants;
2. merged `main` defines executable repository authority;
3. `ARCHITECTURE_CURRENT.md` defines the current merged + shadow/experimental overlay;
4. open draft PR evidence is non-main and non-production until separately accepted/merged;
5. historical stage delta documents preserve their original scope and do not override the current overlay.

Current merged baseline at the 2026-08-23 synchronization is `a6a40b218a95c72349984ee2aee7262f467021fc`.

Important current overlay facts:

- NoteHead has PASS shadow evidence but is not production-wired.
- Rest R4 `half|quarter|eighth` value-specific specialists plus deterministic arbitration have PASS shadow evidence; Resolver connection remains closed.
- Accidental has PASS shadow evidence; deterministic association remains separate.
- Meter has a merged runtime foundation through real D11 Presence inference implementation, while later V4/V5 work remains draft/experimental unless separately merged.
- the V4-5 final holdout is consumed and failed; it cannot be reused for tuning.
- Meter V5-2B is the newest inspected bounded adaptation lane, with TRAIN-safe preflight/annotation gates and VAL/FINAL_HOLDOUT still closed.
- sealed TEST remains unavailable for development/tuning; shadow PASS is not production PASS.

The 2026-08-23 architecture incompatibility audit is recorded in `ARCHITECTURE_COMPATIBILITY_AUDIT_2026-08-23.md`.

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
Stage 8-0 real-data + fine-tuning contract gate
        ↓
Stage 8-1 quarantine/intake + byte-validation gate
        ↓
Stage 8-2 paired experiment profile freeze
        ↓
Stage 8-3A source→training-PNG preparation + admission
        ↓
Stage 8-3B paired real train/validation execution
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
13. Real data must remain distinct from synthetic data and pass the closed Stage 8-0 rights/provenance/permission/pairing/admission/leakage contract plus the closed Stage 8-1 exact-byte handoff before any later Stage 8 train/validation loader may trust it.
14. Stage 8-2 requires the primary Candidate A/B comparison to use the same admitted manifest, Stage 8-1 receipt set, tokenizer, preprocessing, model/trainer surface, deterministic data order, metric family, and resource budget; only initialization may differ.
15. ScoreMosaic uploads and teacher corrections are not automatic training data; no online/automatic learning path is permitted.
16. ST-OMR candidates never enter ScoreMosaic automatically. Integration is a later, independent decision after sealed held-out evaluation and regression evidence.

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

Stage 8-0 adds deterministic metadata identities for real-data admission. A real sample identity excludes split and review-state changes, while the admitted manifest canonicalizes exact evidence/status fields and split assignments.

Stage 8-1 adds a deterministic byte-validation policy fingerprint, a supported-V1 semantic fingerprint bound to the exact tokenizer fingerprint/token-id sequence, a canonical hash-only validation receipt, and deterministic dHash64 near-duplicate candidates. These identities establish reproducible validation evidence; they do not convert a heuristic perceptual match into proof of musical identity or a receipt into a digital signature.

Stage 8-2 adds a deterministic paired-profile fingerprint plus an order-independent receipt-set fingerprint and candidate bindings. The two primary candidate bindings must carry the same profile, development-manifest, receipt-set, and sealed-test commitment hashes. Candidate A additionally binds the exact accepted Stage 7-C checkpoint/model-state hashes; Candidate B binds no checkpoint.

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

Stage 7 is governed by [TRAINING_CONTRACT.md](TRAINING_CONTRACT.md). The closed Stage 7-B implementation profile is recorded in [TRAINING_IMPLEMENTATION.md](TRAINING_IMPLEMENTATION.md); the closed Stage 7-C execution profile and accepted evidence are recorded in [STAGE7C_RUNBOOK.md](STAGE7C_RUNBOOK.md) and [STAGE7C_EVIDENCE.md](STAGE7C_EVIDENCE.md).

Stage 7 is decomposed so the contract, implementation, and actual baseline run cannot be silently combined:

```text
Stage 7-A Training Contract Freeze
        ↓
Stage 7-B Tokenizer + Data + Model + Trainer smoke implementation
        ↓
Stage 7-C Bounded baseline training run + evidence
        ↓
Stage 8 real-data gates
        ↓
Stage 9 sealed-test benchmark gate
```

Stage 7-A, Stage 7-B, and Stage 7-C are closed and exact-main CI verified. Stage 7-C merged through PR #23 at exact `main` commit `2c2c478eb361fa90a3bccd819b623680eb12de0b`; post-merge run #68 (`31692849892`) passed **361/361 tests** on that SHA. Its accepted synthetic-only evidence is permanently summarized by hash without treating the model as a production candidate.

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

Stage 7-C reuses this exact model/trainer surface. It does not add a second architecture, external teacher, hidden augmentation path, or test-driven optimization path.

### Stage 7-C execution boundary

The first real baseline run is implemented by `st_omr_training/training_run.py` under the tighter execution policy in `STAGE7C_RUNBOOK.md`.

The run:

- uses only Stage 7-B-admitted train and validation samples;
- keeps fixed deterministic sample ordering;
- records deterministic untrained validation loss before optimization;
- selects exactly one checkpoint by minimum validation loss;
- requires the best trained validation loss to be strictly lower than the untrained baseline;
- performs bounded incremental greedy validation decoding constrained to the frozen supported-V1 grammar and exact eight-measure profile;
- reports token error rate, exact sequence accuracy, detokenization success, semantic validity, and MusicXML regeneration validity;
- requires at least one semantically valid validation prediction;
- writes hash-addressed checkpoint and canonical metrics/provenance artifacts;
- uses a fresh run directory with explicit `INCOMPLETE`/`COMPLETE` state and no silent resume.

### Split and evaluation boundary

Only the `train` split may update parameters. Only `validation` may select checkpoints and supply Stage 7 development metrics. The Stage 6 `test` split remains sealed until Stage 9 and cannot influence Stage 7 architecture, optimization, thresholds, or checkpoint choice.

Stage 7-C records validation loss, token error rate, exact sequence accuracy, detokenization success, semantic-validity rate, and MusicXML regeneration validity. These metrics establish trainability and evidence only; Stage 9 owns production-candidate thresholds and sealed-test decisions.

The decoding constraint is an inference-time state machine over the existing 35-token vocabulary. It enforces measure order, meter capacity, event/duration structure, pitch-field order, accidental coherence, chord-pitch uniqueness, the frozen eight-measure count, and terminal `EOS`. It masks invalid next-token logits but does not inspect validation targets, introduce a recognition teacher, change model parameters, or access the sealed test split.

### Reproducibility and artifact boundary

Every real Stage 7-C run records repository SHA, dataset build identity, manifest SHA, run/tokenizer/preprocess/model/trainer fingerprints, dependency/runtime/device identity, seeds, parameter count, epoch/step counts, checkpoint SHA-256, and metrics SHA-256. Checkpoints are hash-addressed derived artifacts and remain outside normal Git content.

Stage 7-B demonstrated same-seed deterministic CPU smoke replay for the exact verified CPU runtime. Stage 7-C then used one explicitly approved, exact-head, benchmark-gated GitHub-hosted execution. That exception is historical and has been removed from the active workflow; ordinary CI is again limited to bounded regression/runtime/compile validation.

## Stage 8 real-data and fine-tuning boundary

Stage 8-0 is governed by [STAGE8_REAL_DATA_CONTRACT.md](STAGE8_REAL_DATA_CONTRACT.md) and is closed/main-CI verified. Stage 8-1 is governed by [STAGE8_1_INTAKE_CONTRACT.md](STAGE8_1_INTAKE_CONTRACT.md) and is closed/main-CI verified. Stage 8-2 is governed by [STAGE8_2_RUN_PROFILE.md](STAGE8_2_RUN_PROFILE.md) and is the only active package.

```text
Stage 7-C accepted baseline evidence
        ↓
Stage 8-0 rights/provenance/pairing contract       CLOSED — CI VERIFIED
        ↓
quarantine/admission metadata validator
        ↓
family/hash/semantic split-leakage veto
        ↓
Stage 8-1 bytes-only quarantine validator         CLOSED — CI VERIFIED
        ↓
exact source/image/MusicXML hash binding
        ↓
grayscale-PNG full verification
        ↓
supported-V1 semantic/token fingerprint
        ↓
hash-only byte-validation receipt
        ↓
perceptual near-duplicate candidate veto
        ↓
admitted manifest + receipt development handoff
        ↓
Stage 8-2 paired experiment run profile            ACTIVE — NO TRAINING
        ↓
Stage 8-3A source→PNG preparation + admission      LOCKED
        ↓
Stage 8-3B paired real train/validation execution  LOCKED
        ↓
Stage 9 sealed benchmark                           LOCKED
```

The status labels in this Stage 8 section are preserved as historical closed-stage architecture evidence. Current operational status is maintained in `ARCHITECTURE_CURRENT.md` and `STATUS.md`.

Stage 8-0 freezes two experiment candidates: (A) fine-tuning from the exact Stage 7-C checkpoint identified by its accepted checkpoint/model-state hashes, and (B) the same frozen architecture initialized from scratch. The Stage 7-C Actions artifact is only a temporary location and is scheduled to expire on 2026-09-12. Stage 8-2 does not access or relocate it. If exact Candidate A bytes are unavailable or fail hash/state verification at a later execution gate, Candidate A is blocked rather than silently substituted.

Real sample identity is independent of split/review state. Stage 8-0 rejects exact duplicate and family/source/target/semantic aliases. Stage 8-1 independently rebinds exact source, training-image, and MusicXML bytes to those hashes and recomputes the supported-V1 semantic fingerprint before a later loader may trust the sample. Stage 8-1 additionally performs bounded deterministic dHash64 near-duplicate review and requires one independently revalidated receipt per admitted development sample.

Stage 8-2 freezes the first pilot at exactly **50 admitted development pairs: 40 train + 10 validation**. Raw files do not count until they pass Stage 8-0 and Stage 8-1. Both paired candidates must bind the same admitted manifest hash, receipt-set hash, sealed-test commitment, model/tokenizer/preprocess/trainer identities, canonical data order, metrics, and CPU/time budget. The shared profile is 40 epochs, batch size 4, AdamW at 0.001, no scheduler, one retained min-validation-loss checkpoint, 8-measure/1536-token decode, one CPU thread, 1800 seconds maximum per candidate and 3600 seconds maximum for the pair.

Source JPEG/PDF bytes remain unchanged outside Git. The Stage 8 model input remains the Stage 8-1-valid grayscale PNG. Source-document→training-PNG normalization is intentionally not improvised inside Stage 8-2: Stage 8-3A must freeze and verify crop/render/orientation/grayscale/provenance behavior before a real pair may enter the 40/10 handoff.

User-derived material still requires separate explicit training permission and privacy review. ScoreMosaic uploads and teacher corrections cannot bypass quarantine, and no runtime action may trigger model updates.

Both the Stage 6 synthetic test split and the later Stage 8 real test split remain sealed until Stage 9. The Stage 8 development manifest carries no test records, only the opaque sealed-test manifest commitment. `STAGE8_TEST_SEALING_BOUNDARY.md` remains contract-only; no current Stage 8 code creates, enumerates, validates, or opens real test material.

## Verification boundary

GitHub Actions CI is active for the public repository. The baseline uses GitHub-hosted Ubuntu with Python 3.13, pinned runtime dependencies, complete unittest discovery including real-runtime integration tests, `pip check`, and Python compile validation.

Stage 4 merged through PR #14 at exact `main` commit `f0fd8a732b51b4aa95a66c3a780d0cefa6661361`; post-merge GitHub Actions run `31660215130` passed on that exact commit.

Stage 5-A merged through PR #15 at exact `main` commit `d677f3d27ac710c56c5ce677a46dc62bcf77bd84`; Stage 5 closure documentation merged through PR #16 at exact `main` `ed343d4984aae4507a2dd3238cfd1a98fb25b4b7`; the corresponding post-merge runs passed.

Stage 6 merged through PR #17 to exact `main` `7c3c736e6d3755d1bd098e2874d73ce5ed41e39f`; closure PR #18 merged to `046c9a4e7e41e94b0b4465a2610f30361055a3ed`; post-merge runs passed.

Stage 7-A merged through PR #19 and closure PR #20; Stage 7-B merged through PR #21 and closure PR #22. The exact Stage 7-B final source passed **336/336 tests** before merge.

Stage 7-C source `7a993304218fa19609ea512665148dac3eea503a` passed run #67 (`31691794239`) with **361/361 tests**, guarded benchmark, and authoritative run. PR #23 merged to `2c2c478eb361fa90a3bccd819b623680eb12de0b`; run #68 passed. Closure PR #24 merged to `e56fad04e43bc6302a8cda60c6f382e83a23d734`; run #70 passed.

Stage 8-0 merged through PR #25 to `86487a4c3c41264b02bd159cd647a1318d9b9b88`; run #75 passed. Closure PR #26 merged to `99ffaf41f9c8919827ca97edc1bc3900db29eea2`; run #77 passed.

Stage 8-1 final source `ed4113a25f6e12055b9277f959a4580259d37d40` merged through PR #29 to `d551cb27e0244477379100c45e06193ea7ca0cf8`; post-merge run #92 (`31704137450`) passed **394/394 tests**, `pip check`, pinned runtime verification, and `compileall`. Closure synchronization PR #30 merged to exact `main` `de5d66d30c66c97c03bc1dc60fce094a9f0d64e7`; post-merge run #94 (`31705382162`) passed. This is the exact Stage 8-2 starting baseline.

## Stage roadmap

The roadmap below is preserved as historical long-form stage history. Current operational status is maintained in `ARCHITECTURE_CURRENT.md`.

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
Stage 7-C Bounded baseline training run + evidence      ✅ CLOSED — CI VERIFIED
Stage 8-0 Real data & fine-tuning contract freeze       ✅ CLOSED — CI VERIFIED
Stage 8-1 Quarantine/intake + byte validation           ✅ CLOSED — CI VERIFIED
Stage 8-2 Paired experiment run profile                 🔄 ACTIVE — NO TRAINING
Stage 8-3A Pilot source→PNG preparation + admission     🔒 NOT STARTED
Stage 8-3B Paired real train/validation execution       🔒 NOT STARTED
Stage 9   Benchmark and candidate decision              🔒 TEST SEALED
Stage 10  ScoreMosaic candidate integration             🔒
```
