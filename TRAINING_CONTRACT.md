# ST-OMR Baseline Training Contract

Status: Stage 7-A contract freeze. This document defines the bounded interface and verification rules for Baseline ST-OMR Training. It does **not** implement or run model training.

## Purpose

Stage 7 establishes the first reproducible synthetic-only ST-OMR baseline behind the already-closed Stage 6 dataset boundary.

The baseline exists to prove that the validated synthetic pipeline can train one bounded model and produce auditable validation evidence. It is **not** a production candidate decision. Production candidacy remains a later Stage 9 responsibility.

## Stage decomposition

```text
Stage 7-A  Training contract freeze                  ← this package
Stage 7-B  Deterministic tokenizer/data/model/trainer implementation
Stage 7-C  Bounded baseline training run + evidence
Stage 8    Real-data fine-tuning                     locked
Stage 9    Sealed benchmark / candidate decision     locked
```

Stage 7-B and Stage 7-C require separate bounded packages and separate merge approvals.

## Trusted input boundary

Stage 7 may consume only a Stage 6 `SyntheticDatasetBuild` that has already passed the independent Stage 5 dataset validator.

Each training-eligible sample must be resolved from the exact persisted artifacts recorded by the build:

- one Stage 4 final grayscale PNG derivative;
- one exact Stage 2-C-valid MusicXML target associated with the same symbolic family;
- immutable Stage 5 sample metadata;
- immutable split assignment;
- exact SHA-256 lineage for target and image artifacts.

Stage 7 must re-check persisted artifact hashes before use. A missing artifact, hash mismatch, invalid manifest, unsupported source class, or split inconsistency is a hard failure.

Stage 7 must not ingest loose images, arbitrary MusicXML, user files, downloaded scores, rights-unclear corpora, or files that bypass the Stage 5/6 gates.

## Split isolation

The Stage 5/6 family-exclusive split is authoritative.

- `train` may update model parameters.
- `validation` may select checkpoints and report development metrics.
- `test` is sealed during Stage 7 and must not be used for optimization, early stopping, architecture choice, threshold choice, or routine metric inspection.

The Stage 9 benchmark is the first stage allowed to open the sealed test split for candidate evaluation.

No Stage 7 code may manufacture a new split, move a sample between splits, or derive a new sample identity from a split change.

## Input tensor contract

The model input is the exact Stage 4 final PNG artifact referenced by the Stage 5 sample.

V1 preprocessing rules:

1. decode PNG only;
2. require grayscale mode `L` after verification;
3. require dimensions and pixel count to remain inside the Stage 5 limits;
4. normalize pixel values deterministically to floating point `[0, 1]`;
5. preserve aspect ratio;
6. do not crop musical content;
7. do not apply hidden/random training augmentation in Stage 7;
8. padding, if required by the concrete model, must be deterministic and must not change the source image identity;
9. any later resize policy must be explicitly frozen and fingerprinted before implementation.

All appearance variation used by the V1 baseline comes from the already-audited Stage 4 degradation pipeline.

## Target semantics

The baseline must **not** learn raw XML syntax as the primary prediction target.

The training target is a compact deterministic `ST-OMR V1 token sequence` derived from the independently parsed supported-V1 musical projection of the exact Stage 6 MusicXML target.

The token sequence must preserve every V1 notation semantic required to reconstruct the supported canonical score:

- measure boundaries;
- effective time signature for each measure;
- event order;
- note/rest/chord event type;
- exact V1 duration class;
- chord size and member order;
- pitch step;
- pitch alter `-1/0/+1`;
- octave;
- visible accidental intent: none/sharp/flat/natural.

The fixed V1 single-part, single-staff, treble-clef, one-voice, key-signature-zero policy may remain implicit because those values are invariant in the frozen V1 domain.

## Token vocabulary contract

Stage 7-B must implement a finite, explicit vocabulary containing only these semantic classes:

- sequence: `BOS`, `EOS`, `PAD`;
- structure: `MEASURE_START`, `MEASURE_END`;
- meter: `TS_2_4`, `TS_3_4`, `TS_4_4`;
- event: `NOTE`, `REST`, `CHORD_2`, `CHORD_3`, `CHORD_4`;
- duration: `DUR_WHOLE`, `DUR_HALF`, `DUR_QUARTER`, `DUR_EIGHTH`;
- pitch step: `STEP_A` through `STEP_G`;
- alter: `ALTER_M1`, `ALTER_0`, `ALTER_P1`;
- octave: `OCT_3`, `OCT_4`, `OCT_5`, `OCT_6`;
- display accidental: `ACC_NONE`, `ACC_SHARP`, `ACC_FLAT`, `ACC_NATURAL`.

No vocabulary learning, BPE, external tokenizer model, or text normalization is permitted for V1.

## Target round-trip gate

Tokenizer implementation is not trusted by construction.

Before a sample becomes training-eligible, Stage 7-B must independently verify:

```text
exact Stage 6 MusicXML
        ↓
Stage 2-D supported-V1 semantic projection
        ↓
Stage 7 tokenizer
        ↓
Stage 7 detokenizer
        ↓
reconstructed supported-V1 projection
        ↓
independent semantic comparison
```

The source and reconstructed projections must be exactly equal. Any tokenizer/detokenizer mismatch is a hard veto.

## Model contract

Stage 7-B may implement exactly one bounded baseline model family with the interface:

```text
grayscale page tensor
        ↓
visual encoder
        ↓
sequence decoder
        ↓
logits over frozen ST-OMR V1 token vocabulary
```

Baseline constraints:

- one model, not an ensemble;
- initialized from scratch;
- no external/pretrained weights;
- no network access required for training;
- no OCR/OMR teacher model providing hidden labels;
- no inference-time call to Audiveris, Scan2Notes, an LLM, or another external recognition engine;
- maximum **25,000,000 trainable parameters** for V1;
- architecture, exact parameter count, initialization policy, and framework/runtime versions must be fingerprinted before the first real training run.

Stage 7-A intentionally does not choose PyTorch, TensorFlow, JAX, or another framework. Dependency/framework selection belongs to Stage 7-B and must be based on a separate current-runtime compatibility review before a pin is committed.

## Loss and optimization boundary

Stage 7-B must use one explicit token-sequence training objective with masked padding positions. The exact loss, optimizer, scheduler if any, gradient-clipping policy, batch policy, and checkpoint-selection rule must be frozen in configuration and included in the run fingerprint.

Requirements:

- NaN or Infinity in inputs, logits, loss, gradients, optimizer state, or reported metrics is a hard failure;
- gradient updates may use only the `train` split;
- checkpoint selection may use only `validation` metrics;
- no test-set feedback is permitted in Stage 7;
- an interrupted run must not silently resume from an unverified checkpoint.

## Reproducibility contract

Every Stage 7 run must record at minimum:

- repository commit SHA;
- Stage 6 build identity;
- Stage 5 manifest SHA-256;
- exact hashes of consumed target/image artifacts or an immutable reference to the verified build manifest;
- training configuration fingerprint;
- tokenizer/version fingerprint;
- model architecture/version fingerprint;
- trainable parameter count;
- framework and relevant dependency versions;
- Python version;
- operating system and machine/accelerator identity;
- master seed and all derived seeds;
- epoch/step count;
- selected checkpoint SHA-256;
- metrics file SHA-256.

The implementation must seed every RNG it uses. Data ordering must be deterministic for a fixed run configuration. Where the selected framework/device cannot guarantee bit-identical accelerator execution, that limitation must be stated explicitly and CPU deterministic smoke evidence must remain separate from accelerator reproducibility claims.

## Resource limits

Stage 7 V1 must fail closed rather than silently exceed configured limits.

The frozen outer ceilings are:

- at most the Stage 6 ceiling of `50,000` dataset samples;
- at most `25,000,000` trainable parameters;
- at most `100` configured epochs for a Stage 7-C baseline run;
- at most `10` retained checkpoints per run;
- checkpoint and metric artifacts remain outside normal Git content;
- ordinary recurring GitHub-hosted CI may run only bounded smoke training; any one-shot full Stage 7-C execution requires separate explicit scope, an exact-head fail-closed benchmark, and removal of that execution path at Stage 7-C closure.

Stage 7-B may choose tighter limits but may not enlarge these ceilings without a new contract change.

## Checkpoint and evidence boundary

Model checkpoints are derived artifacts, not source code.

- checkpoints must be written to a fresh/no-overwrite run directory;
- selected checkpoints must be SHA-256 hashed;
- large checkpoints must not be committed to Git;
- no production cloud bucket, credential, secret, signed URL, deployment job, or model registry is introduced in Stage 7 V1;
- a small deterministic metrics/provenance record may be committed as evidence if it contains no large binary artifact or secret.

## Required Stage 7 metrics

Stage 7-C must report validation-only development metrics at minimum:

1. token cross-entropy/loss;
2. token error rate;
3. exact token-sequence accuracy;
4. detokenization success rate;
5. reconstructed supported-V1 semantic-validity rate;
6. MusicXML regeneration validity rate after reconstructed canonical scores are passed through the existing writer/validators.

Stage 7 is a baseline/trainability stage, not the final candidate gate. Therefore Stage 7-C acceptance requires:

- all reported values finite;
- best validation loss strictly better than the deterministic untrained baseline measured before optimization;
- at least one validation prediction successfully detokenizes and passes the supported-V1 semantic gate;
- no family leakage or test access;
- exact checkpoint/config/dataset provenance;
- full repository regression and exact-head GitHub CI for source changes.

No Stage 7 metric alone is sufficient to declare a production-quality OMR model. Comparative quality thresholds and sealed-test candidate decisions belong to Stage 9.

## Stage 7-B verification gate

Before Stage 7-B may merge, it must demonstrate at minimum:

```text
frozen token vocabulary
        ↓
tokenizer/detokenizer exact semantic round trip
        ↓
Stage 5/6 artifact + split revalidation
        ↓
deterministic input preprocessing
        ↓
bounded baseline model construction
        ↓
parameter ceiling check
        ↓
CPU smoke forward/backward/update
        ↓
NaN/Infinity fail-closed tests
        ↓
train-only update / validation-only selection / test-sealed tests
        ↓
same-seed deterministic CPU smoke replay
        ↓
full repository regression
        ↓
compile validation
        ↓
exact PR-head GitHub CI
        ↓
separate merge approval
        ↓
post-merge exact-main CI
```

## Explicitly out of scope

Stage 7-A does not implement:

- tokenizer code;
- model code;
- training framework dependencies;
- dataset generation changes;
- real/user data;
- teacher-correction learning;
- pretrained models;
- hyperparameter search services;
- distributed training;
- cloud training/storage;
- production inference service;
- benchmark opening of the Stage 6 test split;
- Guitar TAB training;
- ScoreMosaic integration.

These boundaries require later separately approved packages.
