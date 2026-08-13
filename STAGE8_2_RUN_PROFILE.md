# Stage 8-2 — Paired Experiment Run Profile Freeze

Status: **active package — contract/configuration only; no real data and no training**.

Stage 8-2 freezes the first bounded real-data pilot comparison before Stage 8-3 may prepare or execute any real-data training. It does not ingest user/real files, convert source documents, load the Stage 7-C checkpoint, run optimization, open either sealed test split, publish a model, or integrate with ScoreMosaic.

## Exact starting baseline

Stage 8-2 starts from exact verified `main` `de5d66d30c66c97c03bc1dc60fce094a9f0d64e7`, where Stage 8-1 is closed and main-CI verified.

The accepted Stage 7-C initialization identity remains:

- checkpoint SHA-256: `75c33cefeb970305f5f9171b3274dbe8b785cbcf1e8d6851de7848e66d24efa4`;
- checkpoint model-state SHA-256: `79d354f2582f3f7cc106564b40f07a6027b62b2a74c9efe13a7b2437a6c3f7a0`;
- temporary GitHub Actions artifact: `9177923796`;
- artifact expiry: `2026-09-12T10:44:21Z`;
- Stage 7-C exact-sequence accuracy: `0.0`;
- Stage 7-C token error rate: `0.8049901510177282`.

The Stage 7-C model remains a trainability baseline, not a production candidate.

## Findings addressed by Stage 8-2

Stage 8-2 closes configuration/evidence gaps before training:

- **P1 — comparison drift:** Candidate A and Candidate B previously had no exact shared run profile, allowing data/order/optimizer/resource differences to confound the comparison;
- **P1 — post-hoc data selection:** the first real-data pilot size and train/validation counts were not frozen;
- **P1 — evidence mismatch:** no Stage 8 binding required both candidates to use the exact same admitted manifest, Stage 8-1 receipt set, and sealed-test commitment;
- **P1 — initialization substitution:** Candidate A must fail closed if the exact accepted Stage 7-C checkpoint/model-state bytes are unavailable or mismatched; Candidate B must not load any checkpoint;
- **P1 — source-to-training-image gap:** Stage 8-1 deliberately left JPEG/PDF-to-training-PNG normalization unresolved. Stage 8-2 makes that a hard pre-training blocker rather than permitting ad-hoc conversion;
- **P2 — resource/evidence ambiguity:** the first pilot lacked a frozen CPU/time/checkpoint-retention/metric budget.

No P0 finding is introduced or left open by this package. The source-to-PNG gap is intentionally carried as a pre-execution Stage 8-3A requirement rather than bypassed.

## First real-data pilot size

The first paired experiment requires exactly **50 admitted development image–MusicXML pairs** after Stage 8-0/8-1 validation:

- `40` train samples;
- `10` validation samples;
- `0` test samples visible to Stage 8 development.

These are **admitted** counts, not raw files placed in a folder. If one of the first 50 raw pairs fails rights, pairing, byte, semantic, duplicate, family-leakage, or near-duplicate checks, that pair does not count toward the 50 and must be replaced before the pilot can execute.

Family/source/target/semantic leakage rules from Stage 8-0 remain mandatory. The 40/10 assignment is fixed before either candidate trains and may not be adjusted in response to model results.

The real test manifest remains separately sealed; the development manifest carries only its opaque SHA-256 commitment.

## What the user may prepare now

Real files remain outside this Git repository. A convenient external working layout may be:

```text
ST-OMR-real-pilot-v1/
├── source/          # original JPEG/PDF/source bytes; unchanged
├── musicxml/        # human-verified ground-truth MusicXML
├── rights/          # provenance/rights/permission evidence, outside Git
├── pairing/         # image↔MusicXML review evidence, outside Git
└── derived/         # reserved for later Stage 8-3A validated PNG derivatives
```

Filenames are convenience only; hashes and manifest identities are authoritative. Matching numeric names such as `0001.jpg` and `0001.musicxml` are recommended for human organization but do not create trust by themselves.

Do **not** commit these files to the repository.

## JPEG/PDF and training-PNG boundary

The original source document may be JPEG, PDF, scan, photograph, or another provenance object accepted by the Stage 8-1 source-byte boundary. It is retained unchanged and hash-bound.

The model input is still the Stage 8-1 training image contract: 8-bit grayscale, non-interlaced, single-frame PNG that passes the exact Pillow/runtime/hash/semantic handoff gates.

Stage 8-2 does **not** define an ad-hoc conversion from JPEG/PDF to PNG. Before the first training run, Stage 8-3 must begin with a small **Stage 8-3A — Pilot Data Preparation + Admission** sub-gate that freezes the source-to-training-PNG derivation/provenance procedure and produces the admitted 40/10 handoff. Only after 8-3A is verified may the paired training execution sub-gate run.

This prevents manual crop, DPI, orientation, grayscale, PDF-rendering, or resize choices from silently changing the training dataset.

## Frozen paired experiment

The primary comparison contains exactly two candidates.

### Candidate A — exact Stage 7-C checkpoint fine-tuning

- same frozen Stage 7-B CNN/GRU architecture;
- same tokenizer/vocabulary;
- same Stage 7 preprocessing;
- initialization only from the exact accepted Stage 7-C checkpoint/model-state hashes above;
- strict checkpoint/state compatibility required before any optimizer step;
- no fallback checkpoint, external pretrained model, teacher model, or silently recreated substitute.

If the exact checkpoint bytes are unavailable or fail hash/state verification, Candidate A is **blocked**. Stage 8-2 does not move or preserve the expiring artifact.

### Candidate B — deterministic from scratch

- exact same architecture/configuration as Candidate A;
- same frozen tokenizer/vocabulary and preprocessing;
- deterministic Stage 7 initialization using the frozen trainer master seed;
- no checkpoint and no external pretrained weights.

## Shared execution profile

The two candidates differ only at initialization. The first pilot freezes:

- total admitted samples: `50`;
- train/validation: `40 / 10`;
- epochs: `40`;
- batch size: `4`;
- deterministic canonical sample-id data order;
- model config: exact Stage 7-C baseline config;
- trainer config: exact Stage 7-C baseline `TrainerConfig()`;
- optimizer: AdamW;
- learning rate: `0.001` (`1000` micros);
- weight decay: `0`;
- gradient clipping: `1.0`;
- scheduler: none;
- checkpoint selection: minimum validation loss;
- retained checkpoints: `1` per candidate;
- decode measure count: `8`;
- max decode tokens: `1536`;
- device: CPU only;
- deterministic CPU threads: `1`;
- wall-clock ceiling: `1800` seconds per candidate;
- paired wall-clock ceiling: `3600` seconds total.

A resource-budget failure is a failed/incomplete pilot, not permission to change the profile mid-run.

## Frozen metric family

Each candidate must report the same Stage 7 metric family on the exact same validation set:

- validation loss;
- token error rate;
- exact sequence accuracy;
- detokenization success rate;
- semantic validity rate;
- MusicXML regeneration validity rate.

Stage 8-3 may record additional diagnostics only if they do not change candidate selection or expose sealed-test information. Stage 9 owns production-quality thresholds and the first sealed-test decision.

No Stage 8 validation result, including a large improvement over Stage 7-C, is sufficient by itself to call a model production-ready.

## Evidence binding

Before a paired run can be executable, both candidates must be bound to exactly the same:

- Stage 8-2 profile fingerprint;
- admitted development manifest SHA-256;
- one-receipt-per-sample Stage 8-1 receipt-set SHA-256;
- sealed real-test manifest SHA-256 commitment;
- tokenizer fingerprint;
- preprocessing fingerprint;
- model fingerprint;
- trainer fingerprint;
- intake-policy fingerprint.

`stage8_experiment_profile.py` supplies deterministic profile/receipt-set/binding identities and rejects A/B drift. These hash-only bindings do not replace Stage 8-0 rights approval or the Stage 8-1 full development-handoff validator.

## Stage 8-3 execution acceptance boundary

Stage 8-2 does not run this gate; it freezes what Stage 8-3 must prove:

```text
Stage 8-2 exact profile
        ↓
Stage 8-3A frozen source→training-PNG derivation
        ↓
50 admitted pairs after Stage 8-0/8-1 validation
        ↓
exact 40 train / 10 validation manifest
        ↓
exact one-receipt-per-sample handoff
        ↓
Candidate A exact checkpoint hash/state verification
        +
Candidate B deterministic from-scratch initialization
        ↓
same manifest + same order + same optimizer/budget
        ↓
finite train/validation numeric state
        ↓
one retained min-validation-loss checkpoint each
        ↓
same validation metric family
        ↓
hash-addressed paired evidence
        ↓
NO sealed-test access
        ↓
NO production/ScoreMosaic promotion
```

Stage 8-3 evidence must record the pre-update validation loss and best validation loss for each candidate. Failure to improve, non-finite state, budget exhaustion, checkpoint mismatch, manifest/receipt drift, or any sealed-test access fails closed. Stage 9 still owns candidate-quality thresholds.

## ScoreMosaic boundary

Nothing in Stage 8-2 changes the frozen rules:

- ScoreMosaic user uploads are not automatic training data;
- teacher corrections are not automatic training data;
- explicit permission/privacy review remains mandatory for user-derived material;
- no online, background, or automatic learning;
- no Stage 8 model enters ScoreMosaic before Stage 9 quality evidence and Stage 10 integration approval.

## Repository and storage boundary

This repository remains code/contracts/tests/hash-only evidence only. Real score images, real MusicXML corpora, private permission documents, datasets, and model checkpoints must not be committed.

Stage 8-2 adds no upload service, cloud bucket, credential, model registry, or persistent real-data store. The user's initial files remain in an external local workspace until the later Stage 8-3A preparation/admission procedure is explicitly implemented.

## Stage 8-2 closure gate

```text
exact Stage 8-1 closed main baseline
        ↓
50-pair pilot count/split freeze
        ↓
A/B same-data same-budget contract
        ↓
exact Candidate A checkpoint identity
        ↓
Candidate B no-checkpoint rule
        ↓
manifest + receipt-set + sealed-test hash bindings
        ↓
CPU/time/checkpoint/metric evidence budget
        ↓
source→PNG gap made explicit fail-closed prerequisite
        ↓
focused tests + full regression + pip check + compileall
        ↓
exact PR-head GitHub CI
        ↓
separate merge approval
        ↓
post-merge exact-main CI
        ↓
Stage 8-2 CLOSED
```

## Explicitly out of scope

Stage 8-2 does not ingest, upload, copy, normalize, or persist real/user/teacher data; does not load/move/publish/preserve the Stage 7-C checkpoint; does not run training/fine-tuning; does not open/create/enumerate a sealed test; does not start Stage 8-3 execution, Stage 9, or Stage 10; does not integrate with ScoreMosaic; and does not enable automatic learning.
