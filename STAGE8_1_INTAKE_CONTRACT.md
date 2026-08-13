# Stage 8-1 — Quarantine / Intake + Byte-Level Validation

Status: **closed — main CI verified; no real data, no training, no sealed-test access occurred**.

Stage 8-1 implements and closes the byte-level gate promised by the Stage 8-0 contract. It does not create a real dataset, choose training hyperparameters, load a checkpoint, run fine-tuning, open either test partition, or integrate with ScoreMosaic.

## Exact baseline and closure evidence

Stage 8-1 started from exact verified `main` `99ffaf41f9c8919827ca97edc1bc3900db29eea2`.

Final Stage 8-1 implementation evidence:

- PR: #29;
- final source head: `ed4113a25f6e12055b9277f959a4580259d37d40`;
- exact `main` merge commit: `d551cb27e0244477379100c45e06193ea7ca0cf8`;
- post-merge GitHub Actions run #92: `31704137450`, success;
- runtime: Ubuntu 24.04 / CPython 3.13.14;
- pinned dependencies: `lxml==6.1.1`, `verovio==6.2.1`, `CairoSVG==2.8.2`, `Pillow==12.3.0`, `torch==2.13.0+cpu`;
- `pip check`: passed;
- full regression: **394/394 tests passed**;
- `compileall`: passed.

The Stage 7-C checkpoint remains outside this package and was not loaded, copied, moved, published, or preserved.

## Stage boundary

```text
Stage 8-0 rights/provenance/pairing metadata contract   CLOSED
        ↓
Stage 8-1 quarantined bytes + exact hash checks         CLOSED — CI VERIFIED
        ↓
training-image structural/decode verification
        ↓
MusicXML XSD + ST semantic + supported-V1/token gate
        ↓
semantic fingerprint recomputation
        ↓
hash-only byte-validation receipt
        ↓
perceptual near-duplicate review gate
        ↓
validated development handoff
        ↓
Stage 8-2 paired experiment run profile                 LOCKED
        ↓
Stage 8-3 real train/validation experiments             LOCKED
        ↓
Stage 9 sealed benchmark                                LOCKED
```

## No repository intake

The repository remains code/contract/test/evidence only. Real score files, user uploads, teacher corrections, permission documents, real MusicXML corpora, private material, datasets, and checkpoints must not be committed.

The Stage 8-1 implementation accepts in-memory `bytes` only. It adds no filesystem writer, upload endpoint, network fetcher, scraper, cloud storage, credential path, or ScoreMosaic connector. Tests use generated MusicXML and synthetic PNG/source-byte fixtures only.

## Quarantine-first rule

`validate_quarantined_sample_bytes(...)` accepts only a `RealDataSample` that still satisfies the Stage 8-0 quarantine contract. A `test` split record is rejected before caller-provided bytes are inspected.

Byte validation alone does not grant training access. The Stage 8-0 admitted development manifest remains the rights/provenance/pairing/leakage authority. A later loader must additionally require the one-to-one Stage 8-1 receipt handoff.

## Source-document byte binding

The source document is format-agnostic at this gate because it may be a PDF, scan, photograph, or other provenance object. It is not used directly as the model input.

The gate requires:

- non-empty immutable `bytes`;
- maximum size 64 MiB;
- exact SHA-256 equality with `source_document_sha256`.

This check proves byte identity only. It does not itself prove rights, privacy clearance, musical correctness, or image–MusicXML pairing.

## Training-image contract

The Stage 8-1 training image is restricted to:

- PNG;
- 8-bit grayscale mode `L`;
- non-interlaced;
- single frame;
- maximum 64 MiB encoded bytes;
- maximum 16,000,000 pixels.

The encoded-size guard runs before digest computation. The expected image SHA-256 is then verified before any untrusted PNG parsing/decode. The validator independently checks the PNG signature, canonical first `IHDR`, `IHDR` CRC, dimensions, bit depth/color type/compression/filter/interlace fields, then performs full verification and decode.

Image validation and perceptual hashing require exact `Pillow==12.3.0`; runtime drift fails closed. The expected Pillow version and frozen perceptual-hash semantics are part of the Stage 8-1 policy identity.

Stage 8-1 does not convert arbitrary source formats into this PNG. Any future source-to-training-image normalization procedure requires a separate frozen transform/provenance contract.

## MusicXML and semantic identity

MusicXML must be non-empty and remain inside the existing `MAX_MUSICXML_BYTES` bound. The size guard runs before digest computation; the exact SHA-256 must then match `musicxml_sha256` before parsing.

The exact bytes must pass the existing frozen ST-OMR V1 path used by `tokenize_musicxml(...)`:

1. MusicXML byte preflight;
2. offline MusicXML 4.0 XSD validation;
3. independent ST-OMR V1 semantic validation;
4. supported-V1 semantic projection;
5. frozen Stage 7 tokenizer;
6. tokenizer/detokenizer semantic round trip.

Stage 8-1 freezes the real-data semantic fingerprint as SHA-256 over canonical JSON containing the semantic-fingerprint version, exact tokenizer fingerprint, and exact supported-V1 token-id sequence. The recomputed value must match the Stage 8-0 metadata record.

Automated OMR/OCR/LLM output is still not ground truth merely because it passes these structural gates. Stage 8-0 pairing-review evidence remains mandatory for admission.

## Hash-only byte-validation receipt

Successful validation emits `RealDataByteReceipt`. The receipt contains no source, image, MusicXML, permission text, personal data, or model bytes. It binds:

- sample/family/split identity;
- source/image/MusicXML SHA-256 values;
- supported-V1 semantic fingerprint;
- tokenizer fingerprint;
- image dimensions;
- token count;
- perceptual hash;
- frozen Stage 8-1 policy fingerprint;
- canonical receipt SHA-256.

Receipt consumers do not trust construction history. Field bounds, current tokenizer/policy fingerprints, and canonical receipt self-hash are independently revalidated before search or development handoff.

A receipt is deterministic integrity/provenance metadata, not a digital signature or independent authorization credential. Rights/pairing admission remains a separate Stage 8-0 gate.

## Near-duplicate review

Exact hashes do not detect alternate encodings or minor scan differences. Stage 8-1 therefore adds a deterministic review surface:

- algorithm: 64-bit difference hash (`dHash64`);
- input: fully verified grayscale training PNG;
- resize: 9×8 using pinned Pillow LANCZOS;
- candidate threshold: Hamming distance ≤ 4.

The final implementation does not perform an unbounded all-pairs scan. It uses five disjoint dHash segments to generate radius-4 candidates, then verifies exact Hamming distance. Comparison and emitted-result budgets are independently bounded; pathological candidate sets fail closed rather than consuming unbounded work.

This is a conservative heuristic, not proof that two pages are musically identical and not a claim of perfect perceptual-duplicate recall. A candidate under different `family_id` values is a fail-closed development-handoff veto until family/leakage review is corrected. Same-family candidates remain visible for audit and inherit the Stage 8-0 family-exclusive split rule.

## Development handoff

`validate_stage8_development_handoff(...)` is the Stage 8-1 handoff surface intended for any later loader. It requires:

- a valid Stage 8-0 admitted train/validation manifest;
- exactly one matching Stage 8-1 receipt per manifest sample;
- exact sample/receipt hash and semantic binding;
- frozen intake-policy and tokenizer identities;
- no test records;
- no unresolved cross-family perceptual near-duplicate candidate;
- all Stage 8-0 exact duplicate/family/source/target/semantic leakage vetoes.

The development manifest still contains train and validation metadata only plus the opaque `sealed_test_manifest_sha256` commitment. Stage 8-1 does not enumerate, fetch, validate, or inspect real test records.

## Future real-test sealing boundary

`STAGE8_TEST_SEALING_BOUNDARY.md` defines the contract-only safety boundary for a future separately authorized test-sealing operation. Stage 8-1 does not implement or execute that operation.

The current code contains no test writer, test loader, test-byte validator, or test enumeration path. Both held-out test partitions remain sealed until Stage 9 explicitly authorizes benchmark access.

## ScoreMosaic boundary

Nothing changes the Stage 8-0 rules:

- ScoreMosaic uploads are not automatic training data;
- teacher corrections are not automatic training data;
- explicit training permission and privacy review remain mandatory for user-derived material;
- no online/background/automatic learning exists;
- no Stage 8 model enters ScoreMosaic before Stage 9 quality evidence and the later Stage 10 integration decision.

## Findings closed in Stage 8-1

No P0 remained. The package closed the following significant implementation risks:

- P1 — image decode was initially reachable before expected-image SHA verification;
- P1 — an initial all-pairs near-duplicate scan was unsafe against the Stage 8-0 manifest ceiling;
- P1 — receipt objects initially needed stronger independent consumer-side integrity revalidation;
- P1/P2 — Pillow/dHash runtime semantics needed explicit policy identity;
- P2 — future real-test sealing needed an explicit contract boundary before any test material could exist;
- P2 from automated review — image/MusicXML encoded-size guards needed to run before digest computation.

The final source includes regressions proving oversized image/MusicXML payloads fail before hashing and preserves hash-before-parse/decode behavior for in-bound data.

## Residual risks

Stage 8-1 deliberately does not resolve:

- dHash false positives/false negatives;
- source-document-to-training-PNG normalization and provenance;
- substantive legal validity of rights/permission evidence;
- cryptographic signing of validation receipts;
- real-corpus quality or distribution representativeness;
- Stage 7-C checkpoint retention/expiry;
- experiment resource budgets, optimization, or quality thresholds.

These are later or organizational gates and must not be inferred as solved by Stage 8-1 CI.

## Stage 8-1 closure gate

```text
exact Stage 8-0 closed main baseline
        ↓
bytes-only quarantined validator
        ↓
pre-hash size bounds + exact source/image/MusicXML binding
        ↓
full grayscale-PNG verification + exact Pillow pin
        ↓
MusicXML supported-V1/token semantic gate
        ↓
deterministic semantic + policy fingerprints
        ↓
independently revalidated hash-only receipt
        ↓
bounded perceptual near-duplicate candidate gate
        ↓
sealed-test early veto + future sealing contract
        ↓
focused tests + full regression + pip check + compileall
        ↓
PR #29 merged to `d551cb27e0244477379100c45e06193ea7ca0cf8`
        ↓
post-merge run #92 SUCCESS — 394/394
        ↓
Stage 8-1 CLOSED — MAIN CI VERIFIED
```

## Explicitly out of scope

Stage 8-1 did not ingest real/user/teacher data; download or scrape scores; add storage or upload infrastructure; normalize arbitrary source formats; create/open/enumerate either test partition; access or relocate the Stage 7-C checkpoint; run training/fine-tuning; alter model/tokenizer surfaces; define Stage 8 experiment profiles; start Stage 8-2/8-3/9/10; integrate with ScoreMosaic; or enable automatic learning.
