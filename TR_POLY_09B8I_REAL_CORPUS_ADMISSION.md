# TR-POLY-09B8I — Real Corpus Admission Bridge

Status: implementation package for the first real Native Polyphonic V2 quality-training corpus.

## Purpose

B8A can prove that a persisted Native V2 root is internally exact, but a valid
`manifest.json` alone does **not** prove that its real source material was
lawfully/properly admitted or that its image/target pairing was reviewed.

The repository already has a frozen Stage 8-0/8-1 real-data boundary for:

- provenance and rights-review metadata;
- explicit training permission for user-derived / ScoreMosaic / teacher material;
- privacy evidence where required;
- image–MusicXML pairing review;
- exact source/image/MusicXML hashes;
- byte receipts and supported semantic validation;
- near-duplicate/family leakage protection;
- sealed TEST isolation.

B8I bridges that mature real-data admission surface to the newer Native
Polyphonic V2 quality lane instead of creating a second, weaker legal/provenance
system.

## Required inputs

`admit_native_poly_v2_real_corpus(...)` accepts only:

1. one already verified B8A `LoadedNativePolyV2Dataset`;
2. one admitted Stage 8 `RealDataManifest` containing TRAIN/VALIDATION only;
3. one verified Stage 8-1 `RealDataByteReceipt` for every real manifest sample;
4. one `NativePolyV2RealDataBinding` for every Native V2 TRAIN/VALIDATION sample.

Every lineage binding hash-binds:

- exact Native V2 sample ID;
- exact real-data sample ID;
- one V2 conversion-profile SHA-256;
- one independent V2 target-review evidence SHA-256.

The machine gate binds those evidence identities. It does not claim that a
human/legal/musical review was substantively correct.

## One-to-one lineage requirements

The bridge fails closed unless the real-data and Native V2 development
populations are exactly one-to-one.

For every bound pair it requires:

- TRAIN/VALIDATION split equality;
- `family_id` equality;
- exact image SHA-256 equality;
- exact image dimensions equal to the Stage 8-1 byte receipt;
- complete B8A TRAIN/VALIDATION population coverage;
- complete Stage 8 real-data manifest/receipt coverage;
- no duplicate Native binding;
- no real sample reused for multiple Native V2 targets.

The upstream Stage 8-1 handoff remains authoritative for rights/provenance,
permission/privacy, byte validation, pairing evidence and perceptual
near-duplicate vetoes.

## TEST boundary

B8I does not open TEST.

It records only:

- the opaque Stage 8 sealed-test-manifest SHA-256; and
- a hash commitment to the already-known B8A sealed Native V2 TEST metadata
  population.

No TEST source, image, MusicXML, V2 target or image artifact bytes are read by
this package.

## Output

A successful `NativePolyV2RealCorpusAdmissionReceipt` binds:

- Native V2 manifest SHA-256 and build ID;
- B8A preflight receipt fingerprint;
- admitted real-data manifest fingerprint;
- Stage 8-1 byte-receipt manifest fingerprint;
- Native↔real binding-manifest fingerprint;
- sealed real TEST commitment;
- sealed Native V2 TEST metadata commitment;
- exact TRAIN and VALIDATION Native sample populations;
- exact admitted real-data sample population;
- accepted same-family/same-split near-duplicate candidate count.

A completed receipt sets `quality_training_eligible=True` only for the B8
research/quality-training lane. It always keeps:

```text
test_artifact_bytes_accessed = false
production_authority = false
commercial_use_authority = false
```

Therefore B8I admission must not be interpreted as a production-release or
commercial-license decision.

## Current blocker

As of 2026-09-16 the connected `ScoreMosaic_Teacher_Gold` Drive structure
contains only empty organizational folders. No admitted Native V2 persisted
root, admitted Stage 8 real-data manifest, byte receipts or B8I bindings are
physically available.

Consequently:

- B8I code can be regression-tested with synthetic fixtures;
- a real B8I receipt cannot yet be emitted;
- B6 real quality training must not yet run;
- no real accuracy/quality number may be claimed.

## Next executable path when data exists

```text
real source + rights/provenance/pairing evidence
        ↓
Stage 8-0/8-1 admission + byte receipts
        ↓
Native V2 materialization
        ↓
B8A persisted-root preflight
        ↓
B8I exact real↔Native lineage admission
        ↓
complete B7 descriptors
        ↓
B8B experiment recipe fingerprint
        ↓
B6/B4 real TRAIN-only optimization
        ↓
B5 checkpoint verification
        ↓
B7 complete VALIDATION
```

Repository fixtures remain test evidence only and may not be reported as real
model-quality evidence.
