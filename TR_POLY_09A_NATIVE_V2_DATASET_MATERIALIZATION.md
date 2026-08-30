# TR-POLY-09A — Native Polyphonic V2 Dataset Materialization

## Purpose

TR-POLY-08C proved that persisted Stage 6 artifacts can reach the bounded 2D Transformer, but its source target is intentionally the single-voice V1 MusicXML bridge. TR-POLY-09A adds a separate native Polyphonic Representation V2 dataset boundary. It does not relabel the V1 dataset and does not change the TR-POLY-08C path.

The new path is:

`explicit canonical V2 target + grayscale PNG` → `native V2 manifest/build SHA-256` → `TRAIN/VALIDATION artifact hash verification` → `lossless V2 JSON/token roundtrip` → `deterministic image preprocessing` → `existing Poly2DTrainingBatch` → `TR-POLY-08A trainer` → `TR-POLY-08B checkpoint/reload`.

No MusicXML parser is used in this path. Unknown voice, onset, duration or staff values are never inferred.

## Native V2 admission contract

`st_omr_training/poly_v2_dataset_materialization.py` introduces an additive source class:

- source class: `explicit_polyphonic_v2`;
- target profile: `native_polyphonic_v2`;
- manifest schema: `st-omr-native-poly-v2-manifest-v1`;
- family split policy: `family-exclusive-v1`;
- TEST policy: `sealed-no-artifact-read`.

TRAIN/VALIDATION artifact inputs must contain exact canonical Polyphonic V2 JSON and exact grayscale PNG bytes. Canonical parsing itself requires explicit V2 onset, duration, staff and voice fields. Each admitted sample must contain explicit voice 2; the complete TRAIN/VALIDATION manifest must additionally cover voice 3, voice 4+, note/rest/chord, simultaneous independent voices, and a same-onset chord versus independent-voice case.

The target profile also records tie, beam and tuplet metadata counts. These metadata already belong to the frozen V2 representation/tokenizer and are verified by the lossless roundtrip; TR-POLY-09A does not invent missing notation metadata.

## Chord versus simultaneous independent voices

The representation distinction remains structural:

- multiple noteheads in one `PolyEvent` with one logical voice are a chord;
- separate `PolyEvent` values with different `voice` values may share the same exact onset;
- the native dataset profile explicitly verifies that at least one same-onset case contains a chord event alongside an independent voice event.

No serialization order heuristic is used to infer this relationship.

## Split and TEST safety

Every manifest contains metadata for TRAIN, VALIDATION and TEST. Whole `family_id` values are exclusive to one split. Identical target or image hashes may not cross splits.

TEST artifact bytes are deliberately outside the native build input. The builder accepts only TRAIN/VALIDATION artifact bytes plus sealed TEST metadata. A TEST artifact input is rejected before target/image inspection, and `materialize_native_poly_v2_samples(..., DatasetSplit.TEST)` rejects before build/root access.

TRAIN/VALIDATION materialization verifies only their selected target/image bytes. The regression suite writes deliberately corrupt files under the sealed TEST hashes and proves that TRAIN/VALIDATION materialization still succeeds, demonstrating that TEST artifacts are not read.

## Hash binding and deterministic rebuild

Each admitted sample binds:

- canonical target JSON SHA-256;
- canonical V2 representation SHA-256;
- PNG SHA-256;
- width/height;
- exact token count;
- derived polyphony profile;
- split-independent deterministic sample identity.

The manifest uses deterministic canonical JSON and has a SHA-256 identity. The build identity binds the manifest SHA-256 plus sorted target/image artifact hashes. Rebuilding from identical explicit inputs therefore produces the same manifest bytes, manifest hash and build identity.

Persisted layout:

- `manifest.json`;
- `manifest.sha256`;
- `build.json`;
- `targets/<sha256>.json` for TRAIN/VALIDATION targets only;
- `images/<sha256>.png` for TRAIN/VALIDATION images only.

Persistence is one-shot/non-overwriting. Materialization rejects tampered manifest, target or image bytes.

## Tokenizer and truncation gate

Every target is parsed as exact canonical V2, serialized again, tokenized, reconstructed from tokens/IDs and compared to the original V2 object/representation hash.

Semantic truncation is forbidden. If the complete decoder target exceeds the selected `Poly2DTransformerConfig.max_target_tokens`, materialization fails rather than clipping a target.

## Existing 2D training integration

Verified native V2 samples are transformed into the unchanged `Poly2DTrainingBatch` teacher-forcing contract. The existing TR-POLY-08A provenance builder and TR-POLY-08B checkpoint writer/reloader are reused without modifying V1 dataset, V1 model, registry authority, ScoreMosaic behavior or production behavior.

A successful TR-POLY-09A execution receipt means only that:

- the native polyphonic dataset contract was verified;
- TRAIN/VALIDATION native V2 artifacts entered the existing 2D training/checkpoint path;
- TEST was not accessed.

It does **not** mean benchmark success, model-quality improvement, ScoreMosaic shadow readiness or production authority.

## Regression coverage

The TR-POLY-09A tests cover:

- materialized explicit voice 2;
- voice 3 and voice 4+ coverage;
- chord versus same-onset independent voices;
- exact onset/duration/staff/voice reconstruction;
- tie/beam/tuplet metadata through the V2 contract;
- TRAIN/VALIDATION/TEST family leakage rejection;
- fail-closed TEST input/materialization;
- corrupt sealed TEST artifacts remaining unread;
- deterministic manifest/build identity;
- tampered target/image/manifest rejection;
- V2 JSON/tokenizer lossless roundtrip;
- semantic target truncation rejection;
- successful entry into the existing 2D batch and bounded checkpoint training path.

## Gate after TR-POLY-09A

TR-POLY-09B common benchmark work must remain closed until this native V2 dataset/materialization contract and its 2D training-entry regression are green on protected main.
