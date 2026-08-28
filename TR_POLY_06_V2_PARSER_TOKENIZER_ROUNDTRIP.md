# TR-POLY-06 — V2 Parser / Tokenizer / Lossless Roundtrip

## Status

Research contract implementation for the additive Polyphonic Representation V2.

This package does **not** change the frozen Stage 7-B V1 tokenizer, the V1 CNN-GRU baseline, training data, sealed TEST data, or ScoreMosaic runtime behavior.

## Why this package exists

TR-POLY-05 froze the semantic object model. TR-POLY-06 proves that the model can cross deterministic serialization boundaries without losing polyphonic evidence.

A 2D model must not be trained against an ambiguous target whose parser and inverse have not been proven first. Therefore this package precedes the Model-A Transformer prototype.

## Two roundtrip surfaces

### 1. Canonical JSON

`PolyScore.canonical_json()` remains the normative V2 serialization.

`parse_canonical_polyphonic_json()` accepts only the exact canonical form:

- sorted object keys;
- compact separators;
- ASCII JSON with deterministic escaping;
- exact field set;
- exact enum values;
- exact integer types (booleans are not integers);
- V2 object invariants revalidated by the TR-POLY-05 constructors.

Equivalent JSON with extra whitespace, unknown fields, alternate key order, or coercible scalar types is rejected.

This is intentional: benchmark/model artifacts must have reproducible content hashes.

### 2. Structured closed-vocabulary token codec

The V2 tokenizer does not use arbitrary strings as vocabulary items.

It uses:

- structural tokens (`OBJ_START`, `ARR_START`, etc.);
- first-class semantic key tokens (`KEY:voice`, `KEY:staff`, `KEY:onset`, ...);
- canonical integer digit tokens;
- UTF-8 byte tokens for open text such as event IDs and source measure labels;
- explicit null/boolean tokens;
- BOS/EOS and batching-only PAD.

This keeps the vocabulary closed while preserving arbitrary identifiers and Unicode text exactly.

The tokenizer therefore avoids two bad alternatives:

1. an unbounded vocabulary created by event IDs/source labels;
2. a raw-character JSON tokenizer that hides semantic field boundaries inside punctuation.

## Frozen versions

- Representation: `st-omr-polyphonic-representation-v2`
- Serialization: `st-omr-polyphonic-serialization-v1`
- Tokenizer: `st-omr-polyphonic-tokenizer-v1`

The tokenizer fingerprint binds:

- serialization version;
- tokenizer version;
- representation version;
- complete ordered vocabulary.

## Exact evidence preserved

The roundtrip covers all TR-POLY-05 semantics, including:

- parts and measure identity;
- staff count and per-staff clefs;
- additive-capable meter;
- key signature;
- explicit voice and staff;
- exact onset and duration rational values;
- note/rest/chord structure;
- notehead IDs and pitch spelling;
- visible accidentals;
- ties;
- notehead-level staff override / cross-staff chords;
- visible note type and dots;
- stem direction;
- beams;
- tuplets;
- zero-duration grace evidence and slash state;
- barlines and repeats.

The token target stores the canonical representation SHA-256. `detokenize_polyphonic_target()` rejects a token sequence if the reconstructed score does not match that hash.

## Fail-closed boundaries

The codec rejects:

- missing/unknown object fields;
- unsupported enum values;
- bool-as-int coercion;
- noncanonical JSON text;
- unknown token IDs;
- PAD inside semantic targets;
- nested or malformed BOS/EOS envelopes;
- duplicate or non-sorted object keys in token streams;
- noncanonical integers such as leading-zero values and negative zero;
- invalid UTF-8 text payloads;
- excessive nesting, text size, JSON size, or token count;
- any reconstructed score that violates the TR-POLY-05 semantic contract;
- representation-hash drift after detokenization.

## Explicit non-goals

TR-POLY-06 does not yet:

- implement a general MusicXML 4.0 importer;
- normalize arbitrary third-party MusicXML conventions;
- train a Transformer;
- choose between Model A/B/C;
- access external benchmark bytes;
- open sealed TEST data;
- promote a model to ScoreMosaic.

A general MusicXML-to-V2 adapter is intentionally a separate parser bridge because MusicXML permits multiple equivalent encodings for the same notation. That bridge must resolve `<backup>/<forward>`, chord continuation, staff/voice state, tuplets and cross-staff semantics without weakening the already-frozen V2 target contract.

## Gate to the next package

TR-POLY-06 is complete only when:

1. exact-head CI passes;
2. all review threads are resolved;
3. canonical JSON roundtrip preserves object equality + SHA-256;
4. token and token-ID roundtrip preserve object equality + SHA-256;
5. V1 tokenizer/model files remain untouched.

After this gate, the safe next step is the V2 MusicXML parser bridge / controlled normalization fixture set, followed by baseline registry consolidation and then the tiny 2D Transformer prototype under the common benchmark contract.
