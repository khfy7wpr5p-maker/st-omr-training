# Stage 7-D3 — Validation Error Diagnostics

Status: **active — validation analysis only; no training**.

Stage 7-D3 diagnoses why the accepted Stage 7-D2 synthetic model is still inaccurate. It does not alter model weights, retrain, open TEST, ingest real data, or integrate with ScoreMosaic.

## Starting baseline

D3 starts after the verified merge of PR #39:

- `main`: `9843d86bbb0599a938336c823d73a0ca53efa8d3`
- post-merge CI #142: SUCCESS
- regression: 471/471 PASS
- D2 authoritative run: 40/40 epochs, 12,320 optimizer steps
- D2 best epoch: 20
- D2 best validation loss: `0.9379074645594616`
- D2 exact-sequence accuracy: `0.0`
- D2 token error rate: `0.847364947676063`
- D2 semantic/MusicXML validity: `1.0`
- D2 TEST samples exposed to model development: `0`

Accepted D2 model identity:

```text
run id                 14d63841254c03463ad76bbed83df95045742c23f71ad91d7b0c5dc19495a373
checkpoint SHA-256     239cf3dbdf80235bfc7e4a68fe5fecc03e8cd6fefc8a9ff6e27a2ca879ed5291
checkpoint state SHA   466cefcd40887cb0578b7bbc87c6a1b5f676dc0272ab5eee1142e45e7da8e17d
metrics SHA-256        e80b8aed13cc8c7aafae283f4306f1f60821fbf75faaaf568ddff7b132c318bd
verification SHA-256   6743425d42da77dfacef50388e879d45aa01f01b740cfd2deb381a55436500c3
```

## Data boundary

D3 is **validation-only** after the existing D1 whole-corpus integrity recheck.

```text
D1 integrity re-verification
        ↓
D3 manifest loader
        ├── TRAIN row → skip before artifact path/byte derivation
        ├── TEST row  → skip before artifact path/byte derivation
        └── VALIDATION → 51 families / 153 images only
```

D1 may hash TRAIN/TEST bytes for complete archive integrity, but the D3 diagnostic code receives no TRAIN or TEST sample artifact data. D3 reports `optimizer_steps = 0`.

## Diagnostic surface

For each of the 153 validation predictions, D3 records deterministic error counts for:

- token edit distance / token error rate;
- exact full-sequence match;
- exact measure match;
- time-signature accuracy;
- event-level edit rate;
- event-type accuracy (`note`, `rest`, `chord`);
- onset accuracy;
- duration/rhythm accuracy;
- pitch identity accuracy (`step`, `alter`, `octave`);
- visible accidental accuracy;
- rest-recognition accuracy;
- chord-size accuracy;
- missing and extra events.

The report also groups errors by ground-truth feature tags:

- meter (`2/4`, `3/4`, `4/4`);
- event type and mixed-event samples;
- duration (`whole`, `half`, `quarter`, `eighth`);
- chord size (`2`, `3`, `4`);
- accidental present/absent;
- degradation profile (`clean`, `light`, `medium`).

This is a diagnosis layer, not a new benchmark. Feature-bucket metrics use the existing validation set and therefore may guide development.

## Accepted source model gate

D3 refuses arbitrary checkpoints. Before decoding it must independently verify:

1. exact D2 checkpoint file SHA-256;
2. exact D2 verification artifact SHA-256 and canonical JSON;
3. D2 run/build/manifest/artifact-binding identities;
4. D2 best epoch and frozen model fingerprint;
5. strict checkpoint state loading;
6. exact model-state SHA-256;
7. finite model tensors;
8. D2 proof that TEST stayed sealed.

## Output

D3 writes only small hash-addressed evidence outside normal Git:

```text
<run-root>/<run-id>/
├── diagnostics-<sha256>.json
├── verification-<sha256>.json
└── COMPLETE
```

The diagnostics file may contain synthetic validation sample/family hash identities and per-sample error counts. It contains no PNG or MusicXML bytes.

## Closure gate

Stage 7-D3 may close only after:

1. diagnostic code and focused tests pass on the exact PR head;
2. full repository regression and compile checks pass in GitHub CI;
3. the real frozen corpus passes D1 re-verification;
4. the exact accepted D2 checkpoint and verification artifact pass the D3 source-model gate;
5. all 153 validation samples are decoded and diagnosed;
6. TRAIN artifact bytes are not exposed to diagnostics after D1;
7. TEST artifact paths/bytes are not exposed after D1;
8. optimizer steps remain zero;
9. diagnostics and verification artifact hashes are independently checked;
10. the resulting error map is used to choose the next small model/data improvement package;
11. explicit merge approval is obtained;
12. post-merge exact-main CI succeeds.

## Explicitly out of scope

D3 does not retrain, fine-tune, change architecture, change tokenizer vocabulary, alter the synthetic corpus, use real data, open the sealed TEST split, define production quality by itself, or integrate a model into ScoreMosaic.
