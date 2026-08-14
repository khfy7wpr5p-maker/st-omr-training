# Stage 7-D3 — Validation Error Diagnostics

Status: **closed — PR #40 merged; post-merge main CI #146 SUCCESS**.

Stage 7-D3 diagnosed why the accepted Stage 7-D2 synthetic model remained inaccurate. It altered no model weights, performed no retraining, opened no TEST data, ingested no real data, and integrated nothing into ScoreMosaic.

## Starting baseline

D3 started after the verified merge of PR #39:

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

D3 was **validation-only** after the existing D1 whole-corpus integrity recheck.

```text
D1 integrity re-verification
        ↓
D3 manifest loader
        ├── TRAIN row → skip before artifact path/byte derivation
        ├── TEST row  → skip before artifact path/byte derivation
        └── VALIDATION → 51 families / 153 images only
```

D1 could hash TRAIN/TEST bytes for complete archive integrity, but the D3 diagnostic code received no TRAIN or TEST sample artifact data. D3 reported `optimizer_steps = 0`.

## Authoritative external result

```text
run id                    22b7d63f5112fb9d41fa72d502c7a3648781d692949bedf5fbbad8142e910ab7
diagnostics SHA-256       b5843f896a2f75f8c0b111a8d1dd562a74b15cf67d48c0d4e1dfa8655ed41a6b
verification SHA-256      558fb0a6e0bfe7e7f461361773a9f8a08b48c5dc4613bd1a3d3a73da7e5186e9
validation samples        153 / 51 families
TRAIN diagnostic exposure 0
TEST diagnostic exposure  0
optimizer steps            0
```

Aggregate validation diagnostics:

```text
exact sequence accuracy      0.0000000000
token error rate             0.8473649477
measure exact accuracy       0.0049019608
meter accuracy               0.3014705882
event error rate             1.0348047538
event type accuracy          0.1799660441
onset accuracy               0.1494057725
duration accuracy            0.1842105263
pitch identity accuracy      0.0000000000
display accidental accuracy  0.0000000000
chord-size accuracy          0.0000000000
rest recognition accuracy    0.6563467492
reference events             3534
predicted events             2601
missing events               1200
extra events                 267
```

All 51 validation families contained clean/light/medium derivatives, and every family's three variants produced identical diagnostic counts. Aggregate degradation buckets were also exactly identical. The dominant failure was therefore not degradation sensitivity alone.

## Diagnostic surface

For each of the 153 validation predictions, D3 recorded deterministic error counts for:

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

The report also grouped errors by ground-truth feature tags:

- meter (`2/4`, `3/4`, `4/4`);
- event type and mixed-event samples;
- duration (`whole`, `half`, `quarter`, `eighth`);
- chord size (`2`, `3`, `4`);
- accidental present/absent;
- degradation profile (`clean`, `light`, `medium`).

## Accepted source model gate

D3 refused arbitrary checkpoints. Before decoding it independently verified:

1. exact D2 checkpoint file SHA-256;
2. exact D2 verification artifact SHA-256 and canonical JSON;
3. D2 run/build/manifest/artifact-binding identities;
4. D2 best epoch and frozen model fingerprint;
5. strict checkpoint state loading;
6. exact model-state SHA-256;
7. finite model tensors;
8. D2 proof that TEST stayed sealed.

## Output

D3 wrote only small hash-addressed evidence outside normal Git:

```text
<run-root>/<run-id>/
├── diagnostics-<sha256>.json
├── verification-<sha256>.json
└── COMPLETE
```

The diagnostics file contains synthetic validation sample/family hash identities and per-sample error counts, but no PNG or MusicXML bytes.

## Closure evidence

D3 closure requirements are complete:

1. diagnostic code and focused tests passed on exact PR head `c25caddeaa897df5eeaad545e68f51aafc19c1f6`;
2. exact-head CI #145 passed with 483/483 tests;
3. real frozen corpus passed D1 re-verification;
4. exact accepted D2 checkpoint and verification artifact passed the D3 source-model gate;
5. all 153 validation samples were decoded and diagnosed;
6. TRAIN artifact bytes were not exposed to diagnostics after D1;
7. TEST artifact paths/bytes were not exposed after D1;
8. optimizer steps remained zero;
9. diagnostics/verification hashes were independently checked after Drive persistence;
10. the error map selected **specialist musical-task decomposition** as the next architecture axis;
11. explicit merge approval was obtained;
12. PR #40 merged as `168c03755f0e06e8042fc0a391a357c71c6288fe`;
13. post-merge main CI #146 succeeded with 483/483 tests and compile checks.

## Accepted decision

D3 rejects a simple "more epochs" response. Broad failures across pitch, rhythm, event type, chord grouping and event completeness require decomposition into small specialist perception tasks with deterministic musical fusion and validation.

The follow-on contract is Stage 7-D4 — Specialist OMR Architecture Contract.

## Explicitly out of scope

D3 did not retrain, fine-tune, change architecture, change tokenizer vocabulary, alter the synthetic corpus, use real data, open the sealed TEST split, define production quality by itself, or integrate a model into ScoreMosaic.