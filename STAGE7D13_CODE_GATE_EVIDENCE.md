# Stage 7-D13 — pre-optimizer code-gate evidence

This file records the D13-4/D13-5 code surface that must pass exact-head CI and the external preflight before any optimizer is authorized.

## Frozen authoritative derivative dependency

```text
derivative build id
44f1932532fb511dfa59a164f94be6b899f3aa0594c0ac0a6f499a38e5fb5697

manifest SHA-256
8cfb87b5c6135be14b4c9ad488868c0edb0d37bb3bb18ad1b5e79d04fdf24f7b

artifact binding SHA-256
c42c1f69e21d61d3eefdcafc40dabf2f0fcd6ac2ceb4d5cf88d8e158246dd33e

TRAIN records       9840
VALIDATION records  1224
TOTAL records      11064
images             11062
labels             11064
TEST                   0
```

## Frozen future optimizer count

```text
batch size       16
epochs           10
batches/epoch   615

NoteHeadSet      6150 steps
RestSet          6150 steps
AccidentalSet    6150 steps
TOTAL           18450 steps
```

No optimizer step is executed by the derivative, model-construction, preflight, CI, or verifier code gates.

## D13-4 implementation surface

The branch contains:

- `stage7d13_verified_surface.py` — frozen authoritative derivative identity;
- `stage7d13_symbol_models.py` — three independent compact stride-4 center detectors, target encoding, frozen objective, decoder and metrics;
- `stage7d13_training.py` — deterministic persisted-derivative loader, sequential independent optimizers, read-only validation, minimum-validation-loss checkpoint selection, exact step enforcement and persistence;
- `stage7d13_run_verifier.py` — independent persisted-run verifier using `torch.load(..., weights_only=True)`, state-hash reproduction, metric/acceptance reproduction and optimizer-count verification;
- `stage7d13_authoritative_training.py` — exact repository/runtime-bound orchestration; independent verification precedes `verification.json` / `COMPLETE` persistence;
- `stage7d13_training_preflight.py` — full pre-optimizer scan of record/image/label hashes, family split isolation, target encodability/collision freedom and combined parameter budget.

## Fail-closed preflight requirements

Before model optimizers are created, the authoritative launcher must prove:

- exact D13 manifest/build identity;
- 9,840 TRAIN + 1,224 VALIDATION records;
- 11,064 labels and 11,062 content-addressed images;
- every referenced image and label SHA-256 matches persisted bytes;
- no TEST/unknown split enters the training surface;
- no family crosses TRAIN/VALIDATION;
- every NoteHead/Rest/Accidental target is encodable on the frozen stride-4 grid;
- no two same-specialist targets collide in one class-agnostic regression cell;
- each model remains below its individual cap and all three remain below the combined 4.5M parameter cap;
- repository identity and pinned runtime remain unchanged.

Any preflight failure leaves optimizer steps at zero.

## Training-run closure requirements

A completed authoritative run is not accepted merely because training exits normally. The independent persisted-run verifier must reproduce:

- exact 6,150 optimizer steps for each specialist / 18,450 total;
- exact checkpoint SHA-256 and each restored model-state SHA-256;
- exactly 10 epoch-history entries per specialist;
- best epoch selected by minimum validation loss, including the untrained epoch-0 baseline candidate;
- final state reproduces the selected minimum validation loss;
- frozen metric calculation and acceptance decision;
- TEST opened = false;
- safe checkpoint reload;
- persisted run/metrics/checkpoint/verification bindings.

Only after independent verification may the authoritative gate write `COMPLETE`. Metric failure may produce a verified completed run, but it does not satisfy D13 technical acceptance or authorize merge.

## Remaining gate before optimizer authorization

1. exact final branch HEAD CI: pinned runtime + full regression + compileall PASS;
2. fresh architecture/safety review with no unresolved P1/P2 blocker;
3. run the external D13 preflight on the accepted derivative bundle and require PASS;
4. only then authorize D13-6 external training on that exact reviewed HEAD.

Merge remains separately gated on the later D13 training results, independent persisted-run verification, closure evidence and explicit user merge approval.
