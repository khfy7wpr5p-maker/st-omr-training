# TR-POLY-09B1 — Bounded Polyphonic V2 Greedy Inference

Status: implementation package; common benchmark and sealed TEST remain closed.

## Purpose

TR-POLY-09A made native explicit Polyphonic V2 TRAIN/VALIDATION targets available to the existing bounded 2D Transformer training/checkpoint chain. The remaining blocker before meaningful end-to-end comparison is free-running inference: the model must produce its own V2 sequence from an image without receiving the gold target prefix.

TR-POLY-09B1 adds that narrow capability.

It does not claim accuracy improvement, benchmark success, TEST performance, ScoreMosaic readiness, or production authority.

## Inference path

```text
verified grayscale tensor
        ↓
TinyPoly2DTransformer.encode_images()
        ↓
full row × column visual memory (computed once)
        ↓
BOS-only decoder prefix
        ↓
deterministic greedy argmax next-token loop
        ↓
EOS / invalid-control / bounded max-step termination
        ↓
strict V2 detokenization + parser
        ↓
canonical PolyScore OR explicit invalid/abstain evidence
```

The gold target is never supplied to the free-running loop.

## Model compatibility

`TinyPoly2DTransformer` gains `decode_from_memory(memory, decoder_input_ids)` so the image encoder does not rerun on every autoregressive token.

This is an additive execution surface only:

- no parameter is added or removed;
- the state-dict layout is unchanged;
- `POLY_2D_TRANSFORMER_VERSION` remains unchanged;
- the model configuration fingerprint remains unchanged;
- existing TR-POLY-08B checkpoint artifacts remain structurally compatible;
- the frozen V2 representation and tokenizer remain unchanged.

`forward(images, decoder_input_ids)` now delegates to the same decoder-from-memory path after encoding the image, preserving teacher-forced behavior.

## Frozen first decoder

Inference version:

`st-omr-poly-2d-greedy-inference-v1`

Policies:

- search: deterministic greedy argmax;
- tie behavior: lowest token id selected by the pinned argmax behavior;
- start token: exactly one `BOS`;
- termination: emitted `EOS` or explicit bounded failure;
- generated `PAD`: fail closed;
- generated `BOS`: fail closed;
- no silent control-token masking;
- no fabricated EOS when the step budget is exhausted;
- strict canonical V2 parse after EOS;
- one image per call in v1;
- CPU-only under the repository-pinned deterministic runtime;
- model state must be byte-identical before/after inference.

Beam search, constrained semantic search, sampling and repair heuristics are deliberately excluded so the first common benchmark can distinguish model behavior from search-algorithm behavior.

## Result states

The bounded result surface is:

- `eos_valid` — EOS reached and the complete token sequence reconstructs a canonical V2 `PolyScore`;
- `eos_semantic_invalid` — EOS reached but strict V2 reconstruction fails;
- `invalid_control_token` — the model emits batching-only `PAD` or a second `BOS`;
- `max_steps` — the fixed decode budget is exhausted without EOS.

Only `eos_valid` contains a parsed score and canonical representation SHA-256. All other states contain an explicit stable error code and no invented musical content.

## Evidence identity

Every inference result binds:

- exact model-state SHA-256;
- exact model-profile SHA-256;
- exact inference-profile SHA-256;
- V2 tokenizer fingerprint;
- representation/tokenizer versions;
- pinned PyTorch runtime;
- exact maximum decode-step policy.

When inference starts from `load_and_verify_poly_2d_checkpoint(...)`, the result additionally binds:

- checkpoint file SHA-256;
- metadata file SHA-256;
- receipt SHA-256;
- checkpoint metadata fingerprint;
- dataset-manifest SHA-256;
- preprocessing fingerprint;
- trainer-profile SHA-256;
- training provenance SHA-256;
- model-registry record fingerprint;
- repository SHA.

This provides the model/checkpoint side of the P09B-0 candidate freeze. The common benchmark identity itself remains a later P09B-2/P09B-3 binding.

## Determinism and mutation guard

The decoder:

1. records exact model-state SHA-256;
2. switches temporarily to evaluation mode;
3. runs under `torch.inference_mode()`;
4. restores the caller's original train/eval mode;
5. recomputes model-state SHA-256;
6. fails if any parameter/buffer state changed.

The encoded image memory is reused for the complete token loop.

## Semantic safety

TR-POLY-09B1 does not repair malformed outputs.

In particular it does not:

- infer a missing onset;
- infer a missing duration;
- assign a missing voice;
- assign a missing staff;
- close an incomplete JSON/token structure;
- add EOS when the model failed to emit it;
- reinterpret PAD as content;
- collapse a chord and independent same-onset voice into one event.

A malformed result is evidence about the candidate, not an invitation to silently normalize the benchmark output.

## TEST sealing

This package contains no dataset loader and no TEST path. The verified checkpoint wrapper only verifies an already-provided checkpoint directory and consumes one caller-provided image tensor.

TR-POLY-09B1 therefore does not authorize:

- TEST artifact reads;
- benchmark-driven checkpoint selection;
- architecture/hyperparameter tuning;
- registry promotion;
- ScoreMosaic integration;
- production inference.

## Regression coverage

Tests verify:

- decoder-from-precomputed-memory equals the existing forward path;
- invalid memory geometry fails closed;
- BOS-only scripted greedy decoding reconstructs an exact canonical V2 score;
- generated PAD/BOS fails explicitly rather than being masked;
- EOS with malformed V2 content becomes semantic-invalid evidence;
- max-step exhaustion does not fabricate EOS;
- same model/input/profile yields identical token/evidence identities;
- inference does not mutate model state;
- inference-profile identity changes when decode bounds change;
- batch and invalid-resource bounds fail closed;
- verified-checkpoint inference binds checkpoint, dataset, preprocessing, provenance and repository identities.

## Exit condition

TR-POLY-09B1 is complete only when exact-head CI is green and the package is merged to protected `main`.

Completion means only that a hash-bound checkpoint can support deterministic free-running V2 inference with explicit semantic failure behavior. It does not mean the model is accurate.

## Next gate

TR-POLY-09B2 — deterministic common metric/adaptor implementation.

The next package should connect free-running prediction/reference pairs to the frozen TR-POLY-02 evaluation contract, implement only missing versioned metrics/adapters, and keep TEST sealed. After that, TR-POLY-09B3 may run the first common VALIDATION comparison by 1/2/3/4+ voice strata.
