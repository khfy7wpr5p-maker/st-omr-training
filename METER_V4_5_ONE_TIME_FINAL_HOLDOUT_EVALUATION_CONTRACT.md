# Meter V4-5 — One-Time Independent Final Holdout Evaluation Contract

Status: **PREREGISTERED / NO FINAL INFERENCE YET**

## Purpose

V4-5 is the first and only model evaluation on the frozen V4-3/V4-4 150-family final holdout. It evaluates the already-frozen V4-2 numerator specialist. It is not a training, tuning, threshold-search, checkpoint-selection, calibration, or runtime-integration stage.

## Frozen parents

V4-4 completion is accepted only when all of the following exact bindings match:

- frozen selection SHA-256: `4335a48a091912ba422c16d8fcbaaa7bbf5f7a0a43f088146a50a3e02e3ed7dc`
- V4-4 image binding SHA-256: `73c932e8cbb55c9b57a482bb05fbf7e033cdd375f4929e34851cca01f9c1cd66`
- V4-4 bbox manifest SHA-256: `0242fe99d39393a78d5b5d69ed95dbfc6d65975a3bef21ec43368386cdfdca70`
- V4-4 completion receipt file SHA-256: `24aea7f900128858a1fecdb5ed01427161e1ba02768da4a191be3e70226d2548`
- V4-4 human-review evidence schema: `st-omr-meter-v4-4-human-visual-review-evidence-v1`
- human review status: `PASS`
- holdout cardinality: 150 unique families, classes 50/50/50

V4-2 candidate is accepted only when all of the following exact bindings match:

- V4-2 result file SHA-256: `bf32ccd4bf9512ad6a39b3e34ac2b7b0ab708f140d8e8bcf48721fab28cab04b`
- checkpoint file SHA-256: `2dc820bc0cbadf5db90a7ddee7f5a9daba06e546dcae1da560d1ac9718e3692a`
- model-state SHA-256: `0ca4831729ba4723d6aac71c73dc24501569492ebb72cf97e6f4bcc33596ead1`
- configuration fingerprint: `414625aa8d2617cf89b324263205e8bbdbc92536a081e8b245823325bd78a4ba`
- classes: numerator `2`, `3`, `4`

Any mismatch fails closed before the checkpoint is deserialized.

## Frozen representation policy

The V4-2 model consumes one gray8 64x64 numerator crop. V4-5 applies the already-frozen V4-0 geometric policy directly in each original `image.png` pixel coordinate system; no holdout-derived parameter is introduced:

1. Start from the human-approved full-meter bbox `(x, y, w, h)` in original integer pixels.
2. Horizontal padding = 15% of full-meter bbox width.
3. Vertical padding = 5% of full-meter bbox height.
4. Numerator region ends at 50% of full-meter bbox height.
5. Clamp derived bounds to the original image.
6. Convert the source PNG deterministically to grayscale (`L`).
7. Crop the derived numerator region.
8. Aspect-fit with bilinear resampling onto a white 64x64 canvas, centered exactly as V4-0 does.
9. Convert to float32 ink tensor `(255-gray)/255`, shape `[1,64,64]`.

No crop size, margin, threshold, interpolation method, or normalization may be changed after final inference begins.

## One-shot barrier

V4-5 has two phases.

### Preflight — checkpoint remains closed

Preflight must verify all immutable V4-4 evidence, all 150 current image hashes and bbox-file hashes, bbox validity, class/family cardinality, V4-2 result bytes, and the candidate checkpoint file hash. It may decode images and construct numerator tensors, but it must not deserialize the checkpoint or invoke the model.

### Locked evaluation — exactly once

Immediately before checkpoint deserialization, the runner creates an exclusive one-shot lock bound to all accepted input hashes and the preregistered protocol. If the lock already exists, the runner refuses to evaluate. The lock remains after success or failure. A process crash after lock creation therefore does not silently authorize a second final-holdout attempt.

Only after the lock exists may the exact trusted-hash checkpoint be opened using the bounded PyTorch `weights_only=True` loader. The frozen model architecture is instantiated, the exact state dict is loaded, and the state hash is verified before inference.

Exactly 150 forward-pass records are permitted. No optimizer, loss backward pass, training mode, augmentation, test-time augmentation, ensemble, retry, model selection, or prediction-dependent preprocessing is permitted.

## Frozen metrics and decision gate

The result records:

- record count and unique family count;
- class counts;
- accuracy;
- macro-F1;
- recall for numerator classes 2, 3, and 4;
- fixed 3x3 confusion matrix ordered `[2,3,4]`;
- per-record true/predicted class and logits/probabilities for audit;
- exact input/evidence/checkpoint/repository bindings;
- `inference_count=150`.

The final gate was fixed before the result is observed:

- accuracy **>= 0.90**;
- macro-F1 **>= 0.90**;
- recall(2) **>= 0.90**;
- recall(3) **>= 0.90**;
- recall(4) **>= 0.90**.

If every condition passes, decision = `FINAL_HOLDOUT_PASS`. Otherwise decision = `FINAL_HOLDOUT_FAIL`. No threshold may be relaxed after observing results.

Neither decision authorizes production promotion. A PASS only allows the next bounded integration/promotion review. A FAIL closes this candidate for production; this final holdout must not be reused to tune or select a replacement model.

## Immutable output

A successful run writes a fresh output directory atomically containing `result.json` and `COMPLETE`. Existing output, partial output, or lock causes fail-closed behavior. Result JSON is canonical ASCII JSON and `COMPLETE` binds its SHA-256.

## Forbidden surfaces

V4-5 must keep all of the following closed:

- training and optimizer updates;
- tuning/calibration/threshold search;
- checkpoint replacement or selection;
- sealed TEST surface;
- runtime/Resolver connection;
- production promotion;
- network-dependent model behavior;
- writing to any selected holdout sample, image, or bbox file.

The final holdout is read-only in V4-5.

## CI / negative security coverage

CI uses synthetic fixtures only; it must never open the real final holdout or real candidate checkpoint. Tests cover at minimum wrong parent hashes, wrong human-review evidence, wrong checkpoint hash, missing/corrupt checkpoint, malformed/missing bbox, image mutation, class/family mismatch, symlink/path escape, deterministic crop/metric computation, existing lock/output/partial output, result tampering, and one-shot second-run refusal.
