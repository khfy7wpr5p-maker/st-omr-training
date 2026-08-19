# Meter Real Inference v1 Contract

## Scope

This stage is the first ST-OMR runtime boundary allowed to execute an audited real frozen checkpoint. It is deliberately limited to the D11 Meter checkpoint used as a temporary Meter Presence bridge.

```text
verified normalized source image
        -> accepted runtime geometry
        -> historical D9/D10/D11 Meter ROI reconstruction
        -> exact ROI PNG SHA verification
        -> exact D11 checkpoint SHA audit
        -> weights-only CPU state load
        -> frozen D11 MeterRefiner runtime mirror
        -> class probabilities + bbox
        -> Presence score = 1 - P(none)
        -> provenance-bound inference fingerprint
```

## Frozen identities

- D11 Presence-bridge checkpoint SHA-256: `cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3`
- historical ROI profile: `measure-start-meter-roi-v1`
- input: grayscale 256x192 PNG, converted exactly as historical D11 (`1 - uint8/255`)
- classes: `none | 2/4 | 3/4 | 4/4`
- runtime: CPU inference only, `eval()` + `torch.inference_mode()`

## Fail-closed requirements

1. ROI bytes must match the `HistoricalMeterRoiArtifactV1` SHA and profile.
2. Checkpoint must first pass `audit_presence_d11_checkpoint_v1` including exact SHA, regular-file, key/shape and finite-tensor checks.
3. Runtime architecture must accept the audited state strictly.
4. Input must decode as exact grayscale 256x192 PNG.
5. Logits, bbox and probabilities must be finite.
6. The output fingerprint binds ROI SHA, checkpoint SHA, probabilities, bbox, x-center and runtime architecture identity.
7. 10/10 repeated inference with the same audited state and ROI must be identical.

## Explicitly out of scope

- no training or optimizer access;
- no threshold tuning;
- no TRAIN/VALIDATION/TEST dataset reads;
- no digit 2/3/4 checkpoint execution in this stage;
- no Meter composition;
- no Resolver wiring;
- no production promotion;
- no claim that CI synthetic-state tests equal private-checkpoint execution.

## Closure meaning

A green CI closes the runtime implementation and determinism contract only. A claim of **real checkpoint execution** additionally requires running `infer_presence_from_checkpoint_v1` with the exact private checkpoint file and a real reconstructed historical ROI in an authorized runtime. Private checkpoint binaries remain outside GitHub and CI.
