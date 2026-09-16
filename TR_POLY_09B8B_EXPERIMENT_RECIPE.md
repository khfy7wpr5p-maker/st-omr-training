# TR-POLY-09B8B — First Real Native V2 Experiment Recipe Freeze

Updated: 2026-09-16

## Purpose

B8B freezes the first real Native Polyphonic V2 baseline recipe before any quality result is observed. It is a control-plane gate, not model training and not benchmark evidence.

The package binds one B8A-verified persisted dataset to the already merged B4/B6/B7 contracts. Any change to dataset identity, population, descriptor metadata, model/trainer configuration, batch policy, or decode bound changes the recipe fingerprint.

## Frozen baseline defaults

The default first-baseline recipe reuses existing merged contracts:

- model: `FROZEN_POLY_2D_CONFIG`;
- quality trainer: `FROZEN_POLY_2D_QUALITY_CONFIG`;
- epochs: `8`;
- optimizer: AdamW;
- scheduler: none;
- maximum optimizer steps: `8192`;
- validation: complete read-only VALIDATION after every epoch;
- checkpoint selection: minimum mean VALIDATION loss, earliest exact tie;
- B6 batch size: maximum admitted batch `8`;
- TRAIN selection: complete population only;
- VALIDATION selection: complete population only;
- B7 decode bound: at least the longest admitted VALIDATION target and no greater than the model target boundary;
- TEST: sealed;
- production authority: false.

B8B does not invent a new optimizer or architecture. It combines the already merged contracts into one dataset-specific, descriptor-specific identity.

## Hash-bound evidence

`NativePolyV2ExperimentRecipe` binds:

- exact repository SHA;
- B8A preflight receipt fingerprint;
- dataset manifest SHA-256;
- deterministic dataset build ID;
- exact TRAIN sample IDs and family IDs;
- exact VALIDATION sample IDs and family IDs;
- deterministic TRAIN/VALIDATION batch-size plans;
- model profile SHA-256;
- quality trainer profile SHA-256;
- Native V2 materialization fingerprint;
- B6 execution profile fingerprint;
- complete B7 descriptor-manifest SHA-256;
- B7 validation split-manifest SHA-256;
- benchmark identity SHA-256;
- exact max decode steps;
- required optimizer-step count.

The final recipe has its own canonical SHA-256 fingerprint.

## Fail-closed rules

Freeze fails before training if:

- B8A receipt population differs from the loaded build;
- TRAIN and VALIDATION samples or families overlap;
- B7 descriptors do not cover the exact VALIDATION population;
- descriptor family identity differs from the dataset manifest;
- batch size exceeds the B6 boundary;
- decode bound would truncate an admitted VALIDATION target;
- decode bound exceeds the model target boundary;
- the configured optimizer-step ceiling cannot cover every TRAIN batch for every frozen epoch;
- TEST bytes or production authority would be implied.

## Artifact blocker

The repository still does not contain the first admitted real Native V2 corpus. Therefore B8B can define and test the freeze contract, but it cannot emit the real dataset-specific recipe fingerprint until a B8A-verified corpus and complete B7 descriptor manifest are physically available.

Synthetic unit-test fixtures remain regression evidence only and must not be reported as model quality.

## Next execution order

```text
real admitted persisted Native V2 root
        ↓
B8A reload/preflight
        ↓
complete explicit B7 descriptor manifest
        ↓
B8B freeze exact experiment recipe
        ↓
B6 full TRAIN-only candidate creation
        ↓
B5 verified checkpoint
        ↓
B7 full VALIDATION
        ↓
B2/B3 failure profile
        ↓
P09C evidence-driven refinement
```

TEST remains sealed throughout B8.
