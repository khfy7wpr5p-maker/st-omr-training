# TR-POLY-09B8A — Native V2 Artifact Reload / Baseline Preflight

## Purpose

TR-POLY-09B8 begins the first real Native Polyphonic V2 quality baseline. The B6/B5/B7 execution chain already exists, but a real baseline may be materialized outside the repository and later mounted into an execution environment. Before B8A there was no independent reload boundary that reconstructed the validated `NativePolyV2DatasetBuild` from the persisted TR-POLY-09A directory.

B8A adds that missing boundary without changing the frozen TR-POLY-09A persistence schema, B4 training semantics, B5 checkpoint schema, B7 benchmark semantics, or TEST policy.

The reload path is:

```text
persisted native V2 root
(manifest.json + manifest.sha256 + build.json
 + TRAIN/VALIDATION targets + TRAIN/VALIDATION images)
        ↓
exact canonical metadata validation
        ↓
exact TRAIN/VALIDATION artifact-set validation
        ↓
per-artifact SHA-256 + V2 semantic roundtrip + PNG validation
        ↓
reconstructed NativePolyV2DatasetBuild
        ↓
hash-bound B8A preflight receipt
        ↓
B6 quality training can consume the verified build
```

## Persisted artifact contract

The accepted root remains the TR-POLY-09A layout:

```text
manifest.json
manifest.sha256
build.json
targets/<sha256>.json
images/<sha256>.png
```

Only TRAIN/VALIDATION target and image bytes may exist. The manifest still contains sealed TEST metadata, but TEST target/image bytes must not be present.

## Verification performed by B8A

`st_omr_training/poly_v2_dataset_reload.py` performs a fail-closed reload:

1. dataset root and metadata files must be regular non-symlink files/directories;
2. `manifest.json` must use the exact frozen native V2 fields and exact canonical encoding;
3. `manifest.sha256` must bind the exact manifest bytes;
4. `build.json` must use the exact frozen builder/source/target/TEST-policy metadata;
5. `targets/` and `images/` must contain exactly the hashes admitted for TRAIN/VALIDATION — no missing or extra entries;
6. every admitted target is SHA-256 verified, parsed as canonical Polyphonic V2, losslessly round-tripped, and checked against representation hash, token count and polyphony profile;
7. every admitted image is SHA-256 verified as a bounded grayscale PNG with exact dimensions;
8. the deterministic `NativePolyV2DatasetBuild` identity is reconstructed and revalidated;
9. the existing TR-POLY-09A root verifier is run again after reconstruction;
10. a deterministic preflight receipt records manifest/build identity and TRAIN/VALIDATION/sealed-TEST sample IDs.

## TEST safety

B8A does not open TEST artifacts.

The exact directory-set check requires the artifact directories to contain only TRAIN/VALIDATION hashes. If a file matching a sealed TEST hash — or any other extra artifact — appears, reload fails before that extra file is parsed or admitted.

The preflight receipt therefore remains explicit:

```text
test_artifact_bytes_accessed = false
production_authority = false
```

## What B8A does not prove

A green B8A implementation proves the reload/preflight mechanism. It does **not** prove that a real quality corpus has already been admitted, that an external dataset has acceptable rights/provenance, that model quality improved, or that B7 metrics have been produced.

The first real baseline still requires one physically available, admitted Native V2 root whose provenance/license conditions are already accepted, plus explicit B7 VALIDATION descriptors. Dataset values must not be invented from repository fixtures, teacher corrections, or unlabeled Drive folders.

## Current B8 execution gate

At the start of B8A, protected `main` contains B1–B7 and B7 was merged as PR #161. The repository itself does not contain a real admitted baseline corpus. Repository tests contain synthetic fixtures only and are not quality evidence.

Therefore the correct continuation is:

```text
B8A reload/preflight implementation + CI
        ↓
locate/provide one admitted real Native V2 artifact root
        ↓
record exact manifest SHA-256 + build ID + provenance
        ↓
freeze B4/B6 recipe and explicit B7 descriptors
        ↓
B6 TRAIN-only multi-epoch quality run
        ↓
B5 checkpoint verification
        ↓
B7 complete VALIDATION execution
        ↓
B2/B3 failure profile
        ↓
P09C evidence-driven refinement
```

If no admitted root is available, B8 remains artifact-blocked. No model accuracy number may be fabricated and sealed TEST remains untouched.
