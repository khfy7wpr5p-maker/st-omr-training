# TR-POLY-04 — External Benchmark Harness

Status: research contract implementation

## Purpose

TR-POLY-04 creates one deterministic adapter surface for external OMR benchmarks so future Model A, Model B and Model C experiments can be compared against hash-identical benchmark evidence.

The harness is intentionally separate from model code, training loops and third-party evaluator implementations.

## Supported benchmark families

1. OLiMPiC synthetic 1.0
2. OLiMPiC scanned 1.0
3. GrandStaff-LMX
4. Muse OMR Benchmark

OLiMPiC and GrandStaff-LMX are represented as system-level LMX benchmark sources. Muse OMR Benchmark is represented as a score-level MuseScore source.

## What the harness freezes

Each benchmark source has a versioned specification containing:

- benchmark id;
- benchmark version;
- exact external dataset name/component identity;
- target format;
- declared train/validation/test split surface;
- system-level vs score-level evaluation unit.

Each admitted benchmark installation is bound to:

- the TR-POLY-03 registry-record SHA-256;
- a local data-artifact/tree SHA-256;
- a canonical dataset-manifest SHA-256;
- a canonical split-manifest SHA-256;
- an admission mode;
- an explicit research-override reference when applicable.

The resulting dataset/split hashes are converted directly into the `BenchmarkIdentity` frozen by TR-POLY-02.

## Manifest row contract

Every external sample row requires:

- `sample_id` — lowercase SHA-256;
- `family_id` — score/family identity used for leakage protection;
- `split` — train, validation or test;
- `image_relpath` — safe relative file path;
- `target_relpath` — safe relative file path;
- `system_id` — stable system/sample grouping identity.

The validator fails closed on:

- duplicate sample ids;
- duplicate image paths;
- duplicate target paths;
- path traversal;
- undeclared split use;
- family leakage across splits;
- malformed JSONL;
- empty manifests.

## Research override

TR-POLY-03 rights metadata remains unchanged.

`RESEARCH_OVERRIDE` exists only to allow explicitly approved scientific benchmark work when a registry record is not yet `INSTALL_PINNED` or remains `LICENSE_REVIEW_REQUIRED`.

A research override:

- must carry an explicit approval/reference string;
- must still hash-bind the exact local dataset tree/artifact;
- must still hash-bind the dataset manifest and split assignments;
- cannot pass `validate_commercial_evidence()`;
- must not be represented as production/commercial/shadow-readiness evidence.

This preserves provenance while allowing the approved research program to continue.

## Strict registry mode

`STRICT_REGISTRY` remains the stronger path. It requires:

- a matching TR-POLY-03 dataset record;
- `INSTALL_PINNED` state;
- evaluation permission;
- exact artifact SHA-256 equality.

This mode is suitable for later evidence that may contribute to commercial/shadow-readiness decisions.

## Dataset bytes

This PR does not vendor or commit OLiMPiC, GrandStaff, Muse, or any other external dataset bytes.

The harness can hash a local installation deterministically with `directory_tree_sha256()`. Symlinks are rejected so a benchmark identity cannot silently depend on files outside the declared root.

## Evaluation relationship

TR-POLY-04 does not implement TEDn or copy third-party evaluators. It only produces the deterministic benchmark identity and validated sample inventory that later evaluation runners will consume.

The required common metric surface remains owned by TR-POLY-02.

## Non-goals

TR-POLY-04 does not:

- train a model;
- create a checkpoint;
- change the V1 tokenizer;
- implement LMX V2;
- implement a Transformer;
- alter Stage 7/8 splits;
- open the existing sealed ST-OMR TEST set;
- change ScoreMosaic;
- promote any model to production or shadow authority.

## Next dependency

With TR-POLY-02, TR-POLY-03 and TR-POLY-04 present, the next critical package is TR-POLY-05: the versioned structured LMX-like polyphonic representation.
