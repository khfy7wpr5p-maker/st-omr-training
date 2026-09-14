# TR-POLY-09B7 — Hash-Bound Quality VALIDATION Execution

Updated: 2026-09-14

## Purpose

B7 connects one verified TR-POLY-09B5 quality checkpoint to the frozen evaluation chain:

```text
native V2 VALIDATION artifacts
        +
explicit descriptor metadata
        +
verified B5 quality checkpoint
        ↓
B1 free-running greedy inference
        ↓
B2 per-sample metric reports
        ↓
B3 candidate-level VALIDATION aggregation
```

It is an execution/control-plane package. It does not train, tune, rank, promote or deploy a model.

## Exact VALIDATION population

B7 accepts the complete native V2 VALIDATION population only. There is no `max_samples`, random sampling or prefix-selection argument in the benchmark execution API.

Every `BenchmarkSampleDescriptor` must:

- use `split="validation"`;
- match an exact native V2 `sample_id`;
- carry the exact manifest `family_id`;
- appear exactly once;
- cover the whole VALIDATION population.

TEST descriptors are rejected before dataset-root or checkpoint access.

## Descriptor metadata is explicit

TR-POLY-02 defines the descriptor schema but does not define a universal algorithm for deriving all complexity densities or robustness labels.

B7 therefore does **not** guess them. In particular it does not silently label an image `clean` when robustness provenance is absent.

The caller supplies explicit:

- `PolyphonicComplexityProfile`;
- `RobustnessBucket`.

B7 binds those declared values to the exact sample and artifact identities.

## Split-manifest identity

`native_poly_v2_validation_split_manifest_sha256(...)` hashes, in canonical sample-id order:

- B7 execution version;
- descriptor policy;
- full-population selection policy;
- native dataset manifest SHA-256;
- native dataset build ID;
- sample ID and family ID;
- explicit complexity descriptor;
- explicit robustness bucket;
- target SHA-256;
- canonical representation SHA-256;
- image SHA-256;
- source width/height;
- target token count.

The resulting SHA becomes `BenchmarkIdentity.split_manifest_sha256`.

Changing a robustness label, complexity value, image, target, family assignment or selected population changes the benchmark identity.

## Checkpoint gate

Before inference B7 requires the B5 loader to verify the checkpoint. It then requires:

- checkpoint dataset manifest == native V2 build manifest;
- checkpoint preprocess/materialization fingerprint == the current native V2 materialization profile;
- checkpoint model-profile fingerprint == loaded model config;
- one explicit decode bound inside the verified model boundary.

The checkpoint is loaded once. B1 inference is then run read-only for each VALIDATION sample and every prediction identity is rebound to the same verified B5 artifact hashes/provenance.

## Inference policy

One `max_decode_steps` value is used for the complete execution. It participates in the B1 inference profile and therefore the candidate identity.

Invalid output, malformed EOS, invalid control tokens and decode exhaustion remain explicit B1 outcomes. B2 retains them in the metric population rather than dropping them.

## Metric and aggregation boundary

B7 does not redefine metrics.

It calls the frozen surfaces directly:

- `evaluate_poly_v2_validation_sample(...)` from B2;
- `aggregate_poly_v2_validation_reports(...)` from B3.

Current numeric surface remains 11 metrics. Five required TR-POLY-02 metrics remain explicitly unsupported:

- `musicxml_validity`;
- `tedn`;
- `notehead_stem_f1`;
- `beam_relation_f1`;
- `tie_relation_f1`.

Therefore a B7 run can produce genuine VALIDATION evidence while `full_metric_contract_ready` and common-comparison/promotion gates remain false.

## Result evidence

`PolyV2ValidationExecutionResult` binds:

- exact `BenchmarkIdentity`;
- checkpoint / metadata / receipt hashes;
- checkpoint metadata fingerprint;
- dataset manifest and build ID;
- VALIDATION split-manifest SHA;
- materialization fingerprint;
- B1 candidate and inference-profile identities;
- fixed decode bound;
- sorted B2 sample reports;
- B3 aggregate fingerprint;
- `validation_benchmark_evidence=true`;
- `test_split_accessed=false`;
- `production_authority=false`.

## Claims explicitly not granted

A green B7 merge means the repository can execute an admitted quality checkpoint over one exact hash-bound VALIDATION benchmark. It does **not** mean:

- the first external/full quality corpus has already been trained and measured;
- the model is accurate enough for deployment;
- the five missing metrics are available;
- a winning candidate exists;
- TEST may be opened;
- ScoreMosaic production authority exists.

## Next gate

After B7 is green, execute a real B6-produced quality checkpoint against a rights/admission-valid VALIDATION corpus with an explicit descriptor manifest. Use that evidence to choose P09C changes. Keep TEST sealed and admit the five missing metric surfaces separately before any complete comparison/promotion claim.
