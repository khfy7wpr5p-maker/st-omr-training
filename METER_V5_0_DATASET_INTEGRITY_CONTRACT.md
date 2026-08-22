# METER V5-0 — METER_V2_1500 Dataset Integrity Contract

Status: **HOLD — NOT TRAINING READY**

This stage is a data-only safety gate. It must not train, tune, deserialize a
checkpoint, run inference, open sealed TEST, connect runtime/Resolver, or
authorize production promotion.

## Frozen target

`TEST/METER_V2_1500`

Expected composition:

| Meter | train | val | final_holdout | total |
|---|---:|---:|---:|---:|
| 2/4 | 400 | 50 | 50 | 500 |
| 3/4 | 400 | 50 | 50 | 500 |
| 4/4 | 400 | 50 | 50 | 500 |
| total | 1200 | 150 | 150 | 1500 |

Every row must be a distinct sample/folder. Family identity is the
contamination boundary: one `FamilyId` may not appear in more than one class or
split.

## Manifest schema

Canonical columns, in order:

`Split,Meter,FamilyId,SampleId,Folder,SourceImage,SourceSemantic,SourceAgnostic,SplitRank`

`SelectionRank` is not accepted as a frozen V5 schema alias. Inputs may be
parsed for audit, but the gate remains HOLD until all three manifests use
`SplitRank`.

## Historical contamination boundary

The complete METER_V1 historical family surface is frozen in
`evidence/METER_V1_HISTORICAL_FAMILIES.txt`.

It contains **325 unique families**, derived from
`METER_V1/00_AUDIT/admission_template.csv`:

- source CSV SHA-256:
  `9eb03e743ab1dc2e99dd3d1a8858dab7136bd1d13a008aab7daf1858f0aef202`
- sorted 325-family list SHA-256:
  `3231134495c3993b9d0d17355c8758bff2b879513289baad62d3dec03b641fc9`

The consumed V4-5 final-holdout set of 150 families is a subset of this
325-family historical blacklist. No V5 train, validation, or final-holdout row
may overlap the 325-family blacklist.

## Source-domain policy

The preferred V5 repair is now **single-domain package_ab-only** for all three
classes, not an AA/AB mixture.

A single source domain is acceptable because it is identical across classes:
package identity cannot predict the class when 2/4, 3/4, and 4/4 are all 100%
`package_ab`. Under this plan the class-to-class source share gap is exactly
`0.0`.

The earlier whole-archive AA/AB inventory is therefore **not required** unless a
future mixed-domain dataset is deliberately proposed.

Source domain is taken from the actual `SourceImage`/`SourceSemantic`/
`SourceAgnostic` path, never from a folder-name or `FamilyId` prefix.

## Current Drive audit

Current Drive manifests are structurally 500/class with 400/50/50 splits, but
are not the intended package_ab-only rebuild.

Observed blockers:

1. 4/4 schema drift: `SelectionRank` instead of `SplitRank`.
2. `ab_000102539` spans 2/4 train and 3/4 final_holdout.
3. Actual source paths are:
   - 2/4: `package_aa=500`
   - 3/4: `package_aa=500`
   - 4/4: `package_aa=24`, `package_ab=476`

Therefore the existing Drive surface remains **HOLD** and must not be patched
in place.

Current Drive manifest SHA-256 values:

- 2/4:
  `688aa6c229eb01cb88a58d9c2bd4225f47af48475a7c72b3d737922016a9d1ae`
- 3/4:
  `6db58cbacc48b41d46a856618432a6d134ff5b4cb6344fc5b7cc06b87b40d5ed`
- 4/4:
  `5d103cc4dd53a4648a776caaae62b752cd3debbb6bbfa0a35b6016e1fc4db3c2`

## Package_ab-only feasibility gate

The supplied `MASTER_INDEX.tsv` was audited read-only before selector
implementation.

Frozen source index:

- row count: `87678`
- SHA-256:
  `03fe74ff13e12c9b4c4500083812240e90c88ecf17df16d495ebe9d8f6b1ef3e`

Candidate policy:

- `Package=ab`
- `Complete=1`
- exactly one of `Meter2_4`, `Meter3_4`, `Meter4_4` equals `1`
- no clef restriction
- all source paths must resolve under `package_ab`
- exclude all 325 historical METER_V1 families
- exclude any family eligible for more than one target meter class
- choose at most one sample per family

After historical blacklist and global cross-meter ambiguity exclusion, clean
unique-family capacity is:

| Meter | clean package_ab families |
|---|---:|
| 2/4 | 3701 |
| 3/4 | 6216 |
| 4/4 | 725 |

Therefore the 500/500/500 package_ab-only target is **feasible**.

The exact feasibility evidence is frozen in
`evidence/METER_V5_0_PACKAGE_AB_SELECTION_FEASIBILITY.json`.

## Deterministic selector

The selector is implemented in:

- `st_omr_training/meter_v5_0_package_ab_selector.py`
- `tools/meter_v5_0_package_ab_selector.py`

Frozen seed:

`st-omr-meter-v5-0-package-ab-selector-v1`

Rules:

1. fail if required index columns are missing;
2. fail if a `Package=ab` row points outside actual `package_ab` source paths;
3. exclude the complete 325-family historical blacklist;
4. exclude cross-meter ambiguous families globally;
5. deterministically choose one sample per remaining family;
6. deterministically choose exactly 500 families per class;
7. assign exactly 400 train + 50 val + 50 final_holdout per class;
8. require 1500 globally unique families;
9. write canonical `SplitRank` manifests only;
10. refuse an existing output directory.

The dry-run on the frozen index + 325-family blacklist produced:

- blacklist overlap: `0`
- cross-class family overlap: `0`
- cross-split family overlap: `0`
- source-domain share gap: `0.0`

Expected deterministic manifest SHA-256 values:

- 2/4:
  `d07ca3d0f7104ac1e5ed551886d80f5971da50b19ef65345c1fd6fa5ebbfb38e`
- 3/4:
  `5509bed3ba11dccbed7c277e90fb5e39e9ae6890bb7f460f0f24e41bb16bf2e8`
- 4/4:
  `cb8d036d1f0629eb6a14dbd57c887a5cec0d405d0e668ca403af8901080adc22`

These hashes are selection evidence only. They do not authorize bbox or
training.

## Repair order

1. selector code/tests must pass exact-head CI;
2. regenerate the three manifests from the exact source index and frozen
   325-family blacklist;
3. verify their SHA-256 values match the preregistered values above;
4. materialize source files only into a **fresh staging destination**;
5. independently audit copied files, source hashes/provenance, 500/class,
   400/50/50, 1500 global unique families, zero historical overlap, zero
   cross-class/split family overlap, and canonical schema;
6. only after that audit PASS may bbox annotation be authorized;
7. training remains closed until bbox completion and its own admission gate.

The current `METER_V2_1500` Drive surface is not modified in place.

## 4/4 policy

4/4 remains a protected regression/reference class. The package_ab-only plan
prevents source package identity from becoming a class shortcut while
preserving 500 independent 4/4 families.

## Safety invariant

`training=false`, `tuning=false`, `model_evaluated=false`,
`checkpoint_opened=false`, `inference_count=0`, `dataset_mutated=false` until
the explicit downstream gates pass.
