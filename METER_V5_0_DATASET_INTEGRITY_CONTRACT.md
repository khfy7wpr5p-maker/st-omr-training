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
contamination boundary: one `FamilyId` may not span multiple splits.

## Manifest schema

Canonical columns, in order:

`Split,Meter,FamilyId,SampleId,Folder,SourceImage,SourceSemantic,SourceAgnostic,SplitRank`

`SelectionRank` is not accepted as a frozen V5 schema alias. Inputs may be
parsed for audit, but the gate remains HOLD until all three manifests use
`SplitRank`.

## Historical contamination boundary

The consumed V4-5 final holdout is permanently excluded from all V5 train,
validation, and final-holdout surfaces. The current audit found **zero**
overlap with the 150 consumed V4-5 families.

The exact consumed-family exclusion set is frozen in
`evidence/METER_V4_5_CONSUMED_FAMILIES.txt` and is bound to:

- V4-5 selection SHA-256:
  `4335a48a091912ba422c16d8fcbaaa7bbf5f7a0a43f088146a50a3e02e3ed7dc`
- sorted 150-family list SHA-256:
  `9d969b6bf5749bae7003c45644c50be36495ccb9b10fe3e7569ace5d413adea3`

Any later inventory, selection, split, or rebuild must reproduce this exclusion
set exactly before proceeding.

## Source-domain shortcut gate

A class-balanced count is not sufficient if source package/domain predicts the
label. Source domain is extracted from the `SourceImage` path (for example
`package_aa`, `package_ab`).

For each material source domain, compare its share within each meter class.
The maximum class-to-class share gap must be **<= 0.20** before bbox annotation
or training is authorized. This threshold is frozen before resampling.

Reason: a model must learn the meter glyph, not a source-package rendering
style.

## Current Drive audit

Manifest SHA-256:

- 2/4: `688aa6c229eb01cb88a58d9c2bd4225f47af48475a7c72b3d737922016a9d1ae`
- 3/4: `6db58cbacc48b41d46a856618432a6d134ff5b4cb6344fc5b7cc06b87b40d5ed`
- 4/4: `5d103cc4dd53a4648a776caaae62b752cd3debbb6bbfa0a35b6016e1fc4db3c2`

Observed structural counts are correct: 500/class with 400/50/50 split counts,
1500 folders/samples total.

Observed integrity failures:

1. **Schema drift** — 4/4 uses `SelectionRank` instead of `SplitRank`.
2. **One family split leak** — `ab_000102539` is present as 2/4 `train` and
   3/4 `final_holdout`.
3. **Severe source-domain/class confounding**:
   - 2/4: `package_aa=500`, `package_ab=0`
   - 3/4: `package_aa=500`, `package_ab=0`
   - 4/4: `package_aa=24`, `package_ab=476`
   - share gap for both material domains: `0.952`

Therefore the current dataset is **HOLD**. Bbox annotation and model training
are not authorized yet.

Current audit receipt SHA-256:
`f30b3717654d253e9296db26b7ab965b4683cfaa32921bd1c7dc2fcc9c6c31d9`

## Read-only source inventory gate

Before any regeneration, reselection, copying, or split mutation, source
capacity must be measured from the existing Windows `MASTER_INDEX.tsv`.

The approved inventory implementation is
`tools/st_omr_meter_v5_0_source_inventory_win7.ps1`. It is observation-only:

- reads `MASTER_INDEX.tsv` and referenced semantic/agnostic files only;
- admits only `package_aa` / `package_ab` candidates;
- requires G2, the target meter flag, exact semantic meter consistency, and the
  matching agnostic digit pair;
- excludes the exact frozen 150 consumed V4-5 families;
- computes symbolic-content SHA-256 identities;
- reports per-meter/per-package raw, unique-family, and unique-content capacity;
- reports cross-meter family and content collisions;
- writes only a fresh isolated `V5_0_SOURCE_INVENTORY` audit directory;
- does not modify the current dataset.

The inventory also reports a **count-only** common package mix interval. This
is feasibility evidence only. Even a feasible 250/250-style count does not
authorize a rebuild because global family/content collisions may reduce the
real selectable surface.

## Repair policy

Do not patch only the single split leak and proceed. The source-domain
confound is the dominant blocker.

Required order:

1. run the read-only source inventory and freeze its receipt;
2. require a feasible common source mix before selector design proceeds;
3. build a deterministic global selector from the inventory, with source-domain
   stratification and family/content exclusion enforced before any copy;
4. preserve exactly 500/class and 400/50/50 per class;
5. enforce family-disjoint splits globally across all classes;
6. keep consumed V4-5 families at zero overlap;
7. normalize all manifest schemas to `SplitRank`;
8. materialize only into a fresh staging destination and independently audit it;
9. rerun this integrity audit;
10. only if status is `PASS`, freeze manifest hashes and begin bbox annotation.

The current `METER_V2_1500` surface remains unchanged until the inventory and
selector gates have both passed.

## 4/4 policy

4/4 is a protected regression/reference class, not a reason to accept source
imbalance. Its representation must remain strong while source-domain cues are
balanced enough that 4/4 cannot be recognized merely from package identity.

## Safety invariant

`training=false`, `tuning=false`, `model_evaluated=false`,
`checkpoint_opened=false`, `inference_count=0`, `dataset_mutated=false` during
this audit stage.
