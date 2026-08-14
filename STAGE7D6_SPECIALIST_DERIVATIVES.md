# Stage 7-D6 — StaffSet + StructureSet specialist derivatives

## Purpose

Stage 7-D6 materializes the first specialist OMR development dataset from the accepted Synthetic Curriculum v1. It does **not** train a model.

The accepted D5 geometry contract is applied only to development splits:

```text
Frozen Synthetic Curriculum v1
        ↓
D1 whole-corpus byte/integrity acceptance
        ↓
read manifest split
        ├── TEST → skip immediately
        └── TRAIN / VALIDATION
                ↓
        pinned MusicXML + Verovio lineage
                ↓
        D5 StaffSet / StructureSet geometry
                ↓
        replay exact final-PNG coordinate transform
                ↓
        canonical hash-addressed JSON sidecar
```

## Frozen development surface

| Split | Families | PNG samples | D6 labels |
|---|---:|---:|---:|
| TRAIN | 410 | 1,230 | 1,230 |
| VALIDATION | 51 | 153 | 153 |
| TEST | 51 | 153 | **0** |
| Development total | 461 | 1,383 | 1,383 |

The source corpus remains frozen. D6 does not create new PNG or MusicXML copies.

## Output layout

```text
<fresh-output>/
  manifest.json
  manifest.sha256
  build.json
  labels/
    <label_sha256>.json
```

Each label sidecar binds:

- source `sample_id` and `family_id`;
- inherited `train|validation` split;
- page number;
- exact source PNG SHA-256 and dimensions;
- source MusicXML SHA-256;
- frozen normal Verovio SVG SHA-256;
- renderer and degradation fingerprints;
- D5 geometry/transform versions;
- final-PNG-coordinate StaffSet and StructureSet geometry.

The existing frozen PNG remains the image authority. `image_policy = reference-frozen-source-png-by-sha256` prevents duplication of the large corpus.

## StaffSet label surface

For each graphical staff instance:

- `staff_instance_id`;
- owning `system_id`;
- exactly five staff-line segments;
- staff bounding box;
- staff spacing.

One V1 logical staff may have multiple graphical instances when notation wraps across systems.

## StructureSet label surface

For each page/system/measure:

- system id and bounding box;
- measure id and canonical measure number;
- measure content bounding box;
- trailing barline segment;
- visible G2 clef box when present;
- visible meter box when present;
- canonical `2/4|3/4|4/4` meter class.

The barline authority is a two-point segment, not a scalar x coordinate, because controlled rotation can make the line slanted in final PNG space.

## Ground-truth authority

AI never creates D6 ground truth.

- Canonical symbolic measure/meter identity comes from the deterministic canonical music/MusicXML path.
- Synthetic spatial geometry comes from pinned Verovio 6.2.1 geometry instrumentation accepted in D5.
- Final coordinates are obtained by replaying the accepted deterministic SVG/raster/rotation transform.
- If any lineage, cardinality, coordinate, shape, hash or V1 invariant is ambiguous, D6 fails closed.

## TEST seal

D6 has a dedicated split boundary. For every source manifest row it reads `split` first. A `test` row is skipped before any specialist image hash, MusicXML/SVG field, artifact path, geometry or label is derived.

D1 may still hash TEST artifacts as frozen archive/corpus integrity evidence. That does not expose TEST to specialist development and produces no D6 TEST record.

## Independent derivative gate

Persisted artifacts are reparsed rather than trusting builder objects. The gate verifies:

- exact frozen source manifest identity;
- canonical JSON bytes;
- exact 1,383 record and label cardinality;
- 1,230/153 sample split counts;
- 410/51 family counts and family exclusivity;
- no forbidden split;
- label filename = label SHA-256;
- source sample/image/provenance binding;
- exact label set with no extras/missing files;
- StaffSet five-line structure;
- system/staff/measure cross-reference consistency;
- supported V1 meter classes;
- finite, in-bounds final-PNG geometry;
- deterministic manifest/build/artifact binding receipt.

## Runtime / storage boundary

D6 uses the same pinned repository runtime as the accepted synthetic pipeline. The authoritative frozen-corpus build must be executed from an exact PR head and saved outside normal Git. Only code, tests and small documentation are versioned in the repository.

## Closure requirements

Stage 7-D6 closes only when all are true:

1. exact-head focused/full regression and CI PASS;
2. independent P1/P2 review has no blocker;
3. authoritative frozen-corpus D6 build succeeds;
4. returned receipt proves 1,383 development labels / 461 families / TEST records = 0;
5. persisted derivative output passes the independent D6 gate;
6. evidence hashes are recorded in PR closure evidence;
7. explicit user merge approval is received;
8. post-merge `main` CI passes.

No StaffSet/StructureSet model training begins before this data gate closes.
