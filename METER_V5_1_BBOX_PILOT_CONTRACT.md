# METER V5-1 — Clean package_ab BBox Pilot Contract

Status: **PREREGISTERED / TRAIN-ONLY PILOT**

## Authoritative dataset

Only the Google Drive dataset named:

`METER_V2_1500_PACKAGE_AB_CLEAN`

is admissible for this stage.

The previously used `METER_V2_1500` surface is **REJECTED_SOURCE_CONFOUND** and must not be used for bbox annotation, training, validation, or final evaluation.

Drive admission evidence for the selected TEST child:
- selected folder id: `1OY6ZInOGh0Xtqb4Lkw78zx2rZTfpUasG`
- parent TEST folder id: `13PKxIyjRgZRhqF1C9n9XRMQCYFn-aSdX`
- verification report: 1500 total; 500 each for 2/4, 3/4, 4/4; 400/50/50 per class; 1500 unique families; 1500 unique samples; 1500 unique source images; `package_ab_only=true`; `bbox_status=NOT_ANNOTATED`.

The Colab runtime must not hard-code the Drive path. It searches `/content/drive/MyDrive` for the exact dataset directory name and:
- fails on 0 matches;
- accepts exactly 1 match;
- fails on >1 matches.

## Dataset gate before annotation

The three root selection manifests are authoritative:
- `2_4_SELECTION_MANIFEST.csv`
- `3_4_SELECTION_MANIFEST.csv`
- `4_4_SELECTION_MANIFEST.csv`

Required invariants:
- 500 rows per meter;
- each class split is exactly 400 train + 50 val + 50 final_holdout;
- global unique `FamilyId` = 1500;
- global unique `SampleId` = 1500;
- global unique `SourceImage` = 1500;
- cross-split family leakage = 0;
- cross-meter family overlap = 0;
- `Package=package_ab` for every row;
- `SourceImage`, `SourceSemantic`, and `SourceAgnostic` all contain a `package_ab` path segment;
- copied dataset directories exist with exact 400/50/50 cardinalities and every manifest row maps to an existing `image.png`.

This gate may count final_holdout folders and check existence of final_holdout `image.png`, but it must not decode/hash/display final_holdout image bytes.

## Final holdout lock

Immediately after the dataset gate passes, write:

`annotations/FINAL_HOLDOUT_LOCK.json`

outside the final_holdout tree.

The lock binds the current manifest/directory fingerprint and asserts:
- `locked=true`
- `annotation_opened=false`
- `training_opened=false`
- `tuning_opened=false`
- `model_evaluated=false`
- `inference_count=0`

The V5-1 pilot code exposes only train samples.

## Pilot selection

Only 30 TRAIN samples:
- 2/4: 10
- 3/4: 10
- 4/4: 10

Selection is deterministic and persisted once as:

`annotations/bbox_pilot_30_selection.csv`

Each selected image is bound to:
- sample id
- family id
- meter
- split=train
- copied image path
- SHA-256
- original width/height

Once this selection exists, resume must validate it rather than silently reselect.

## BBox contract

Exactly one human bbox row per pilot sample.

The single rectangle must contain the full meter pair (top + bottom digit). Clef, key signature and the first note to the right should remain outside whenever possible.

Original `image.png` files are read-only. A red rectangle exists only on the browser canvas preview and is never painted into the source image.

Persistent annotation:

`annotations/bbox_pilot_30.csv`

Required columns begin with:

`sample_id,meter,split,x,y,w,h,status`

and additionally bind image SHA/dimensions and update time.

Allowed statuses:
- `PASS`: one valid bbox in original-image integer coordinates
- `REVIEW`: no bbox coordinates; human follow-up required

Each save atomically rewrites the checkpoint CSV. The row for a sample may be corrected, but duplicate rows are forbidden. Backups are written at 10/20/30 handled samples.

## Resume safety

Resume is keyed by `sample_id`.
- completed PASS/REVIEW rows are not restarted;
- first unhandled sample becomes the resume point;
- image SHA/dimensions are revalidated before display and before each write;
- a changed original image fails closed.

## Pilot mechanical audit

After all 30 samples have PASS or REVIEW rows, write:

`annotations/bbox_pilot_30_audit.json`

Audit includes:
- annotation count
- unique sample count
- PASS/REVIEW counts
- per-class PASS/REVIEW
- bbox width/height min/median/max/mean
- zero/negative bbox count
- bbox-outside-image count
- suspiciously small/large bbox warnings
- final-holdout lock state

Mechanical PASS requires:
- annotation_count = 30
- unique_sample_id = 30
- zero/negative bbox = 0
- bbox outside image = 0

`annotation_contract_freeze_ready` additionally requires no unresolved REVIEW or suspicious-size warnings. Human-drawn boxes are never auto-adjusted.

## Explicitly forbidden in V5-1

- scanning the 80k+ PrIMuS archive;
- using the rejected old `METER_V2_1500`;
- mixing package_aa and package_ab;
- running the failed PowerShell R05 locator over the dataset;
- modifying original image.png files;
- annotating val or final_holdout;
- training, tuning, checkpoint opening, or inference;
- using final_holdout for model selection or threshold tuning.
