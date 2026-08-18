# ST-OMR — Isolated Runtime Page + Geometry Contracts

Status: **contract package + first isolated Page Normalizer implementation slice**.

This package defines two future runtime boundaries without changing the active D10/D13 identities or the Rest R2 lane.

```text
Raster page
   ↓
ST Page Normalizer
   ↓
Normalized page + reversible transform
   ↓
ST Geometry Engine
   ↓
System / staff / measure proposals
   ↓
STOP
```

No specialist model is connected by this package.

## 1. ST Page Normalizer boundary

The Page Normalizer receives one already-rasterized page. PDF rasterization is intentionally outside V1 so the rasterizer cannot silently change the normalizer's deterministic identity.

Allowed responsibilities:

- orientation normalization;
- grayscale conversion;
- deskew;
- safe crop;
- illumination normalization;
- contrast normalization;
- bounded perspective correction;
- bounded resolution normalization.

The accepted output must remain a staff-preserved grayscale page and must carry both original→normalized and normalized→original transforms.

The normalizer must not recognize staff, measure, meter, notehead, rest, accidental, pitch, duration, or MusicXML. Staff-line removal and other destructive symbol removal are forbidden.

If the page cannot be normalized safely, the contract requires `ambiguous` or `rejected` instead of invented certainty.

### First implementation slice

`runtime_page_normalizer_v1.py` implements only a deliberately small subset:

- one already-rasterized PNG or JPEG page;
- exact input-byte SHA / dimensions / pixel-mode verification;
- pinned Pillow `12.3.0` runtime;
- normal EXIF orientation values `1`, `3`, `6`, `8` with reversible coordinate transforms;
- mirrored EXIF orientations `2`, `4`, `5`, `7` rejected fail-closed in this first slice;
- staff-preserving conversion to gray8;
- deterministic global linear autocontrast (`cutoff=0`);
- deterministic metadata-free PNG output.

This slice does **not** implement automatic skew-angle detection, deskew, crop, dewarp, perspective correction, resolution changes, or semantic recognition.

Test score images are generated in memory by the unit tests. No binary image corpus is added to Git.

## 2. ST Geometry Engine boundary

The Geometry Engine accepts only an accepted Normalized Page identity plus its reversible transform.

Allowed observations:

- system bounding boxes;
- five staff lines;
- staff bounding boxes;
- staff spacing;
- measure bounding-box proposals;
- measure left/right boundary proposals.

The engine does not decide meter, notehead class, rest class, accidental class, pitch, duration, chord, voice, or MusicXML semantics.

Measure outputs are deliberately called **proposals**. Ambiguous geometry must remain ambiguous instead of being silently promoted to a confident measure.

## 3. Isolation from the active training lane

Both contracts and the first normalizer slice preserve the following boundaries:

```text
Stage 7-D10 read       false
Stage 7-D10 write      false
Stage 7-D13 read       false
Stage 7-D13 write      false
checkpoint access      false
optimizer access       false
TEST split access      false
```

They import neither D10 nor D13 modules. They introduce no model, optimizer, checkpoint loader, training runner, or dataset derivative writer.

## 4. First gates

The tests check three user-visible safety ideas:

1. **Do not damage the score.** Five synthetic staff lines and a notehead remain visible after the first grayscale/contrast normalization slice.
2. **Keep direction traceable.** A page carrying standard phone/camera orientation metadata is reoriented and its coordinate mapping can be replayed back to the source.
3. **Do not invent certainty.** An orientation this slice does not safely support is rejected rather than guessed.

Additional technical gates verify:

- canonical deterministic fingerprints;
- identical normalized PNG bytes for repeated identical inputs;
- exact raster byte SHA, dimensions, and pixel-mode binding;
- bounded raster dimensions and pixel modes;
- finite, invertible forward/inverse transforms;
- coordinate round-trip replay;
- transparent RGBA pages composite onto white before grayscale conversion;
- finite positive staff spacing in the Geometry contract;
- unique system/staff/measure identities;
- page-bound geometry;
- explicit D10/D13/checkpoint/optimizer/TEST isolation.

## 5. Explicit non-goals

This package does **not**:

- add OpenCV or another heavy image-processing dependency;
- rasterize PDFs;
- execute a real-image runtime pilot;
- build local ROIs;
- call Meter/NoteHead/Rest/Accidental specialists;
- alter D10/D13 manifests, derivative identities, checkpoints, or Rest R2 evidence;
- open sealed TEST;
- authorize training or production integration.

## 6. Next gate

After focused tests and repository CI pass for this first image slice, the next bounded implementation package may begin the smallest deterministic Geometry Engine observation on isolated fixtures. A real-image shadow/runtime pilot remains later and must still stop before D10/D13 or specialist inference.
