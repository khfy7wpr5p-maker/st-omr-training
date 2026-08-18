# ST-OMR — Isolated Runtime Page + Geometry Contracts

Status: **isolated runtime development package; no D10/D13/Rest R2 integration**.

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
- deskew;
- safe crop;
- illumination normalization;
- contrast normalization;
- bounded perspective correction;
- bounded resolution normalization.

The accepted output must remain a staff-preserved grayscale page and must carry both original→normalized and normalized→original transforms.

The normalizer must not recognize staff, measure, meter, notehead, rest, accidental, pitch, duration, or MusicXML. Staff-line removal and other destructive symbol removal are forbidden.

If the page cannot be normalized safely, the contract requires `ambiguous` or `rejected` instead of invented certainty.

### First Page Normalizer implementation slice

The current bounded V1 implementation only:

- validates raster byte SHA, dimensions and declared pixel mode;
- accepts PNG/JPEG raster bytes;
- applies non-mirrored EXIF orientation 1/3/6/8 with reversible coordinate mapping;
- rejects mirrored EXIF orientation 2/4/5/7 instead of guessing;
- converts RGB/RGBA/gray input to staff-preserving gray8;
- applies deterministic global linear autocontrast;
- emits deterministic metadata-free PNG bytes.

It deliberately does **not** auto-detect skew, deskew, crop, dewarp, change resolution, or recognize music symbols yet.

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

### First Geometry Engine implementation slice

The current V1 implementation answers only one narrow question:

> Does this normalized page contain exactly one clear horizontal five-line staff?

It uses deterministic row/run geometry only. It:

- finds long dark horizontal line candidates;
- merges line thickness into one row center;
- accepts only five near-equidistant candidate lines;
- reports staff spacing and five line segments;
- emits no measure proposals;
- emits no musical-symbol semantics;
- treats four-line input as ambiguous;
- treats multiple plausible staffs as ambiguous in this first slice rather than guessing a grouping.

Multi-staff/system grouping is intentionally a later package.

## 3. Isolation from the active training lane

Both runtime surfaces remain isolated from the active training lane:

```text
Stage 7-D10 read       false
Stage 7-D10 write      false
Stage 7-D13 read       false
Stage 7-D13 write      false
checkpoint access      false
optimizer access       false
TEST split access      false
```

The runtime implementation imports neither D10 nor D13 modules. It introduces no model, optimizer, checkpoint loader, training runner, or dataset derivative writer.

## 4. Current test gates

The tests check user-visible safety ideas:

1. **Do not damage the score.** A low-contrast synthetic score keeps its five staff lines and notehead after normalization.
2. **Respect page orientation.** Standard camera/phone EXIF orientation is applied with reversible coordinates.
3. **Do not guess unsupported transforms.** Mirrored orientation is rejected.
4. **Find one simple staff correctly.** A synthetic five-line staff is detected with the expected spacing.
5. **Do not invent a staff.** Four lines remain ambiguous.
6. **Do not overreach yet.** Two plausible staffs remain ambiguous until multi-staff grouping gets its own bounded package.
7. **Stay deterministic.** Repeating identical input produces identical normalized bytes and geometry output.

Additional technical gates verify:

- canonical deterministic fingerprints;
- bounded raster dimensions and pixel modes;
- finite, invertible forward/inverse transforms;
- coordinate round-trip replay;
- finite positive staff spacing;
- page-bound geometry;
- explicit D10/D13/checkpoint/optimizer/TEST isolation.

## 5. Explicit non-goals

This package does **not**:

- add OpenCV;
- rasterize PDFs;
- execute a real-image runtime pilot;
- build local ROIs;
- detect measures in the first Geometry V1 slice;
- call Meter/NoteHead/Rest/Accidental specialists;
- alter D10/D13 manifests, derivative identities, checkpoints, or Rest R2 evidence;
- open sealed TEST;
- authorize training or production integration.

## 6. Next gate

After review, the next bounded geometry package may expand from one-staff detection to safe multi-staff/system grouping. Measure-boundary proposals remain a separate later gate. Merge remains explicitly approval-gated.
