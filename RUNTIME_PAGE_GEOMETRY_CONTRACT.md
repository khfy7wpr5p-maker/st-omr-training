# ST-OMR — Isolated Runtime Page + Geometry Contracts

Status: **contract-only development package**.

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

Allowed future responsibilities:

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

Both contracts freeze the following values to false:

```text
Stage 7-D10 read       false
Stage 7-D10 write      false
Stage 7-D13 read       false
Stage 7-D13 write      false
checkpoint access      false
optimizer access       false
TEST split access      false
```

The package imports neither D10 nor D13 modules. It introduces no model, optimizer, checkpoint loader, training runner, or dataset derivative writer.

## 4. First contract gates

The first tests check three user-visible safety ideas:

1. **Do not damage the score.** A contract operation cannot authorize destructive symbol removal.
2. **Only describe page geometry.** A staff must contain exactly five ordered lines and a measure remains a geometry proposal, not a musical interpretation.
3. **Do not invent certainty.** Ambiguous/rejected outputs must state a reason; accepted outputs cannot carry hidden rejection reasons.

Additional technical gates verify:

- canonical deterministic contract fingerprints;
- bounded raster dimensions and pixel modes;
- finite, invertible forward/inverse transforms;
- coordinate round-trip replay;
- finite positive staff spacing;
- unique system/staff/measure identities;
- page-bound geometry;
- explicit D10/D13/checkpoint/optimizer/TEST isolation.

## 5. Explicit non-goals

This package does **not**:

- implement OpenCV processing;
- rasterize PDFs;
- execute a real-image runtime pilot;
- build local ROIs;
- call Meter/NoteHead/Rest/Accidental specialists;
- alter D10/D13 manifests, derivative identities, checkpoints, or Rest R2 evidence;
- open sealed TEST;
- authorize training or production integration.

## 6. Next gate

After focused tests and the repository CI pass, the next package may implement the smallest deterministic Page Normalizer behavior against isolated fixtures. A real-image shadow/runtime pilot remains a later, separately bounded step and must still stop before D10/D13 or specialist inference.
