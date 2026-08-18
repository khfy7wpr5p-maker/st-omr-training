# ST-OMR — Isolated Runtime Page → Deterministic Resolver Lane

Status: **isolated draft runtime lane; not merged; no D10/D13 specialist checkpoint integration**.

```text
Raster image
   ↓
ST Page Normalizer v1
   ↓
Multi-Staff Geometry v2
   ↓
A01–A07 deterministic ambiguity resolver
   ↓
Measure Geometry v1
   ↓
Runtime Local ROI v1
   ↓
Model-agnostic specialist evidence boundary
   ↓
Deterministic Resolver v1
   ↓
STOP
```

The lane is intentionally separate from the active training identities. It does not import or mutate Stage 7-D10 or Stage 7-D13, does not load checkpoints, does not run an optimizer and does not access the sealed TEST split.

## 1. Page Normalizer

The runtime normalizer receives an already-rasterized PNG/JPEG page. PDF rasterization remains a separate upstream concern.

Implemented bounded behavior:

- verify raster SHA/dimensions/pixel mode;
- apply supported non-mirrored EXIF orientation;
- reject mirrored orientations rather than guessing;
- convert to staff-preserving gray8;
- deterministic global autocontrast;
- deterministic metadata-free PNG output;
- reversible source↔normalized coordinate transform.

It does not yet auto-deskew, crop, dewarp or apply perspective correction.

## 2. Multi-Staff Geometry

Geometry v2 detects multiple clearly separated horizontal five-line staffs and returns stable top-to-bottom identities (`staff-1`, `staff-2`, ...).

A staff remains geometry only. This layer does not infer meter, notes, rests, accidentals, pitch, duration or MusicXML.

### A01–A07 ambiguity priority

The canonical primary/secondary ordering is frozen:

1. `A04_PAGE_CROPPED`
2. `A03_LOW_VISIBILITY`
3. `A01_INCOMPLETE_STAFF`
4. `A05_OVERLAPPING_CANDIDATES`
5. `A02_STAFFS_TOO_CLOSE`
6. `A07_EXTRA_LINE_CANDIDATES`
7. `A06_IRREGULAR_SPACING`

The same active code set must produce the same primary reason, ordered secondary reasons and report fingerprint on every run.

## 3. Measure Geometry

Measure Geometry v1 searches for strong vertical separators spanning each accepted staff. These are **measure-boundary proposals**, not semantic barline recognition.

The first bounded gates require:

- at least two separator candidates per staff;
- positive measure widths;
- cross-staff separator alignment inside one system;
- fail-closed output when boundaries are missing or incoherent.

## 4. Runtime Local ROI

Runtime Local ROI v1 is a new in-memory inference crop layer. It is explicitly **not D10**.

It emits deterministic hash-bound:

- `measure-full` crops for future NoteHead/Rest/Accidental adapters;
- `measure-start` crops for future Meter adapters;
- source→ROI reversible transforms;
- stable ROI identities and PNG hashes.

No dataset derivative is written.

## 5. Specialist evidence boundary

The runtime lane defines a model-agnostic evidence contract for:

- Meter: `none | 2/4 | 3/4 | 4/4`
- NoteHead: `open | filled`
- Rest: `half | quarter | eighth`
- Accidental: `sharp | flat | natural`

Each observation is bound to one measure/staff and has status, confidence, optional bbox and class.

**Important:** this boundary does not mean the D13 specialist checkpoints are production-connected. D13-R2 remains authoritative for specialist recovery/acceptance. Shadow fixtures can exercise the resolver without promoting an unaccepted model.

## 6. Deterministic Resolver v1

The resolver consumes accepted measure geometry plus explicit specialist observations. It performs deterministic validation/ordering and accidental→following-notehead association on the same staff/measure.

Fail-closed reasons include:

- conflicting meter evidence;
- accidental association tie;
- multiple accidentals competing for one notehead;
- unassociated accidental;
- upstream specialist ambiguity.

The resolver does **not** yet compose pitch, duration, voice or MusicXML. Those remain later deterministic composer/validator layers.

## 7. Shadow orchestrator

The isolated orchestrator provides the tested runtime seam:

```text
raster bytes
→ normalized page
→ multi-staff geometry
→ measure proposals
→ runtime ROI batch
→ explicit specialist evidence
→ deterministic resolver
```

It stops there. No production specialist model is auto-loaded.

## 8. Safety / isolation evidence

The branch-level isolation gates require:

```text
D10 import/write          0
D13 import/write          0
checkpoint load           0
optimizer                 0
sealed TEST access        0
training derivative write 0
```

The branch comparison against `main` must contain only isolated runtime/documentation/test additions. Existing training files and Rest R2 evidence remain untouched.

## 9. What is proven vs not proven

Proven in deterministic synthetic/in-memory fixtures:

- Page Normalizer repeatability;
- single and multi-staff separation;
- A01–A07 report ordering and 10/10 repeatability;
- aligned measure-boundary proposals;
- deterministic runtime ROI extraction;
- model-agnostic Meter/NoteHead/Rest/Accidental evidence validation;
- deterministic accidental→notehead association;
- fail-closed resolver conflicts;
- full isolated raster→resolver orchestration.

Not yet proven:

- production accuracy on scanned/photographed real scores;
- robust deskew/perspective/dewarp;
- complex multi-system score geometry;
- semantic barline accuracy in real notation;
- accepted Rest R2 / Accidental / NoteHead runtime checkpoint adapters;
- end-to-end pitch/duration/MusicXML correctness.

## 10. Promotion boundary

Passing this draft lane does not authorize automatic model integration or merge. Before specialist runtime promotion, each specialist adapter must be independently bound to an accepted checkpoint/evidence lineage and exercised first in read-only shadow mode. Merge remains an explicit user approval gate.
