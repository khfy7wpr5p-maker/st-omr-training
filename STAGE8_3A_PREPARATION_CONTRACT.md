# Stage 8-3A — Pilot Data Preparation + Admission

Status: **active — preparation/admission only; no training**.

Stage 8-3A is the first real-data execution gate after the closed Stage 8-2 paired-run profile. Its job is to freeze deterministic source-image preparation, triage external symbolic auxiliaries without silently trusting them, and then run the already-closed Stage 8-0/8-1 admission boundary on supported-V1 pairs. It does not perform optimizer steps, load a model checkpoint, open a sealed test split, publish a model, or integrate with ScoreMosaic.

## Exact starting baseline

Stage 8-3A starts from exact verified `main`:

`99a32ef917ba4ba5c72ef6a537c24c90a1b0c47f`

That commit is the Stage 8-2 closure synchronization from PR #32. Exact-main GitHub Actions run #104 (`31712294207`) succeeded.

The first Stage 8 paired experiment remains frozen at exactly **50 admitted development pairs = 40 train + 10 validation**. Raw files, auxiliary annotations, or folders do not count toward those totals until admission is complete.

## Repository and real-data boundary

Real score images and real symbolic annotation files remain outside Git. Stage 8-3A repository changes may contain only code, contracts, synthetic test fixtures created in memory, and hash-only evidence where appropriate.

The implementation performs no filesystem scanning, Drive access, network access, dataset persistence, checkpoint access, model loading, or training. External orchestration supplies bytes explicitly to the in-memory preparation/intake functions.

## First-pilot source-image policy

The first pilot intentionally accepts **source PNG only**. This is narrower than the general Stage 8-1 source-document boundary and prevents PDF/JPEG renderer, DPI, crop, EXIF orientation, codec, and color-management choices from entering the pilot without a separately frozen policy.

The exact source bytes are retained unchanged outside Git and hash-bound. `prepare_training_png(...)` may derive a training image only under the following frozen policy:

- Pillow must be exactly the Stage 8-1 runtime version (`12.3.0`);
- input must be a non-empty PNG within the existing Stage 8 source-byte bound;
- one frame only;
- no transparency/alpha;
- accepted source image modes are only `1`, `L`, or `P`;
- `RGB`, `RGBA`, CMYK, floating-point, and other source modes are rejected in this first pilot rather than silently color-normalized;
- width and height must already be positive and within the Stage 8-1 pixel bound;
- **no crop**;
- **no resize**;
- **no rotation/orientation rewrite**;
- convert accepted pixels deterministically to Pillow mode `L`;
- construct a new `L` image from the resulting pixel bytes so source metadata is not copied;
- write PNG with `optimize=False` and `compress_level=9`;
- independently verify the derived IHDR CRC and verify the derivative as 8-bit grayscale, non-interlaced, single-frame PNG;
- verify derivative geometry equals source geometry;
- return the derivative bytes to the caller plus hash-only preparation evidence.

The preparation policy has a deterministic fingerprint. Any change to accepted modes, geometry behavior, Pillow version, conversion behavior, metadata handling, PNG encoding, limits, or trust boundary requires a new policy version rather than silent drift.

## PrIMuS-style auxiliary package boundary

Some external pilot candidates are available as packages containing:

- PNG notation image;
- MEI;
- semantic token annotation;
- agnostic graphical-symbol annotation;
- optionally MIDI and Plaine & Easie data.

Stage 8-3A treats MEI, semantic, and agnostic files as **auxiliary triage evidence only**. MIDI and Plaine & Easie are not part of the admission target contract. macOS `._*` sidecar/resource files are ignored by external orchestration and are not data samples.

`inspect_primus_auxiliary_package(...)` may check bounded parseability, MEI/semantic header agreement, and obvious supported-V1 exclusions. It fails triage for empty event streams, unsupported accidentals/durations, deferred structures, unsupported staff/layer structure, unsupported clef/key/meter, or header mismatch. A successful auxiliary triage result does **not** establish:

- legal rights or training permission;
- image↔symbolic pairing approval;
- full MEI↔semantic equivalence;
- MusicXML validity;
- Stage 8 admission;
- production quality.

No MEI, semantic, agnostic, MIDI, PAE, model prediction, OMR output, or parser success is silently promoted to trusted MusicXML.

## Supported-V1 triage

The current model/tokenizer boundary remains unchanged. For a candidate to proceed toward the current V1 admission path it must remain compatible with the existing frozen semantics, including:

- exactly one part/staff;
- treble clef (`G2`);
- one voice;
- key signature `0`;
- meter only `2/4`, `3/4`, or `4/4`;
- note durations only whole, half, quarter, or eighth;
- rest durations only half, quarter, or eighth;
- supported pitch spelling and only the frozen sharp/flat/natural accidental surface;
- no deferred structure such as explicit beam semantics, ties/slurs, tuplets, multi-measure rests, implicit measure-space proxies, or other unsupported notation.

Auxiliary packages outside this surface remain useful future-scope data but do not enter the first 40/10 pilot.

## MusicXML target boundary

Stage 8-1 still requires a supported-V1 MusicXML target before a real sample can receive a byte-validation receipt. Stage 8-3A does not weaken that requirement.

A future deterministic adapter from a supported auxiliary representation may be added only as a separately tested, fail-closed Stage 8-3A change. Such an adapter must terminate in the existing canonical ST music model, pass independent canonical validation, serialize through the existing deterministic MusicXML writer, and then pass the complete Stage 8-1 MusicXML/XSD/semantic/token round-trip gate. Unsupported source notation must be rejected rather than approximated or silently dropped.

Until that adapter is proven, auxiliary files are triage evidence and not training labels.

## Admission sequence

```text
external exact source PNG bytes
        ↓
Stage 8-3A deterministic source→training-PNG preparation
        ↓
hash-bound preparation evidence
        +
supported-V1 ground-truth MusicXML
        +
rights/provenance/pairing metadata
        ↓
Stage 8-0 quarantine/admission contract
        ↓
Stage 8-1 exact-byte + grayscale-PNG + MusicXML/token validation
        ↓
Stage 8-1 hash-only receipt
        ↓
near-duplicate + family/leakage veto
        ↓
50 admitted development pairs
        ↓
exact family-exclusive 40 train / 10 validation handoff
        ↓
Stage 8-3B may become eligible to start
```

A candidate that fails any step remains quarantined/rejected and does not count toward 50.

## External pilot smoke finding

The first externally inspected PrIMuS-style packages demonstrated that the uploaded package layout is usable for format triage and that MEI/semantic annotations can be internally coherent. The inspected examples also contain notation outside the current V1 surface (for example non-zero key signatures, non-treble clefs, 6/8 or meter symbols, sixteenth durations, beams, or multi-measure rests). They are therefore **not admitted training samples** and no real-data hashes or bytes are committed by this package.

This smoke result is intentionally not a rights review, pairing approval, or corpus-wide quality claim.

## Stage 8-3A acceptance boundary

Stage 8-3A may close only after:

1. deterministic PNG preparation code and policy are merged and exact-main CI verified;
2. unsupported/ambiguous preparation inputs fail closed;
3. auxiliary triage cannot bypass MusicXML/rights/pairing admission;
4. enough external candidates cross the full Stage 8-0/8-1 boundary to produce exactly 50 admitted pairs;
5. family/duplicate/leakage review permits an exact 40 train / 10 validation handoff;
6. the handoff is hash-bound to the frozen Stage 8-2 profile;
7. neither real nor synthetic sealed test material has been opened;
8. no optimizer/model-training action has occurred.

The present implementation package establishes items 1–3 only. Actual 50-pair admission remains active work and Stage 8-3A stays open until items 4–8 are proven.

## Explicitly out of scope

Stage 8-3A does not run fine-tuning or training, compare Candidate A/B model results, access/move/publish the Stage 7-C checkpoint, open or enumerate either sealed test split, expand the V1 model/tokenizer vocabulary, generalize PDF/JPEG/RGB preparation, integrate with ScoreMosaic, or enable automatic/online learning.
