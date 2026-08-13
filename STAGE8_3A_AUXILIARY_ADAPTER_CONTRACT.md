# Stage 8-3A — Auxiliary Adapter Contract

Status: **active subpackage; preparation only; no training**.

PR #34 starts from exact main `43591d289b1bcd283a971d59d623730426d7f22b`. Exact-main run #110 (`31732007702`) succeeded.

This package adds a narrow conversion path for supported-V1 auxiliary notation. It does not change Stage 8-0, Stage 8-1, Stage 8-2, sealed-test, or ScoreMosaic boundaries.

## Canonical entrypoints

- `inspect_guarded_primus_auxiliary_package(...)` is the Stage 8-3A auxiliary triage entrypoint.
- `adapt_guarded_primus_v1_to_musicxml(...)` is the Stage 8-3A auxiliary conversion entrypoint.
- Lower-level parser functions are implementation details, not admission gates.

## Frozen conversion surface

The guarded conversion accepts only the existing V1 surface: one staff, treble G2, one voice, key 0, 2/4 or 3/4 or 4/4, supported note/rest durations, supported pitches and the frozen accidental surface. Deferred notation is rejected rather than approximated or dropped.

MEI and semantic representations must corroborate the same meter, measure structure, event order, event kind, duration and sounding pitch. Visible accidental evidence must also agree with the required per-measure sounding-state transition. Ambiguous, redundant or contradictory accidental evidence is rejected.

The conversion chain is:

```text
bounded auxiliary bytes
    ↓
guarded preflight
    ↓
MEI / semantic corroboration
    ↓
canonical ST music model
    ↓
independent canonical validation
    ↓
deterministic MusicXML writer
    ↓
MusicXML validation + supported-V1 round trip
    ↓
hash-only guarded conversion evidence
```

A successful conversion is **not Stage 8 admission**. The exact source image, rights/provenance/pairing evidence, deterministic training-PNG preparation, Stage 8-0 admission, Stage 8-1 exact-byte/semantic receipt, duplicate/leakage vetoes and the exact 50-pair 40/10 handoff are still mandatory.

No real corpus bytes, model checkpoints, optimizer steps, training, sealed-test access, Stage 9/10 work, ScoreMosaic integration, or automatic learning are authorized by this package.

Stage 8-3A remains **ACTIVE** until exactly 50 real pairs cross the complete admission boundary and the frozen family-exclusive 40 train / 10 validation handoff is proven.
