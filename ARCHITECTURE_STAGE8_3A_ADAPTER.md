# Stage 8-3A Adapter Architecture Delta

> Historical/frozen stage delta. This file preserves the original Stage 8-3A adapter architecture and is not current operational-status authority. Use `ARCHITECTURE_CURRENT.md` for the live merged + shadow/experimental overlay.

PR #34 added one guarded auxiliary target path inside the then-active Stage 8-3A package.

```text
source PNG -> deterministic training PNG

MEI + semantic
    -> guarded corroboration
    -> canonical ST music model
    -> independent validation
    -> deterministic MusicXML
    -> MusicXML validation and V1 round trip

training PNG + MusicXML + required evidence
    -> existing Stage 8 admission gates
    -> 50 admitted pairs
    -> 40 train / 10 validation handoff
```

The adapter does not change the frozen V1 surface. Representation disagreement and unsupported notation fail closed. Conversion success is not data admission.

Historical gate status for this package: Stage 0 through 8-2 closed; Stage 8-3A active; Stage 8-3B, Stage 9 and Stage 10 locked. Current project status is intentionally not inferred from this historical delta.
