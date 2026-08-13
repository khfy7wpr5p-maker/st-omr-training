# Stage 8-3A Adapter Architecture Delta

PR #34 adds one guarded auxiliary target path inside the already-active Stage 8-3A package.

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

Gate status remains: Stage 0 through 8-2 closed; Stage 8-3A active; Stage 8-3B, Stage 9 and Stage 10 locked.
