# TR-POLY-09B8O — OSSQ Pairing-Source Preflight

B8O is the bridge between the B8N exact source-PDF byte pins and the later creation of real system-image ↔ MusicXML training candidates.

It does **not** claim that a system image and a MusicXML segment are already paired correctly. It only decides whether the exact upstream source metadata for a score is sufficiently understood to proceed to deterministic materialization.

## Frozen inputs

B8O binds:

- merged B8N receipt SHA-256: `9f9b678e2365ec849cc19424b28d8dbdb435af3a5a9ef5b47cf7a460e72a801c`;
- OSSQ camera-ready source commit: `7a17e45cddc0b7064fc3a179b62caeb57595e993`;
- OSSQ preprocessor snapshot: `bdea0d1829c9db84480ebd2e0385f6f5fe324274`;
- exact `sq<id>_scanned.csv` Git blob identity for each score;
- exact `sq<id>_cleaned.musicxml` Git blob identity for each score;
- full SHA-256 identities of the live-fetched alignment and cleaned-MusicXML bytes.

The canonical B8O receipt is stored at:

`evidence/ossq_b8o_pairing_preflight.json`

Receipt SHA-256:

`b77718f90f5865082a36f18da8701457baa5014d278e4ad33e49e727abbab65a`

## Camera-ready result

Eight B8N score records are covered.

Ready for later system-image materialization:

- `7070781`
- `7075297`
- `7078259`
- `7093885`
- `7103818`
- `7108150`
- `8071278`

Blocked:

- `7397765` — Schubert D.810

The Schubert record is **not** classified as an empty alignment file. Its exact camera-ready alignment payload is `:\na`. The literal `a` marker is present upstream, but its semantics are not established by the pinned preprocessor code inspected for B8O. B8O therefore records it as `blocked-uninterpreted-alignment-marker` instead of inventing a meaning.

The literal `x` marker found in several Mozart K.464 alignment files is retained in evidence but does not itself block materialization because those files also contain numeric alignment values. No numeric value is fabricated for `x`.

## What the pinned preprocessor actually supports

The inspected OSSQ preprocessing path confirms that scanned-score system detection reads the first line of `sq<id>_scanned.csv` as the PDF page-range selector. Later systemwise alignment code requires the scanned alignment file to exist and pairs generated system images with generated symbolic segments under its own pipeline invariants.

B8O therefore does not infer undocumented semantics for later CSV marker rows. Exact raw metadata identity is retained, and uncertain marker semantics fail closed.

## Safety boundary

B8O grants none of the following:

- no system-image bytes are materialized;
- no image/MusicXML pair is independently approved;
- no Stage 8 real-data admission is granted;
- no TRAIN or VALIDATION split is assigned;
- no model training runs;
- no TEST artifact bytes are opened;
- no production authority is granted;
- no commercial-use authority is granted.

`READY` means only **ready to attempt deterministic materialization**.

## Next admissible step

For the seven READY records:

1. re-fetch the exact B8N-pinned source PDF bytes and verify their SHA-256;
2. bind the exact per-score YOLO reproduction metadata and model identity;
3. reproduce page images and system crops using the pinned OSSQ preprocessing path;
4. produce the corresponding systemwise MusicXML segments;
5. hash every image and MusicXML segment;
6. independently review pair correctness;
7. only accepted pairs may proceed to leakage-safe TRAIN/VALIDATION assignment and Stage 8 quarantine/intake.

Schubert `7397765` stays excluded from that batch until the upstream `a` marker meaning is independently resolved or a separate verified pairing path is supplied.
