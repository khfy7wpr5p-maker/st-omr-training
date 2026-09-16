# TR-POLY-09B8L — OSSQ per-score rights/provenance audit

B8K selected the OSSQ-OMR camera-ready source family at commit
`7a17e45cddc0b7064fc3a179b62caeb57595e993` but intentionally kept the
IMSLP-derived scanned track blocked. B8L adds the next fail-closed gate.

## Pinned upstream metadata

B8L binds the camera-ready source inventory to:

- `data/scanned_score_types.tsv`
  - Git blob: `35ebe84f0c9f03a41231c92fa21183242bf12774`
  - 122 score entries
  - 116 unique work paths
- `data/scores_w_pub.yaml`
  - Git blob: `7a72b220f4faa897900f51bcb85e49b845cb1304`

The TSV supplies score identity, score path, source type, MuseScore URL, IMSLP
file identity and set identity. The publication metadata supplies publisher,
publication year, upstream copyright text and source URL context for later
review.

The public loader recomputes the Git blob SHA-1 from the caller-supplied TSV
bytes. A same-named or modified file cannot silently replace the B8K-pinned
camera-ready metadata.

## Rights policy

An OSSQ/IMSLP upstream copyright label is metadata, not independent rights
approval. In particular, text such as `Public Domain` or `Creative Commons Zero
1.0` does not by itself create a B8L training decision.

Every scanned/manuscript/part-book score entry must receive one
`OssqScoreRightsReview` with:

- exact OSSQ score ID;
- exact IMSLP file ID;
- exact OSSQ source type;
- hash of the extracted upstream metadata evidence;
- separate provenance-evidence SHA-256;
- separate independent rights-evidence SHA-256 for approved records;
- explicit `RightsBasis`;
- explicit review state;
- explicit training/commercial/redistribution permission booleans when approved.

`rights_evidence_sha256` may not equal the upstream metadata-evidence hash for an
approved record. This prevents the OSSQ metadata itself from being treated as
an independent legal/source-rights review.

## Source types

B8L requires per-score rights review for the real-image source types:

- `0` — manuscript;
- `1`, `1\``, `1\`\`` — scanned score variants;
- `3` — part-book source;
- `4` — first-violin-only source.

Type `2` is the digitally engraved source and is not part of the
IMSLP-scanned-track rights-review population.

## Shared IMSLP files

Some OSSQ score entries share one IMSLP source file. Those entries are allowed,
but they may not carry contradictory rights/provenance decisions. This is
important for known clusters such as multiple works/movements sourced from one
physical scan.

## Output

`audit_ossq_scanned_score_rights(...)` requires complete review coverage for the
entire scanned candidate population and emits an immutable receipt containing:

- inventory fingerprint;
- B8K source-selection fingerprint;
- review-manifest fingerprint;
- reviewed population count;
- training-candidate score IDs;
- commercial-candidate score IDs;
- blocked score IDs;
- count of shared IMSLP source groups.

A pending or rejected score remains blocked. An approved score may become a
candidate for later Stage 8 byte admission, but the B8L receipt itself grants
neither production authority nor commercial-use authority.

## What B8L does not do

B8L does not:

- download IMSLP PDFs;
- open score-image bytes;
- install-pin external artifacts;
- create Stage 8 byte receipts;
- create Native V2 samples;
- train a model;
- open TEST;
- declare the whole OSSQ scanned corpus legally cleared;
- grant production or commercial deployment authority.

## Current blocker after B8L

The next evidence-bearing step is to populate the per-score review records from
independently captured source/provenance/rights evidence. Only approved scores
can then proceed to exact PDF/image byte acquisition, SHA-256 install pinning,
Stage 8-1 byte receipts and the B8A → B8I → B8B → B8J baseline path.
