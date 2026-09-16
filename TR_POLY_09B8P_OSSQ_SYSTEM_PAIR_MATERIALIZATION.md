# TR-POLY-09B8P — OSSQ System-Pair Materialization

B8P deterministically materializes real scanned system-image ↔ systemwise-MusicXML candidate pairs for the exact seven B8O READY OSSQ scores and records hash-only evidence.

It is a materialization/reproducibility package, not a pairing-approval or training-admission package.

## Frozen inputs

B8P binds:

- B8N source-byte receipt: `9f9b678e2365ec849cc19424b28d8dbdb435af3a5a9ef5b47cf7a460e72a801c`;
- B8O pairing-preflight receipt: `b77718f90f5865082a36f18da8701457baa5014d278e4ad33e49e727abbab65a`;
- OSSQ camera-ready commit: `7a17e45cddc0b7064fc3a179b62caeb57595e993`;
- preprocessor commit: `bdea0d1829c9db84480ebd2e0385f6f5fe324274`;
- the exact B8N-pinned source-document SHA-256 for every included score;
- exact upstream YOLO-info bytes and exclusions;
- exact materialized PNG and MusicXML byte identities.

READY score IDs:

- `7070781`
- `7075297`
- `7078259`
- `7093885`
- `7103818`
- `7108150`
- `8071278`

Blocked and excluded:

- `7397765` — the upstream `a` alignment marker remains uninterpreted and fail-closed.

## Canonical materialization evidence

Discovery run #4 on PR #171 head `88c371aedec0c8a737ad08dd0ce3701697727a6c` completed successfully and emitted the canonical B8P receipt fingerprint:

`3f9f2df43b287dd95e03eaf1ffc4c5a3c9e25bdced8f09675918bc9b1f99e0cf`

The canonical hash-only receipt is committed at:

`evidence/ossq_b8p_system_pair_materialization.json`

Total materialized candidate pairs: **412**.

Per-score pair counts:

- `7070781`: 9
- `7075297`: 28
- `7078259`: 26
- `7093885`: 26
- `7103818`: 87
- `7108150`: 117
- `8071278`: 119

The receipt was pinned only after an exact rerun reproduced the historical discovery fingerprint. The repository verification workflow then requires the live canonical receipt to equal the committed canonical receipt; changing the expected evidence to bless drift is not an allowed repair.

## Evidence policy

Raw third-party source PDF, generated PNG and generated MusicXML bytes are processing inputs only. B8P does not commit or upload those raw bytes as evidence. The committed evidence consists of hashes, byte counts, dimensions, segment identities, upstream identities, exclusion evidence and authority flags.

## Authority boundary

B8P explicitly grants none of the following:

- independent image ↔ MusicXML pairing approval;
- Stage 8 real-data admission;
- TRAIN/VALIDATION assignment;
- model-training authority;
- access to sealed TEST artifact bytes;
- production authority;
- commercial-use authority.

Therefore the 412 pairs are **real materialized candidates**, not yet 412 approved training examples.

Real OMR accuracy remains **UNKNOWN / NOT MEASURED** until an admitted real TRAIN corpus is trained and the frozen checkpoint is evaluated on the complete frozen VALIDATION population.

## Next admissible step

After PR #171 is merged on an exact HEAD with both standard CI and the dedicated B8P VERIFIED job green:

1. independently verify image ↔ MusicXML correspondence for each candidate pair;
2. classify each pair as `VERIFIED`, `REVIEW_REQUIRED` or `BLOCKED`;
3. admit only `VERIFIED` pairs through the existing Stage 8 real-data intake path;
4. preserve exact provenance, rights/training permission and byte lineage;
5. split by whole musical-work/source-document family so TRAIN and VALIDATION cannot share a family;
6. keep TEST sealed.
