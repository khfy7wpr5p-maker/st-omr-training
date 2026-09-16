# TR-POLY-09B8Q — OSSQ Pair Review → Stage 8 Admission Boundary

B8Q begins only after the exact B8P materialization receipt is merged and live-reproduced.

## Frozen input

- merged B8P receipt SHA-256: `3f9f2df43b287dd95e03eaf1ffc4c5a3c9e25bdced8f09675918bc9b1f99e0cf`
- B8P pair population: 412 system-image ↔ MusicXML candidates
- READY score records: `7070781`, `7075297`, `7078259`, `7093885`, `7103818`, `7108150`, `8071278`
- excluded score: `7397765`

B8P proves deterministic reproduction of the exact pair bytes. It explicitly does **not** grant independent pairing-review authority or Stage 8 admission authority.

## B8Q rule

Every exact B8P pair starts as:

`REVIEW_REQUIRED / unreviewed-v1 / independent-review-not-yet-performed`

A pair may become `VERIFIED` only if all of the following are present:

1. the exact B8P `segment_id` matches;
2. the exact B8P image SHA-256 matches;
3. the exact B8P MusicXML SHA-256 matches;
4. the decision uses a separately defined independent review method;
5. independent review evidence has its own SHA-256;
6. the reviewer implementation/person identity evidence has its own SHA-256;
7. the canonical reason is `independent-pair-review-passed`.

B8P materialization identity alone can never satisfy those conditions.

## Initial scientific status

At package creation, the only justified classification is:

- VERIFIED: 0
- REVIEW_REQUIRED: 412
- BLOCKED: 0 among the 412 B8P candidates
- score `7397765`: remains excluded upstream and is not part of the 412-candidate population

This is deliberate fail-closed behavior, not a failure of materialization.

## Independent review methods

The contract currently reserves two explicit independent methods:

- `independent-visual-v1`
- `independent-cross-render-v1`

Merely naming a method is insufficient. VERIFIED/BLOCKED records must bind separate review evidence and reviewer identity hashes. `unreviewed-v1` can only produce `REVIEW_REQUIRED`.

## Stage 8 boundary

Even a B8Q `VERIFIED` pair is only eligible to continue to Stage 8 quarantine preparation. B8Q grants none of the following:

- no Stage 8 admission;
- no TRAIN/VALIDATION assignment;
- no byte receipt;
- no leakage clearance;
- no TEST access;
- no training authority;
- no production authority;
- no commercial-use authority.

The existing Stage 8-0/8-1 contracts remain authoritative for rights evidence, family-exclusive split assignment, source/image/MusicXML byte verification, semantic fingerprinting, perceptual near-duplicate checks, quarantine receipts, and sealed TEST behavior.

## Next executable work

1. produce independent review evidence for B8P pairs without using B8P reproducibility itself as the verifier;
2. classify each pair as `VERIFIED`, `REVIEW_REQUIRED`, or `BLOCKED`;
3. send only exact `VERIFIED` pair identities to Stage 8 quarantine construction;
4. bind B8M rights evidence and exact source-document identity;
5. assign complete work/source families to TRAIN or VALIDATION without leakage;
6. run Stage 8-1 byte/semantic/perceptual validation on those quarantined records;
7. admit only records that pass every frozen Stage 8 gate.

Until independent pair review exists, real model training remains `NOT STARTED` and real model accuracy remains `NOT MEASURED`.
