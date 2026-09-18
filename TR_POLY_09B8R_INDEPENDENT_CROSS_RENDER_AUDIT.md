# TR-POLY-09B8R — Independent cross-render audit

## Purpose

B8R provides an independent image ↔ MusicXML review path for the exact 412 B8P candidate pairs. It does not trust matching filenames as review evidence.

## Frozen upstream identity

- B8P receipt: `3f9f2df43b287dd95e03eaf1ffc4c5a3c9e25bdced8f09675918bc9b1f99e0cf`
- candidate population: 412 systems across the seven B8P READY score records
- upstream score `7397765` remains excluded

## Independent review method

For every candidate, B8R:

1. reproduces the exact B8P scan and MusicXML bytes in ephemeral CI storage;
2. verifies those bytes against the committed B8P hashes;
3. independently renders each MusicXML candidate with pinned Verovio 6.2.1 and CairoSVG 2.8.2;
4. removes staff-line-dominated rows and computes two bounded structural signatures:
   - horizontal ink-distribution profile;
   - coarse two-dimensional ink occupancy;
5. compares every scanned system against every independently rendered MusicXML system **within the same score**;
6. accepts a `VERIFIED_CANDIDATE` only when the declared pair is the unique mutual nearest neighbour under both signatures;
7. if a wrong scan/XML combination is a mutual-nearest match under both signatures, the complete score remains fail-closed from automatic verification.

Pairs that do not satisfy the conservative rule remain `REVIEW_REQUIRED`; B8R does not infer correctness from filename identity or B8P reproduction alone.

## Evidence boundary

The committed evidence, when discovery succeeds, is hash-only. Raw source PDFs, scanned-system PNGs, MusicXML bytes and independent render PNGs are not committed or uploaded as evidence.

B8R does **not** by itself grant:

- Stage 8 real-data admission;
- TRAIN/VALIDATION assignment;
- TEST access;
- model-training authority;
- production authority;
- commercial-use authority.

The B8Q contract remains the gate that converts independently supported review evidence into pair-review decisions, and existing Stage 8-0/8-1 rights/intake/leakage contracts remain authoritative after that.
