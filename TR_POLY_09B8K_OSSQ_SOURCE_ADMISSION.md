# TR-POLY-09B8K — OSSQ-OMR source selection and admission boundary

Status: implementation package; no external corpus bytes are admitted by this package.

## Purpose

B8A–B8J made the first real Native Polyphonic V2 baseline executable once a real corpus exists. B8K addresses the remaining source-selection blocker without weakening any Stage 8 rights/provenance rules.

The primary external source family selected for further admission work is **OSSQ-OMR**, the camera-ready dataset accompanying the ISMIR 2026 string-quartet OMR benchmark.

Exact upstream snapshot reviewed:

`MALerLab/ossq-omr@7a17e45cddc0b7064fc3a179b62caeb57595e993`

The upstream camera-ready release documents paired scanned/synthetic score images, symbolic targets, score-level split design, source/provenance metadata, and a CC0 dataset license.

## Why OSSQ-OMR is technically relevant

The published release describes:

- 116 string-quartet works by 47 composers;
- 122 MuseScore source files;
- 24,544 system images total;
- 13,240 synthetic system images;
- 11,304 scanned system images;
- 98,172 staff/part images total;
- 52,960 synthetic staff images;
- 45,212 scanned staff images;
- score-level splits intended to prevent cross-split leakage;
- tracked MuseScore/MusicXML annotation sources and scanned alignment metadata.

These properties make OSSQ-OMR materially closer to the ST-OMR Native V2 goal than symbol-only datasets or synthetic-only corpora.

## License boundary

The reviewed upstream `LICENSE.txt` is CC0 1.0 Universal. The upstream README states that the annotation sources and bulk-distributed derived formats are released under CC0.

B8K nevertheless separates two components:

### 1. Annotation sources + publisher-created synthetic/derived artifacts

Registry state: `LICENSE_VERIFIED`

Use class: `COMMERCIAL_CLEAN`

Reason: the dataset publisher expressly releases these materials under CC0. They are still not training-ready because ST-OMR has not install-pinned an exact artifact SHA-256.

### 2. IMSLP-derived scanned-image track

Registry state: `CANDIDATE`

Use class: `LICENSE_REVIEW_REQUIRED`

Reason: the upstream pipeline identifies IMSLP PDFs as the source of the scanned track. CC0 only covers rights held by the affirmer and its own legal text disclaims responsibility for clearing third-party rights. ST-OMR therefore requires per-score source/provenance and rights evidence before any scanned bytes can be admitted.

This is intentionally stricter than simply trusting a repository-level license label.

## What B8K does not do

B8K does **not**:

- download OSSQ-OMR data;
- download IMSLP PDFs;
- open or enumerate any sealed TEST bytes;
- create TRAIN/VALIDATION splits;
- install-pin an external artifact;
- admit any scanned image for training;
- run B8A/B8I/B8B/B8J on external data;
- train a model;
- claim real quality numbers;
- grant production authority;
- grant commercial-use authority for the scanned track.

## Next admissible step

The next data step is a deterministic, metadata-first rights/provenance audit over the OSSQ-OMR scanned source inventory. Only records whose exact scan source and rights evidence pass the existing Stage 8 admission rules may proceed to byte acquisition and Stage 8-1 validation.

After that:

```text
per-score scanned-source rights/provenance audit
        ↓
approved source inventory only
        ↓
exact artifact acquisition + SHA-256 install pin
        ↓
Stage 8-1 byte receipts
        ↓
Native V2 materialization
        ↓
B8A → B8I → B8B → B8J
        ↓
B6 real baseline training
```

TEST remains sealed throughout B8K.
