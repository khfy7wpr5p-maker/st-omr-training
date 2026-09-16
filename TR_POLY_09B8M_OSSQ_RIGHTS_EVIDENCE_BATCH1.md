# TR-POLY-09B8M — OSSQ independent rights evidence batch 1

Date: 2026-09-16

## Purpose

B8L requires one independent rights/provenance decision for every OSSQ scanned-track score before any score can become a Stage 8 training candidate. B8M records the first independently checked source batch.

This package does **not** download, open, install-pin, admit, train on, or evaluate any score PDF/image bytes.

## Batch scope

Five exact IMSLP source files were independently checked on their official IMSLP work pages. Because Mozart K.464 has four OSSQ movement records sharing one IMSLP source, the batch covers eight OSSQ score records.

| IMSLP file | OSSQ score IDs | Work | IMSLP evidence result | B8M decision |
|---|---|---|---|---|
| `#64141` | `7070781`, `7075297`, `7078259`, `7093885` | Mozart, String Quartet No.18 K.464 | exact file entry shows Breitkopf & Härtel 1882 and `Public Domain` | research-training candidate only |
| `#64136` | `7103818` | Mozart, String Quartet No.14 K.387 | exact file entry shows Breitkopf & Härtel 1882 and `Public Domain` | research-training candidate only |
| `#04755` | `8071278` | Beethoven, String Quartet No.1 Op.18 No.1 | exact file entry shows Breitkopf & Härtel 1862 / Dover reprint and `Public Domain` | research-training candidate only |
| `#04047` | `7397765` | Schubert, String Quartet D.810 | exact file entry shows Breitkopf & Härtel 1890 / Dover reprint and `Public Domain` | research-training candidate only |
| `#242305` | `7108150` | Brahms, String Quartet No.1 Op.51 No.1 | exact file entry is the 1926–27 Breitkopf complete-edition scan and the score block is marked `Public Domain` | research-training candidate only |

Official work-page evidence URLs are frozen in `poly_v2_ossq_rights_evidence_batch1.py`. Exact source-PDF URLs are separately bound to the B8K camera-ready OSSQ metadata snapshot.

## Conservative permission surface

For all eight batch records:

- `rights_basis = PUBLIC_DOMAIN`
- `rights_review = APPROVED`
- `independent_rights_review = True`
- `training_allowed = True`
- `commercial_use_allowed = False`
- `redistribution_allowed = False`

This is intentionally narrower than the upstream `Public Domain` label. B8M does not make a jurisdiction-global copyright conclusion and does not grant commercial product authority.

All other scanned-track OSSQ records remain `PENDING` with no usage permissions.

## Identity and leakage rules

B8M binds each reviewed score to:

- the exact B8K camera-ready OSSQ commit;
- the B8L publication-metadata Git blob;
- exact OSSQ score ID;
- exact IMSLP file ID;
- exact source type;
- exact IMSLP work-page URL;
- exact source-PDF URL;
- independent rights evidence fingerprint;
- provenance evidence fingerprint.

The four K.464 score records share one identical source policy, satisfying B8L shared-IMSLP consistency.

## What B8M does not do

- no PDF/image bytes downloaded or opened;
- no artifact SHA-256 install pin;
- no Stage 8 byte receipt;
- no Native V2 materialization;
- no B8I lineage receipt;
- no B8J start permit;
- no model training;
- no VALIDATION quality result;
- no TEST access;
- no production authority;
- no commercial-use authority.

## Next executable step

Acquire only these five approved source PDFs, compute exact SHA-256 identities, verify the bytes correspond to the reviewed IMSLP file IDs, and create a byte-install receipt. Only after byte identity is pinned can the data proceed toward Stage 8 admission and Native V2 conversion.
