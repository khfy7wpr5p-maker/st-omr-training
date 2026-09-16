# TR-POLY-09B8N — Exact OSSQ source PDF byte pins

Date: 2026-09-16

## Purpose

B8M approved five exact OSSQ/IMSLP source PDFs as research-training candidates at the rights/provenance layer. B8N binds those reviewed source identities to the actual PDF bytes with SHA-256 and byte counts.

B8N is deliberately source-document-only. It does not make the PDFs themselves training samples and does not create Stage 8-1 sample receipts.

## Exact byte receipt

Canonical receipt SHA-256:

`9f9b678e2365ec849cc19424b28d8dbdb435af3a5a9ef5b47cf7a460e72a801c`

Rights-evidence manifest SHA-256:

`806b6d538cea090ebea8a7e61a54dcd15003a3d99dbb9cfa5e38956e59410e38`

| IMSLP file | OSSQ score IDs | Bytes | SHA-256 |
|---|---|---:|---|
| `#04047` | `7397765` | 7,131,205 | `55db640f6b3ec6715a0bd5527ed3f26cca181036a4e848a32027dd7d29bef48e` |
| `#04755` | `8071278` | 6,487,137 | `e2e554b844dc16b40f479939126ff5d71fa44263a7338b993a9ee1dbbcca3796` |
| `#64136` | `7103818` | 3,703,239 | `64406ae67f690b32f689bb60169287d0a6d514d13437b6027ee999381a43cb01` |
| `#64141` | `7070781`, `7075297`, `7078259`, `7093885` | 3,589,762 | `6724c52cb2795f9724455bc73184ad7d35de45b61e554f5c2efa99facc680ae5` |
| `#242305` | `7108150` | 4,788,630 | `129a4e626db0db6cd2771b0bfaec7356001ae324bbdfef06b6132be75806f145` |

The hash-only receipt is stored at `evidence/ossq_b8n_source_byte_pins.json`.

## Two-phase verification

The dedicated `OSSQ B8N Source Byte Pin` workflow operates in two modes:

1. **DISCOVERY** — when no committed receipt exists, fetch only the five exact B8M-reviewed HTTPS URLs, require PDF magic, enforce the 64 MiB source limit, and print the canonical hash-only receipt.
2. **VERIFIED** — once the receipt is committed, re-fetch all five exact URLs and fail if any SHA-256, byte count, URL, IMSLP ID or score population differs.

Raw PDF bytes are not committed and are not uploaded as workflow artifacts.

## Authority boundary

B8N explicitly keeps all of the following false:

- `raw_source_bytes_persisted`
- `raw_source_bytes_redistributed`
- `stage8_admission_authority`
- `test_artifact_bytes_accessed`
- `production_authority`
- `commercial_use_authority`

Therefore B8N proves exact source-document identity only. It does not prove image/MusicXML pairing quality and does not authorize model training by itself.

## Next step

For these byte-pinned source documents, construct reviewed image/MusicXML pairs under the existing Stage 8 quarantine contract. Only after exact image bytes, exact MusicXML bytes, semantic identity, pairing evidence, split assignment and Stage 8-1 validation exist can samples become an admitted real TRAIN/VALIDATION corpus.
