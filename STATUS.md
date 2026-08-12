# ST-OMR Training Lab Status

This file is the current stage-status source for this repository.

## Current repository phase

The repository is public. Stage 0 safety/architecture baseline, Stage 1 ST Music Generator, Stage 2 MusicXML Pipeline, and Stage 3 Renderer Integration are complete on `main`.

The exact pre-Stage-4 `main` commit `23739ddfab618a0406836e94bb0ced1a124f8886` completed GitHub-hosted CI successfully in run `31648164533`. The integrated state through Stage 3 is therefore `CI VERIFIED`.

Stage 4 Controlled Degradation is now the active bounded implementation package on branch `stage-4-controlled-degradation`. Stage 5 Dataset Validation has not started and remains locked.

## Stage status

| Stage | Description | Status |
|---|---|---|
| 0-A | Repository baseline | ✅ Complete |
| 0-B | Architecture and data boundaries | ✅ Complete |
| 0-C | Canonical data contract | ✅ Complete |
| 0-D | Local verification strategy | ✅ Complete |
| 0-E | GitHub CI | ✅ Complete — main CI green |
| 0-F | Architecture consistency audit | ✅ Complete |
| 0-G | Baseline documentation | ✅ Complete |
| 0 | Safety and architecture baseline | ✅ Closed |
| 1-A | Canonical core model | ✅ Complete |
| 1-B | Independent V1 validator | ✅ Complete |
| 1-C | Score structure model and validator | ✅ Complete |
| 1-D | Deterministic ST Music Generator v1 | ✅ Complete |
| 1 | ST Music Generator | ✅ Closed — integrated main CI verified |
| 2-A | MusicXML contract freeze | ✅ Complete |
| 2-B | Deterministic MusicXML 4.0 writer | ✅ Complete |
| 2-C | Offline XSD + independent MusicXML validator | ✅ Complete |
| 2-D | Supported-V1 semantic round-trip verifier | ✅ Complete |
| 2 | MusicXML pipeline | ✅ Closed — integrated main CI verified |
| 3 | Renderer integration | ✅ Closed — integrated main CI verified |
| 4 | Controlled degradation | 🔄 Active package — PR/CI gate pending |
| 5 | Dataset validation | 🔒 Not started |
| 6 | Synthetic Dataset v1 | 🔒 Not started |
| 7 | Baseline ST-OMR training | 🔒 Not started |
| 8 | Real-data fine-tuning | 🔒 Not started |
| 9 | Benchmark and candidate decision | 🔒 Not started |
| 10 | ScoreMosaic candidate integration | 🔒 Not started |

## Completed evidence through Stage 3

- Stage 2-B merged through PR #8 at main commit `1940d43b3986e5fe359aa79b86cc2af26e96fe98`.
- Stage 2-C merged through PR #9 at main commit `81cbfdb2958b8b8b8f4ee5cbd50960a7a75049f0`.
- Stage 2-D merged through PR #10 at main commit `4f36da277540d5a1b7a074215f2def968db73739`.
- Stage 3 merged through PR #11 at main commit `3b1e94cea8ac3a26a5df1e2038acc4331a24d371`.
- CI baseline merged through PR #12 at main commit `5abbc9859a4a69bf9a17936bc41e722256f87472` and post-merge run `31647615123` succeeded.
- Current-state documentation synchronized through PR #13 at main commit `23739ddfab618a0406836e94bb0ced1a124f8886` and post-merge run `31648164533` succeeded.

The `CI VERIFIED` statement applies to the exact integrated `main` content GitHub-hosted CI exercised. Stage 4 requires its own new evidence.

## Stage 4 package

Branch: `stage-4-controlled-degradation`

Base: `23739ddfab618a0406836e94bb0ced1a124f8886`

Contract: [DEGRADATION_CONTRACT.md](DEGRADATION_CONTRACT.md)

Implemented V1 scope:

- Stage 3 self-contained SVG page + explicit family/hash lineage input;
- independent SVG/hash/resource preflight before rasterization;
- direct runtime pins `CairoSVG==2.8.2` and `Pillow==12.3.0`;
- clean deterministic grayscale PNG rasterization with bounded width and pixel budget;
- conservative deterministic transforms: expanded-canvas rotation, Gaussian blur, brightness, contrast, deterministic grayscale noise, and optional JPEG round-trip compression;
- integer public configuration and deterministic `clean`, `light`, and `medium` seeded profiles;
- exact source SVG, clean raster, configuration, final PNG, runtime/platform, and family provenance;
- fail-closed blank/dark/resource/ink-retention checks;
- real Verovio → Stage 4 integration tests.

Explicitly deferred from Stage 4 V1:

- arbitrary crop;
- perspective warp;
- shear or elastic deformation;
- synthetic shadows/occlusion;
- staff/symbol deletion or content-aware erasing;
- dataset manifests/splitting/storage;
- model training;
- real/user score ingestion;
- ScoreMosaic integration.

## Current Stage 4 verification evidence

Available local focused/prototype evidence before GitHub-hosted PR verification:

- focused Stage 4 unit suite: 25 tests passed on the branch-equivalent implementation during development;
- all six Stage 2 golden MusicXML fixtures crossed real Verovio 6.2.1 → Stage 4 medium degradation locally;
- 30 generated scores crossed MusicXML → real Verovio → medium Stage 4 degradation locally;
- 50/50 seeded medium derivatives of a real Verovio SVG produced valid distinct outputs;
- 10/10 deterministic repeats produced byte-identical output locally.

A complete local repository regression is not claimed for this package. The authoritative full-suite integration gate will be GitHub-hosted PR CI in a fresh environment with the branch `requirements.txt`.

Current Stage 4 status is `UNVERIFIED — FINAL PR CI PENDING` until the exact final branch head passes the GitHub-hosted workflow.

## CI baseline

The active `.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual dispatch using GitHub-hosted Ubuntu 24.04 / Python 3.13, read-only repository contents permission, pinned GitHub action commits, dependency installation from `requirements.txt`, pinned runtime checks, `pip check`, complete unittest discovery, and Python compile validation.

The Stage 4 branch extends the pinned runtime check to CairoSVG 2.8.2 and Pillow 12.3.0 so the new image path is exercised in the same full-suite gate.

## Next gate

Complete Stage 4 scope/diff review, open the bounded Stage 4 PR, and require GitHub-hosted CI to pass on the exact final PR head. Merge then requires separate explicit approval. After merge, the exact resulting `main` commit must also pass GitHub-hosted CI before Stage 4 closes and Stage 5 may begin.
