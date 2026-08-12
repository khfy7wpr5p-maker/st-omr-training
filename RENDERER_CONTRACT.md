# ST-OMR Renderer Contract

## Status

This document freezes the Stage 3 renderer boundary for ST-OMR Training Lab.

Stage 2 is complete: the canonical score is deterministically serialized to MusicXML 4.0, independently XSD/semantic validated, and verified through the supported-V1 semantic round-trip gate. Stage 3 consumes only MusicXML that passes the existing Stage 2-C validation boundary.

Stage 3 implementation is complete and merged through PR #11. Its exact renderer runtime was locally verified with Verovio 6.2.1, and the current integrated main state containing Stage 3 passed GitHub-hosted CI after PR #12. Runtime evidence remains recorded in [VEROVIO_RUNTIME_EVIDENCE.md](VEROVIO_RUNTIME_EVIDENCE.md).

Changing this renderer contract requires an explicit compatibility/architecture decision and regression review.

## 1. Renderer backend

Stage 3 V1 uses the Verovio Python toolkit as the notation renderer.

Pinned dependency:

```text
verovio==6.2.1
```

The renderer adapter must fail closed when the package is missing or when the installed package/runtime version does not match the pin.

Verovio remains behind `st_omr_training.renderer`; generator, canonical-model, MusicXML writer, validator, and round-trip code must not import or call Verovio directly.

## 2. Input boundary

Renderer input is uncompressed MusicXML 4.0 bytes from the existing Stage 2 boundary.

Before Verovio is imported or called, input must pass `validate_musicxml()`.

Therefore the renderer does not accept:

- arbitrary XML;
- unsupported MusicXML constructs;
- MusicXML with a wrong namespace/version/part structure;
- malformed or oversized XML;
- MusicXML that fails the frozen ST-OMR V1 semantic rules.

The adapter explicitly selects Verovio's direct MusicXML input mode (`xml`). It must not silently switch to another importer.

## 3. Output boundary

Stage 3 V1 emits SVG pages only.

Each output page records:

- one-based page number;
- exact UTF-8 SVG bytes;
- SHA-256 of those SVG bytes.

The render result also records:

- source MusicXML SHA-256;
- renderer name;
- pinned package version;
- runtime version string;
- renderer-adapter version;
- renderer-configuration fingerprint.

PNG/JPEG rasterization is intentionally not part of Stage 3. Rasterization and appearance degradation belong to the separate Stage 4 package so clean vector rendering remains independently auditable.

## 4. Frozen V1 renderer configuration

V1 defaults are explicit rather than relying on mutable library defaults:

```text
pageHeight       = 2970
pageWidth        = 2100
page margins     = 50 / 50 / 50 / 50
scale            = 100
breaks           = auto
font             = Leipzig
fontFallback     = Leipzig
smuflTextFont    = embedded
xmlIdChecksum    = true
svgFormatRaw     = true
svgViewBox       = true
svgHtml5         = false
svgRemoveXlink   = true
adjustPageHeight = false
adjustPageWidth  = false
landscape        = false
mmOutput         = false
scaleToPageSize  = false
```

The configuration is serialized canonically and SHA-256 fingerprinted together with the adapter version, renderer name, and pinned renderer package version.

The V1 font is fixed to Leipzig. Adding another music font is a later explicit compatibility/dataset decision, not an implicit renderer option.

## 5. Determinism scope

`xmlIdChecksum=true` is required so generated Verovio XML/SVG identifiers are derived from input content rather than a time-based random seed.

For identical MusicXML bytes, pinned Verovio package/runtime, renderer configuration, platform/resource bundle, and adapter version, repeated rendering must produce identical SVG bytes in the verified environment.

Cross-platform byte identity is not assumed merely because the Verovio version is pinned. If later dataset production spans multiple operating systems/architectures, renderer-platform evidence must be recorded and compared before artifacts are mixed.

SVG hashes are derived-artifact identities. They do not replace canonical musical content identity or the MusicXML derived-artifact hash.

## 6. SVG safety boundary

Stage 3 accepts only an SVG document root and rejects renderer output containing active/external surfaces outside the V1 need, including:

- `script`;
- `foreignObject`;
- `iframe`;
- `object`;
- `embed`;
- `image`;
- external `href` references;
- external stylesheet imports or HTTP/file URL references.

Internal fragment references such as `href="#glyph-id"` remain allowed because Verovio uses them to reuse SVG glyph definitions.

## 7. Resource and network policy

Stage 3 code must not download fonts, renderer assets, scores, or other resources at render time.

The pinned Verovio Python wheel includes its renderer resources. The V1 SVG text/music-font policy is self-contained/embedded. Missing resources are a render failure, not permission to fetch replacements from the network.

Large renderer binaries/wheels are dependencies, not normal Git repository content.

## 8. Page limits

The adapter renders all pages reported by Verovio and never silently truncates output.

V1 uses a fail-closed upper guard of 64 pages per render call. A page count outside `1..64` is rejected. Changing this limit requires explicit review because it affects resource consumption and downstream dataset assumptions.

## 9. Verification gate

Stage 3 cannot close on mocked adapter tests alone.

Required evidence:

1. focused adapter tests pass;
2. full existing regression suite passes;
3. Python compile validation passes;
4. the exact pinned real Verovio 6.2.1 runtime renders all Stage 2 golden MusicXML fixtures successfully;
5. repeated real-runtime rendering of the same inputs produces identical SVG bytes/digests in the verified local environment;
6. generated ST-OMR scores pass MusicXML validation and real rendering across a bounded stress sample;
7. unsafe/invalid renderer inputs and unsafe SVG surfaces fail closed.

These gates were satisfied before PR #11 merged. The current integrated main state containing the renderer is additionally exercised by GitHub-hosted CI.

If the exact required renderer runtime is unavailable or mismatched in a future environment, rendering must fail closed rather than silently use a different backend/version.

## 10. Explicitly out of scope

Stage 3 does not add:

- controlled blur/noise/skew/perspective/compression;
- SVG-to-PNG/JPEG rasterization;
- dataset creation or splitting;
- model training;
- real-score/user-file ingestion;
- ScoreMosaic runtime integration;
- Guitar TAB;
- deployment/release automation.

Stage 4 Controlled Degradation is the next separate development package.
