"""Live TR-POLY-09B8R independent cross-render audit runner.

The runner re-materializes the exact seven B8P READY OSSQ scores in an ephemeral
checkout, verifies the committed B8P receipt byte-for-byte, independently renders
each systemwise MusicXML through pinned Verovio/CairoSVG, and emits hash-only
cross-render audit evidence. Raw PDFs, system images, MusicXML and rendered PNGs
are never committed or uploaded as evidence.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import sys

import cairosvg
import verovio

from b8p_materialize_ossq_system_pairs import materialize, prepare_sources
from st_omr_training.poly_v2_ossq_pairing_preflight import B8O_PAIRING_SOURCE_SPECS
from st_omr_training.poly_v2_ossq_system_pair_materialization import (
    B8P_EXPECTED_B8N_RECEIPT_SHA256,
    B8P_EXPECTED_B8O_RECEIPT_SHA256,
    build_b8p_materialization_receipt,
    receipt_to_json as b8p_receipt_to_json,
)
from st_omr_training.poly_v2_ossq_cross_render_audit import (
    B8R_EXPECTED_B8P_RECEIPT_SHA256,
    build_cross_render_audit,
    extract_structural_signature,
    receipt_to_json,
    reviewer_identity_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
B8P_RECEIPT = ROOT / "evidence" / "ossq_b8p_system_pair_materialization.json"
B8R_RECEIPT = ROOT / "evidence" / "ossq_b8r_cross_render_audit.json"

EXPECTED_VEROVIO_VERSION = "6.2.1"
EXPECTED_CAIROSVG_VERSION = "2.8.2"
EXPECTED_PILLOW_VERSION = "12.3.0"

RENDER_OPTIONS = {
    "adjustPageHeight": True,
    "adjustPageWidth": True,
    "breaks": "none",
    "font": "Leipzig",
    "fontFallback": "Leipzig",
    "pageHeight": 10000,
    "pageWidth": 60000,
    "pageMarginTop": 20,
    "pageMarginRight": 20,
    "pageMarginBottom": 20,
    "pageMarginLeft": 20,
    "scale": 100,
    "svgFormatRaw": True,
    "svgViewBox": True,
    "svgHtml5": False,
    "svgRemoveXlink": True,
    "xmlIdChecksum": True,
}


def _runtime_versions() -> tuple[str, str, str]:
    verovio_package = metadata.version("verovio")
    cairosvg_package = metadata.version("CairoSVG")
    pillow_package = metadata.version("Pillow")
    if verovio_package != EXPECTED_VEROVIO_VERSION:
        raise RuntimeError(f"expected verovio {EXPECTED_VEROVIO_VERSION}, got {verovio_package}")
    if cairosvg_package != EXPECTED_CAIROSVG_VERSION:
        raise RuntimeError(f"expected CairoSVG {EXPECTED_CAIROSVG_VERSION}, got {cairosvg_package}")
    if pillow_package != EXPECTED_PILLOW_VERSION:
        raise RuntimeError(f"expected Pillow {EXPECTED_PILLOW_VERSION}, got {pillow_package}")
    runtime = str(verovio.toolkit().getVersion())
    if not runtime.startswith(EXPECTED_VEROVIO_VERSION):
        raise RuntimeError(f"Verovio runtime differs from pinned package: {runtime}")
    return verovio_package, cairosvg_package, pillow_package


def _render_musicxml_to_png(musicxml_bytes: bytes) -> bytes:
    toolkit = verovio.toolkit()
    runtime = str(toolkit.getVersion())
    if not runtime.startswith(EXPECTED_VEROVIO_VERSION):
        raise RuntimeError(f"Verovio runtime drift: {runtime}")
    if toolkit.setInputFrom("xml") is False:
        raise RuntimeError("Verovio rejected explicit MusicXML input mode")
    if toolkit.setOptions(RENDER_OPTIONS) is False:
        raise RuntimeError("Verovio rejected B8R render options")
    text = musicxml_bytes.decode("utf-8", errors="strict")
    if toolkit.loadData(text) is False:
        raise RuntimeError("Verovio rejected systemwise MusicXML")
    page_count = toolkit.getPageCount()
    if page_count != 1:
        raise RuntimeError(f"B8R expects one rendered system page, got {page_count}")
    svg = toolkit.renderToSVG(1, True)
    if not isinstance(svg, str) or "<svg" not in svg:
        raise RuntimeError("Verovio returned invalid SVG")
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), background_color="white")
    if not isinstance(png, bytes) or not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("CairoSVG did not return PNG bytes")
    return png


def _pair_paths(dataset_root: Path, score_id: str, segment_id: str) -> tuple[Path, Path]:
    specs = {item.score_id: item for item in B8O_PAIRING_SOURCE_SPECS}
    spec = specs[score_id]
    score_dir = dataset_root / "scores" / spec.work_path
    return (
        score_dir / "images" / "scanned" / "systemwise" / f"{segment_id}.png",
        score_dir / "musicxml" / "scanned" / "systemwise" / f"{segment_id}.musicxml",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--preprocessor-root", required=True, type=Path)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    preprocessor_root = args.preprocessor_root.resolve()
    if not B8P_RECEIPT.is_file():
        raise RuntimeError("committed B8P receipt is missing")

    committed_b8p = B8P_RECEIPT.read_text(encoding="utf-8").strip()
    committed_payload = json.loads(committed_b8p)
    if committed_payload.get("receipt_sha256") != B8R_EXPECTED_B8P_RECEIPT_SHA256:
        raise RuntimeError("committed B8P receipt identity differs from B8R freeze")

    source_hashes, task_lines = prepare_sources(dataset_root)
    task_file = dataset_root / "b8r_ready_scores.txt"
    task_file.write_text("\n".join(task_lines) + "\n", encoding="utf-8")
    materialize(dataset_root, preprocessor_root, task_file)

    live_b8p = build_b8p_materialization_receipt(
        dataset_root=dataset_root,
        b8n_source_sha256_by_score=source_hashes,
        b8n_receipt_sha256=B8P_EXPECTED_B8N_RECEIPT_SHA256,
        b8o_receipt_sha256=B8P_EXPECTED_B8O_RECEIPT_SHA256,
    )
    if b8p_receipt_to_json(live_b8p) != committed_b8p:
        raise RuntimeError("live B8P materialization differs from committed receipt before B8R")

    verovio_version, cairosvg_version, pillow_version = _runtime_versions()
    reviewer = reviewer_identity_sha256(
        verovio_version=verovio_version,
        cairosvg_version=cairosvg_version,
        pillow_version=pillow_version,
    )

    pairs = []
    scanned_signatures = {}
    render_signatures = {}
    for pair in live_b8p.pairs:
        image_path, xml_path = _pair_paths(dataset_root, pair.score_id, pair.segment_id)
        image_bytes = image_path.read_bytes()
        xml_bytes = xml_path.read_bytes()
        if sha256(image_bytes).hexdigest() != pair.image_sha256:
            raise RuntimeError(f"B8R image hash mismatch for {pair.segment_id}")
        if sha256(xml_bytes).hexdigest() != pair.musicxml_sha256:
            raise RuntimeError(f"B8R MusicXML hash mismatch for {pair.segment_id}")
        rendered_png = _render_musicxml_to_png(xml_bytes)
        render_sha = sha256(rendered_png).hexdigest()
        pairs.append(
            {
                "score_id": pair.score_id,
                "segment_id": pair.segment_id,
                "image_sha256": pair.image_sha256,
                "musicxml_sha256": pair.musicxml_sha256,
                "independent_render_png_sha256": render_sha,
            }
        )
        scanned_signatures[pair.segment_id] = extract_structural_signature(image_bytes)
        render_signatures[pair.segment_id] = extract_structural_signature(rendered_png)

    audit = build_cross_render_audit(
        b8p_receipt_sha256=live_b8p.receipt_sha256,
        pairs=pairs,
        scanned_signatures=scanned_signatures,
        render_signatures=render_signatures,
        reviewer_identity_sha=reviewer,
    )
    canonical = receipt_to_json(audit)
    if B8R_RECEIPT.exists():
        committed = B8R_RECEIPT.read_text(encoding="utf-8").strip()
        if committed != canonical:
            raise RuntimeError("live B8R cross-render audit differs from committed receipt")
        print("B8R_MODE=VERIFIED")
        print("B8R_LIVE_AUDIT_MATCHES_COMMITTED_RECEIPT=true")
    else:
        print("B8R_MODE=DISCOVERY")
        print("B8R_RECEIPT_JSON=" + canonical)

    print(f"B8R_RECEIPT_SHA256={audit.receipt_sha256}")
    print("B8R_DECISION_COUNTS=" + ",".join(f"{name}:{count}" for name, count in audit.decision_counts))
    print("B8R_SCORE_CROSS_CONFLICTS=" + ",".join(f"{score}:{count}" for score, count in audit.score_cross_conflicts))
    print(f"B8R_REVIEWER_IDENTITY_SHA256={audit.reviewer_identity_sha256}")
    print("B8R_RAW_PDF_BYTES_PERSISTED_AS_EVIDENCE=false")
    print("B8R_RAW_PAIR_BYTES_PERSISTED_AS_EVIDENCE=false")
    print("B8R_STAGE8_ADMISSION_AUTHORITY=false")
    print("B8R_TRAIN_VALIDATION_ASSIGNMENT_AUTHORITY=false")
    print("B8R_TEST_ARTIFACT_BYTES_ACCESSED=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
