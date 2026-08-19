from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import math
import unittest
import xml.etree.ElementTree as ET

from st_omr_training.renderer import RendererConfig, _load_verovio_runtime
from st_omr_training.system_geometry_evidence_v1 import StaffSystemRelation
from st_omr_training.system_geometry_spatial_evidence_v1 import (
    extract_system_geometry_spatial_evidence_v1,
    system_geometry_spatial_rule_design_allowed,
    system_geometry_spatial_runtime_connection_allowed,
)


# This is deliberately a test-only observation surface.  SVG topology is used
# only to locate the known synthetic staff geometry.  Every candidate value
# below is measured from a clean grayscale raster, not from SVG class metadata.
# No candidate threshold or grouping decision is implemented here.
_VARIANTS = (
    {
        "name": "same-system-brace",
        "staff_count": 2,
        "measure_count": 1,
        "breaks_before": (),
        "part_symbol": "brace",
        "page_height": 3200,
        "spacing_staff": 12,
        "spacing_system": 8,
    },
    {
        "name": "same-system-bracket",
        "staff_count": 2,
        "measure_count": 1,
        "breaks_before": (),
        "part_symbol": "bracket",
        "page_height": 3200,
        "spacing_staff": 12,
        "spacing_system": 8,
    },
    {
        "name": "same-system-no-brace-or-bracket",
        "staff_count": 2,
        "measure_count": 1,
        "breaks_before": (),
        "part_symbol": "none",
        "page_height": 3200,
        "spacing_staff": 12,
        "spacing_system": 8,
    },
    {
        # Hard negative with no hidden intermediate staff: the two graphical
        # staves belong to different systems of a one-staff score.
        "name": "different-system-single-staff",
        "staff_count": 1,
        "measure_count": 2,
        "breaks_before": (2,),
        "part_symbol": None,
        "page_height": 4200,
        "spacing_staff": 12,
        "spacing_system": 8,
    },
    {
        # Multi-staff negative preserves the real adjacency problem: the bottom
        # staff of system 1 and top staff of system 2 are a DIFFERENT_SYSTEM
        # pair even though each system independently contains a valid pair.
        "name": "different-system-two-grand-staff-systems-none",
        "staff_count": 2,
        "measure_count": 2,
        "breaks_before": (2,),
        "part_symbol": "none",
        "page_height": 4600,
        "spacing_staff": 12,
        "spacing_system": 8,
    },
)

_RASTER_WIDTH = 1200


def _attributes(staff_count: int, part_symbol: str | None) -> str:
    symbol = ""
    if staff_count == 2 and part_symbol is not None:
        symbol = (
            f'<part-symbol top-staff="1" bottom-staff="2">'
            f"{part_symbol}</part-symbol>"
        )
    clefs = '<clef number="1"><sign>G</sign><line>2</line></clef>'
    if staff_count == 2:
        clefs += '<clef number="2"><sign>F</sign><line>4</line></clef>'
    return f"""
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>{staff_count}</staves>
        {symbol}
        {clefs}
      </attributes>"""


def _notes(staff_count: int) -> str:
    result = (
        '<note><pitch><step>C</step><octave>5</octave></pitch>'
        '<duration>4</duration><voice>1</voice><type>whole</type>'
        '<staff>1</staff></note>'
    )
    if staff_count == 2:
        result += (
            '<backup><duration>4</duration></backup>'
            '<note><pitch><step>C</step><octave>3</octave></pitch>'
            '<duration>4</duration><voice>2</voice><type>whole</type>'
            '<staff>2</staff></note>'
        )
    return result


def _score(variant: dict[str, object]) -> str:
    staff_count = int(variant["staff_count"])
    measure_count = int(variant["measure_count"])
    breaks_before = tuple(int(value) for value in variant["breaks_before"])
    part_symbol = variant["part_symbol"]
    if part_symbol is not None:
        part_symbol = str(part_symbol)

    measures = []
    for number in range(1, measure_count + 1):
        new_system = number in breaks_before
        print_tag = '<print new-system="yes"/>' if new_system else ""
        # Repeat attributes at every explicit break to keep layout conditions
        # symmetric and observable rather than relying on inherited state.
        attributes = (
            _attributes(staff_count, part_symbol)
            if number == 1 or new_system
            else ""
        )
        measures.append(
            f"""
    <measure number="{number}">
      {print_tag}{attributes}
      {_notes(staff_count)}
      <barline location="right"><bar-style>regular</bar-style></barline>
    </measure>"""
        )

    return """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Fixture</part-name></score-part></part-list>
  <part id="P1">""" + "".join(measures) + """
  </part>
</score-partwise>"""


def _render_svg(variant: dict[str, object], *, bounding_boxes: bool) -> str:
    verovio, package_version = _load_verovio_runtime()
    toolkit = verovio.toolkit()
    if not str(toolkit.getVersion()).startswith(package_version):
        raise AssertionError("pinned Verovio runtime mismatch")
    if toolkit.setInputFrom("xml") is False:
        raise AssertionError("fixture input mode rejected")

    config = RendererConfig(
        page_height=int(variant["page_height"]),
        page_width=2400,
        scale=100,
        breaks="encoded",
    )
    options = dict(config.verovio_options())
    options.update(
        {
            "spacingStaff": int(variant["spacing_staff"]),
            "spacingSystem": int(variant["spacing_system"]),
            "svgBoundingBoxes": bounding_boxes,
            "svgContentBoundingBoxes": bounding_boxes,
        }
    )
    if toolkit.setOptions(options) is False:
        raise AssertionError("fixture layout options rejected")
    if toolkit.loadData(_score(variant)) is False:
        raise AssertionError("fixture MusicXML rejected")
    if toolkit.getPageCount() != 1:
        raise AssertionError("raster-observable fixture must remain on one page")
    return toolkit.renderToSVG(1, True)


def _viewbox(svg: str) -> tuple[float, float, float, float]:
    root = ET.fromstring(svg.encode("utf-8"))
    raw = root.attrib.get("viewBox")
    if raw is None:
        raise AssertionError("fixture SVG requires viewBox")
    values = tuple(float(part) for part in raw.replace(",", " ").split())
    if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
        raise AssertionError("fixture SVG viewBox invalid")
    return values  # type: ignore[return-value]


def _clean_raster(svg: str):
    import cairosvg  # type: ignore
    from PIL import Image  # type: ignore

    # Verovio SVG pages have a transparent page background.  Rendering that
    # surface without an explicit background and then dropping alpha turns the
    # transparent page black, which would make every darkness observation 1.0.
    # Composite on white at rasterization time so this audit observes the same
    # black-ink-on-white-page convention as the runtime image pipeline.
    png = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=_RASTER_WIDTH,
        background_color="#ffffff",
    )
    image = Image.open(BytesIO(png)).convert("L")
    if image.width != _RASTER_WIDTH or image.height < 1:
        raise AssertionError("unexpected clean raster geometry")
    low, high = image.getextrema()
    if low >= high or high != 255:
        raise AssertionError("clean raster must contain dark ink on white background")
    return image


def _pixel_box(
    box: tuple[float, float, float, float],
    *,
    viewbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x0, y0, vw, vh = viewbox
    width, height = image_size
    sx = width / vw
    sy = height / vh
    left = max(0, min(width - 1, math.floor((box[0] - x0) * sx)))
    top = max(0, min(height - 1, math.floor((box[1] - y0) * sy)))
    right = max(left + 1, min(width, math.ceil((box[2] - x0) * sx)))
    bottom = max(top + 1, min(height, math.ceil((box[3] - y0) * sy)))
    return left, top, right, bottom


def _darkness_metrics(image, pixel_box: tuple[int, int, int, int]) -> dict[str, float]:
    crop = image.crop(pixel_box)
    width, height = crop.size
    pixels = list(crop.get_flattened_data())
    darkness = [255 - int(value) for value in pixels]
    mean_darkness = sum(darkness) / (255.0 * width * height)
    column_fractions = []
    for x in range(width):
        total = sum(darkness[y * width + x] for y in range(height))
        column_fractions.append(total / (255.0 * height))
    return {
        "mean_darkness_fraction": round(mean_darkness, 9),
        "max_column_darkness_fraction": round(max(column_fractions), 9),
    }


def _pair_raster_observation(report, image, viewbox, pair) -> dict[str, object]:
    staffs = {
        staff.staff_instance_id: staff
        for system in report.systems
        for staff in system.staffs
    }
    first = staffs[pair.staff_a_id]
    second = staffs[pair.staff_b_id]
    upper, lower = sorted((first, second), key=lambda staff: staff.center_y)
    spacing = (upper.staff_spacing + lower.staff_spacing) / 2.0
    if spacing <= 0:
        raise AssertionError("fixture staff spacing must be positive")

    # Left-of-staff raster observation window.  Its geometry is a fixed
    # measurement protocol expressed in staff-space units; it is not a
    # SAME/DIFFERENT decision threshold.
    anchor_x = min(upper.bbox.x_min, lower.bbox.x_min)
    left_x0 = anchor_x - (6.0 * spacing)
    left_x1 = anchor_x + (1.5 * spacing)
    pair_y0 = min(upper.bbox.y_min, lower.bbox.y_min) - spacing
    pair_y1 = max(upper.bbox.y_max, lower.bbox.y_max) + spacing

    pair_box = _pixel_box(
        (left_x0, pair_y0, left_x1, pair_y1),
        viewbox=viewbox,
        image_size=image.size,
    )
    pair_metrics = _darkness_metrics(image, pair_box)

    gap_y0 = upper.bbox.y_max + (0.5 * spacing)
    gap_y1 = lower.bbox.y_min - (0.5 * spacing)
    if gap_y1 <= gap_y0:
        # Preserve geometry truth rather than inventing a positive-height gap.
        gap_metrics = {
            "mean_darkness_fraction": 0.0,
            "max_column_darkness_fraction": 0.0,
        }
        gap_height_staff_spaces = 0.0
    else:
        gap_box = _pixel_box(
            (left_x0, gap_y0, left_x1, gap_y1),
            viewbox=viewbox,
            image_size=image.size,
        )
        gap_metrics = _darkness_metrics(image, gap_box)
        gap_height_staff_spaces = (gap_y1 - gap_y0) / spacing

    return {
        "relation": pair.relation.value,
        "staff_a_id": pair.staff_a_id,
        "staff_b_id": pair.staff_b_id,
        "pair_left_mean_darkness_fraction": pair_metrics["mean_darkness_fraction"],
        "pair_left_max_column_darkness_fraction": pair_metrics[
            "max_column_darkness_fraction"
        ],
        "gap_left_mean_darkness_fraction": gap_metrics["mean_darkness_fraction"],
        "gap_left_max_column_darkness_fraction": gap_metrics[
            "max_column_darkness_fraction"
        ],
        "gap_height_staff_spaces": round(gap_height_staff_spaces, 9),
    }


def _variant_observation(variant: dict[str, object]) -> dict[str, object]:
    evidence_svg = _render_svg(variant, bounding_boxes=True)
    raster_svg = _render_svg(variant, bounding_boxes=False)
    evidence_viewbox = _viewbox(evidence_svg)
    raster_viewbox = _viewbox(raster_svg)
    if evidence_viewbox != raster_viewbox:
        raise AssertionError("bounding-box instrumentation changed fixture viewBox")

    report = extract_system_geometry_spatial_evidence_v1(
        page_id=f"raster-observable:{variant['name']}",
        svg=evidence_svg,
    )
    image = _clean_raster(raster_svg)
    pairs = [
        _pair_raster_observation(report, image, raster_viewbox, pair)
        for pair in report.pair_observations
    ]
    pairs.sort(key=lambda item: (item["relation"], item["staff_a_id"], item["staff_b_id"]))
    return {
        "name": variant["name"],
        "raster_sha256": sha256(image.tobytes()).hexdigest(),
        "raster_size": list(image.size),
        "system_count": len(report.systems),
        "pair_observations": pairs,
    }


def _audit() -> dict[str, object]:
    variants = [_variant_observation(variant) for variant in _VARIANTS]
    metric_names = (
        "pair_left_mean_darkness_fraction",
        "pair_left_max_column_darkness_fraction",
        "gap_left_mean_darkness_fraction",
        "gap_left_max_column_darkness_fraction",
        "gap_height_staff_spaces",
    )
    by_relation: dict[str, list[dict[str, object]]] = {
        StaffSystemRelation.SAME_SYSTEM.value: [],
        StaffSystemRelation.DIFFERENT_SYSTEM.value: [],
    }
    for variant in variants:
        for pair in variant["pair_observations"]:
            assert isinstance(pair, dict)
            by_relation[str(pair["relation"])].append(pair)

    ranges: dict[str, dict[str, list[float]]] = {}
    for metric in metric_names:
        ranges[metric] = {}
        for relation, rows in by_relation.items():
            values = [float(row[metric]) for row in rows]
            if values:
                ranges[metric][relation] = [round(min(values), 9), round(max(values), 9)]

    return {
        "claim_boundary": "FIXTURE_ONLY_RASTER_OBSERVATIONS_NO_GROUPING_RULE",
        "raster_width": _RASTER_WIDTH,
        "relation_counts": {key: len(value) for key, value in by_relation.items()},
        "feature_ranges_by_relation": ranges,
        "variants": variants,
    }


def _fingerprint(payload: dict[str, object]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256(raw).hexdigest()


class SystemGeometryRasterObservableAuditV1Tests(unittest.TestCase):
    def test_surface_contains_positive_and_hard_negative_pairs(self) -> None:
        audit = _audit()
        self.assertGreater(audit["relation_counts"]["SAME_SYSTEM"], 0)
        self.assertGreater(audit["relation_counts"]["DIFFERENT_SYSTEM"], 0)

    def test_raster_audit_is_observation_only(self) -> None:
        audit = _audit()
        self.assertEqual(
            audit["claim_boundary"],
            "FIXTURE_ONLY_RASTER_OBSERVATIONS_NO_GROUPING_RULE",
        )
        self.assertFalse(system_geometry_spatial_rule_design_allowed())
        self.assertFalse(system_geometry_spatial_runtime_connection_allowed())
        print(
            "SYSTEM_GEOMETRY_RASTER_OBSERVABLE_AUDIT",
            json.dumps(audit, sort_keys=True),
        )

    def test_raster_observations_are_deterministic_5_of_5(self) -> None:
        fingerprints = {_fingerprint(_audit()) for _ in range(5)}
        self.assertEqual(len(fingerprints), 1)


if __name__ == "__main__":
    unittest.main()
