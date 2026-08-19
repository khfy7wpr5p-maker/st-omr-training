from __future__ import annotations

from hashlib import sha256
import json
import unittest

from st_omr_training.runtime_geometry_engine_v1 import ROW_DARK_THRESHOLD
from st_omr_training.system_geometry_evidence_v1 import StaffSystemRelation
from st_omr_training.system_geometry_spatial_evidence_v1 import extract_system_geometry_spatial_evidence_v1
from test_system_geometry_raster_broad_robustness_v1 import _clean_raster, _render_svg, _variants
from test_system_geometry_raster_observable_audit_v1 import _pixel_box, _viewbox


# Diagnostic bridge between the merged fixture-only System Geometry evidence and
# a future runtime System Grouper.  Unlike the previous system-bbox corridor,
# this probe anchors on the actual staff-line x extents exposed by the fixture
# staff geometry.  It records raster-only column continuity; it does not define
# a grouping threshold or production rule.


def _pair_endpoint_probe(report, image, viewbox, pair) -> dict[str, object]:
    staffs = {
        staff.staff_instance_id: staff
        for system in report.systems
        for staff in system.staffs
    }
    first = staffs[pair.staff_a_id]
    second = staffs[pair.staff_b_id]
    upper, lower = sorted((first, second), key=lambda item: item.center_y)
    spacing = (upper.staff_spacing + lower.staff_spacing) / 2.0
    if spacing <= 0:
        raise AssertionError("staff spacing must be positive")

    # StaffSpatialObservationV1.bbox.x_min comes from the direct staff-line
    # extents, not the enclosing SVG system bbox.
    upper_left = upper.bbox.x_min
    lower_left = lower.bbox.x_min
    anchor_x = (upper_left + lower_left) / 2.0
    x0 = min(upper_left, lower_left) - spacing
    x1 = max(upper_left, lower_left) + spacing
    y0 = upper.bbox.y_max
    y1 = lower.bbox.y_min

    if y1 <= y0:
        return {
            "relation": pair.relation.value,
            "staff_a_id": pair.staff_a_id,
            "staff_b_id": pair.staff_b_id,
            "upper_left_x": round(upper_left, 9),
            "lower_left_x": round(lower_left, 9),
            "left_delta_staff_spaces": round(abs(upper_left - lower_left) / spacing, 9),
            "max_gap_column_coverage": 0.0,
            "max_gap_column_dark_pixels": 0,
            "gap_height_pixels": 0,
            "best_x": None,
        }

    box = _pixel_box((x0, y0, x1, y1), viewbox=viewbox, image_size=image.size)
    crop = image.crop(box)
    width, height = crop.size
    pixels = crop.load()
    best_dark = -1
    best_x = 0
    for x in range(width):
        dark = sum(1 for y in range(height) if int(pixels[x, y]) <= ROW_DARK_THRESHOLD)
        if dark > best_dark:
            best_dark = dark
            best_x = x
    coverage = 0.0 if height <= 0 else best_dark / height
    sx = image.width / viewbox[2]
    best_viewbox_x = (box[0] + best_x) / sx if sx > 0 else 0.0
    return {
        "relation": pair.relation.value,
        "staff_a_id": pair.staff_a_id,
        "staff_b_id": pair.staff_b_id,
        "upper_left_x": round(upper_left, 9),
        "lower_left_x": round(lower_left, 9),
        "left_delta_staff_spaces": round(abs(upper_left - lower_left) / spacing, 9),
        "anchor_x": round(anchor_x, 9),
        "max_gap_column_coverage": round(coverage, 9),
        "max_gap_column_dark_pixels": int(best_dark),
        "gap_height_pixels": int(height),
        "best_x": round(best_viewbox_x, 9),
    }


def _variant_probe(variant: dict[str, object]) -> dict[str, object]:
    evidence_svg = _render_svg(variant, bounding_boxes=True)
    raster_svg = _render_svg(variant, bounding_boxes=False)
    evidence_viewbox = _viewbox(evidence_svg)
    raster_viewbox = _viewbox(raster_svg)
    if evidence_viewbox != raster_viewbox:
        raise AssertionError("bounding-box instrumentation changed fixture viewBox")
    report = extract_system_geometry_spatial_evidence_v1(
        page_id=f"endpoint-probe:{variant['name']}",
        svg=evidence_svg,
    )
    image = _clean_raster(raster_svg, raster_width=int(variant["raster_width"]))
    pairs = [
        _pair_endpoint_probe(report, image, raster_viewbox, pair)
        for pair in report.pair_observations
    ]
    pairs.sort(key=lambda item: (item["relation"], item["staff_a_id"], item["staff_b_id"]))
    return {
        "name": variant["name"],
        "part_symbol": variant["part_symbol"],
        "raster_sha256": sha256(image.tobytes()).hexdigest(),
        "pairs": pairs,
    }


def _audit(variants: tuple[dict[str, object], ...] | None = None) -> dict[str, object]:
    selected = _variants() if variants is None else variants
    rows = [_variant_probe(variant) for variant in selected]
    values: dict[str, list[float]] = {
        StaffSystemRelation.SAME_SYSTEM.value: [],
        StaffSystemRelation.DIFFERENT_SYSTEM.value: [],
    }
    left_deltas: dict[str, list[float]] = {
        StaffSystemRelation.SAME_SYSTEM.value: [],
        StaffSystemRelation.DIFFERENT_SYSTEM.value: [],
    }
    for row in rows:
        for pair in row["pairs"]:
            relation = str(pair["relation"])
            values[relation].append(float(pair["max_gap_column_coverage"]))
            left_deltas[relation].append(float(pair["left_delta_staff_spaces"]))
    return {
        "claim_boundary": "FIXTURE_ONLY_STAFF_ENDPOINT_RASTER_PROBE_NO_GROUPING_RULE",
        "variant_count": len(rows),
        "relation_counts": {key: len(item) for key, item in values.items()},
        "coverage_ranges": {
            key: [round(min(item), 9), round(max(item), 9)] if item else None
            for key, item in values.items()
        },
        "left_delta_ranges": {
            key: [round(min(item), 9), round(max(item), 9)] if item else None
            for key, item in left_deltas.items()
        },
        "variants": rows,
    }


def _fingerprint(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return sha256(raw).hexdigest()


class SystemGrouperEndpointProbeV1Tests(unittest.TestCase):
    def test_probe_keeps_positive_and_hard_negative_surface(self) -> None:
        audit = _audit()
        self.assertEqual(audit["variant_count"], 20)
        self.assertGreater(audit["relation_counts"]["SAME_SYSTEM"], 0)
        self.assertGreater(audit["relation_counts"]["DIFFERENT_SYSTEM"], 0)

    def test_probe_is_observation_only(self) -> None:
        audit = _audit()
        self.assertEqual(
            audit["claim_boundary"],
            "FIXTURE_ONLY_STAFF_ENDPOINT_RASTER_PROBE_NO_GROUPING_RULE",
        )
        print("SYSTEM_GROUPER_ENDPOINT_PROBE", json.dumps(audit, sort_keys=True))

    def test_probe_sentinel_is_deterministic_5_of_5(self) -> None:
        variants = _variants()
        sentinel = (variants[0], variants[2], variants[12], variants[-1])
        fingerprints = {_fingerprint(_audit(sentinel)) for _ in range(5)}
        self.assertEqual(len(fingerprints), 1)


if __name__ == "__main__":
    unittest.main()
