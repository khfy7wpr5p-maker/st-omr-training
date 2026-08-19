from __future__ import annotations

from hashlib import sha256
import json
import unittest

from st_omr_training.system_geometry_evidence_v1 import StaffSystemRelation
from st_omr_training.system_geometry_spatial_evidence_v1 import (
    extract_system_geometry_spatial_evidence_v1,
    system_geometry_spatial_rule_design_allowed,
    system_geometry_spatial_runtime_connection_allowed,
)
from test_system_geometry_raster_broad_robustness_v1 import (
    _clean_raster,
    _render_svg,
    _variants,
)
from test_system_geometry_raster_observable_audit_v1 import _pixel_box, _viewbox


# Fixture-only structural raster audit.  SVG geometry is used only to locate
# the known synthetic staff/system corridor.  The observations themselves come
# from exact white-vs-rendered-ink topology in the clean raster.  No learned or
# fitted threshold, grouping rule, or runtime connection is defined here.


def _component_stats(
    image,
    pixel_box: tuple[int, int, int, int],
    *,
    upper_band: tuple[int, int],
    lower_band: tuple[int, int],
) -> dict[str, object]:
    crop = image.crop(pixel_box)
    width, height = crop.size
    pixels = list(crop.get_flattened_data())
    ink = [int(value) < 255 for value in pixels]
    seen = bytearray(width * height)
    components: list[dict[str, object]] = []

    for start in range(width * height):
        if seen[start] or not ink[start]:
            continue
        stack = [start]
        seen[start] = 1
        count = 0
        min_y = height
        max_y = -1
        touches_upper = False
        touches_lower = False
        while stack:
            index = stack.pop()
            y, x = divmod(index, width)
            count += 1
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            if upper_band[0] <= y < upper_band[1]:
                touches_upper = True
            if lower_band[0] <= y < lower_band[1]:
                touches_lower = True
            for dy in (-1, 0, 1):
                ny = y + dy
                if ny < 0 or ny >= height:
                    continue
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    if nx < 0 or nx >= width:
                        continue
                    neighbor = ny * width + nx
                    if not seen[neighbor] and ink[neighbor]:
                        seen[neighbor] = 1
                        stack.append(neighbor)
        components.append(
            {
                "pixel_count": count,
                "min_y": min_y,
                "max_y": max_y,
                "touches_upper": touches_upper,
                "touches_lower": touches_lower,
            }
        )

    bridges = [
        component
        for component in components
        if component["touches_upper"] and component["touches_lower"]
    ]
    max_bridge_height = max(
        (int(component["max_y"]) - int(component["min_y"]) + 1 for component in bridges),
        default=0,
    )
    return {
        "component_count": len(components),
        "bridge_component_count": len(bridges),
        "max_bridge_height_pixels": max_bridge_height,
    }


def _gap_row_topology(
    image,
    pixel_box: tuple[int, int, int, int],
) -> dict[str, float]:
    crop = image.crop(pixel_box)
    width, height = crop.size
    pixels = list(crop.get_flattened_data())
    row_has_ink: list[bool] = []
    for y in range(height):
        start = y * width
        row_has_ink.append(any(int(value) < 255 for value in pixels[start : start + width]))

    ink_rows = sum(row_has_ink)
    longest = 0
    current = 0
    for value in row_has_ink:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {
        "gap_row_ink_coverage_fraction": round(ink_rows / height, 9),
        "gap_longest_ink_run_fraction": round(longest / height, 9),
    }


def _pair_topology_observation(report, image, viewbox, pair) -> dict[str, object]:
    staffs = {
        staff.staff_instance_id: staff
        for system in report.systems
        for staff in system.staffs
    }
    systems = {system.system_id: system for system in report.systems}
    first = staffs[pair.staff_a_id]
    second = staffs[pair.staff_b_id]
    upper, lower = sorted((first, second), key=lambda staff: staff.center_y)
    spacing = (upper.staff_spacing + lower.staff_spacing) / 2.0
    if spacing <= 0:
        raise AssertionError("fixture staff spacing must be positive")

    upper_system = systems[upper.system_id]
    lower_system = systems[lower.system_id]
    anchor_x = min(upper_system.bbox.x_min, lower_system.bbox.x_min)
    x0 = anchor_x - (2.0 * spacing)
    x1 = anchor_x + (2.0 * spacing)
    y0 = upper.bbox.y_min - (0.5 * spacing)
    y1 = lower.bbox.y_max + (0.5 * spacing)
    pair_box = _pixel_box(
        (x0, y0, x1, y1),
        viewbox=viewbox,
        image_size=image.size,
    )

    upper_box = _pixel_box(
        (x0, upper.bbox.y_min - 0.25 * spacing, x1, upper.bbox.y_max + 0.25 * spacing),
        viewbox=viewbox,
        image_size=image.size,
    )
    lower_box = _pixel_box(
        (x0, lower.bbox.y_min - 0.25 * spacing, x1, lower.bbox.y_max + 0.25 * spacing),
        viewbox=viewbox,
        image_size=image.size,
    )
    upper_band = (
        max(0, upper_box[1] - pair_box[1]),
        min(pair_box[3] - pair_box[1], upper_box[3] - pair_box[1]),
    )
    lower_band = (
        max(0, lower_box[1] - pair_box[1]),
        min(pair_box[3] - pair_box[1], lower_box[3] - pair_box[1]),
    )
    components = _component_stats(
        image,
        pair_box,
        upper_band=upper_band,
        lower_band=lower_band,
    )

    gap_y0 = upper.bbox.y_max + 0.25 * spacing
    gap_y1 = lower.bbox.y_min - 0.25 * spacing
    if gap_y1 <= gap_y0:
        gap = {
            "gap_row_ink_coverage_fraction": 0.0,
            "gap_longest_ink_run_fraction": 0.0,
        }
    else:
        gap_box = _pixel_box(
            (x0, gap_y0, x1, gap_y1),
            viewbox=viewbox,
            image_size=image.size,
        )
        gap = _gap_row_topology(image, gap_box)

    sy = image.height / viewbox[3]
    bridge_height_spaces = (
        float(components["max_bridge_height_pixels"]) / sy / spacing
        if sy > 0
        else 0.0
    )
    return {
        "relation": pair.relation.value,
        "staff_a_id": pair.staff_a_id,
        "staff_b_id": pair.staff_b_id,
        "system_start_x": round(anchor_x, 9),
        "component_count": int(components["component_count"]),
        "bridge_component_count": int(components["bridge_component_count"]),
        "max_bridge_height_staff_spaces": round(bridge_height_spaces, 9),
        **gap,
    }


def _variant_observation(variant: dict[str, object]) -> dict[str, object]:
    evidence_svg = _render_svg(variant, bounding_boxes=True)
    raster_svg = _render_svg(variant, bounding_boxes=False)
    evidence_viewbox = _viewbox(evidence_svg)
    raster_viewbox = _viewbox(raster_svg)
    if evidence_viewbox != raster_viewbox:
        raise AssertionError("bounding-box instrumentation changed fixture viewBox")

    report = extract_system_geometry_spatial_evidence_v1(
        page_id=f"raster-topology:{variant['name']}",
        svg=evidence_svg,
    )
    image = _clean_raster(raster_svg, raster_width=int(variant["raster_width"]))
    pairs = [
        _pair_topology_observation(report, image, raster_viewbox, pair)
        for pair in report.pair_observations
    ]
    pairs.sort(
        key=lambda item: (
            item["relation"],
            item["staff_a_id"],
            item["staff_b_id"],
        )
    )
    return {
        "name": variant["name"],
        "part_symbol": variant["part_symbol"],
        "spacing_staff": variant["spacing_staff"],
        "spacing_system": variant["spacing_system"],
        "scale": variant["scale"],
        "page_width": variant["page_width"],
        "raster_width": variant["raster_width"],
        "raster_sha256": sha256(image.tobytes()).hexdigest(),
        "system_count": len(report.systems),
        "pair_observations": pairs,
    }


def _intervals_overlap(first: list[float], second: list[float]) -> bool:
    return max(first[0], second[0]) <= min(first[1], second[1])


def _audit(variants: tuple[dict[str, object], ...] | None = None) -> dict[str, object]:
    selected = _variants() if variants is None else variants
    observations = [_variant_observation(variant) for variant in selected]
    metrics = (
        "bridge_component_count",
        "max_bridge_height_staff_spaces",
        "gap_row_ink_coverage_fraction",
        "gap_longest_ink_run_fraction",
    )
    by_relation: dict[str, list[dict[str, object]]] = {
        StaffSystemRelation.SAME_SYSTEM.value: [],
        StaffSystemRelation.DIFFERENT_SYSTEM.value: [],
    }
    for observation in observations:
        for pair in observation["pair_observations"]:
            assert isinstance(pair, dict)
            by_relation[str(pair["relation"])].append(pair)

    ranges: dict[str, dict[str, list[float]]] = {}
    assessments: dict[str, str] = {}
    for metric in metrics:
        ranges[metric] = {}
        for relation, rows in by_relation.items():
            values = [float(row[metric]) for row in rows]
            if values:
                ranges[metric][relation] = [
                    round(min(values), 9),
                    round(max(values), 9),
                ]
        same = ranges[metric].get(StaffSystemRelation.SAME_SYSTEM.value)
        different = ranges[metric].get(StaffSystemRelation.DIFFERENT_SYSTEM.value)
        if same is None or different is None:
            assessments[metric] = "INSUFFICIENT_RELATION_COVERAGE"
        elif _intervals_overlap(same, different):
            assessments[metric] = "OVERLAP_REJECT_STANDALONE"
        else:
            assessments[metric] = "DISJOINT_ON_FIXTURES_NOT_A_RULE"

    return {
        "claim_boundary": "FIXTURE_ONLY_RASTER_TOPOLOGY_OBSERVATIONS_NO_GROUPING_RULE",
        "pixel_interpretation": "EXACT_WHITE_255_VS_RENDERED_INK_NOT_A_FITTED_THRESHOLD",
        "variant_count": len(observations),
        "relation_counts": {key: len(value) for key, value in by_relation.items()},
        "feature_ranges_by_relation": ranges,
        "feature_assessments": assessments,
        "variants": observations,
    }


def _fingerprint(payload: dict[str, object]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256(raw).hexdigest()


class SystemGeometryRasterTopologyEvidenceV1Tests(unittest.TestCase):
    def test_topology_audit_keeps_broad_positive_and_hard_negative_surface(self) -> None:
        audit = _audit()
        self.assertEqual(audit["variant_count"], 20)
        self.assertGreater(audit["relation_counts"]["SAME_SYSTEM"], 0)
        self.assertGreater(audit["relation_counts"]["DIFFERENT_SYSTEM"], 0)

    def test_topology_audit_is_observation_only(self) -> None:
        audit = _audit()
        self.assertEqual(
            audit["claim_boundary"],
            "FIXTURE_ONLY_RASTER_TOPOLOGY_OBSERVATIONS_NO_GROUPING_RULE",
        )
        self.assertFalse(system_geometry_spatial_rule_design_allowed())
        self.assertFalse(system_geometry_spatial_runtime_connection_allowed())
        print(
            "SYSTEM_GEOMETRY_RASTER_TOPOLOGY_EVIDENCE_AUDIT",
            json.dumps(audit, sort_keys=True),
        )

    def test_overlap_is_evidence_not_a_test_failure(self) -> None:
        audit = _audit()
        self.assertTrue(
            all(
                value in {
                    "OVERLAP_REJECT_STANDALONE",
                    "DISJOINT_ON_FIXTURES_NOT_A_RULE",
                }
                for value in audit["feature_assessments"].values()
            )
        )

    def test_topology_sentinel_is_deterministic_5_of_5(self) -> None:
        variants = _variants()
        sentinel = (variants[0], variants[2], variants[12], variants[-1])
        fingerprints = {_fingerprint(_audit(sentinel)) for _ in range(5)}
        self.assertEqual(len(fingerprints), 1)


if __name__ == "__main__":
    unittest.main()
