from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import unittest

from st_omr_training.renderer import RendererConfig, _load_verovio_runtime
from st_omr_training.system_geometry_evidence_v1 import StaffSystemRelation
from st_omr_training.system_geometry_spatial_evidence_v1 import (
    extract_system_geometry_spatial_evidence_v1,
    system_geometry_spatial_rule_design_allowed,
    system_geometry_spatial_runtime_connection_allowed,
)
from test_system_geometry_raster_observable_audit_v1 import (
    _darkness_metrics,
    _pair_raster_observation,
    _pixel_box,
    _score,
    _viewbox,
)


# Broad fixture-only raster robustness surface.  This intentionally widens the
# synthetic layout conditions without fitting a threshold or implementing a
# grouping rule.  Any interval overlap is evidence against a standalone cue;
# it is not a test failure.
_LAYOUTS = (
    {
        "spacing_staff": 8,
        "spacing_system": 4,
        "scale": 75,
        "page_width": 1800,
        "raster_width": 900,
    },
    {
        "spacing_staff": 12,
        "spacing_system": 8,
        "scale": 100,
        "page_width": 2400,
        "raster_width": 1200,
    },
    {
        "spacing_staff": 18,
        "spacing_system": 20,
        "scale": 125,
        "page_width": 3200,
        "raster_width": 1600,
    },
)


def _fixture(
    name: str,
    *,
    staff_count: int,
    measure_count: int,
    breaks_before: tuple[int, ...],
    part_symbol: str | None,
    layout: dict[str, int],
    page_height: int,
) -> dict[str, object]:
    return {
        "name": name,
        "staff_count": staff_count,
        "measure_count": measure_count,
        "breaks_before": breaks_before,
        "part_symbol": part_symbol,
        "page_height": page_height,
        **layout,
    }


def _variants() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []

    # Same-system positives cover brace, bracket and no visible part symbol at
    # three staff/system spacing, scale, page-width and raster-width settings.
    for layout_index, layout in enumerate(_LAYOUTS, start=1):
        for symbol in ("brace", "bracket", "none"):
            rows.append(
                _fixture(
                    f"same-{symbol}-layout-{layout_index}",
                    staff_count=2,
                    measure_count=1,
                    breaks_before=(),
                    part_symbol=symbol,
                    layout=layout,
                    page_height=5200,
                )
            )

    # Multi-measure positives make sure a cue is not merely a one-measure page
    # artifact.  Use the middle layout to keep this expansion bounded.
    for symbol in ("brace", "bracket", "none"):
        rows.append(
            _fixture(
                f"same-{symbol}-two-measure",
                staff_count=2,
                measure_count=2,
                breaks_before=(),
                part_symbol=symbol,
                layout=_LAYOUTS[1],
                page_height=5200,
            )
        )

    # One-staff, two-system hard negatives span all layout settings.  There is
    # no hidden intermediate staff and therefore no same-system ambiguity.
    for layout_index, layout in enumerate(_LAYOUTS, start=1):
        rows.append(
            _fixture(
                f"different-single-staff-layout-{layout_index}",
                staff_count=1,
                measure_count=2,
                breaks_before=(2,),
                part_symbol=None,
                layout=layout,
                page_height=7600,
            )
        )

    # Real adjacency hard negatives: each system is a valid two-staff system,
    # while the bottom staff of system 1 and top staff of system 2 are not.
    for layout_index, layout in enumerate(_LAYOUTS, start=1):
        rows.append(
            _fixture(
                f"different-grand-none-layout-{layout_index}",
                staff_count=2,
                measure_count=2,
                breaks_before=(2,),
                part_symbol="none",
                layout=layout,
                page_height=9000,
            )
        )

    # Keep brace/bracket system-start ink in the negative surface too so a
    # connector cue is challenged when each neighboring system has one.
    for symbol in ("brace", "bracket"):
        rows.append(
            _fixture(
                f"different-grand-{symbol}-middle-layout",
                staff_count=2,
                measure_count=2,
                breaks_before=(2,),
                part_symbol=symbol,
                layout=_LAYOUTS[1],
                page_height=9000,
            )
        )

    return tuple(rows)


def _render_svg(variant: dict[str, object], *, bounding_boxes: bool) -> str:
    verovio, package_version = _load_verovio_runtime()
    toolkit = verovio.toolkit()
    if not str(toolkit.getVersion()).startswith(package_version):
        raise AssertionError("pinned Verovio runtime mismatch")
    if toolkit.setInputFrom("xml") is False:
        raise AssertionError("fixture input mode rejected")

    config = RendererConfig(
        page_height=int(variant["page_height"]),
        page_width=int(variant["page_width"]),
        scale=int(variant["scale"]),
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
        raise AssertionError("broad raster fixture must remain on one page")
    return toolkit.renderToSVG(1, True)


def _clean_raster(svg: str, *, raster_width: int):
    import cairosvg  # type: ignore
    from PIL import Image  # type: ignore

    png = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=raster_width,
        background_color="#ffffff",
    )
    image = Image.open(BytesIO(png)).convert("L")
    if image.width != raster_width or image.height < 1:
        raise AssertionError("unexpected broad raster geometry")
    low, high = image.getextrema()
    if low >= high or high != 255:
        raise AssertionError("broad raster must contain dark ink on white background")
    return image


def _variant_observation(variant: dict[str, object]) -> dict[str, object]:
    evidence_svg = _render_svg(variant, bounding_boxes=True)
    raster_svg = _render_svg(variant, bounding_boxes=False)
    evidence_viewbox = _viewbox(evidence_svg)
    raster_viewbox = _viewbox(raster_svg)
    if evidence_viewbox != raster_viewbox:
        raise AssertionError("bounding-box instrumentation changed fixture viewBox")

    report = extract_system_geometry_spatial_evidence_v1(
        page_id=f"raster-broad:{variant['name']}",
        svg=evidence_svg,
    )
    image = _clean_raster(raster_svg, raster_width=int(variant["raster_width"]))
    pairs = [
        _pair_raster_observation(report, image, raster_viewbox, pair)
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
        "staff_count": variant["staff_count"],
        "measure_count": variant["measure_count"],
        "spacing_staff": variant["spacing_staff"],
        "spacing_system": variant["spacing_system"],
        "scale": variant["scale"],
        "page_width": variant["page_width"],
        "raster_width": variant["raster_width"],
        "raster_sha256": sha256(image.tobytes()).hexdigest(),
        "raster_size": list(image.size),
        "system_count": len(report.systems),
        "pair_observations": pairs,
    }


def _intervals_overlap(first: list[float], second: list[float]) -> bool:
    return max(first[0], second[0]) <= min(first[1], second[1])


def _audit(variants: tuple[dict[str, object], ...] | None = None) -> dict[str, object]:
    selected = _variants() if variants is None else variants
    observations = [_variant_observation(variant) for variant in selected]
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
    for observation in observations:
        for pair in observation["pair_observations"]:
            assert isinstance(pair, dict)
            by_relation[str(pair["relation"])].append(pair)

    ranges: dict[str, dict[str, list[float]]] = {}
    assessments: dict[str, str] = {}
    for metric in metric_names:
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
        "claim_boundary": "FIXTURE_ONLY_BROAD_RASTER_OBSERVATIONS_NO_THRESHOLD_NO_GROUPING_RULE",
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


class SystemGeometryRasterBroadRobustnessV1Tests(unittest.TestCase):
    def test_broad_surface_covers_requested_layout_axes(self) -> None:
        variants = _variants()
        self.assertGreaterEqual(len(variants), 18)
        self.assertEqual(
            {variant["part_symbol"] for variant in variants if variant["staff_count"] == 2},
            {"brace", "bracket", "none"},
        )
        self.assertGreaterEqual(len({variant["spacing_staff"] for variant in variants}), 3)
        self.assertGreaterEqual(len({variant["spacing_system"] for variant in variants}), 3)
        self.assertGreaterEqual(len({variant["scale"] for variant in variants}), 3)
        self.assertGreaterEqual(len({variant["page_width"] for variant in variants}), 3)
        self.assertGreaterEqual(len({variant["raster_width"] for variant in variants}), 3)
        self.assertEqual({variant["measure_count"] for variant in variants}, {1, 2})

    def test_broad_raster_audit_is_observation_only(self) -> None:
        audit = _audit()
        self.assertEqual(
            audit["claim_boundary"],
            "FIXTURE_ONLY_BROAD_RASTER_OBSERVATIONS_NO_THRESHOLD_NO_GROUPING_RULE",
        )
        self.assertGreater(audit["relation_counts"]["SAME_SYSTEM"], 0)
        self.assertGreater(audit["relation_counts"]["DIFFERENT_SYSTEM"], 0)
        self.assertFalse(system_geometry_spatial_rule_design_allowed())
        self.assertFalse(system_geometry_spatial_runtime_connection_allowed())
        print(
            "SYSTEM_GEOMETRY_RASTER_BROAD_ROBUSTNESS_AUDIT",
            json.dumps(audit, sort_keys=True),
        )

    def test_overlap_is_evidence_not_a_test_failure(self) -> None:
        audit = _audit()
        assessments = audit["feature_assessments"]
        self.assertTrue(
            all(
                value in {
                    "OVERLAP_REJECT_STANDALONE",
                    "DISJOINT_ON_FIXTURES_NOT_A_RULE",
                }
                for value in assessments.values()
            )
        )

    def test_broad_raster_sentinel_is_deterministic_5_of_5(self) -> None:
        variants = _variants()
        sentinel = (variants[0], variants[2], variants[12], variants[-1])
        fingerprints = {_fingerprint(_audit(sentinel)) for _ in range(5)}
        self.assertEqual(len(fingerprints), 1)


if __name__ == "__main__":
    unittest.main()
