from __future__ import annotations

import json
import unittest

from st_omr_training.renderer import RendererConfig, _load_verovio_runtime
from st_omr_training.system_geometry_evidence_v1 import StaffSystemRelation
from st_omr_training.system_geometry_spatial_evidence_v1 import (
    audit_system_geometry_spatial_stability_v1,
    extract_system_geometry_spatial_evidence_v1,
    system_geometry_spatial_rule_design_allowed,
    system_geometry_spatial_runtime_connection_allowed,
)


# Fixture-only robustness matrix.  Values vary renderer layout but are never
# interpreted as runtime thresholds or grouping rules.
_VARIANTS = (
    {
        "name": "default-brace-positive",
        "measure_count": 2,
        "break_before": None,
        "part_symbol": "brace",
        "page_width": 2100,
        "page_height": 2970,
        "scale": 100,
        "spacing_staff": 12,
        "spacing_system": 4,
    },
    {
        "name": "compact-brace-positive",
        "measure_count": 1,
        "break_before": None,
        "part_symbol": "brace",
        "page_width": 1600,
        "page_height": 2600,
        "scale": 80,
        "spacing_staff": 8,
        "spacing_system": 2,
    },
    {
        "name": "wide-bracket-positive",
        "measure_count": 4,
        "break_before": None,
        "part_symbol": "bracket",
        "page_width": 3200,
        "page_height": 3600,
        "scale": 120,
        "spacing_staff": 20,
        "spacing_system": 12,
    },
    {
        "name": "default-brace-hard-negative",
        "measure_count": 4,
        "break_before": 3,
        "part_symbol": "brace",
        "page_width": 2100,
        "page_height": 3600,
        "scale": 100,
        "spacing_staff": 12,
        "spacing_system": 4,
    },
    {
        "name": "compact-bracket-hard-negative",
        "measure_count": 4,
        "break_before": 3,
        "part_symbol": "bracket",
        "page_width": 1800,
        "page_height": 3600,
        "scale": 90,
        "spacing_staff": 8,
        "spacing_system": 2,
    },
    {
        "name": "expanded-brace-hard-negative",
        "measure_count": 6,
        "break_before": 4,
        "part_symbol": "brace",
        "page_width": 3200,
        "page_height": 5000,
        "scale": 120,
        "spacing_staff": 20,
        "spacing_system": 16,
    },
)


def _grand_staff_measure(
    number: int,
    *,
    part_symbol: str,
    include_attributes: bool = False,
    new_system: bool = False,
    final: bool = False,
) -> str:
    attributes = ""
    if include_attributes:
        attributes = f"""
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>2</staves>
        <part-symbol top-staff="1" bottom-staff="2">{part_symbol}</part-symbol>
        <clef number="1"><sign>G</sign><line>2</line></clef>
        <clef number="2"><sign>F</sign><line>4</line></clef>
      </attributes>"""
    print_tag = '<print new-system="yes"/>' if new_system else ""
    upper_step = ("C", "D", "E", "F")[((number - 1) % 4)]
    lower_step = ("C", "B", "A", "G")[((number - 1) % 4)]
    bar_style = "light-heavy" if final else "regular"
    return f"""
    <measure number="{number}">
      {print_tag}{attributes}
      <note><pitch><step>{upper_step}</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type><staff>1</staff></note>
      <backup><duration>4</duration></backup>
      <note><pitch><step>{lower_step}</step><octave>3</octave></pitch><duration>4</duration><voice>2</voice><type>whole</type><staff>2</staff></note>
      <barline location="right"><bar-style>{bar_style}</bar-style></barline>
    </measure>"""


def _grand_staff_score(
    *,
    measure_count: int,
    break_before: int | None,
    part_symbol: str,
) -> str:
    measures = [
        _grand_staff_measure(
            number,
            part_symbol=part_symbol,
            include_attributes=number == 1,
            new_system=break_before == number,
            final=number == measure_count,
        )
        for number in range(1, measure_count + 1)
    ]
    return """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">""" + "".join(measures) + """
  </part>
</score-partwise>"""


def _render_variant(variant: dict[str, object]) -> str:
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
            "svgBoundingBoxes": True,
            "svgContentBoundingBoxes": True,
            "spacingStaff": int(variant["spacing_staff"]),
            "spacingSystem": int(variant["spacing_system"]),
        }
    )
    if toolkit.setOptions(options) is False:
        raise AssertionError("fixture layout options rejected")

    xml_text = _grand_staff_score(
        measure_count=int(variant["measure_count"]),
        break_before=variant["break_before"],
        part_symbol=str(variant["part_symbol"]),
    )
    if toolkit.loadData(xml_text) is False:
        raise AssertionError("fixture MusicXML rejected")
    if toolkit.getPageCount() != 1:
        raise AssertionError("robustness fixture must remain on one page")
    return toolkit.renderToSVG(1, True)


def _reports():
    return tuple(
        extract_system_geometry_spatial_evidence_v1(
            page_id=f"spatial-robustness:{variant['name']}",
            svg=_render_variant(variant),
        )
        for variant in _VARIANTS
    )


class SystemGeometrySpatialRobustnessV1Tests(unittest.TestCase):
    def test_frozen_matrix_covers_requested_layout_dimensions(self) -> None:
        self.assertEqual(len(_VARIANTS), 6)
        self.assertEqual({x["part_symbol"] for x in _VARIANTS}, {"brace", "bracket"})
        self.assertGreater(len({x["page_width"] for x in _VARIANTS}), 1)
        self.assertGreater(len({x["scale"] for x in _VARIANTS}), 1)
        self.assertGreater(len({x["spacing_staff"] for x in _VARIANTS}), 1)
        self.assertGreater(len({x["spacing_system"] for x in _VARIANTS}), 1)
        self.assertGreater(len({x["measure_count"] for x in _VARIANTS}), 1)
        self.assertTrue(any(x["break_before"] is None for x in _VARIANTS))
        self.assertTrue(any(x["break_before"] is not None for x in _VARIANTS))

    def test_broad_surface_contains_positive_and_hard_negative_geometry(self) -> None:
        reports = _reports()
        same = [
            pair
            for report in reports
            for pair in report.pair_observations
            if pair.relation is StaffSystemRelation.SAME_SYSTEM
        ]
        different = [
            pair
            for report in reports
            for pair in report.pair_observations
            if pair.relation is StaffSystemRelation.DIFFERENT_SYSTEM
        ]
        self.assertGreaterEqual(len(same), 6)
        self.assertGreaterEqual(len(different), 8)
        # Prove the surface actually perturbs geometry; do not require separation.
        self.assertGreater(
            len({round(pair.normalized_center_distance, 6) for pair in same}),
            1,
        )
        self.assertGreater(
            len({round(pair.normalized_center_distance, 6) for pair in different}),
            1,
        )

    def test_broad_spatial_audit_observes_ranges_without_authorizing_rule(self) -> None:
        audit = audit_system_geometry_spatial_stability_v1(_reports())
        self.assertGreater(audit.relation_counts["SAME_SYSTEM"], 0)
        self.assertGreater(audit.relation_counts["DIFFERENT_SYSTEM"], 0)
        self.assertFalse(system_geometry_spatial_rule_design_allowed())
        self.assertFalse(system_geometry_spatial_runtime_connection_allowed())
        print(
            "SYSTEM_GEOMETRY_BROAD_SPATIAL_AUDIT",
            json.dumps(audit.canonical_payload(), sort_keys=True),
        )

    def test_broad_surface_is_deterministic_5_of_5(self) -> None:
        fingerprints = {
            audit_system_geometry_spatial_stability_v1(_reports()).fingerprint()
            for _ in range(5)
        }
        self.assertEqual(len(fingerprints), 1)


if __name__ == "__main__":
    unittest.main()
