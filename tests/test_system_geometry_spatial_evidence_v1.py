from __future__ import annotations

import json
import unittest

from st_omr_training.renderer import RendererConfig, _load_verovio_runtime
from st_omr_training.system_geometry_evidence_v1 import StaffSystemRelation
from st_omr_training.system_geometry_spatial_evidence_v1 import (
    SYSTEM_GEOMETRY_SPATIAL_EVIDENCE_CLAIM_BOUNDARY,
    SYSTEM_GEOMETRY_SPATIAL_EVIDENCE_VERSION,
    audit_system_geometry_spatial_stability_v1,
    extract_system_geometry_spatial_evidence_v1,
    system_geometry_spatial_rule_design_allowed,
    system_geometry_spatial_runtime_connection_allowed,
)


def _grand_staff_measure(
    number: int,
    *,
    include_attributes: bool = False,
    new_system: bool = False,
    final: bool = False,
) -> str:
    attributes = ""
    if include_attributes:
        attributes = """
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>2</staves>
        <part-symbol top-staff="1" bottom-staff="2">brace</part-symbol>
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


def _grand_staff_score(*, measure_count: int, break_before: int | None = None) -> str:
    measures = []
    for number in range(1, measure_count + 1):
        measures.append(
            _grand_staff_measure(
                number,
                include_attributes=number == 1,
                new_system=break_before == number,
                final=number == measure_count,
            )
        )
    return """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">""" + "".join(measures) + """
  </part>
</score-partwise>"""


def _render_encoded(xml_text: str) -> str:
    verovio, package_version = _load_verovio_runtime()
    toolkit = verovio.toolkit()
    if not str(toolkit.getVersion()).startswith(package_version):
        raise AssertionError("pinned Verovio runtime mismatch")
    if toolkit.setInputFrom("xml") is False:
        raise AssertionError("fixture input mode rejected")
    options = dict(RendererConfig(breaks="encoded").verovio_options())
    options.update({"svgBoundingBoxes": True, "svgContentBoundingBoxes": True})
    if toolkit.setOptions(options) is False:
        raise AssertionError("fixture options rejected")
    if toolkit.loadData(xml_text) is False:
        raise AssertionError("fixture MusicXML rejected")
    if toolkit.getPageCount() != 1:
        raise AssertionError("spatial fixture must remain on one page")
    return toolkit.renderToSVG(1, True)


def _reports():
    positive = extract_system_geometry_spatial_evidence_v1(
        page_id="spatial-multi-measure-grand-staff",
        svg=_render_encoded(_grand_staff_score(measure_count=2)),
    )
    hard = extract_system_geometry_spatial_evidence_v1(
        page_id="spatial-two-grand-staff-systems",
        svg=_render_encoded(_grand_staff_score(measure_count=4, break_before=3)),
    )
    return positive, hard


class SystemGeometrySpatialEvidenceV1Tests(unittest.TestCase):
    def test_multi_measure_grand_staff_extracts_raw_spatial_observations(self) -> None:
        positive, _ = _reports()
        self.assertEqual(len(positive.systems), 1)
        system = positive.systems[0]
        self.assertEqual(len(system.staffs), 2)
        self.assertEqual(len(system.measures), 2)
        self.assertTrue(all(len(staff.measure_x_spans) == 2 for staff in system.staffs))
        self.assertTrue(all(staff.staff_spacing > 0 for staff in system.staffs))
        self.assertEqual(len(positive.pair_observations), 1)
        pair = positive.pair_observations[0]
        self.assertIs(pair.relation, StaffSystemRelation.SAME_SYSTEM)
        self.assertGreater(pair.normalized_center_distance, 0.0)
        self.assertGreater(pair.x_overlap_ratio, 0.9)
        self.assertGreaterEqual(pair.measure_boundary_exact_match_fraction, 0.0)
        self.assertLessEqual(pair.measure_boundary_exact_match_fraction, 1.0)

    def test_two_grand_staff_systems_yield_hard_cross_system_pairs(self) -> None:
        _, hard = _reports()
        self.assertEqual(len(hard.systems), 2)
        self.assertEqual([len(system.staffs) for system in hard.systems], [2, 2])
        same = [x for x in hard.pair_observations if x.relation is StaffSystemRelation.SAME_SYSTEM]
        different = [
            x for x in hard.pair_observations
            if x.relation is StaffSystemRelation.DIFFERENT_SYSTEM
        ]
        self.assertEqual(len(same), 2)
        self.assertEqual(len(different), 4)

    def test_spatial_stability_audit_records_ranges_without_fitting_rule(self) -> None:
        positive, hard = _reports()
        audit = audit_system_geometry_spatial_stability_v1((positive, hard))
        self.assertEqual(audit.relation_counts, {"SAME_SYSTEM": 3, "DIFFERENT_SYSTEM": 4})
        for feature in (
            "normalized_center_distance",
            "normalized_edge_gap",
            "x_overlap_ratio",
            "grouping_span_cover_count",
            "barline_span_cover_count",
            "measure_boundary_exact_match_fraction",
        ):
            self.assertIn(feature, audit.feature_ranges_by_relation)
        self.assertFalse(system_geometry_spatial_rule_design_allowed())
        self.assertFalse(system_geometry_spatial_runtime_connection_allowed())
        print("SYSTEM_GEOMETRY_SPATIAL_AUDIT", json.dumps(audit.canonical_payload(), sort_keys=True))

    def test_spatial_report_and_audit_are_deterministic_10_of_10(self) -> None:
        svg = _render_encoded(_grand_staff_score(measure_count=2))
        report_fingerprints = {
            extract_system_geometry_spatial_evidence_v1(
                page_id="repeat-spatial", svg=svg
            ).fingerprint()
            for _ in range(10)
        }
        self.assertEqual(len(report_fingerprints), 1)
        positive, hard = _reports()
        audit_fingerprints = {
            audit_system_geometry_spatial_stability_v1((positive, hard)).fingerprint()
            for _ in range(10)
        }
        self.assertEqual(len(audit_fingerprints), 1)

    def test_version_and_claim_boundary_are_explicit(self) -> None:
        self.assertEqual(
            SYSTEM_GEOMETRY_SPATIAL_EVIDENCE_VERSION,
            "system-geometry-spatial-evidence-v1",
        )
        self.assertEqual(
            SYSTEM_GEOMETRY_SPATIAL_EVIDENCE_CLAIM_BOUNDARY,
            "FIXTURE_ONLY_RAW_SPATIAL_OBSERVATIONS_NO_GROUPING_RULE",
        )


if __name__ == "__main__":
    unittest.main()
