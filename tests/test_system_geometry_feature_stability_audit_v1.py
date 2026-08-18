from __future__ import annotations

import unittest

from st_omr_training.renderer import RendererConfig, _load_verovio_runtime
from st_omr_training.system_geometry_evidence_dataset_v1 import (
    build_system_geometry_evidence_dataset_v1,
)
from st_omr_training.system_geometry_evidence_extractor_v1 import (
    extract_system_geometry_evidence_v1,
)
from st_omr_training.system_geometry_feature_stability_audit_v1 import (
    SYSTEM_GEOMETRY_FEATURE_STABILITY_AUDIT_CLAIM_BOUNDARY,
    SYSTEM_GEOMETRY_FEATURE_STABILITY_AUDIT_VERSION,
    audit_system_geometry_feature_stability_v1,
    system_geometry_feature_stability_rule_design_allowed,
    system_geometry_feature_stability_runtime_connection_allowed,
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
        raise AssertionError("expanded fixture must remain on one page")
    return toolkit.renderToSVG(1, True)


def _expanded_reports():
    multi_measure_positive = extract_system_geometry_evidence_v1(
        page_id="multi-measure-grand-staff-positive",
        svg=_render_encoded(_grand_staff_score(measure_count=2)),
    )
    multi_system_hard_surface = extract_system_geometry_evidence_v1(
        page_id="multi-system-grand-staff-hard-surface",
        svg=_render_encoded(_grand_staff_score(measure_count=4, break_before=3)),
    )
    return multi_measure_positive, multi_system_hard_surface


class SystemGeometryFeatureStabilityAuditV1Tests(unittest.TestCase):
    def test_expanded_fixture_surface_has_multi_measure_positive_and_hard_negative(self) -> None:
        positive, hard_surface = _expanded_reports()

        self.assertEqual(len(positive.systems), 1)
        self.assertEqual(len(positive.systems[0].staff_instance_ids), 2)
        self.assertEqual(len(positive.systems[0].measures), 2)

        self.assertEqual(len(hard_surface.systems), 2)
        self.assertEqual(
            [len(system.staff_instance_ids) for system in hard_surface.systems],
            [2, 2],
        )
        self.assertEqual(
            [len(system.measures) for system in hard_surface.systems],
            [2, 2],
        )

        dataset = build_system_geometry_evidence_dataset_v1(
            dataset_id="expanded-system-geometry-fixture-v1",
            reports=(positive, hard_surface),
        )
        self.assertEqual(
            dataset.relation_counts(),
            {"SAME_SYSTEM": 3, "DIFFERENT_SYSTEM": 4},
        )
        self.assertEqual(len(dataset.records), 7)

    def test_feature_stability_audit_proves_coarse_raw_features_are_not_sufficient(self) -> None:
        dataset = build_system_geometry_evidence_dataset_v1(
            dataset_id="expanded-system-geometry-fixture-v1",
            reports=_expanded_reports(),
        )
        audit = audit_system_geometry_feature_stability_v1(dataset)

        self.assertEqual(audit.hard_positive_count, 3)
        self.assertEqual(audit.hard_adjacent_negative_count, 4)
        for feature_name in (
            "staff_count_pair",
            "measure_count_pair",
            "grouping_token_presence_pair",
            "grouping_token_signature_pair",
            "barline_group_count_pair",
        ):
            self.assertIn(feature_name, audit.overlapping_features)
        self.assertEqual(
            audit.claim_boundary,
            "FIXTURE_ONLY_NO_RUNTIME_GROUPING_RULE",
        )
        self.assertFalse(system_geometry_feature_stability_rule_design_allowed())
        self.assertFalse(system_geometry_feature_stability_runtime_connection_allowed())

    def test_expanded_audit_fingerprint_repeats_10_of_10(self) -> None:
        dataset = build_system_geometry_evidence_dataset_v1(
            dataset_id="expanded-system-geometry-fixture-v1",
            reports=_expanded_reports(),
        )
        fingerprints = {
            audit_system_geometry_feature_stability_v1(dataset).fingerprint()
            for _ in range(10)
        }
        self.assertEqual(len(fingerprints), 1)
        self.assertEqual(len(next(iter(fingerprints))), 64)

    def test_version_boundary_is_explicit(self) -> None:
        self.assertEqual(
            SYSTEM_GEOMETRY_FEATURE_STABILITY_AUDIT_VERSION,
            "system-geometry-feature-stability-audit-v1",
        )
        self.assertEqual(
            SYSTEM_GEOMETRY_FEATURE_STABILITY_AUDIT_CLAIM_BOUNDARY,
            "FIXTURE_ONLY_NO_RUNTIME_GROUPING_RULE",
        )


if __name__ == "__main__":
    unittest.main()
