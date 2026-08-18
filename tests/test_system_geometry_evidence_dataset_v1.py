from __future__ import annotations

import unittest

from st_omr_training.renderer import RendererConfig, _load_verovio_runtime
from st_omr_training.system_geometry_evidence_dataset_v1 import (
    SYSTEM_GEOMETRY_EVIDENCE_DATASET_VERSION,
    SystemGeometryEvidenceDatasetError,
    build_system_geometry_evidence_dataset_v1,
    system_geometry_evidence_dataset_runtime_connection_allowed,
)
from st_omr_training.system_geometry_evidence_extractor_v1 import (
    extract_system_geometry_evidence_v1,
)
from st_omr_training.system_geometry_evidence_v1 import StaffSystemRelation


def _grand_staff_musicxml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions><key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>2</staves>
        <part-symbol top-staff="1" bottom-staff="2">brace</part-symbol>
        <clef number="1"><sign>G</sign><line>2</line></clef>
        <clef number="2"><sign>F</sign><line>4</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type><staff>1</staff></note>
      <backup><duration>4</duration></backup>
      <note><pitch><step>C</step><octave>3</octave></pitch><duration>4</duration><voice>2</voice><type>whole</type><staff>2</staff></note>
      <barline location="right"><bar-style>light-heavy</bar-style></barline>
    </measure>
  </part>
</score-partwise>"""


def _two_system_musicxml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Fixture</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions><key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type><staff>1</staff></note>
      <barline location="right"><bar-style>regular</bar-style></barline>
    </measure>
    <measure number="2">
      <print new-system="yes"/>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type><staff>1</staff></note>
      <barline location="right"><bar-style>light-heavy</bar-style></barline>
    </measure>
  </part>
</score-partwise>"""


def _render(xml_text: str, *, encoded_breaks: bool = False) -> str:
    verovio, package_version = _load_verovio_runtime()
    toolkit = verovio.toolkit()
    if not str(toolkit.getVersion()).startswith(package_version):
        raise AssertionError("pinned Verovio runtime mismatch")
    if toolkit.setInputFrom("xml") is False:
        raise AssertionError("fixture input mode rejected")
    config = RendererConfig(breaks="encoded" if encoded_breaks else "auto")
    options = dict(config.verovio_options())
    options.update({"svgBoundingBoxes": True, "svgContentBoundingBoxes": True})
    if toolkit.setOptions(options) is False:
        raise AssertionError("fixture options rejected")
    if toolkit.loadData(xml_text) is False:
        raise AssertionError("fixture MusicXML rejected")
    if toolkit.getPageCount() != 1:
        raise AssertionError("fixture must remain on one page")
    return toolkit.renderToSVG(1, True)


def _reports():
    positive = extract_system_geometry_evidence_v1(
        page_id="grand-staff-positive",
        svg=_render(_grand_staff_musicxml()),
    )
    negative = extract_system_geometry_evidence_v1(
        page_id="adjacent-systems-negative",
        svg=_render(_two_system_musicxml(), encoded_breaks=True),
    )
    return positive, negative


class SystemGeometryEvidenceDatasetV1Tests(unittest.TestCase):
    def test_pilot_surface_contains_required_positive_and_negative(self) -> None:
        dataset = build_system_geometry_evidence_dataset_v1(
            dataset_id="fixture-pilot-v1", reports=_reports()
        )
        self.assertEqual(len(dataset.records), 2)
        self.assertEqual(
            dataset.relation_counts(),
            {"SAME_SYSTEM": 1, "DIFFERENT_SYSTEM": 1},
        )

        positive = next(
            x for x in dataset.records if x.relation is StaffSystemRelation.SAME_SYSTEM
        )
        self.assertEqual(positive.system_a_staff_count, 2)
        self.assertEqual(positive.system_b_staff_count, 2)
        self.assertIn("grpSym", positive.system_a_grouping_tokens)
        self.assertFalse(positive.adjacent_different_systems)

        negative = next(
            x
            for x in dataset.records
            if x.relation is StaffSystemRelation.DIFFERENT_SYSTEM
        )
        self.assertEqual(negative.system_order_gap, 1)
        self.assertTrue(negative.adjacent_different_systems)
        self.assertEqual(negative.system_a_staff_count, 1)
        self.assertEqual(negative.system_b_staff_count, 1)

    def test_dataset_keeps_svg_and_extractor_provenance(self) -> None:
        reports = _reports()
        dataset = build_system_geometry_evidence_dataset_v1(
            dataset_id="fixture-pilot-v1", reports=reports
        )
        expected = {
            (report.page_id, report.source_svg_sha256, report.fingerprint())
            for report in reports
        }
        observed = {
            (
                record.source_page_id,
                record.source_svg_sha256,
                record.source_extractor_fingerprint,
            )
            for record in dataset.records
        }
        self.assertEqual(observed, expected)

    def test_dataset_fingerprint_repeats_10_of_10(self) -> None:
        reports = _reports()
        fingerprints = {
            build_system_geometry_evidence_dataset_v1(
                dataset_id="fixture-pilot-v1", reports=reports
            ).fingerprint()
            for _ in range(10)
        }
        self.assertEqual(len(fingerprints), 1)
        self.assertEqual(len(next(iter(fingerprints))), 64)

    def test_incomplete_one_sided_surface_fails_closed(self) -> None:
        positive, negative = _reports()
        with self.assertRaisesRegex(
            SystemGeometryEvidenceDatasetError, "DIFFERENT_SYSTEM"
        ):
            build_system_geometry_evidence_dataset_v1(
                dataset_id="positive-only", reports=(positive,)
            )
        with self.assertRaisesRegex(SystemGeometryEvidenceDatasetError, "SAME_SYSTEM"):
            build_system_geometry_evidence_dataset_v1(
                dataset_id="negative-only", reports=(negative,)
            )

    def test_duplicate_source_page_id_fails_closed(self) -> None:
        positive, _ = _reports()
        with self.assertRaisesRegex(SystemGeometryEvidenceDatasetError, "page ids"):
            build_system_geometry_evidence_dataset_v1(
                dataset_id="duplicate-pages", reports=(positive, positive)
            )

    def test_version_and_runtime_boundary_are_explicit(self) -> None:
        self.assertEqual(
            SYSTEM_GEOMETRY_EVIDENCE_DATASET_VERSION,
            "system-geometry-evidence-dataset-v1",
        )
        self.assertFalse(system_geometry_evidence_dataset_runtime_connection_allowed())


if __name__ == "__main__":
    unittest.main()
