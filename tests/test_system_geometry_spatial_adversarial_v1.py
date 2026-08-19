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


# Fixture-only adversarial probes for the three candidates that remained
# disjoint on the broad robustness surface.  These cases are observations,
# never runtime thresholds or grouping rules.
_VARIANTS = (
    {
        "name": "same-system-no-brace-or-bracket",
        "measure_count": 1,
        "breaks_before": (),
        "repeat_attributes_after_break": False,
        "part_symbol": "none",
        "page_width": 2400,
        "page_height": 3200,
        "scale": 100,
        "spacing_staff": 12,
        "spacing_system": 8,
    },
    {
        # Three one-measure systems are deliberate: the first two are both
        # non-final systems, which attacks the possibility that a short last
        # system alone explains cross-system x-overlap differences.
        "name": "different-system-symmetric-three-one-measure-systems",
        "measure_count": 3,
        "breaks_before": (2, 3),
        "repeat_attributes_after_break": True,
        "part_symbol": "none",
        "page_width": 2400,
        "page_height": 5200,
        "scale": 100,
        "spacing_staff": 12,
        "spacing_system": 8,
    },
    {
        # Systems 1 and 2 each contain two identical measures and are both
        # non-final.  Repeated attributes at the second system start attack
        # exact cross-system measure-boundary alignment as a standalone cue.
        # The fifth measure creates a third system so the first two are not a
        # special final-system comparison.
        "name": "different-system-repeated-two-measure-layouts",
        "measure_count": 5,
        "breaks_before": (3, 5),
        "repeat_attributes_after_break": True,
        "part_symbol": "none",
        "page_width": 2400,
        "page_height": 7000,
        "scale": 100,
        "spacing_staff": 12,
        "spacing_system": 8,
    },
)


def _attributes(part_symbol: str) -> str:
    return f"""
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>2</staves>
        <part-symbol top-staff="1" bottom-staff="2">{part_symbol}</part-symbol>
        <clef number="1"><sign>G</sign><line>2</line></clef>
        <clef number="2"><sign>F</sign><line>4</line></clef>
      </attributes>"""


def _measure(
    number: int,
    *,
    part_symbol: str,
    include_attributes: bool,
    new_system: bool,
) -> str:
    print_tag = '<print new-system="yes"/>' if new_system else ""
    attributes = _attributes(part_symbol) if include_attributes else ""
    return f"""
    <measure number="{number}">
      {print_tag}{attributes}
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type><staff>1</staff></note>
      <backup><duration>4</duration></backup>
      <note><pitch><step>C</step><octave>3</octave></pitch><duration>4</duration><voice>2</voice><type>whole</type><staff>2</staff></note>
      <barline location="right"><bar-style>regular</bar-style></barline>
    </measure>"""


def _score(variant: dict[str, object]) -> str:
    measure_count = int(variant["measure_count"])
    breaks_before = tuple(int(value) for value in variant["breaks_before"])
    repeat_attributes = bool(variant["repeat_attributes_after_break"])
    part_symbol = str(variant["part_symbol"])
    measures = []
    for number in range(1, measure_count + 1):
        new_system = number in breaks_before
        include_attributes = number == 1 or (new_system and repeat_attributes)
        measures.append(
            _measure(
                number,
                part_symbol=part_symbol,
                include_attributes=include_attributes,
                new_system=new_system,
            )
        )
    return """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">""" + "".join(measures) + """
  </part>
</score-partwise>"""


def _render(variant: dict[str, object]) -> str:
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
    if toolkit.loadData(_score(variant)) is False:
        raise AssertionError("fixture MusicXML rejected")
    if toolkit.getPageCount() != 1:
        raise AssertionError("adversarial fixture must remain on one page")
    return toolkit.renderToSVG(1, True)


def _report(variant: dict[str, object]):
    return extract_system_geometry_spatial_evidence_v1(
        page_id=f"spatial-adversarial:{variant['name']}",
        svg=_render(variant),
    )


def _reports():
    return tuple(_report(variant) for variant in _VARIANTS)


class SystemGeometrySpatialAdversarialV1Tests(unittest.TestCase):
    def test_adversarial_surface_has_intended_topology(self) -> None:
        positive, symmetric_negative, repeated_negative = _reports()

        self.assertEqual(len(positive.systems), 1)
        self.assertEqual(len(positive.systems[0].staffs), 2)
        self.assertEqual(len(positive.systems[0].measures), 1)

        self.assertEqual(len(symmetric_negative.systems), 3)
        self.assertTrue(
            all(len(system.staffs) == 2 for system in symmetric_negative.systems)
        )
        self.assertTrue(
            all(len(system.measures) == 1 for system in symmetric_negative.systems)
        )

        self.assertEqual(len(repeated_negative.systems), 3)
        self.assertTrue(all(len(system.staffs) == 2 for system in repeated_negative.systems))
        self.assertEqual(
            [len(system.measures) for system in repeated_negative.systems],
            [2, 2, 1],
        )

        same = [
            pair
            for pair in positive.pair_observations
            if pair.relation is StaffSystemRelation.SAME_SYSTEM
        ]
        different = [
            pair
            for report in (symmetric_negative, repeated_negative)
            for pair in report.pair_observations
            if pair.relation is StaffSystemRelation.DIFFERENT_SYSTEM
        ]
        self.assertEqual(len(same), 1)
        self.assertGreaterEqual(len(different), 1)

    def test_no_brace_or_bracket_records_renderer_group_metadata(self) -> None:
        positive = _report(_VARIANTS[0])
        same = [
            pair
            for pair in positive.pair_observations
            if pair.relation is StaffSystemRelation.SAME_SYSTEM
        ]
        self.assertEqual(len(same), 1)

        tokens = tuple(
            token
            for span in positive.systems[0].grouping_spans
            for token in span.tokens
        )
        lowered = tuple(token.lower() for token in tokens)
        self.assertFalse(any("brace" in token or "bracket" in token for token in lowered))

        # Verovio 6.2.1 still exposes generic grpSym metadata for a two-staff
        # part whose MusicXML part-symbol is "none".  Preserve that finding
        # instead of pretending the SVG evidence disappears with the glyph.
        self.assertEqual(same[0].grouping_span_cover_count, 1)
        print(
            "SYSTEM_GEOMETRY_NO_BRACE_BRACKET_GROUP_METADATA",
            json.dumps(
                {
                    "grouping_span_cover_count": same[0].grouping_span_cover_count,
                    "grouping_tokens": list(tokens),
                },
                sort_keys=True,
            ),
        )

    def test_symmetric_negative_is_observed_without_requiring_separation(self) -> None:
        negative = _report(_VARIANTS[1])
        different = [
            pair
            for pair in negative.pair_observations
            if pair.relation is StaffSystemRelation.DIFFERENT_SYSTEM
        ]
        self.assertGreaterEqual(len(different), 1)
        # Record exact candidate values; overlap or continued separation are
        # both evidence.  Neither outcome is encoded as a pass criterion.
        print(
            "SYSTEM_GEOMETRY_ADVERSARIAL_DIFFERENT_SYSTEM",
            json.dumps(
                [pair.canonical_payload() for pair in different],
                sort_keys=True,
            ),
        )

    def test_repeated_measure_layout_negative_is_observed_without_fitting_rule(self) -> None:
        negative = _report(_VARIANTS[2])
        different = [
            pair
            for pair in negative.pair_observations
            if pair.relation is StaffSystemRelation.DIFFERENT_SYSTEM
        ]
        self.assertGreaterEqual(len(different), 1)
        print(
            "SYSTEM_GEOMETRY_REPEATED_MEASURE_DIFFERENT_SYSTEM",
            json.dumps(
                [pair.canonical_payload() for pair in different],
                sort_keys=True,
            ),
        )

    def test_adversarial_audit_records_ranges_without_authorizing_rule(self) -> None:
        audit = audit_system_geometry_spatial_stability_v1(_reports())
        self.assertGreater(audit.relation_counts["SAME_SYSTEM"], 0)
        self.assertGreater(audit.relation_counts["DIFFERENT_SYSTEM"], 0)
        self.assertFalse(system_geometry_spatial_rule_design_allowed())
        self.assertFalse(system_geometry_spatial_runtime_connection_allowed())
        print(
            "SYSTEM_GEOMETRY_ADVERSARIAL_SPATIAL_AUDIT",
            json.dumps(audit.canonical_payload(), sort_keys=True),
        )

    def test_adversarial_surface_is_deterministic_5_of_5(self) -> None:
        fingerprints = {
            audit_system_geometry_spatial_stability_v1(_reports()).fingerprint()
            for _ in range(5)
        }
        self.assertEqual(len(fingerprints), 1)


if __name__ == "__main__":
    unittest.main()
