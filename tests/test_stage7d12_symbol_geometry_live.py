from __future__ import annotations

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from st_omr_training.stage7d12_symbol_geometry import (
    Stage7D12SymbolGeometryError,
    extract_symbol_geometry,
)
from st_omr_training.stage7d5_geometry import render_musicxml_geometry_svg


GOLDEN = Path(__file__).parent / "golden"
GOLDEN_NAMES = (
    "accidentals.musicxml",
    "basic_2_4.musicxml",
    "basic_4_4.musicxml",
    "chords_2_3_4.musicxml",
    "rest_3_4.musicxml",
    "time_change.musicxml",
)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _expected_counts(musicxml: bytes) -> tuple[int, int, int]:
    root = ET.fromstring(musicxml)
    noteheads = 0
    rests = 0
    accidentals = 0
    for element in root.iter():
        if _local(element.tag) != "note":
            continue
        children = list(element)
        if any(_local(child.tag) == "rest" for child in children):
            rests += 1
        else:
            noteheads += 1
            accidentals += sum(
                _local(child.tag) == "accidental" for child in children
            )
    return noteheads, rests, accidentals


def _flatten(pages):
    measures = [measure for page in pages for measure in page.measures]
    noteheads = [record for measure in measures for record in measure.noteheads]
    rests = [record for measure in measures for record in measure.rests]
    accidentals = [record for measure in measures for record in measure.accidentals]
    return measures, noteheads, rests, accidentals


class Stage7D12LiveSymbolGeometryTests(unittest.TestCase):
    def test_all_goldens_link_exact_symbol_cardinality(self) -> None:
        for name in GOLDEN_NAMES:
            with self.subTest(name=name):
                musicxml = (GOLDEN / name).read_bytes()
                geometry_render = render_musicxml_geometry_svg(musicxml)
                pages = extract_symbol_geometry(geometry_render, musicxml)
                self.assertEqual(len(pages), len(geometry_render.pages))

                measures, noteheads, rests, accidentals = _flatten(pages)
                expected = _expected_counts(musicxml)
                self.assertEqual(
                    (len(noteheads), len(rests), len(accidentals)),
                    expected,
                )
                self.assertGreaterEqual(len(measures), 1)
                self.assertEqual(
                    len({record.renderer_id for record in noteheads}),
                    len(noteheads),
                )
                self.assertEqual(
                    len({record.renderer_id for record in rests}),
                    len(rests),
                )
                self.assertEqual(
                    len({record.renderer_id for record in accidentals}),
                    len(accidentals),
                )

    def test_accidental_classes_and_note_linkage_are_exact(self) -> None:
        musicxml = (GOLDEN / "accidentals.musicxml").read_bytes()
        geometry_render = render_musicxml_geometry_svg(musicxml)
        pages = extract_symbol_geometry(geometry_render, musicxml)
        _, noteheads, rests, accidentals = _flatten(pages)

        self.assertEqual(len(noteheads), 4)
        self.assertEqual(len(rests), 0)
        self.assertEqual(
            [record.class_name for record in accidentals],
            ["sharp", "natural", "flat", "natural"],
        )
        self.assertEqual(
            [record.canonical_event_id for record in accidentals],
            [record.canonical_event_id for record in noteheads],
        )

    def test_chord_members_receive_distinct_audit_ids(self) -> None:
        musicxml = (GOLDEN / "chords_2_3_4.musicxml").read_bytes()
        geometry_render = render_musicxml_geometry_svg(musicxml)
        pages = extract_symbol_geometry(geometry_render, musicxml)
        _, noteheads, rests, accidentals = _flatten(pages)

        self.assertEqual(len(noteheads), 9)
        self.assertEqual(len(rests), 0)
        self.assertEqual(len(accidentals), 0)
        self.assertEqual(
            [record.canonical_event_id for record in noteheads],
            [
                "m1-e0-n0",
                "m1-e0-n1",
                "m1-e1-n0",
                "m1-e1-n1",
                "m1-e1-n2",
                "m1-e2-n0",
                "m1-e2-n1",
                "m1-e2-n2",
                "m1-e2-n3",
            ],
        )
        self.assertEqual(
            [record.class_name for record in noteheads],
            ["filled"] * 5 + ["open"] * 4,
        )

    def test_same_pinned_render_is_byte_identity_replayable_for_linkage(self) -> None:
        musicxml = (GOLDEN / "accidentals.musicxml").read_bytes()
        first_render = render_musicxml_geometry_svg(musicxml)
        second_render = render_musicxml_geometry_svg(musicxml)
        self.assertEqual(
            [page.sha256 for page in first_render.pages],
            [page.sha256 for page in second_render.pages],
        )
        self.assertEqual(
            extract_symbol_geometry(first_render, musicxml),
            extract_symbol_geometry(second_render, musicxml),
        )

    def test_musicxml_provenance_mismatch_fails_before_linkage(self) -> None:
        source = (GOLDEN / "basic_4_4.musicxml").read_bytes()
        wrong = (GOLDEN / "basic_2_4.musicxml").read_bytes()
        geometry_render = render_musicxml_geometry_svg(source)
        with self.assertRaises(Stage7D12SymbolGeometryError):
            extract_symbol_geometry(geometry_render, wrong)


if __name__ == "__main__":
    unittest.main()
