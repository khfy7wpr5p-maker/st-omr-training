from __future__ import annotations

from fractions import Fraction
import unittest

from st_omr_training.musicxml_scoregraph_adapter import parse_musicxml_to_scoregraph
from st_omr_training.polyphonic_representation import EventKind
from st_omr_training.scoregraph_v2 import CapabilityOutcome, SourceNavigationKind


TWO_VOICE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1"><part-name>Piano</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>1</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
      </attributes>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>8</duration><voice>1</voice><type>half</type><staff>1</staff>
      </note>
      <note>
        <pitch><step>D</step><octave>4</octave></pitch>
        <duration>8</duration><voice>1</voice><type>half</type><staff>1</staff>
      </note>
      <backup><duration>16</duration></backup>
      <note>
        <pitch><step>G</step><octave>3</octave></pitch>
        <duration>16</duration><voice>2</voice><type>whole</type><staff>1</staff>
      </note>
    </measure>
  </part>
</score-partwise>
"""


CHORD_AND_VOICE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1"><part-name>Piano</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>1</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
      </attributes>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>4</duration><voice>1</voice><type>quarter</type><staff>1</staff>
      </note>
      <note>
        <chord/>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>4</duration><voice>1</voice><type>quarter</type><staff>1</staff>
      </note>
      <backup><duration>4</duration></backup>
      <note>
        <pitch><step>G</step><octave>3</octave></pitch>
        <duration>4</duration><voice>2</voice><type>quarter</type><staff>1</staff>
      </note>
    </measure>
  </part>
</score-partwise>
"""


class MusicXMLScoreGraphAdapterTests(unittest.TestCase):
    def test_backup_reconstructs_independent_voice_onsets(self) -> None:
        graph = parse_musicxml_to_scoregraph(TWO_VOICE_XML)

        events = graph.core.parts[0].measures[0].events
        self.assertEqual(
            tuple((event.voice, event.onset.fraction, event.duration.fraction) for event in events),
            (
                (1, Fraction(0, 1), Fraction(1, 2)),
                (2, Fraction(0, 1), Fraction(1, 1)),
                (1, Fraction(1, 2), Fraction(1, 2)),
            ),
        )
        self.assertEqual(
            tuple(item.kind for item in graph.source_navigation),
            (SourceNavigationKind.BACKUP,),
        )
        self.assertEqual(graph.source_navigation[0].duration.fraction, Fraction(1, 1))
        self.assertEqual(tuple(node.voice for node in graph.voice_nodes), (1, 2))

    def test_chord_is_not_flattened_into_independent_voice(self) -> None:
        graph = parse_musicxml_to_scoregraph(CHORD_AND_VOICE_XML)
        events = graph.core.parts[0].measures[0].events

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].kind, EventKind.CHORD)
        self.assertEqual(events[0].voice, 1)
        self.assertEqual(len(events[0].noteheads), 2)
        self.assertEqual(events[1].kind, EventKind.NOTE)
        self.assertEqual(events[1].voice, 2)
        self.assertEqual(events[0].onset.fraction, events[1].onset.fraction)

    def test_missing_measure_number_is_review_required_not_blocked(self) -> None:
        data = TWO_VOICE_XML.replace(b'<measure number="1">', b"<measure>", 1)
        graph = parse_musicxml_to_scoregraph(data)

        self.assertEqual(graph.core.parts[0].measures[0].source_number, "1")
        finding = next(item for item in graph.capabilities if item.code == "musicxml.measure_number_missing")
        self.assertEqual(finding.outcome, CapabilityOutcome.REVIEW_REQUIRED)


if __name__ == "__main__":
    unittest.main()
