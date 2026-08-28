from __future__ import annotations

import unittest

from st_omr_training.polyphonic_representation import (
    ClefAssignment,
    EventKind,
    ExactRational,
    KeySignature,
    NoteType,
    PolyEvent,
    PolyMeasure,
    PolyPart,
    PolyScore,
    TimeSignature,
)
from st_omr_training.polyphonic_serialization import (
    PolyphonicSerializationError,
    parse_canonical_polyphonic_json,
    parse_polyphonic_payload,
    serialize_polyphonic_score,
    tokenize_polyphonic_score,
)


def _score_with_part_id(part_id: str) -> PolyScore:
    measure = PolyMeasure(
        measure_index=1,
        source_number="1",
        time_signature=TimeSignature(beats=(4,), beat_type=4),
        key_signature=KeySignature(fifths=0),
        clefs=(ClefAssignment(staff=1, sign="G", line=2),),
        events=(
            PolyEvent(
                event_id="rest-1",
                kind=EventKind.REST,
                onset=ExactRational(0, 1),
                duration=ExactRational(1, 4),
                voice=1,
                staff=1,
                note_type=NoteType.QUARTER,
            ),
        ),
    )
    return PolyScore(parts=(PolyPart(part_id=part_id, staff_count=1, measures=(measure,)),))


class PolyphonicSerializationUtf8Tests(unittest.TestCase):
    def test_unpaired_surrogate_is_rejected_consistently(self) -> None:
        score = _score_with_part_id("bad-\ud800-id")
        with self.assertRaisesRegex(PolyphonicSerializationError, "valid UTF-8"):
            serialize_polyphonic_score(score)
        with self.assertRaisesRegex(PolyphonicSerializationError, "valid UTF-8"):
            tokenize_polyphonic_score(score)
        with self.assertRaisesRegex(PolyphonicSerializationError, "valid UTF-8"):
            parse_canonical_polyphonic_json(score.canonical_json())

    def test_non_text_mapping_key_fails_closed_without_sort_type_error(self) -> None:
        with self.assertRaisesRegex(PolyphonicSerializationError, "object keys must be text"):
            parse_polyphonic_payload({1: "invalid", "parts": []})


if __name__ == "__main__":
    unittest.main()
