from __future__ import annotations

import unittest

from st_omr_training.poly_v2_ossq_pairing_preflight import (
    B8O_EXPECTED_BLOCKED_SCORE_IDS,
    B8O_EXPECTED_READY_SCORE_COUNT,
    B8O_PAIRING_SOURCE_SPECS,
    OssqPairingPreflightError,
    OssqPairingSourcePayload,
    OssqPairingSourceSpec,
    PairingMaterializationState,
    git_blob_sha1,
    inspect_pairing_source,
    parse_scanned_alignment,
)


def h(ch: str) -> str:
    return ch * 64


def spec_for(alignment: bytes, musicxml: bytes) -> OssqPairingSourceSpec:
    return OssqPairingSourceSpec(
        score_id="123",
        imslp_id="#456",
        work_path="Composer/Work",
        alignment_git_blob_sha1=git_blob_sha1(alignment),
        cleaned_musicxml_git_blob_sha1=git_blob_sha1(musicxml),
    )


class OssqPairingPreflightTests(unittest.TestCase):
    def test_exact_first_batch_population_is_frozen(self) -> None:
        self.assertEqual(len(B8O_PAIRING_SOURCE_SPECS), 8)
        self.assertEqual(
            tuple(item.score_id for item in B8O_PAIRING_SOURCE_SPECS),
            (
                "7070781",
                "7075297",
                "7078259",
                "7093885",
                "7103818",
                "7108150",
                "7397765",
                "8071278",
            ),
        )
        self.assertEqual(B8O_EXPECTED_READY_SCORE_COUNT, 7)
        self.assertEqual(B8O_EXPECTED_BLOCKED_SCORE_IDS, ("7397765",))

    def test_parser_accepts_open_page_range_and_alignment_rows(self) -> None:
        profile = parse_scanned_alignment(
            b":\n8, 10, 9,\n7, 8\n\n5, 6, 4\n"
        )
        self.assertIsNone(profile.page_start)
        self.assertIsNone(profile.page_end)
        self.assertEqual(profile.block_count, 2)
        self.assertEqual(profile.row_count, 3)
        self.assertEqual(profile.value_count, 8)
        self.assertEqual(profile.value_sum, 57)
        self.assertTrue(profile.has_alignment_values)

    def test_parser_accepts_bounded_page_range(self) -> None:
        profile = parse_scanned_alignment(b"3:8\n4, 4\n")
        self.assertEqual(profile.page_start, 3)
        self.assertEqual(profile.page_end, 8)
        self.assertEqual(profile.value_count, 2)

    def test_parser_keeps_metadata_only_alignment_empty(self) -> None:
        profile = parse_scanned_alignment(b":\n")
        self.assertEqual(profile.value_count, 0)
        self.assertEqual(profile.row_count, 0)
        self.assertFalse(profile.has_alignment_values)

    def test_parser_rejects_reversed_page_range(self) -> None:
        with self.assertRaisesRegex(OssqPairingPreflightError, "reversed"):
            parse_scanned_alignment(b"8:3\n4, 4\n")

    def test_parser_rejects_non_positive_or_non_integer_values(self) -> None:
        with self.assertRaisesRegex(OssqPairingPreflightError, "positive integers"):
            parse_scanned_alignment(b":\n4, 0\n")
        with self.assertRaisesRegex(OssqPairingPreflightError, "positive integers"):
            parse_scanned_alignment(b":\n4, x\n")

    def test_inspection_marks_nonempty_alignment_ready(self) -> None:
        alignment = b":\n4, 5\n"
        musicxml = b'<?xml version="1.0"?><score-partwise version="4.0"></score-partwise>'
        evidence = inspect_pairing_source(
            spec_for(alignment, musicxml),
            source_document_sha256=h("a"),
            payload=OssqPairingSourcePayload(
                alignment_bytes=alignment,
                cleaned_musicxml_bytes=musicxml,
            ),
        )
        self.assertEqual(evidence.state, PairingMaterializationState.READY)
        self.assertEqual(evidence.alignment_profile.value_count, 2)

    def test_inspection_blocks_empty_alignment_without_granting_readiness(self) -> None:
        alignment = b":\n"
        musicxml = b'<score-partwise version="4.0"></score-partwise>'
        evidence = inspect_pairing_source(
            spec_for(alignment, musicxml),
            source_document_sha256=h("b"),
            payload=OssqPairingSourcePayload(
                alignment_bytes=alignment,
                cleaned_musicxml_bytes=musicxml,
            ),
        )
        self.assertEqual(
            evidence.state,
            PairingMaterializationState.BLOCKED_EMPTY_ALIGNMENT,
        )

    def test_inspection_rejects_alignment_blob_identity_drift(self) -> None:
        alignment = b":\n4, 5\n"
        musicxml = b'<score-partwise version="4.0"></score-partwise>'
        spec = spec_for(alignment, musicxml)
        with self.assertRaisesRegex(OssqPairingPreflightError, "alignment Git blob"):
            inspect_pairing_source(
                spec,
                source_document_sha256=h("c"),
                payload=OssqPairingSourcePayload(
                    alignment_bytes=alignment + b"6\n",
                    cleaned_musicxml_bytes=musicxml,
                ),
            )

    def test_inspection_rejects_musicxml_blob_identity_drift(self) -> None:
        alignment = b":\n4, 5\n"
        musicxml = b'<score-partwise version="4.0"></score-partwise>'
        spec = spec_for(alignment, musicxml)
        with self.assertRaisesRegex(OssqPairingPreflightError, "MusicXML Git blob"):
            inspect_pairing_source(
                spec,
                source_document_sha256=h("d"),
                payload=OssqPairingSourcePayload(
                    alignment_bytes=alignment,
                    cleaned_musicxml_bytes=musicxml + b" ",
                ),
            )

    def test_inspection_rejects_non_musicxml_target(self) -> None:
        alignment = b":\n4, 5\n"
        not_xml = b"plain text target"
        spec = spec_for(alignment, not_xml)
        with self.assertRaisesRegex(OssqPairingPreflightError, "MusicXML envelope"):
            inspect_pairing_source(
                spec,
                source_document_sha256=h("e"),
                payload=OssqPairingSourcePayload(
                    alignment_bytes=alignment,
                    cleaned_musicxml_bytes=not_xml,
                ),
            )


if __name__ == "__main__":
    unittest.main()
