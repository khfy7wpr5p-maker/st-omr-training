from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import unittest

from PIL import Image, ImageDraw

from st_omr_training.poly_v2_ossq_cross_render_audit import (
    B8R_EXPECTED_PAIR_COUNT,
    B8R_EXPECTED_PAIR_COUNT_BY_SCORE,
    B8R_EXPECTED_B8P_RECEIPT_SHA256,
    B8R_READY_SCORE_IDS,
    CrossRenderDecision,
    StructuralSignature,
    audit_policy_fingerprint,
    build_cross_render_audit,
    cosine_similarity_ppm,
    extract_structural_signature,
    reviewer_identity_sha256,
)


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _synthetic_signature(seed: str) -> StructuralSignature:
    raw = sha256(seed.encode("utf-8")).digest() + sha256((seed + ":2").encode("utf-8")).digest()
    primary = tuple(1000 + raw[index % len(raw)] * 30 for index in range(64))
    raw2 = sha256((seed + ":secondary").encode("utf-8")).digest() * 4
    secondary = tuple(1000 + raw2[index] * 30 for index in range(128))
    return StructuralSignature(
        primary=primary,
        secondary=secondary,
        image_width=400,
        image_height=100,
        otsu_threshold=127,
        suppressed_staff_rows=10,
        ink_fraction_ppm=120_000,
    )


def _full_population():
    pairs = []
    scanned = {}
    rendered = {}
    for score_id, count in B8R_EXPECTED_PAIR_COUNT_BY_SCORE:
        for index in range(1, count + 1):
            segment_id = f"sq{score_id}:{1 + (index - 1) // 20:04d}:{1 + (index - 1) % 20:04d}"
            pairs.append(
                {
                    "score_id": score_id,
                    "segment_id": segment_id,
                    "image_sha256": _digest("image:" + segment_id),
                    "musicxml_sha256": _digest("xml:" + segment_id),
                    "independent_render_png_sha256": _digest("render:" + segment_id),
                }
            )
            signature = _synthetic_signature(segment_id)
            scanned[segment_id] = signature
            rendered[segment_id] = signature
    return pairs, scanned, rendered


class B8RCrossRenderAuditTests(unittest.TestCase):
    def test_policy_and_reviewer_identity_are_stable(self) -> None:
        self.assertEqual(len(audit_policy_fingerprint()), 64)
        reviewer = reviewer_identity_sha256(
            verovio_version="6.2.1",
            cairosvg_version="2.8.2",
            pillow_version="12.3.0",
        )
        self.assertEqual(len(reviewer), 64)
        self.assertEqual(
            reviewer,
            reviewer_identity_sha256(
                verovio_version="6.2.1",
                cairosvg_version="2.8.2",
                pillow_version="12.3.0",
            ),
        )

    def test_signature_extraction_is_deterministic_and_suppresses_staff_rows(self) -> None:
        image = Image.new("L", (320, 96), 255)
        draw = ImageDraw.Draw(image)
        for y in (30, 34, 38, 42, 46):
            draw.line((5, y, 314, y), fill=0, width=1)
        for x in (45, 105, 180, 245):
            draw.ellipse((x, 35, x + 8, 41), fill=0)
            draw.line((x + 8, 18, x + 8, 39), fill=0, width=2)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        left = extract_structural_signature(data)
        right = extract_structural_signature(data)
        self.assertEqual(left, right)
        self.assertGreater(left.suppressed_staff_rows, 0)
        self.assertGreater(cosine_similarity_ppm(left.primary, right.primary), 999_999)
        self.assertGreater(cosine_similarity_ppm(left.secondary, right.secondary), 999_999)

    def test_full_frozen_population_requires_unique_mutual_match_on_both_signatures(self) -> None:
        pairs, scanned, rendered = _full_population()
        self.assertEqual(len(pairs), B8R_EXPECTED_PAIR_COUNT)
        self.assertEqual(set(score for score, _ in B8R_EXPECTED_PAIR_COUNT_BY_SCORE), set(B8R_READY_SCORE_IDS))
        reviewer = reviewer_identity_sha256(
            verovio_version="6.2.1",
            cairosvg_version="2.8.2",
            pillow_version="12.3.0",
        )
        receipt = build_cross_render_audit(
            b8p_receipt_sha256=B8R_EXPECTED_B8P_RECEIPT_SHA256,
            pairs=pairs,
            scanned_signatures=scanned,
            render_signatures=rendered,
            reviewer_identity_sha=reviewer,
        )
        self.assertEqual(len(receipt.observations), B8R_EXPECTED_PAIR_COUNT)
        self.assertEqual(dict(receipt.score_cross_conflicts), {score: 0 for score in B8R_READY_SCORE_IDS})
        self.assertEqual(
            dict(receipt.decision_counts)[CrossRenderDecision.VERIFIED_CANDIDATE.value],
            B8R_EXPECTED_PAIR_COUNT,
        )
        self.assertEqual(
            dict(receipt.decision_counts)[CrossRenderDecision.REVIEW_REQUIRED.value],
            0,
        )
        self.assertFalse(receipt.stage8_admission_authority)
        self.assertFalse(receipt.train_validation_assignment_authority)
        self.assertFalse(receipt.test_artifact_bytes_accessed)
        self.assertFalse(receipt.production_authority)
        self.assertFalse(receipt.commercial_use_authority)
        self.assertEqual(len(receipt.receipt_sha256), 64)


if __name__ == "__main__":
    unittest.main()
