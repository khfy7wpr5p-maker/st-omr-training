from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from st_omr_training.stage7d13_measure_derivatives import make_letterbox_plan
from st_omr_training.stage7d13_measure_derivative_verifier import (
    Stage7D13VerificationError,
    _plan,
    verify_stage7d13_measure_derivatives,
)


PNG_SHA = "a" * 64


class Stage7D13MeasureDerivativeVerifierTests(unittest.TestCase):
    def test_independent_plan_agrees_with_frozen_builder_contract(self) -> None:
        measure = {"x_min": 11.2, "y_min": 7.9, "x_max": 265.4, "y_max": 81.1}
        builder = make_letterbox_plan(
            measure,
            image_width=500,
            image_height=180,
            source_png_sha256=PNG_SHA,
        )
        verifier = _plan(
            measure,
            image_width=500,
            image_height=180,
            source_png_sha=PNG_SHA,
        )
        self.assertEqual(
            (builder.crop_left, builder.crop_top, builder.crop_right, builder.crop_bottom),
            (verifier.left, verifier.top, verifier.right, verifier.bottom),
        )
        self.assertEqual(builder.scale, verifier.scale)
        self.assertEqual(builder.pad_x, verifier.pad_x)
        self.assertEqual(builder.pad_y, verifier.pad_y)
        self.assertEqual(builder.transform_fingerprint, verifier.fingerprint)

    def test_premature_complete_or_wrong_top_level_fails_before_dependency_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            corpus = base / "corpus"
            d6 = base / "d6"
            d12 = base / "d12"
            derivative = base / "d13"
            for directory in (corpus, d6, d12, derivative):
                directory.mkdir()
            (derivative / "COMPLETE").write_text("premature", encoding="ascii")
            with self.assertRaisesRegex(Stage7D13VerificationError, "top-level layout|COMPLETE"):
                verify_stage7d13_measure_derivatives(
                    corpus_root=corpus,
                    d6_root=d6,
                    d12_root=d12,
                    derivative_root=derivative,
                )

    def test_symlink_derivative_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            corpus = base / "corpus"
            d6 = base / "d6"
            d12 = base / "d12"
            target = base / "real-d13"
            link = base / "linked-d13"
            for directory in (corpus, d6, d12, target):
                directory.mkdir()
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(Stage7D13VerificationError, "regular non-symlink directory"):
                verify_stage7d13_measure_derivatives(
                    corpus_root=corpus,
                    d6_root=d6,
                    d12_root=d12,
                    derivative_root=link,
                )


if __name__ == "__main__":
    unittest.main()
