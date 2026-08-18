from __future__ import annotations

import inspect
import math
import unittest

from st_omr_training import runtime_specialist_shadow_acceptance_v1 as shadow


class SpecialistShadowAcceptanceTests(unittest.TestCase):
    def test_frozen_d13_io_and_decoder_contract(self) -> None:
        self.assertEqual(shadow.D13_INPUT_SHAPE, (1, 128, 512))
        self.assertEqual(shadow.D13_OUTPUT_STRIDE, 4)
        self.assertEqual(shadow.D13_DECODER_SCORE_THRESHOLD, 0.25)
        self.assertEqual(shadow.D13_LOCAL_MAX_KERNEL, 3)
        self.assertEqual(shadow.D13_TOP_K, 256)
        self.assertEqual(shadow.D13_CENTER_TOLERANCE_PX, 4.0)
        self.assertEqual(shadow.D13_BBOX_IOU_THRESHOLD, 0.50)

    def test_notehead_real_checkpoint_passes_shadow_gate(self) -> None:
        item = shadow.NOTEHEAD_SHADOW
        self.assertTrue(item.smoke.passed)
        self.assertTrue(item.metrics_pass)
        self.assertEqual(item.completed_epochs, 10)
        self.assertEqual(item.shadow_decision, "PASS")
        self.assertEqual(item.classes, ("open", "filled"))

    def test_accidental_real_checkpoint_is_held(self) -> None:
        item = shadow.ACCIDENTAL_SHADOW
        self.assertTrue(item.smoke.passed)
        self.assertFalse(item.metrics_pass)
        self.assertLess(item.macro_f1, item.macro_gate)
        self.assertEqual(item.completed_epochs, 5)
        self.assertEqual(item.shadow_decision, "HOLD")

    def test_rest_r2_class_gates_pass_but_integrated_adapter_is_held(self) -> None:
        item = shadow.REST_R2_SHADOW
        self.assertTrue(item.proposal_smoke.passed)
        self.assertTrue(item.class_gates_pass)
        self.assertFalse(item.integrated_arbitration_ready)
        self.assertEqual(item.shadow_decision, "HOLD")
        self.assertEqual(
            tuple(verifier.class_name for verifier in item.verifiers),
            ("half", "quarter", "eighth"),
        )
        self.assertTrue(all(not verifier.test_opened for verifier in item.verifiers))
        self.assertTrue(all(not verifier.production_promotion for verifier in item.verifiers))

    def test_fail_closed_threshold_boundary(self) -> None:
        self.assertEqual(
            shadow.fail_closed_observation_status(
                score=0.25, threshold=0.25, bbox_finite=True
            ),
            "accepted",
        )
        self.assertEqual(
            shadow.fail_closed_observation_status(
                score=0.249999, threshold=0.25, bbox_finite=True
            ),
            "ambiguous",
        )
        self.assertEqual(
            shadow.fail_closed_observation_status(
                score=0.90, threshold=0.25, bbox_finite=False
            ),
            "rejected",
        )
        self.assertEqual(
            shadow.fail_closed_observation_status(
                score=0.90, threshold=0.25, bbox_finite=True, class_conflict=True
            ),
            "ambiguous",
        )

    def test_non_finite_and_out_of_range_scores_reject(self) -> None:
        for score in (math.nan, math.inf, -math.inf, -0.01, 1.01):
            with self.subTest(score=score):
                self.assertEqual(
                    shadow.fail_closed_observation_status(
                        score=score, threshold=0.25, bbox_finite=True
                    ),
                    "rejected",
                )

    def test_ten_of_ten_real_smoke_evidence_is_frozen(self) -> None:
        for smoke in (
            shadow.NOTEHEAD_SHADOW.smoke,
            shadow.ACCIDENTAL_SHADOW.smoke,
            shadow.REST_R2_SHADOW.proposal_smoke,
        ):
            self.assertEqual(smoke.identical_runs, 10)
            self.assertEqual(smoke.required_runs, 10)
            self.assertEqual(smoke.fail_closed_passed, 4)
            self.assertEqual(smoke.fail_closed_required, 4)

    def test_shadow_package_cannot_authorize_resolver_connection(self) -> None:
        self.assertFalse(shadow.resolver_connection_allowed())
        source = inspect.getsource(shadow)
        self.assertNotIn("runtime_deterministic_resolver", source)
        self.assertNotIn("stage7d10", source.lower())
        self.assertNotIn("optimizer", source.lower())
        self.assertNotIn("torch.load", source)


if __name__ == "__main__":
    unittest.main()
