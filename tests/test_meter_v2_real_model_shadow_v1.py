from __future__ import annotations

import inspect
import math
import unittest

from st_omr_training.meter_v2_deterministic_composer_v1 import (
    ACCEPTED,
    AMBIGUOUS,
    REJECTED,
    MeterBox,
)
from st_omr_training.meter_v2_real_model_shadow_v1 import (
    D11_PRESENCE_BRIDGE_CHECKPOINT_SHA256,
    D11_PRESENCE_BRIDGE_STATUS,
    FROZEN_DIGIT_SPECIALISTS,
    MeterDigitSlotScores,
    checkpoint_loading_allowed_in_ci,
    compose_meter_v2_shadow_from_model_evidence,
    digit_observation_from_slot,
    presence_from_d11_class,
    resolver_connection_allowed,
)
import st_omr_training.meter_v2_real_model_shadow_v1 as real_shadow


class MeterV2RealModelShadowV1Tests(unittest.TestCase):
    def test_frozen_checkpoint_identities_thresholds_and_validation_counts(self) -> None:
        self.assertEqual(
            D11_PRESENCE_BRIDGE_CHECKPOINT_SHA256,
            "cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3",
        )
        self.assertEqual(D11_PRESENCE_BRIDGE_STATUS, "TECHNICAL_BASELINE_ONLY")
        frozen = {
            spec.digit: (
                spec.checkpoint_sha256,
                spec.threshold_milli,
                spec.validation_tp,
                spec.validation_fp,
                spec.validation_fn,
                spec.validation_tn,
            )
            for spec in FROZEN_DIGIT_SPECIALISTS
        }
        self.assertEqual(
            frozen,
            {
                2: (
                    "92b985d989e4338e3ae39b0a984879f4188be32c0d281390839117e1e9a715fa",
                    480,
                    185,
                    4,
                    1,
                    3182,
                ),
                3: (
                    "5ee45faf2efe0e2c83dbad716736d7ae16ad7251730431d368c10c4574836485",
                    600,
                    203,
                    0,
                    1,
                    3168,
                ),
                4: (
                    "dcd582b60b39e65798aa77aacea3cc797cd7513b7925151f0573be4aec6af43f",
                    470,
                    788,
                    23,
                    4,
                    2557,
                ),
            },
        )
        self.assertGreaterEqual(FROZEN_DIGIT_SPECIALISTS[0].validation_f1, 0.98)
        self.assertGreaterEqual(FROZEN_DIGIT_SPECIALISTS[1].validation_f1, 0.99)
        self.assertGreaterEqual(FROZEN_DIGIT_SPECIALISTS[2].validation_f1, 0.98)

    def test_d11_bridge_collapses_only_to_presence(self) -> None:
        absent = presence_from_d11_class(
            status=ACCEPTED,
            meter_class="none",
            confidence_milli=999,
        )
        self.assertEqual(absent.status, ACCEPTED)
        self.assertIs(absent.present, False)

        for meter_class in ("2/4", "3/4", "4/4"):
            present = presence_from_d11_class(
                status=ACCEPTED,
                meter_class=meter_class,
                confidence_milli=900,
            )
            self.assertEqual(present.status, ACCEPTED)
            self.assertIs(present.present, True)

    def test_d11_bridge_fails_closed(self) -> None:
        invalid = presence_from_d11_class(
            status=ACCEPTED,
            meter_class="6/8",
            confidence_milli=900,
        )
        self.assertEqual(invalid.status, REJECTED)
        ambiguous = presence_from_d11_class(
            status=AMBIGUOUS,
            meter_class=None,
            confidence_milli=500,
        )
        self.assertEqual(ambiguous.status, AMBIGUOUS)
        rejected = presence_from_d11_class(
            status=REJECTED,
            meter_class=None,
            confidence_milli=500,
        )
        self.assertEqual(rejected.status, REJECTED)
        nonfinite_like = presence_from_d11_class(
            status=ACCEPTED,
            meter_class="2/4",
            confidence_milli=1001,
        )
        self.assertEqual(nonfinite_like.status, REJECTED)

    @staticmethod
    def _slot(slot_id: str, y0: float, y1: float, *, s2: int, s3: int, s4: int):
        return MeterDigitSlotScores(
            slot_id=slot_id,
            bbox=MeterBox(20.0, y0, 40.0, y1),
            score_2_milli=s2,
            score_3_milli=s3,
            score_4_milli=s4,
        )

    def test_real_score_bridge_composes_2_4_3_4_and_4_4(self) -> None:
        cases = (
            (
                "2/4",
                self._slot("upper", 10.0, 30.0, s2=900, s3=100, s4=100),
                self._slot("lower", 35.0, 55.0, s2=100, s3=100, s4=900),
            ),
            (
                "3/4",
                self._slot("upper", 10.0, 30.0, s2=100, s3=900, s4=100),
                self._slot("lower", 35.0, 55.0, s2=100, s3=100, s4=900),
            ),
            (
                "4/4",
                self._slot("upper", 10.0, 30.0, s2=100, s3=100, s4=900),
                self._slot("lower", 35.0, 55.0, s2=100, s3=100, s4=900),
            ),
        )
        for expected, upper, lower in cases:
            with self.subTest(expected=expected):
                result = compose_meter_v2_shadow_from_model_evidence(
                    d11_status=ACCEPTED,
                    d11_meter_class=expected,
                    d11_confidence_milli=950,
                    slots=(upper, lower),
                )
                self.assertEqual(result.status, ACCEPTED)
                self.assertEqual(result.meter_class, expected)

    def test_slot_specialist_conflict_is_ambiguous_not_confidence_ranked(self) -> None:
        slot = self._slot("upper", 10.0, 30.0, s2=990, s3=100, s4=980)
        observation = digit_observation_from_slot(slot)
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.status, AMBIGUOUS)
        self.assertIsNone(observation.digit)

        result = compose_meter_v2_shadow_from_model_evidence(
            d11_status=ACCEPTED,
            d11_meter_class="2/4",
            d11_confidence_milli=999,
            slots=(
                slot,
                self._slot("lower", 35.0, 55.0, s2=100, s3=100, s4=900),
            ),
        )
        self.assertEqual(result.status, AMBIGUOUS)

    def test_presence_digit_conflict_fails_closed(self) -> None:
        result = compose_meter_v2_shadow_from_model_evidence(
            d11_status=ACCEPTED,
            d11_meter_class="none",
            d11_confidence_milli=999,
            slots=(self._slot("upper", 10.0, 30.0, s2=900, s3=100, s4=100),),
        )
        self.assertEqual(result.status, AMBIGUOUS)
        self.assertIsNone(result.meter_class)

    def test_invalid_slot_or_score_is_rejected(self) -> None:
        invalid_bbox = MeterDigitSlotScores(
            slot_id="bad",
            bbox=MeterBox(40.0, 10.0, 20.0, 30.0),
            score_2_milli=900,
            score_3_milli=0,
            score_4_milli=0,
        )
        result = compose_meter_v2_shadow_from_model_evidence(
            d11_status=ACCEPTED,
            d11_meter_class="2/4",
            d11_confidence_milli=900,
            slots=(invalid_bbox,),
        )
        self.assertEqual(result.status, REJECTED)

        invalid_score = MeterDigitSlotScores(
            slot_id="bad-score",
            bbox=MeterBox(20.0, 10.0, 40.0, 30.0),
            score_2_milli=1001,
            score_3_milli=0,
            score_4_milli=0,
        )
        result = compose_meter_v2_shadow_from_model_evidence(
            d11_status=ACCEPTED,
            d11_meter_class="2/4",
            d11_confidence_milli=900,
            slots=(invalid_score,),
        )
        self.assertEqual(result.status, REJECTED)

    def test_no_passing_digit_is_omitted_and_presence_then_fails_closed(self) -> None:
        slot = self._slot("quiet", 10.0, 30.0, s2=100, s3=100, s4=100)
        self.assertIsNone(digit_observation_from_slot(slot))
        result = compose_meter_v2_shadow_from_model_evidence(
            d11_status=ACCEPTED,
            d11_meter_class="2/4",
            d11_confidence_milli=900,
            slots=(slot,),
        )
        self.assertEqual(result.status, AMBIGUOUS)

    def test_representative_pipeline_is_10_of_10_deterministic(self) -> None:
        kwargs = dict(
            d11_status=ACCEPTED,
            d11_meter_class="3/4",
            d11_confidence_milli=973,
            slots=(
                self._slot("upper", 10.0, 30.0, s2=120, s3=887, s4=100),
                self._slot("lower", 35.0, 55.0, s2=100, s3=80, s4=932),
            ),
        )
        reports = [compose_meter_v2_shadow_from_model_evidence(**kwargs) for _ in range(10)]
        self.assertEqual(len(set(reports)), 1)
        self.assertEqual(reports[0].status, ACCEPTED)
        self.assertEqual(reports[0].meter_class, "3/4")

    def test_shadow_isolation(self) -> None:
        source = inspect.getsource(real_shadow)
        self.assertNotIn("import torch", source)
        self.assertNotIn("stage7d11_barline_meter_training", source)
        self.assertNotIn("runtime_deterministic_resolver", source)
        self.assertFalse(checkpoint_loading_allowed_in_ci())
        self.assertFalse(resolver_connection_allowed())


if __name__ == "__main__":
    unittest.main()
