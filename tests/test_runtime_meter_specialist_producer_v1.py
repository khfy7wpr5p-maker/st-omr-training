from __future__ import annotations

from hashlib import sha256
import unittest

from st_omr_training.runtime_geometry_engine_contract import BoxContract
from st_omr_training.runtime_local_roi_v1 import RuntimeRoiArtifact
from st_omr_training.runtime_meter_integration_v3 import MeterDigitScoresV3
from st_omr_training.runtime_meter_specialist_producer_v1 import (
    MeterRawInferenceV1,
    MeterSpecialistProducerError,
    MeterSpecialistProfileV1,
    checkpoint_loading_allowed,
    produce_meter_evidence_v1,
    resolver_connection_allowed,
    train_validation_test_access_allowed,
)
from st_omr_training.runtime_page_normalizer_contract import HomographyContract


def _sha(token: str) -> str:
    return sha256(token.encode("ascii")).hexdigest()


def _roi(kind: str = "measure-start") -> RuntimeRoiArtifact:
    data = b"exact-runtime-meter-roi-bytes"
    return RuntimeRoiArtifact(
        roi_id=f"staff-1-measure-1:{kind}",
        kind=kind,
        measure_id="staff-1-measure-1",
        staff_id="staff-1",
        source_image_sha256=_sha("source"),
        roi_image_sha256=sha256(data).hexdigest(),
        crop_bbox=BoxContract(10.0, 20.0, 70.0, 100.0),
        source_to_roi=HomographyContract(
            forward=(1.0, 0.0, -10.0, 0.0, 1.0, -20.0, 0.0, 0.0, 1.0),
            inverse=(1.0, 0.0, 10.0, 0.0, 1.0, 20.0, 0.0, 0.0, 1.0),
        ),
        png_bytes=data,
    )


def _profile() -> MeterSpecialistProfileV1:
    return MeterSpecialistProfileV1(
        presence_checkpoint_sha256=_sha("presence"),
        digit2_checkpoint_sha256=_sha("digit2"),
        digit3_checkpoint_sha256=_sha("digit3"),
        digit4_checkpoint_sha256=_sha("digit4"),
    )


def _accepted_raw() -> MeterRawInferenceV1:
    return MeterRawInferenceV1(
        presence_status="accepted",
        presence_score=0.99,
        refined_x_center_roi=20.0,
        numerator_scores=MeterDigitScoresV3(100, 200, 990),
        denominator_scores=MeterDigitScoresV3(100, 200, 995),
    )


class RuntimeMeterSpecialistProducerV1Tests(unittest.TestCase):
    def test_exact_roi_bytes_are_the_only_runner_pixel_input(self) -> None:
        roi = _roi()
        seen: list[bytes] = []

        def runner(data: bytes) -> MeterRawInferenceV1:
            seen.append(data)
            return _accepted_raw()

        result = produce_meter_evidence_v1(
            roi,
            system_id="system-1",
            logical_measure_id="system-1-logical-measure-1",
            profile=_profile(),
            runner=runner,
        )
        self.assertEqual(seen, [roi.png_bytes])
        self.assertEqual(result.roi_image_sha256, roi.roi_image_sha256)
        self.assertEqual(result.source_image_sha256, roi.source_image_sha256)
        self.assertEqual(result.evidence.measure_id, roi.measure_id)
        self.assertEqual(result.evidence.roi_id, roi.roi_id)
        self.assertEqual(result.evidence.presence_score, 0.99)

    def test_same_request_and_output_is_deterministic_10_of_10(self) -> None:
        roi = _roi()
        outputs = [
            produce_meter_evidence_v1(
                roi,
                system_id="system-1",
                logical_measure_id="system-1-logical-measure-1",
                profile=_profile(),
                runner=lambda _: _accepted_raw(),
            )
            for _ in range(10)
        ]
        identities = {
            (
                item.evidence.evidence_id,
                item.specialist_profile_fingerprint,
                item.inference_request_fingerprint,
                item.inference_output_fingerprint,
                item.fingerprint(),
            )
            for item in outputs
        }
        self.assertEqual(len(identities), 1)

    def test_non_measure_start_roi_fails_closed_before_runner(self) -> None:
        called = False

        def runner(_: bytes) -> MeterRawInferenceV1:
            nonlocal called
            called = True
            return _accepted_raw()

        with self.assertRaises(MeterSpecialistProducerError):
            produce_meter_evidence_v1(
                _roi("measure-full"),
                system_id="system-1",
                logical_measure_id="system-1-logical-measure-1",
                profile=_profile(),
                runner=runner,
            )
        self.assertFalse(called)

    def test_runner_must_return_exact_raw_contract(self) -> None:
        with self.assertRaises(MeterSpecialistProducerError):
            produce_meter_evidence_v1(
                _roi(),
                system_id="system-1",
                logical_measure_id="system-1-logical-measure-1",
                profile=_profile(),
                runner=lambda _: object(),
            )

    def test_profile_fingerprint_changes_with_checkpoint_identity(self) -> None:
        first = _profile()
        second = MeterSpecialistProfileV1(
            presence_checkpoint_sha256=_sha("presence-changed"),
            digit2_checkpoint_sha256=first.digit2_checkpoint_sha256,
            digit3_checkpoint_sha256=first.digit3_checkpoint_sha256,
            digit4_checkpoint_sha256=first.digit4_checkpoint_sha256,
        )
        self.assertNotEqual(first.fingerprint(), second.fingerprint())

    def test_nonaccepted_raw_evidence_stays_explicit(self) -> None:
        raw = MeterRawInferenceV1(
            presence_status="ambiguous",
            presence_score=0.51,
            reasons=("runner-ambiguous",),
        )
        result = produce_meter_evidence_v1(
            _roi(),
            system_id="system-1",
            logical_measure_id="system-1-logical-measure-1",
            profile=_profile(),
            runner=lambda _: raw,
        )
        self.assertEqual(result.evidence.presence_status, "ambiguous")
        self.assertEqual(result.evidence.reasons, ("runner-ambiguous",))

    def test_stage_cannot_load_checkpoints_or_access_splits_or_resolver(self) -> None:
        self.assertFalse(checkpoint_loading_allowed())
        self.assertFalse(train_validation_test_access_allowed())
        self.assertFalse(resolver_connection_allowed())


if __name__ == "__main__":
    unittest.main()
