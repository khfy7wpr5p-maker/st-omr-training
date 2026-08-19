from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import inspect
import unittest

from PIL import Image, ImageDraw

from st_omr_training.runtime_meter_integration_v3 import (
    MeterDigitScoresV3,
    MeterModelEvidenceV3,
)
from st_omr_training.runtime_page_normalizer_contract import RasterPageInputContract
from st_omr_training.runtime_resolver_e2e_v2 import (
    BoundMeterEvidenceBatchV2,
    BoundMeterEvidenceRecordV2,
    RuntimeResolverE2EV2Error,
    old_measure_geometry_fallback_allowed,
    prepare_runtime_resolver_e2e_v2,
    real_model_proof_allowed_from_test_fixture,
    resolve_runtime_resolver_e2e_v2,
    sealed_test_access_allowed,
)


SOURCE_SHA = sha256(b"resolver-e2e-v2-source").hexdigest()
PROVIDER_SHA = sha256(b"fixture-meter-provider-v1").hexdigest()


def _raw_page() -> bytes:
    image = Image.new("RGB", (320, 220), "white")
    draw = ImageDraw.Draw(image)
    for y in (30, 40, 50, 60, 70, 140, 150, 160, 170, 180):
        draw.line((20, y, 300, y), fill="black", width=1)
    for x in (20, 160, 300):
        draw.line((x, 30, x, 70), fill="black", width=1)
        draw.line((x, 140, x, 180), fill="black", width=1)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _contract(raw: bytes) -> RasterPageInputContract:
    return RasterPageInputContract(
        source_id="resolver-e2e-v2-fixture",
        source_sha256=SOURCE_SHA,
        page_number=1,
        width=320,
        height=220,
        pixel_mode="rgb8",
        raster_sha256=sha256(raw).hexdigest(),
        dpi=300,
    )


def _logical_by_measure(prepared) -> dict[str, str]:
    result: dict[str, str] = {}
    for logical in prepared.boundary_report.logical_measures:
        for measure_id in logical.member_measure_ids:
            result[measure_id] = logical.logical_measure_id
    return result


def _fixture_meter_batch(prepared, *, ambiguous_measure_id: str | None = None) -> BoundMeterEvidenceBatchV2:
    logical = _logical_by_measure(prepared)
    measure_by_id = {item.measure_id: item for item in prepared.measure_geometry.measure_proposals}
    records: list[BoundMeterEvidenceRecordV2] = []
    for roi in prepared.roi_batch.artifacts:
        if roi.kind != "measure-start":
            continue
        measure = measure_by_id[roi.measure_id]
        ambiguous = roi.measure_id == ambiguous_measure_id
        evidence = MeterModelEvidenceV3(
            evidence_id=f"fixture-meter:{roi.measure_id}",
            system_id=measure.system_id,
            logical_measure_id=logical[roi.measure_id],
            measure_id=roi.measure_id,
            staff_id=roi.staff_id,
            roi_id=roi.roi_id,
            presence_status="ambiguous" if ambiguous else "accepted",
            presence_score=0.5 if ambiguous else 0.1,
            refined_x_center_roi=None,
            numerator_scores=MeterDigitScoresV3(0, 0, 0),
            denominator_scores=MeterDigitScoresV3(0, 0, 0),
            reasons=("fixture-ambiguous-presence",) if ambiguous else (),
        )
        records.append(
            BoundMeterEvidenceRecordV2(
                source_image_sha256=roi.source_image_sha256,
                roi_id=roi.roi_id,
                roi_image_sha256=roi.roi_image_sha256,
                evidence=evidence,
            )
        )
    return BoundMeterEvidenceBatchV2(
        provider_fingerprint=PROVIDER_SHA,
        evidence_origin="test-fixture",
        source_image_sha256=prepared.measure_geometry.normalized_image_sha256,
        records=tuple(records),
    )


class RuntimeResolverE2EV2Tests(unittest.TestCase):
    def test_real_raster_bytes_use_new_system_measure_meter_resolver_lane(self) -> None:
        raw = _raw_page()
        prepared = prepare_runtime_resolver_e2e_v2(raw, _contract(raw), system_policy="monostaff-v1")
        self.assertEqual(len(prepared.detected_geometry.staffs), 2)
        self.assertEqual(len(prepared.grouped_geometry.systems), 2)
        self.assertEqual(tuple(len(item.staff_ids) for item in prepared.grouped_geometry.systems), (1, 1))
        self.assertEqual(len(prepared.measure_geometry.measure_proposals), 4)
        self.assertEqual(len(prepared.boundary_report.logical_measures), 4)
        self.assertEqual(len(prepared.roi_batch.artifacts), 8)

        result = resolve_runtime_resolver_e2e_v2(prepared, _fixture_meter_batch(prepared))
        self.assertEqual(result.resolver_result.status, "accepted")
        self.assertEqual(len(result.resolver_result.measures), 4)
        self.assertTrue(all(item.meter_class == "none" for item in result.resolver_result.measures))
        self.assertEqual(result.evidence_origin, "test-fixture")
        self.assertFalse(result.is_real_model_proof)

    def test_meter_ambiguity_propagates_to_resolver_instead_of_disappearing(self) -> None:
        raw = _raw_page()
        prepared = prepare_runtime_resolver_e2e_v2(raw, _contract(raw), system_policy="monostaff-v1")
        target = prepared.measure_geometry.measure_proposals[0].measure_id
        result = resolve_runtime_resolver_e2e_v2(
            prepared,
            _fixture_meter_batch(prepared, ambiguous_measure_id=target),
        )
        self.assertEqual(result.resolver_result.status, "ambiguous")
        resolved = next(item for item in result.resolver_result.measures if item.measure_id == target)
        self.assertEqual(resolved.status, "ambiguous")
        self.assertTrue(any(reason.endswith("_METER") for reason in resolved.reasons))

    def test_wrong_roi_byte_binding_fails_before_meter_composition(self) -> None:
        raw = _raw_page()
        prepared = prepare_runtime_resolver_e2e_v2(raw, _contract(raw), system_policy="monostaff-v1")
        batch = _fixture_meter_batch(prepared)
        first = batch.records[0]
        forged = BoundMeterEvidenceRecordV2(
            source_image_sha256=first.source_image_sha256,
            roi_id=first.roi_id,
            roi_image_sha256="f" * 64,
            evidence=first.evidence,
        )
        bad = BoundMeterEvidenceBatchV2(
            provider_fingerprint=batch.provider_fingerprint,
            evidence_origin=batch.evidence_origin,
            source_image_sha256=batch.source_image_sha256,
            records=(forged,) + batch.records[1:],
        )
        with self.assertRaises(RuntimeResolverE2EV2Error) as caught:
            resolve_runtime_resolver_e2e_v2(prepared, bad)
        self.assertEqual(caught.exception.stage, "meter-evidence-binding")
        self.assertIn("meter-record-roi-byte-or-owner-mismatch", caught.exception.reasons)

    def test_auto_policy_stops_on_under_determined_two_staff_page(self) -> None:
        raw = _raw_page()
        with self.assertRaises(RuntimeResolverE2EV2Error) as caught:
            prepare_runtime_resolver_e2e_v2(raw, _contract(raw), system_policy="auto-v1")
        self.assertEqual(caught.exception.stage, "system-grouper-v1")

    def test_10_of_10_full_lane_fingerprint_is_stable(self) -> None:
        raw = _raw_page()
        fingerprints: list[str] = []
        for _ in range(10):
            prepared = prepare_runtime_resolver_e2e_v2(raw, _contract(raw), system_policy="monostaff-v1")
            result = resolve_runtime_resolver_e2e_v2(prepared, _fixture_meter_batch(prepared))
            fingerprints.append(result.fingerprint)
        self.assertEqual(len(set(fingerprints)), 1)

    def test_module_has_no_old_measure_model_checkpoint_or_test_split_lane(self) -> None:
        import st_omr_training.runtime_resolver_e2e_v2 as module

        source = inspect.getsource(module)
        for token in (
            "runtime_measure_geometry_v1",
            "stage7d10_",
            "stage7d13_",
            "torch.load(",
            "torch.optim",
            "DataLoader(",
        ):
            self.assertNotIn(token, source)
        self.assertFalse(old_measure_geometry_fallback_allowed())
        self.assertFalse(sealed_test_access_allowed())
        self.assertFalse(real_model_proof_allowed_from_test_fixture())


if __name__ == "__main__":
    unittest.main()
