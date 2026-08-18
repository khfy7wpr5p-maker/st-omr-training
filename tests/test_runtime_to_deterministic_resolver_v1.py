from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import inspect
import unittest

from PIL import Image, ImageDraw

from st_omr_training.runtime_deterministic_resolver_v1 import (
    R01_METER_CONFLICT,
    R04_ACCIDENTAL_UNASSOCIATED,
    resolve_specialist_evidence_v1,
)
from st_omr_training.runtime_geometry_engine_contract import BoxContract, GeometryInputContract
from st_omr_training.runtime_geometry_engine_v2 import detect_multistaff_geometry_v2
from st_omr_training.runtime_local_roi_v1 import extract_runtime_rois_v1
from st_omr_training.runtime_measure_geometry_v1 import propose_measure_geometry_v1
from st_omr_training.runtime_page_normalizer_contract import RasterPageInputContract
from st_omr_training.runtime_page_normalizer_v1 import normalize_raster_page_v1
from st_omr_training.runtime_specialist_evidence_v1 import (
    SpecialistEvidenceBatch,
    SpecialistObservation,
)


SOURCE_SHA = sha256(b"runtime-resolver-fixture-source").hexdigest()


def _raw_page() -> bytes:
    image = Image.new("RGB", (320, 220), "white")
    draw = ImageDraw.Draw(image)
    for y in (30, 40, 50, 60, 70, 140, 150, 160, 170, 180):
        draw.line((20, y, 300, y), fill="black", width=1)
    for x in (20, 160, 300):
        draw.line((x, 30, x, 70), fill="black", width=1)
        draw.line((x, 140, x, 180), fill="black", width=1)
    out = BytesIO()
    image.save(out, format="PNG", optimize=False, compress_level=9)
    return out.getvalue()


def _runtime_geometry():
    raw = _raw_page()
    raster = RasterPageInputContract(
        source_id="resolver-fixture",
        source_sha256=SOURCE_SHA,
        page_number=1,
        width=320,
        height=220,
        pixel_mode="rgb8",
        raster_sha256=sha256(raw).hexdigest(),
        dpi=300,
    )
    normalized = normalize_raster_page_v1(raw, raster)
    assert normalized.normalized_png is not None
    assert normalized.page.normalized_image_sha256 is not None
    assert normalized.page.normalized_width is not None
    assert normalized.page.normalized_height is not None
    assert normalized.page.transform is not None
    geometry_input = GeometryInputContract(
        normalized_image_sha256=normalized.page.normalized_image_sha256,
        normalizer_config_fingerprint=normalized.page.normalizer_config_fingerprint,
        normalized_width=normalized.page.normalized_width,
        normalized_height=normalized.page.normalized_height,
        transform=normalized.page.transform,
    )
    staff_geometry = detect_multistaff_geometry_v2(normalized.normalized_png, geometry_input)
    if staff_geometry.page.status != "accepted":
        raise AssertionError(staff_geometry.page.reasons)
    measure_geometry = propose_measure_geometry_v1(normalized.normalized_png, staff_geometry.page)
    if measure_geometry.page.status != "accepted":
        raise AssertionError(measure_geometry.page.reasons)
    return normalized.normalized_png, measure_geometry.page


def _accepted_evidence() -> SpecialistEvidenceBatch:
    measure_id = "staff-1-measure-1"
    staff_id = "staff-1"
    return SpecialistEvidenceBatch(
        observations=(
            SpecialistObservation(
                observation_id="meter-1",
                task="meter",
                measure_id=measure_id,
                staff_id=staff_id,
                status="accepted",
                confidence_milli=950,
                class_label="4/4",
                bbox=BoxContract(25, 35, 36, 65),
                source_kind="shadow-fixture",
            ),
            SpecialistObservation(
                observation_id="accidental-1",
                task="accidental",
                measure_id=measure_id,
                staff_id=staff_id,
                status="accepted",
                confidence_milli=930,
                class_label="sharp",
                bbox=BoxContract(45, 44, 51, 58),
                source_kind="shadow-fixture",
            ),
            SpecialistObservation(
                observation_id="notehead-1",
                task="notehead",
                measure_id=measure_id,
                staff_id=staff_id,
                status="accepted",
                confidence_milli=970,
                class_label="filled",
                bbox=BoxContract(58, 45, 70, 57),
                source_kind="shadow-fixture",
            ),
            SpecialistObservation(
                observation_id="rest-1",
                task="rest",
                measure_id=measure_id,
                staff_id=staff_id,
                status="accepted",
                confidence_milli=920,
                class_label="quarter",
                bbox=BoxContract(100, 44, 111, 59),
                source_kind="shadow-fixture",
            ),
        )
    )


class RuntimeRoiAndResolverIntegrationTests(unittest.TestCase):
    def test_page_to_staff_to_measure_to_roi_is_deterministic_10_of_10(self) -> None:
        normalized_png, geometry = _runtime_geometry()
        batches = [extract_runtime_rois_v1(normalized_png, geometry) for _ in range(10)]
        for batch in batches:
            self.assertEqual(len(batch.artifacts), 8)
            self.assertEqual(
                tuple(item.kind for item in batch.artifacts),
                (
                    "measure-full", "measure-start",
                    "measure-full", "measure-start",
                    "measure-full", "measure-start",
                    "measure-full", "measure-start",
                ),
            )
        identities = [
            tuple((item.roi_id, item.roi_image_sha256, item.crop_bbox) for item in batch.artifacts)
            for batch in batches
        ]
        self.assertEqual(len(set(identities)), 1)

    def test_shadow_specialist_evidence_resolves_deterministically(self) -> None:
        _, geometry = _runtime_geometry()
        results = [resolve_specialist_evidence_v1(geometry, _accepted_evidence()) for _ in range(10)]
        for result in results:
            self.assertEqual(result.status, "accepted")
            first = next(item for item in result.measures if item.measure_id == "staff-1-measure-1")
            self.assertEqual(first.meter_class, "4/4")
            self.assertEqual(len(first.notes), 1)
            self.assertEqual(first.notes[0].notehead_class, "filled")
            self.assertEqual(first.notes[0].accidental_class, "sharp")
            self.assertEqual(first.notes[0].accidental_observation_id, "accidental-1")
            self.assertEqual(len(first.rests), 1)
            self.assertEqual(first.rests[0].rest_class, "quarter")
        identities = [
            (
                result.status,
                result.config_fingerprint,
                tuple((m.measure_id, m.status, m.meter_class, m.notes, m.rests, m.reasons) for m in result.measures),
            )
            for result in results
        ]
        self.assertEqual(len(set(identities)), 1)

    def test_conflicting_meter_evidence_fails_closed_as_ambiguous(self) -> None:
        _, geometry = _runtime_geometry()
        base = _accepted_evidence().observations
        conflict = SpecialistObservation(
            observation_id="meter-2",
            task="meter",
            measure_id="staff-1-measure-1",
            staff_id="staff-1",
            status="accepted",
            confidence_milli=940,
            class_label="3/4",
            bbox=BoxContract(26, 35, 37, 65),
            source_kind="shadow-fixture",
        )
        result = resolve_specialist_evidence_v1(
            geometry,
            SpecialistEvidenceBatch(observations=(*base, conflict)),
        )
        self.assertEqual(result.status, "ambiguous")
        first = next(item for item in result.measures if item.measure_id == "staff-1-measure-1")
        self.assertEqual(first.meter_class, None)
        self.assertIn(R01_METER_CONFLICT, first.reasons)

    def test_unassociated_accidental_is_not_invented_onto_a_note(self) -> None:
        _, geometry = _runtime_geometry()
        evidence = SpecialistEvidenceBatch(
            observations=(
                SpecialistObservation(
                    observation_id="accidental-alone",
                    task="accidental",
                    measure_id="staff-1-measure-1",
                    staff_id="staff-1",
                    status="accepted",
                    confidence_milli=900,
                    class_label="flat",
                    bbox=BoxContract(100, 44, 106, 58),
                    source_kind="shadow-fixture",
                ),
            )
        )
        result = resolve_specialist_evidence_v1(geometry, evidence)
        first = next(item for item in result.measures if item.measure_id == "staff-1-measure-1")
        self.assertEqual(first.status, "ambiguous")
        self.assertIn(R04_ACCIDENTAL_UNASSOCIATED, first.reasons)
        self.assertEqual(first.notes, ())

    def test_runtime_path_stays_isolated_from_d10_d13_training_and_checkpoints(self) -> None:
        import st_omr_training.runtime_deterministic_resolver_v1 as resolver
        import st_omr_training.runtime_local_roi_v1 as roi
        import st_omr_training.runtime_measure_geometry_v1 as measures
        import st_omr_training.runtime_specialist_evidence_v1 as evidence

        source = "\n".join(inspect.getsource(module) for module in (resolver, roi, measures, evidence))
        forbidden = (
            "from .stage7d10_",
            "from .stage7d13_",
            "import stage7d10_",
            "import stage7d13_",
            "torch.optim",
            "torch.load(",
            ".backward(",
            "DataLoader(",
        )
        for token in forbidden:
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
