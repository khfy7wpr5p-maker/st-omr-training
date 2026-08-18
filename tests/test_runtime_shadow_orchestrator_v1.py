from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import inspect
import unittest

from PIL import Image, ImageDraw

from st_omr_training.runtime_geometry_engine_contract import BoxContract
from st_omr_training.runtime_page_normalizer_contract import RasterPageInputContract
from st_omr_training.runtime_shadow_orchestrator_v1 import (
    prepare_runtime_shadow_v1,
    resolve_runtime_shadow_v1,
)
from st_omr_training.runtime_specialist_evidence_v1 import (
    SpecialistEvidenceBatch,
    SpecialistObservation,
)


SOURCE_SHA = sha256(b"shadow-orchestrator-source").hexdigest()


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


def _raster_contract(raw: bytes) -> RasterPageInputContract:
    return RasterPageInputContract(
        source_id="shadow-orchestrator-fixture",
        source_sha256=SOURCE_SHA,
        page_number=1,
        width=320,
        height=220,
        pixel_mode="rgb8",
        raster_sha256=sha256(raw).hexdigest(),
        dpi=300,
    )


class RuntimeShadowOrchestratorV1Tests(unittest.TestCase):
    def test_raster_to_roi_preparation_reaches_expected_safe_stop(self) -> None:
        raw = _raw_page()
        prepared = prepare_runtime_shadow_v1(raw, _raster_contract(raw))
        self.assertEqual(prepared.normalized_page.status, "accepted")
        self.assertEqual(prepared.staff_geometry.status, "accepted")
        self.assertEqual(len(prepared.staff_geometry.staffs), 2)
        self.assertEqual(prepared.measure_geometry.status, "accepted")
        self.assertEqual(len(prepared.measure_geometry.measure_proposals), 4)
        self.assertEqual(len(prepared.roi_batch.artifacts), 8)

    def test_explicit_shadow_evidence_reaches_deterministic_resolver(self) -> None:
        raw = _raw_page()
        prepared = prepare_runtime_shadow_v1(raw, _raster_contract(raw))
        evidence = SpecialistEvidenceBatch(
            observations=(
                SpecialistObservation(
                    observation_id="note-1",
                    task="notehead",
                    measure_id="staff-1-measure-1",
                    staff_id="staff-1",
                    status="accepted",
                    confidence_milli=970,
                    class_label="filled",
                    bbox=BoxContract(60, 45, 70, 57),
                    source_kind="shadow-fixture",
                ),
                SpecialistObservation(
                    observation_id="acc-1",
                    task="accidental",
                    measure_id="staff-1-measure-1",
                    staff_id="staff-1",
                    status="accepted",
                    confidence_milli=950,
                    class_label="sharp",
                    bbox=BoxContract(48, 44, 54, 58),
                    source_kind="shadow-fixture",
                ),
            )
        )
        results = [resolve_runtime_shadow_v1(prepared, evidence) for _ in range(10)]
        for result in results:
            self.assertEqual(result.status, "accepted")
            first = next(item for item in result.measures if item.measure_id == "staff-1-measure-1")
            self.assertEqual(first.notes[0].accidental_class, "sharp")
        self.assertEqual(len(set(result.config_fingerprint for result in results)), 1)
        self.assertEqual(len(set(str(result.measures) for result in results)), 1)

    def test_orchestrator_has_no_d10_d13_model_or_test_path(self) -> None:
        import st_omr_training.runtime_shadow_orchestrator_v1 as module

        source = inspect.getsource(module)
        for token in (
            "from .stage7d10_",
            "from .stage7d13_",
            "torch.load(",
            "torch.optim",
            "DataLoader(",
            "TEST",
        ):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
