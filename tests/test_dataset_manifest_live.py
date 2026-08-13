import unittest
from dataclasses import replace

from st_omr_training.dataset_manifest import (
    DatasetManifest,
    DatasetManifestInputError,
    DatasetSplit,
    canonical_manifest_bytes,
    dataset_manifest_sha256,
    sample_from_degraded_page,
    validate_dataset_manifest,
)
from st_omr_training.degradation import degrade_render_result_page, sample_degradation_config
from st_omr_training.generator import GeneratorConfig, generate_score
from st_omr_training.musicxml_writer import write_musicxml
from st_omr_training.renderer import render_musicxml_svg


class RealDatasetManifestPipelineTests(unittest.TestCase):
    def _real_sample(self, seed: int, split: DatasetSplit):
        score = generate_score(GeneratorConfig(measure_count=8), seed)
        musicxml = write_musicxml(score)
        rendered = render_musicxml_svg(musicxml)
        degraded = degrade_render_result_page(
            rendered,
            family_id=f"family-{seed}",
            page_number=1,
            config=sample_degradation_config(seed, "light", raster_width=1000),
        )
        return sample_from_degraded_page(degraded, split=split), degraded

    def test_three_real_pipeline_families_form_valid_manifest(self):
        train, _ = self._real_sample(5101, DatasetSplit.TRAIN)
        validation, _ = self._real_sample(5102, DatasetSplit.VALIDATION)
        test, _ = self._real_sample(5103, DatasetSplit.TEST)
        manifest = DatasetManifest(
            dataset_name="st-stage5a-live",
            dataset_version="v1",
            samples=(train, validation, test),
        )
        result = validate_dataset_manifest(manifest)
        self.assertTrue(result.is_valid, result.issues)
        self.assertEqual(len(dataset_manifest_sha256(manifest)), 64)
        self.assertTrue(canonical_manifest_bytes(manifest).startswith(b"{"))

    def test_real_manifest_serialization_is_order_deterministic(self):
        train, _ = self._real_sample(5201, DatasetSplit.TRAIN)
        validation, _ = self._real_sample(5202, DatasetSplit.VALIDATION)
        test, _ = self._real_sample(5203, DatasetSplit.TEST)
        first = DatasetManifest("st-stage5a-order", "v1", (train, validation, test))
        second = DatasetManifest("st-stage5a-order", "v1", (test, train, validation))
        self.assertEqual(canonical_manifest_bytes(first), canonical_manifest_bytes(second))
        self.assertEqual(dataset_manifest_sha256(first), dataset_manifest_sha256(second))

    def test_bridge_rejects_tampered_real_png_hash(self):
        _, degraded = self._real_sample(5301, DatasetSplit.TRAIN)
        tampered = replace(degraded, png_sha256="0" * 64)
        with self.assertRaises(DatasetManifestInputError):
            sample_from_degraded_page(tampered, split=DatasetSplit.TRAIN)


if __name__ == "__main__":
    unittest.main()
