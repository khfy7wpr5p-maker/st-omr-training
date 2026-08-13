import unittest
from dataclasses import replace
from hashlib import sha256

from st_omr_training.dataset_manifest import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    DATASET_SOURCE_CLASS,
    DATASET_SPLIT_POLICY,
    DatasetDegradationConfig,
    DatasetManifest,
    DatasetManifestInputError,
    DatasetSample,
    DatasetSplit,
    canonical_manifest_bytes,
    dataset_degradation_config_fingerprint,
    dataset_derivative_id,
    dataset_manifest_sha256,
    dataset_sample_id,
    sample_from_degraded_page,
    validate_dataset_manifest,
    validate_dataset_sample,
)


def _hex(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _sample(
    label: str,
    split: DatasetSplit,
    *,
    family_id: str | None = None,
    target_hash: str | None = None,
    svg_hash: str | None = None,
) -> DatasetSample:
    family = family_id or f"family-{label}"
    config = DatasetDegradationConfig(
        seed=int.from_bytes(label.encode("ascii"), "big") % 100000,
        raster_width=1400,
        rotation_mdeg=100,
        blur_milli=200,
        noise_level=3,
        brightness_milli=990,
        contrast_milli=1010,
        jpeg_quality=90,
    )
    musicxml = target_hash or _hex(f"musicxml-{label}")
    renderer = _hex(f"renderer-{label}")
    svg = svg_hash or _hex(f"svg-{label}")
    clean = _hex(f"clean-{label}")
    config_hash = dataset_degradation_config_fingerprint(config)
    png = _hex(f"png-{label}")
    derivative = dataset_derivative_id(
        family_id=family,
        page_number=1,
        source_musicxml_sha256=musicxml,
        renderer_config_fingerprint=renderer,
        source_svg_sha256=svg,
        clean_raster_sha256=clean,
        degradation_config_fingerprint=config_hash,
        png_sha256=png,
    )
    sample_id = dataset_sample_id(
        family_id=family,
        page_number=1,
        source_musicxml_sha256=musicxml,
        renderer_config_fingerprint=renderer,
        source_svg_sha256=svg,
        clean_raster_sha256=clean,
        degradation_config_fingerprint=config_hash,
        derivative_id=derivative,
        png_sha256=png,
    )
    return DatasetSample(
        sample_id=sample_id,
        family_id=family,
        split=split,
        page_number=1,
        source_musicxml_sha256=musicxml,
        renderer_config_fingerprint=renderer,
        source_svg_sha256=svg,
        clean_raster_sha256=clean,
        degradation_config_fingerprint=config_hash,
        degradation_config=config,
        derivative_id=derivative,
        png_sha256=png,
        degradation_version="st-controlled-degradation-v1",
        cairosvg_version="2.8.2",
        pillow_version="12.3.0",
        cairo_runtime_version="1.18.0",
        python_version="3.13.14",
        platform_system="Linux",
        platform_machine="x86_64",
        clean_width=1400,
        clean_height=2000,
        width=1410,
        height=2010,
        mode="L",
        image_format="png",
    )


def _manifest(*samples: DatasetSample) -> DatasetManifest:
    return DatasetManifest(dataset_name="st-synthetic", dataset_version="v1", samples=tuple(samples))


class DatasetConfigTests(unittest.TestCase):
    def test_replay_config_is_immutable_and_bounded(self):
        config = DatasetDegradationConfig(1, 1400, 0, 0, 0, 1000, 1000, 0)
        with self.assertRaises(Exception):
            config.seed = 2
        with self.assertRaises(DatasetManifestInputError):
            DatasetDegradationConfig(1, 511, 0, 0, 0, 1000, 1000, 0)
        with self.assertRaises(DatasetManifestInputError):
            DatasetDegradationConfig(1, 1400, 3001, 0, 0, 1000, 1000, 0)

    def test_bool_is_not_accepted_as_integer(self):
        with self.assertRaises(DatasetManifestInputError):
            DatasetDegradationConfig(True, 1400, 0, 0, 0, 1000, 1000, 0)

    def test_config_fingerprint_is_deterministic_and_sensitive(self):
        first = DatasetDegradationConfig(1, 1400, 0, 0, 0, 1000, 1000, 0)
        same = DatasetDegradationConfig(1, 1400, 0, 0, 0, 1000, 1000, 0)
        changed = DatasetDegradationConfig(2, 1400, 0, 0, 0, 1000, 1000, 0)
        self.assertEqual(dataset_degradation_config_fingerprint(first), dataset_degradation_config_fingerprint(same))
        self.assertNotEqual(dataset_degradation_config_fingerprint(first), dataset_degradation_config_fingerprint(changed))


class DatasetSampleTests(unittest.TestCase):
    def test_valid_sample_passes_independent_validator(self):
        result = validate_dataset_sample(_sample("a", DatasetSplit.TRAIN))
        self.assertTrue(result.is_valid, result.issues)

    def test_raw_split_string_is_rejected(self):
        valid = _sample("a", DatasetSplit.TRAIN)
        with self.assertRaises(DatasetManifestInputError):
            replace(valid, split="train")

    def test_sample_identity_is_independent_of_split_assignment(self):
        sample = _sample("a", DatasetSplit.TRAIN)
        moved = replace(sample, split=DatasetSplit.VALIDATION)
        self.assertEqual(sample.sample_id, moved.sample_id)
        self.assertTrue(validate_dataset_sample(moved).is_valid)

    def test_config_fingerprint_tamper_is_detected(self):
        sample = _sample("a", DatasetSplit.TRAIN)
        object.__setattr__(sample, "degradation_config_fingerprint", _hex("tampered-config"))
        codes = {issue.code for issue in validate_dataset_sample(sample).issues}
        self.assertIn("lineage.config_fingerprint", codes)

    def test_derivative_identity_tamper_is_detected(self):
        sample = _sample("a", DatasetSplit.TRAIN)
        object.__setattr__(sample, "derivative_id", _hex("tampered-derivative"))
        codes = {issue.code for issue in validate_dataset_sample(sample).issues}
        self.assertIn("lineage.derivative_id", codes)
        self.assertIn("lineage.sample_id", codes)

    def test_sample_identity_tamper_is_detected(self):
        sample = _sample("a", DatasetSplit.TRAIN)
        object.__setattr__(sample, "sample_id", _hex("tampered-sample"))
        codes = {issue.code for issue in validate_dataset_sample(sample).issues}
        self.assertIn("lineage.sample_id", codes)

    def test_unsupported_stage4_runtime_is_rejected(self):
        sample = _sample("a", DatasetSplit.TRAIN)
        object.__setattr__(sample, "cairosvg_version", "9.9.9")
        object.__setattr__(sample, "pillow_version", "9.9.9")
        codes = {issue.code for issue in validate_dataset_sample(sample).issues}
        self.assertIn("sample.cairosvg_version", codes)
        self.assertIn("sample.pillow_version", codes)

    def test_invalid_image_metadata_is_rejected(self):
        sample = _sample("a", DatasetSplit.TRAIN)
        object.__setattr__(sample, "mode", "RGB")
        object.__setattr__(sample, "image_format", "jpeg")
        codes = {issue.code for issue in validate_dataset_sample(sample).issues}
        self.assertIn("sample.mode", codes)
        self.assertIn("sample.image_format", codes)

    def test_non_sample_object_fails_closed(self):
        result = validate_dataset_sample({"sample": "not trusted"})
        self.assertFalse(result.is_valid)
        self.assertEqual(result.issues[0].code, "sample.type")


class DatasetManifestTests(unittest.TestCase):
    def setUp(self):
        self.train = _sample("train", DatasetSplit.TRAIN)
        self.validation = _sample("validation", DatasetSplit.VALIDATION)
        self.test = _sample("test", DatasetSplit.TEST)

    def test_valid_three_split_manifest_passes(self):
        result = validate_dataset_manifest(_manifest(self.train, self.validation, self.test))
        self.assertTrue(result.is_valid, result.issues)

    def test_manifest_contract_fields_are_frozen(self):
        manifest = _manifest(self.train, self.validation, self.test)
        self.assertEqual(manifest.schema_version, DATASET_MANIFEST_SCHEMA_VERSION)
        self.assertEqual(manifest.source_class, DATASET_SOURCE_CLASS)
        self.assertEqual(manifest.split_policy, DATASET_SPLIT_POLICY)
        with self.assertRaises(Exception):
            manifest.dataset_version = "v2"

    def test_mutable_sample_container_is_rejected(self):
        with self.assertRaises(DatasetManifestInputError):
            DatasetManifest(dataset_name="st", dataset_version="v1", samples=[self.train, self.validation, self.test])

    def test_missing_split_is_rejected(self):
        result = validate_dataset_manifest(_manifest(self.train, self.validation))
        codes = {issue.code for issue in result.issues}
        self.assertIn("manifest.missing_split", codes)

    def test_family_split_leakage_is_rejected(self):
        first = _sample("a", DatasetSplit.TRAIN, family_id="family-shared", target_hash=_hex("target-shared"))
        second = _sample("b", DatasetSplit.TEST, family_id="family-shared", target_hash=_hex("target-shared"))
        result = validate_dataset_manifest(_manifest(first, self.validation, second))
        codes = {issue.code for issue in result.issues}
        self.assertIn("leakage.family_split", codes)
        self.assertIn("leakage.target_split", codes)

    def test_target_hash_alias_across_families_is_rejected(self):
        target = _hex("same-target")
        first = _sample("a", DatasetSplit.TRAIN, family_id="family-a", target_hash=target)
        second = _sample("b", DatasetSplit.TEST, family_id="family-b", target_hash=target)
        result = validate_dataset_manifest(_manifest(first, self.validation, second))
        codes = {issue.code for issue in result.issues}
        self.assertIn("leakage.target_family", codes)
        self.assertIn("leakage.target_split", codes)

    def test_svg_hash_alias_across_families_is_rejected(self):
        svg = _hex("same-svg")
        first = _sample("a", DatasetSplit.TRAIN, family_id="family-a", svg_hash=svg)
        second = _sample("b", DatasetSplit.TEST, family_id="family-b", svg_hash=svg)
        result = validate_dataset_manifest(_manifest(first, self.validation, second))
        codes = {issue.code for issue in result.issues}
        self.assertIn("leakage.svg_family", codes)
        self.assertIn("leakage.svg_split", codes)

    def test_duplicate_sample_derivative_and_png_are_rejected(self):
        duplicate = replace(self.train)
        result = validate_dataset_manifest(_manifest(self.train, self.validation, self.test, duplicate))
        codes = {issue.code for issue in result.issues}
        self.assertIn("duplicate.sample_id", codes)
        self.assertIn("duplicate.derivative_id", codes)
        self.assertIn("duplicate.png_sha256", codes)

    def test_corrupted_source_class_is_rejected_independently(self):
        manifest = _manifest(self.train, self.validation, self.test)
        object.__setattr__(manifest, "source_class", "real")
        codes = {issue.code for issue in validate_dataset_manifest(manifest).issues}
        self.assertIn("manifest.source_class", codes)

    def test_canonical_bytes_ignore_tuple_order_only(self):
        first = _manifest(self.train, self.validation, self.test)
        second = _manifest(self.test, self.train, self.validation)
        self.assertEqual(canonical_manifest_bytes(first), canonical_manifest_bytes(second))
        self.assertEqual(dataset_manifest_sha256(first), dataset_manifest_sha256(second))

    def test_split_assignment_changes_manifest_hash(self):
        first = _manifest(self.train, self.validation, self.test)
        moved_train = replace(self.train, split=DatasetSplit.VALIDATION)
        moved_validation = replace(self.validation, split=DatasetSplit.TRAIN)
        second = _manifest(moved_train, moved_validation, self.test)
        self.assertTrue(validate_dataset_manifest(second).is_valid)
        self.assertNotEqual(dataset_manifest_sha256(first), dataset_manifest_sha256(second))

    def test_invalid_manifest_cannot_be_canonically_serialized(self):
        invalid = _manifest(self.train, self.validation)
        with self.assertRaises(DatasetManifestInputError):
            canonical_manifest_bytes(invalid)

    def test_issue_order_is_deterministic(self):
        manifest = _manifest(self.train, self.validation)
        first = validate_dataset_manifest(manifest).issues
        second = validate_dataset_manifest(manifest).issues
        self.assertEqual(first, second)

    def test_non_manifest_object_fails_closed(self):
        result = validate_dataset_manifest({"samples": []})
        self.assertFalse(result.is_valid)
        self.assertEqual(result.issues[0].code, "manifest.type")


class DatasetBridgeBoundaryTests(unittest.TestCase):
    def test_missing_stage4_fields_fail_closed(self):
        class Incomplete:
            pass

        with self.assertRaises(DatasetManifestInputError):
            sample_from_degraded_page(Incomplete(), split=DatasetSplit.TRAIN)

    def test_bridge_requires_enum_split(self):
        with self.assertRaises(DatasetManifestInputError):
            sample_from_degraded_page(object(), split="train")


if __name__ == "__main__":
    unittest.main()
