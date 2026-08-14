from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from st_omr_training.synthetic_curriculum_corpus_gate import (
    SyntheticCurriculumCorpusAcceptanceError,
    SyntheticCurriculumCorpusReceipt,
    _CorpusExpectations,
    _canonical_json,
    _verify_corpus_directory,
    _verify_manifest,
    _verify_transport_archive,
    canonical_stage7d_corpus_evidence,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
LAYOUT = {
    "manifest": "manifest.json",
    "metadata": "build.json",
    "images": "images/<png_sha256>.png",
    "targets": "targets/<source_musicxml_sha256>.musicxml",
}


def _hex(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _sample(*, family: str, split: str, target_hash: str, index: int, png_hash: str) -> dict[str, object]:
    return {
        "sample_id": _hex(f"sample:{family}:{index}"),
        "family_id": family,
        "split": split,
        "page_number": 1,
        "source_musicxml_sha256": target_hash,
        "renderer_config_fingerprint": _hex(f"renderer:{family}"),
        "source_svg_sha256": _hex(f"svg:{family}"),
        "clean_raster_sha256": _hex(f"clean:{family}"),
        "degradation_config_fingerprint": _hex(f"degradation:{family}:{index}"),
        "degradation_config": {
            "seed": index,
            "raster_width": 1000,
            "rotation_mdeg": 0,
            "blur_milli": 0,
            "noise_level": 0,
            "brightness_milli": 1000,
            "contrast_milli": 1000,
            "jpeg_quality": 0,
        },
        "derivative_id": _hex(f"derivative:{family}:{index}"),
        "png_sha256": png_hash,
        "degradation_version": "st-controlled-degradation-v1",
        "cairosvg_version": "2.8.2",
        "pillow_version": "12.3.0",
        "cairo_runtime_version": "test",
        "python_version": "3.13",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "clean_width": 100,
        "clean_height": 100,
        "width": 100,
        "height": 100,
        "mode": "L",
        "image_format": "png",
    }


def _fixture(root: Path) -> tuple[_CorpusExpectations, dict[str, object]]:
    (root / "images").mkdir(parents=True)
    (root / "targets").mkdir()

    samples: list[dict[str, object]] = []
    splits = (("family-train", "train"), ("family-validation", "validation"), ("family-test", "test"))
    for family, split in splits:
        target_bytes = f"<score-partwise id='{family}'/>".encode("ascii")
        target_hash = sha256(target_bytes).hexdigest()
        (root / "targets" / f"{target_hash}.musicxml").write_bytes(target_bytes)
        for index in range(3):
            png_bytes = PNG_SIGNATURE + f"{family}:{index}".encode("ascii")
            png_hash = sha256(png_bytes).hexdigest()
            (root / "images" / f"{png_hash}.png").write_bytes(png_bytes)
            samples.append(
                _sample(
                    family=family,
                    split=split,
                    target_hash=target_hash,
                    index=index,
                    png_hash=png_hash,
                )
            )

    manifest = {
        "schema_version": "st-dataset-manifest-v1",
        "source_class": "synthetic",
        "split_policy": "family-exclusive-v1",
        "dataset_name": "mini-curriculum",
        "dataset_version": "v1",
        "samples": samples,
    }
    manifest_bytes = _canonical_json(manifest)
    manifest_hash = sha256(manifest_bytes).hexdigest()
    (root / "manifest.json").write_bytes(manifest_bytes)
    (root / "manifest.sha256").write_text(
        f"{manifest_hash}  manifest.json\n",
        encoding="ascii",
        newline="\n",
    )

    expectations = _CorpusExpectations(
        source_commit=_hex("source"),
        build_id=_hex("build"),
        config_fingerprint=_hex("config"),
        manifest_sha256=manifest_hash,
        transport_sha256=_hex("unused-transport"),
        archive_name="mini.tar.gz",
        archive_size_bytes=None,
        dataset_name="mini-curriculum",
        dataset_version="v1",
        builder_version="st-synthetic-dataset-builder-v1",
        sample_counts={"test": 3, "train": 3, "validation": 3},
        family_counts={"test": 1, "train": 1, "validation": 1},
        sample_count=9,
        target_count=3,
        image_count=9,
    )
    build = {
        "builder_version": expectations.builder_version,
        "build_id": expectations.build_id,
        "config_fingerprint": expectations.config_fingerprint,
        "manifest_sha256": manifest_hash,
        "sample_count": 9,
        "target_count": 3,
        "image_count": 9,
        "sample_split_counts": expectations.sample_counts,
        "family_split_counts": expectations.family_counts,
        "layout": LAYOUT,
    }
    (root / "build.json").write_bytes(_canonical_json(build))
    return expectations, manifest


class SyntheticCurriculumCorpusGateTests(unittest.TestCase):
    def test_exact_miniature_corpus_passes_byte_and_family_bindings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"
            root.mkdir()
            expectations, _manifest = _fixture(root)
            family_counts, sample_counts, target_bytes, image_bytes, binding = _verify_corpus_directory(
                root, expectations
            )
            self.assertEqual(family_counts, expectations.family_counts)
            self.assertEqual(sample_counts, expectations.sample_counts)
            self.assertGreater(target_bytes, 0)
            self.assertGreater(image_bytes, 0)
            self.assertEqual(len(binding), 64)

    def test_persisted_image_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"
            root.mkdir()
            expectations, manifest = _fixture(root)
            image_hash = manifest["samples"][0]["png_sha256"]
            (root / "images" / f"{image_hash}.png").write_bytes(PNG_SIGNATURE + b"tampered")
            with self.assertRaisesRegex(SyntheticCurriculumCorpusAcceptanceError, "artifact hash mismatch"):
                _verify_corpus_directory(root, expectations)

    def test_family_split_leakage_is_rejected_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"
            root.mkdir()
            expectations, manifest = _fixture(root)
            tampered = copy.deepcopy(manifest)
            tampered["samples"][0]["split"] = "validation"
            with self.assertRaisesRegex(SyntheticCurriculumCorpusAcceptanceError, "family appears in multiple splits"):
                _verify_manifest(tampered, expectations)

    def test_build_identity_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"
            root.mkdir()
            expectations, _manifest = _fixture(root)
            build = json.loads((root / "build.json").read_bytes())
            build["build_id"] = _hex("wrong-build")
            (root / "build.json").write_bytes(_canonical_json(build))
            with self.assertRaisesRegex(SyntheticCurriculumCorpusAcceptanceError, "build.json build_id mismatch"):
                _verify_corpus_directory(root, expectations)

    def test_missing_target_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"
            root.mkdir()
            expectations, manifest = _fixture(root)
            target_hash = manifest["samples"][0]["source_musicxml_sha256"]
            (root / "targets" / f"{target_hash}.musicxml").unlink()
            with self.assertRaisesRegex(SyntheticCurriculumCorpusAcceptanceError, "filenames do not match"):
                _verify_corpus_directory(root, expectations)

    def test_unexpected_top_level_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"
            root.mkdir()
            expectations, _manifest = _fixture(root)
            (root / "unexpected.bin").write_bytes(b"x")
            with self.assertRaisesRegex(SyntheticCurriculumCorpusAcceptanceError, "top-level layout mismatch"):
                _verify_corpus_directory(root, expectations)

    def test_transport_hash_is_streamed_and_size_can_be_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "mini.tar.gz"
            raw = b"transport bytes"
            archive.write_bytes(raw)
            expectations = _CorpusExpectations(
                source_commit=_hex("source"),
                build_id=_hex("build"),
                config_fingerprint=_hex("config"),
                manifest_sha256=_hex("manifest"),
                transport_sha256=sha256(raw).hexdigest(),
                archive_name=archive.name,
                archive_size_bytes=len(raw),
                dataset_name="mini",
                dataset_version="v1",
                builder_version="builder",
                sample_counts={"test": 0, "train": 0, "validation": 0},
                family_counts={"test": 0, "train": 0, "validation": 0},
                sample_count=0,
                target_count=0,
                image_count=0,
            )
            digest, size = _verify_transport_archive(archive, expectations)
            self.assertEqual(digest, sha256(raw).hexdigest())
            self.assertEqual(size, len(raw))
            archive.write_bytes(raw + b"x")
            with self.assertRaisesRegex(SyntheticCurriculumCorpusAcceptanceError, "byte length mismatch"):
                _verify_transport_archive(archive, expectations)

    def test_evidence_is_canonical_hash_only_json(self):
        receipt = SyntheticCurriculumCorpusReceipt(
            source_commit=_hex("source"),
            build_id=_hex("build"),
            config_fingerprint=_hex("config"),
            manifest_sha256=_hex("manifest"),
            transport_sha256=_hex("transport"),
            transport_archive="mini.tar.gz",
            archive_size_bytes=123,
            sample_count=9,
            target_count=3,
            image_count=9,
            family_split_counts={"test": 1, "train": 1, "validation": 1},
            sample_split_counts={"test": 3, "train": 3, "validation": 3},
            target_bytes_total=42,
            image_bytes_total=84,
            artifact_binding_sha256=_hex("binding"),
        )
        raw = canonical_stage7d_corpus_evidence(receipt)
        self.assertTrue(raw.endswith(b"\n"))
        payload = json.loads(raw)
        self.assertEqual(payload["schema_version"], "st-omr-synthetic-corpus-acceptance-v1")
        self.assertEqual(raw[:-1], _canonical_json(payload))
        self.assertNotIn(b"family-train", raw)
        self.assertNotIn(b"/", raw.replace(b"mini.tar.gz", b""))


if __name__ == "__main__":
    unittest.main()
