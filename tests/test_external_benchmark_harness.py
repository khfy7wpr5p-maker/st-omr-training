from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from st_omr_training.external_benchmark_harness import (
    AdmissionMode,
    BenchmarkDatasetKind,
    BenchmarkManifestRow,
    ExternalBenchmarkHarnessError,
    REQUIRED_BENCHMARK_KINDS,
    benchmark_spec,
    create_admission,
    directory_tree_sha256,
    manifest_sha256,
    matching_registry_record,
    read_manifest_jsonl,
    split_manifest_sha256,
    validate_commercial_evidence,
    validate_manifest_rows,
)
from st_omr_training.external_dataset_registry import EXTERNAL_DATASET_CANDIDATES


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _rows() -> tuple[BenchmarkManifestRow, ...]:
    return (
        BenchmarkManifestRow(
            sample_id=SHA_A,
            family_id="score-1",
            split="validation",
            image_relpath="images/system-1.png",
            target_relpath="targets/system-1.lmx",
            system_id="system-1",
        ),
        BenchmarkManifestRow(
            sample_id=SHA_B,
            family_id="score-2",
            split="test",
            image_relpath="images/system-2.png",
            target_relpath="targets/system-2.lmx",
            system_id="system-2",
        ),
    )


class ExternalBenchmarkHarnessTests(unittest.TestCase):
    def test_required_external_benchmark_kinds_are_frozen(self) -> None:
        self.assertEqual(
            tuple(kind.value for kind in REQUIRED_BENCHMARK_KINDS),
            (
                "olimpic-synthetic",
                "olimpic-scanned",
                "grandstaff-lmx",
                "muse-omr-benchmark",
            ),
        )

    def test_olimpic_scanned_spec_is_system_level_validation_test(self) -> None:
        spec = benchmark_spec(BenchmarkDatasetKind.OLIMPIC_SCANNED)
        self.assertTrue(spec.system_level)
        self.assertEqual(spec.declared_splits, ("validation", "test"))
        self.assertEqual(spec.dataset_name, "OLiMPiC")

    def test_spec_matches_exact_registry_component(self) -> None:
        spec = benchmark_spec(BenchmarkDatasetKind.OLIMPIC_SCANNED)
        record = matching_registry_record(spec, EXTERNAL_DATASET_CANDIDATES)
        self.assertEqual(record.dataset_component, "scanned 1.0")

    def test_manifest_hashes_are_order_independent_and_distinct(self) -> None:
        spec = benchmark_spec(BenchmarkDatasetKind.OLIMPIC_SCANNED)
        rows = _rows()
        self.assertEqual(manifest_sha256(rows, spec), manifest_sha256(tuple(reversed(rows)), spec))
        self.assertEqual(
            split_manifest_sha256(rows, spec),
            split_manifest_sha256(tuple(reversed(rows)), spec),
        )
        self.assertNotEqual(manifest_sha256(rows, spec), split_manifest_sha256(rows, spec))

    def test_manifest_rejects_family_leakage(self) -> None:
        spec = benchmark_spec(BenchmarkDatasetKind.OLIMPIC_SYNTHETIC)
        rows = (
            BenchmarkManifestRow(
                sample_id=SHA_A,
                family_id="same-score",
                split="train",
                image_relpath="images/a.png",
                target_relpath="targets/a.lmx",
                system_id="a",
            ),
            BenchmarkManifestRow(
                sample_id=SHA_B,
                family_id="same-score",
                split="test",
                image_relpath="images/b.png",
                target_relpath="targets/b.lmx",
                system_id="b",
            ),
        )
        with self.assertRaises(ExternalBenchmarkHarnessError):
            validate_manifest_rows(rows, spec)

    def test_manifest_rejects_undeclared_split(self) -> None:
        spec = benchmark_spec(BenchmarkDatasetKind.OLIMPIC_SCANNED)
        row = BenchmarkManifestRow(
            sample_id=SHA_A,
            family_id="score-1",
            split="train",
            image_relpath="images/a.png",
            target_relpath="targets/a.lmx",
            system_id="a",
        )
        with self.assertRaises(ExternalBenchmarkHarnessError):
            validate_manifest_rows((row,), spec)

    def test_manifest_rejects_path_traversal(self) -> None:
        with self.assertRaises(ExternalBenchmarkHarnessError):
            BenchmarkManifestRow(
                sample_id=SHA_A,
                family_id="score-1",
                split="test",
                image_relpath="../secret.png",
                target_relpath="targets/a.lmx",
                system_id="a",
            )

    def test_research_override_admits_candidate_but_is_not_commercial_evidence(self) -> None:
        spec = benchmark_spec(BenchmarkDatasetKind.OLIMPIC_SCANNED)
        record = matching_registry_record(spec, EXTERNAL_DATASET_CANDIDATES)
        admission = create_admission(
            spec=spec,
            registry_record=record,
            rows=_rows(),
            data_artifact_sha256=SHA_C,
            admission_mode=AdmissionMode.RESEARCH_OVERRIDE,
            research_override_reference="approved-research-override-2026-08-28",
        )
        self.assertFalse(admission.commercial_evidence_eligible)
        self.assertEqual(admission.benchmark_identity().benchmark_id, spec.benchmark_id)
        with self.assertRaises(ExternalBenchmarkHarnessError):
            validate_commercial_evidence(admission)

    def test_research_override_requires_explicit_reference(self) -> None:
        spec = benchmark_spec(BenchmarkDatasetKind.OLIMPIC_SCANNED)
        record = matching_registry_record(spec, EXTERNAL_DATASET_CANDIDATES)
        with self.assertRaises(ExternalBenchmarkHarnessError):
            create_admission(
                spec=spec,
                registry_record=record,
                rows=_rows(),
                data_artifact_sha256=SHA_C,
                admission_mode=AdmissionMode.RESEARCH_OVERRIDE,
            )

    def test_strict_mode_rejects_unpinned_registry_candidate(self) -> None:
        spec = benchmark_spec(BenchmarkDatasetKind.OLIMPIC_SCANNED)
        record = matching_registry_record(spec, EXTERNAL_DATASET_CANDIDATES)
        with self.assertRaises(ExternalBenchmarkHarnessError):
            create_admission(
                spec=spec,
                registry_record=record,
                rows=_rows(),
                data_artifact_sha256=SHA_C,
                admission_mode=AdmissionMode.STRICT_REGISTRY,
            )

    def test_directory_hash_is_deterministic_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir()
            (root / "a" / "one.txt").write_text("one", encoding="utf-8")
            (root / "two.txt").write_text("two", encoding="utf-8")
            first = directory_tree_sha256(root)
            second = directory_tree_sha256(root)
            self.assertEqual(first, second)
            (root / "two.txt").write_text("changed", encoding="utf-8")
            self.assertNotEqual(first, directory_tree_sha256(root))

    def test_jsonl_manifest_reader_is_fail_closed(self) -> None:
        spec = benchmark_spec(BenchmarkDatasetKind.OLIMPIC_SCANNED)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(
                "\n".join(
                    (
                        '{"sample_id":"' + SHA_A + '","family_id":"score-1","split":"validation","image_relpath":"images/a.png","target_relpath":"targets/a.lmx","system_id":"a"}',
                        '{"sample_id":"' + SHA_B + '","family_id":"score-2","split":"test","image_relpath":"images/b.png","target_relpath":"targets/b.lmx","system_id":"b"}',
                    )
                ),
                encoding="utf-8",
            )
            rows = read_manifest_jsonl(path, spec)
            self.assertEqual(len(rows), 2)
            path.write_text("not-json", encoding="utf-8")
            with self.assertRaises(ExternalBenchmarkHarnessError):
                read_manifest_jsonl(path, spec)


if __name__ == "__main__":
    unittest.main()
