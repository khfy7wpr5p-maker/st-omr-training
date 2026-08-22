import csv
import tempfile
import unittest
from pathlib import Path

from st_omr_training.meter_v5_0_dataset_integrity import audit_manifests


FIELDS = [
    "Split",
    "Meter",
    "FamilyId",
    "SampleId",
    "Folder",
    "SourceImage",
    "SourceSemantic",
    "SourceAgnostic",
    "SplitRank",
]


def write_manifest(path, meter, package="package_aa", rank_name="SplitRank", family_prefix=None):
    fields = FIELDS[:-1] + [rank_name]
    prefix = family_prefix or meter.replace("/", "")
    rows = []
    for i in range(500):
        split = "train" if i < 400 else ("val" if i < 450 else "final_holdout")
        family = f"{prefix}_{i:04d}"
        row = {
            "Split": split,
            "Meter": meter,
            "FamilyId": family,
            "SampleId": f"{prefix}_sample_{i:04d}",
            "Folder": f"{prefix}_folder_{i:04d}",
            "SourceImage": f"C:\\data\\{package}\\{prefix}_{i:04d}\\image.png",
            "SourceSemantic": "x.semantic",
            "SourceAgnostic": "x.agnostic",
            rank_name: f"{i:064x}",
        }
        rows.append(row)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


class MeterV50DatasetIntegrityTests(unittest.TestCase):
    def _base(self, tmp):
        manifests = {}
        for meter in ("2/4", "3/4", "4/4"):
            path = Path(tmp) / f"{meter.replace('/', '_')}.csv"
            write_manifest(path, meter)
            with open(path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
                fields = rows[0].keys()
            for i, row in enumerate(rows):
                if i >= 250:
                    row["SourceImage"] = row["SourceImage"].replace(
                        "package_aa", "package_ab"
                    )
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(fields))
                writer.writeheader()
                writer.writerows(rows)
            manifests[meter] = path
        return manifests

    def test_balanced_dataset_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = audit_manifests(self._base(tmp))
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["training_authorized"])

    def test_cross_split_family_leak_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifests = self._base(tmp)
            path = manifests["3/4"]
            with open(path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
                fields = rows[0].keys()
            rows[450]["FamilyId"] = "24_0000"
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(fields))
                writer.writeheader()
                writer.writerows(rows)
            result = audit_manifests(manifests)
            self.assertIn("FAMILY_SPLIT_LEAKAGE:1", result["reasons"])

    def test_consumed_holdout_overlap_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifests = self._base(tmp)
            result = audit_manifests(manifests, consumed_family_ids={"24_0000"})
            self.assertIn("CONSUMED_HOLDOUT_OVERLAP:1", result["reasons"])

    def test_noncanonical_rank_column_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifests = self._base(tmp)
            path = Path(tmp) / "4_4_alias.csv"
            write_manifest(path, "4/4", rank_name="SelectionRank")
            manifests["4/4"] = path
            result = audit_manifests(manifests)
            self.assertTrue(
                any(
                    reason.startswith("NON_CANONICAL_RANK_COLUMN:4/4")
                    for reason in result["reasons"]
                )
            )

    def test_source_domain_class_shortcut_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifests = {}
            manifests["2/4"] = Path(tmp) / "2.csv"
            write_manifest(manifests["2/4"], "2/4", "package_aa")
            manifests["3/4"] = Path(tmp) / "3.csv"
            write_manifest(manifests["3/4"], "3/4", "package_aa")
            manifests["4/4"] = Path(tmp) / "4.csv"
            write_manifest(manifests["4/4"], "4/4", "package_ab")
            result = audit_manifests(manifests)
            self.assertEqual(result["status"], "HOLD")
            self.assertTrue(
                any(
                    reason.startswith("SOURCE_DOMAIN_SHARE_GAP:")
                    for reason in result["reasons"]
                )
            )


if __name__ == "__main__":
    unittest.main()
