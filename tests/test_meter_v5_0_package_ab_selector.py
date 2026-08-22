import tempfile
import unittest
from pathlib import Path

from st_omr_training.meter_v5_0_package_ab_selector import (
    PackageAbSelectionError,
    build_package_ab_selection,
    write_selection_manifests,
)


def make_rows(per_class=520):
    rows = []
    for meter, flag in (("2/4", "Meter2_4"), ("3/4", "Meter3_4"), ("4/4", "Meter4_4")):
        for i in range(per_class):
            family = f"{meter.replace('/', '')}_{i:04d}"
            sample = f"{family}-1_1_1"
            rows.append(
                {
                    "Package": "ab",
                    "Sample": sample,
                    "Family": family,
                    "PNG": f"C:\\data\\package_ab\\{sample}\\{sample}.png",
                    "Semantic": f"C:\\data\\package_ab\\{sample}\\{sample}.semantic",
                    "Agnostic": f"C:\\data\\package_ab\\{sample}\\{sample}.agnostic",
                    "Complete": "1",
                    "Meter2_4": "1" if flag == "Meter2_4" else "0",
                    "Meter3_4": "1" if flag == "Meter3_4" else "0",
                    "Meter4_4": "1" if flag == "Meter4_4" else "0",
                }
            )
    return rows


class PackageAbSelectorTests(unittest.TestCase):
    def test_selects_500_per_class_and_canonical_splits(self):
        selected, receipt = build_package_ab_selection(make_rows())
        self.assertEqual(receipt["global_unique_families"], 1500)
        self.assertEqual(receipt["source_domain_share_gap"], 0.0)
        for meter in ("2/4", "3/4", "4/4"):
            self.assertEqual(len(selected[meter]), 500)
            counts = {}
            for row in selected[meter]:
                counts[row["Split"]] = counts.get(row["Split"], 0) + 1
                self.assertIn("\\package_ab\\", row["SourceImage"])
            self.assertEqual(
                counts,
                {"train": 400, "val": 50, "final_holdout": 50},
            )

    def test_blacklist_is_excluded(self):
        blacklist = {"ab_24_0000", "ab_34_0000", "ab_44_0000"}
        selected, receipt = build_package_ab_selection(make_rows(), blacklist=blacklist)
        all_families = {
            row["FamilyId"]
            for meter_rows in selected.values()
            for row in meter_rows
        }
        self.assertTrue(blacklist.isdisjoint(all_families))
        self.assertEqual(receipt["blacklist_overlap"], 0)

    def test_cross_meter_family_is_excluded(self):
        rows = make_rows()
        common = "shared_0001"
        rows.extend(
            [
                {
                    "Package": "ab",
                    "Sample": "shared-2",
                    "Family": common,
                    "PNG": "C:\\data\\package_ab\\shared-2\\image.png",
                    "Semantic": "C:\\data\\package_ab\\shared-2\\x.semantic",
                    "Agnostic": "C:\\data\\package_ab\\shared-2\\x.agnostic",
                    "Complete": "1",
                    "Meter2_4": "1",
                    "Meter3_4": "0",
                    "Meter4_4": "0",
                },
                {
                    "Package": "ab",
                    "Sample": "shared-3",
                    "Family": common,
                    "PNG": "C:\\data\\package_ab\\shared-3\\image.png",
                    "Semantic": "C:\\data\\package_ab\\shared-3\\x.semantic",
                    "Agnostic": "C:\\data\\package_ab\\shared-3\\x.agnostic",
                    "Complete": "1",
                    "Meter2_4": "0",
                    "Meter3_4": "1",
                    "Meter4_4": "0",
                },
            ]
        )
        selected, receipt = build_package_ab_selection(rows)
        all_families = {
            row["FamilyId"]
            for meter_rows in selected.values()
            for row in meter_rows
        }
        self.assertNotIn("ab_shared_0001", all_families)
        self.assertGreaterEqual(receipt["ambiguous_family_exclusion_count"], 1)

    def test_insufficient_capacity_fails_closed(self):
        with self.assertRaises(PackageAbSelectionError):
            build_package_ab_selection(make_rows(per_class=499))

    def test_package_provenance_mismatch_fails(self):
        rows = make_rows()
        rows[0]["PNG"] = rows[0]["PNG"].replace("package_ab", "package_aa")
        with self.assertRaises(PackageAbSelectionError):
            build_package_ab_selection(rows)

    def test_write_refuses_existing_output(self):
        selected, _ = build_package_ab_selection(make_rows())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            write_selection_manifests(selected, out)
            self.assertTrue((out / "2_4_SELECTION_MANIFEST.csv").exists())
            with self.assertRaises(PackageAbSelectionError):
                write_selection_manifests(selected, out)


if __name__ == "__main__":
    unittest.main()
