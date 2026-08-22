import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "st_omr_meter_v5_0_source_inventory_win7.ps1"
FROZEN = ROOT / "evidence" / "METER_V4_5_CONSUMED_FAMILIES.txt"
EXPECTED_SELECTION_SHA = "4335a48a091912ba422c16d8fcbaaa7bbf5f7a0a43f088146a50a3e02e3ed7dc"
EXPECTED_FAMILY_LIST_SHA = "9d969b6bf5749bae7003c45644c50be36495ccb9b10fe3e7569ace5d413adea3"


class MeterV50SourceInventoryScriptTests(unittest.TestCase):
    def setUp(self):
        self.text = SCRIPT.read_text(encoding="utf-8")

    def test_script_is_read_only_against_source_and_existing_datasets(self):
        forbidden_commands = (
            "Copy-Item",
            "Move-Item",
            "Remove-Item",
            "Start-Process",
            "robocopy",
            "xcopy",
            "mklink",
        )
        for command in forbidden_commands:
            self.assertIsNone(
                re.search(rf"(?im)^\s*{re.escape(command)}\b", self.text),
                command,
            )
        self.assertNotIn("cmd.exe", self.text)
        self.assertNotIn("Get-ChildItem", self.text)
        self.assertNotIn("METER_V2_1500", self.text)
        self.assertIn(
            'D:\\veri eğitim seti\\ST_OMR_WORKSPACE\\V5_0_SOURCE_INVENTORY',
            self.text,
        )
        self.assertIn('if(Test-Path -LiteralPath $outRoot)', self.text)
        self.assertEqual(self.text.count("New-Item -ItemType Directory"), 1)

    def test_inventory_is_master_index_driven_and_package_bounded(self):
        self.assertIn(
            'D:\\veri eğitim seti\\ST_OMR_PRIMUS_INDEX\\MASTER_INDEX.tsv',
            self.text,
        )
        self.assertIn('Import-Csv -Delimiter "`t" -LiteralPath $indexPath', self.text)
        self.assertIn('$r.Package -ne "aa"', self.text)
        self.assertIn('$r.Package -ne "ab"', self.text)
        self.assertIn('Flag="Meter2_4"', self.text)
        self.assertIn('Flag="Meter3_4"', self.text)
        self.assertIn('Flag="Meter4_4"', self.text)

    def test_consumed_holdout_family_freeze_is_exact_and_hash_bound(self):
        frozen_text = FROZEN.read_text(encoding="utf-8")
        frozen_lines = [line for line in frozen_text.splitlines() if line]
        self.assertEqual(len(frozen_lines), 150)
        self.assertEqual(len(set(frozen_lines)), 150)
        self.assertEqual(frozen_lines, sorted(frozen_lines))
        self.assertEqual(
            hashlib.sha256(frozen_text.encode("ascii")).hexdigest(),
            EXPECTED_FAMILY_LIST_SHA,
        )

        match = re.search(
            r'\$consumedBlock\s*=\s*@"\n(?P<body>.*?)\n"@',
            self.text,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        embedded = [line for line in match.group("body").splitlines() if line]
        self.assertEqual(embedded, frozen_lines)
        self.assertIn(EXPECTED_SELECTION_SHA, self.text)
        self.assertIn(EXPECTED_FAMILY_LIST_SHA, self.text)
        self.assertIn('$consumedFamilies.Count -ne 150', self.text)
        self.assertIn('$observedConsumedFamilyListSha -ne $expectedConsumedFamilyListSha', self.text)

    def test_semantic_agnostic_and_global_collision_observations_are_required(self):
        self.assertIn("function Test-SingleMeter", self.text)
        self.assertIn("function Test-AgnosticMeterPair", self.text)
        self.assertIn("function Get-ContentKey", self.text)
        self.assertIn("Group-Object FamilyId", self.text)
        self.assertIn("Group-Object ContentKey", self.text)
        self.assertIn("cross_meter_family_collisions.csv", self.text)
        self.assertIn("cross_meter_content_collisions.csv", self.text)

    def test_count_only_mix_cannot_authorize_rebuild_training_or_bbox(self):
        self.assertIn("count_only_common_mix_feasible=", self.text)
        self.assertIn("count_only_aa_low=", self.text)
        self.assertIn("count_only_aa_high=", self.text)
        self.assertIn(
            "count_only_warning=NOT_REBUILD_AUTHORIZATION_GLOBAL_FAMILY_AND_CONTENT_DISJOINTNESS_STILL_REQUIRED",
            self.text,
        )
        self.assertIn('"training_authorized=false"', self.text)
        self.assertIn('"bbox_annotation_authorized=false"', self.text)
        self.assertIn('"checkpoint_opened=false"', self.text)
        self.assertIn('"inference_count=0"', self.text)
        self.assertIn('"dataset_mutated=false"', self.text)


if __name__ == "__main__":
    unittest.main()
