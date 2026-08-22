import re
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "st_omr_meter_v5_0_package_ab_materialize_win7.ps1"
)


class Win7MaterializerStaticSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_old_windows_powershell_compatibility_guards(self):
        self.assertNotIn("Import-Csv -LiteralPath", self.text)
        self.assertNotIn(".Dispose()", self.text)
        self.assertNotIn("ConvertTo-Json", self.text)
        self.assertIn("try { $sha.Clear() } catch {}", self.text)

    def test_preregistered_hashes_are_embedded(self):
        for sha in (
            "d07ca3d0f7104ac1e5ed551886d80f5971da50b19ef65345c1fd6fa5ebbfb38e",
            "5509bed3ba11dccbed7c277e90fb5e39e9ae6890bb7f460f0f24e41bb16bf2e8",
            "cb8d036d1f0629eb6a14dbd57c887a5cec0d405d0e668ca403af8901080adc22",
            "3231134495c3993b9d0d17355c8758bff2b879513289baad62d3dec03b641fc9",
        ):
            self.assertIn(sha, self.text)

    def test_existing_meter_v2_is_explicitly_forbidden(self):
        self.assertIn('"\\test\\meter_v2_1500"', self.text.lower())
        self.assertIn("Mevcut TEST\\\\METER_V2_1500 icine yazmak yasak", self.text)
        self.assertNotIn("Remove-Item", self.text)
        self.assertNotIn("Clear-Content", self.text)

    def test_preflight_precedes_first_output_creation(self):
        precheck = self.text.index('Write-Host "PRECHECK=PASS')
        first_partial_creation = self.text.index(
            "New-Item -ItemType Directory -Path $PartialRoot"
        )
        self.assertLess(precheck, first_partial_creation)

    def test_copy_is_non_overwriting_and_hash_verified(self):
        self.assertIn(
            "[System.IO.File]::Copy($artifact.Source, $artifact.Dest, $false)",
            self.text,
        )
        self.assertIn("$sourceSha = Get-Sha256File $artifact.Source", self.text)
        self.assertIn("$destSha = Get-Sha256File $artifact.Dest", self.text)
        self.assertIn("if($sourceSha -ne $destSha)", self.text)

    def test_training_and_bbox_remain_closed(self):
        self.assertIn('"bbox_annotation_authorized=false"', self.text)
        self.assertIn('"training_authorized=false"', self.text)
        self.assertIn('"checkpoint_opened=false"', self.text)
        self.assertIn('"model_evaluated=false"', self.text)
        self.assertIn('"inference_count=0"', self.text)
        self.assertNotRegex(self.text.lower(), re.compile(r"torch|optimizer|backward\("))


if __name__ == "__main__":
    unittest.main()
