import unittest
from pathlib import Path

from st_omr_training.generator import GeneratorConfig, generate_score
from st_omr_training.musicxml_validator import (
    validate_musicxml,
    validate_musicxml_xsd,
    verify_musicxml_schema_assets,
)
from st_omr_training.musicxml_writer import write_musicxml


GOLDEN_DIR = Path(__file__).with_name("golden")


class VendoredMusicXMLSchemaTests(unittest.TestCase):
    def test_pinned_official_schema_assets_pass_integrity_gate(self):
        result = verify_musicxml_schema_assets()
        self.assertTrue(result.is_valid, result.issues)

    def test_all_stage2b_goldens_pass_real_xsd_and_semantic_gates(self):
        fixtures = sorted(GOLDEN_DIR.glob("*.musicxml"))
        self.assertEqual(len(fixtures), 6)
        for path in fixtures:
            with self.subTest(path=path.name):
                data = path.read_bytes()
                xsd = validate_musicxml_xsd(data)
                combined = validate_musicxml(data)
                self.assertTrue(xsd.is_valid, xsd.issues)
                self.assertTrue(combined.is_valid, combined.issues)

    def test_generated_writer_outputs_pass_real_xsd_and_semantic_gates(self):
        config = GeneratorConfig(measure_count=5)
        for seed in range(25):
            with self.subTest(seed=seed):
                data = write_musicxml(generate_score(config, seed))
                result = validate_musicxml(data)
                self.assertTrue(result.is_valid, result.issues)

    def test_real_xsd_rejects_non_musicxml_root(self):
        result = validate_musicxml_xsd(b'<?xml version="1.0"?><not-musicxml/>')
        self.assertFalse(result.is_valid)
        self.assertEqual([issue.code for issue in result.issues], ["musicxml.xsd_invalid"])


if __name__ == "__main__":
    unittest.main()
