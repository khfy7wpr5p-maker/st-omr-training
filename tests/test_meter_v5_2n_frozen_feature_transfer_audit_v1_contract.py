import unittest

from st_omr_training import meter_v5_2n_frozen_feature_transfer_audit_v1 as m


class TestMeterV52NFrozenFeatureTransferAuditContract(unittest.TestCase):
    def test_exact_surfaces_and_feature_dimension_are_frozen(self):
        self.assertEqual(m.EXPECTED_V5_COUNT, 540)
        self.assertEqual(m.EXPECTED_V5_POSITIVE, 90)
        self.assertEqual(m.EXPECTED_HISTORICAL_COUNT, 26964)
        self.assertEqual(
            m.EXPECTED_HISTORICAL_LABEL_COUNTS,
            {"2": 1527, "3": 1587, "4": 6396, "NONE": 17454},
        )
        self.assertEqual(m.EXPECTED_FEATURE_DIM, 64)
        self.assertEqual(m.HISTORICAL_BATCH_SIZE, 256)

    def test_report_name_and_schema_are_v5_2n_specific(self):
        self.assertEqual(
            m.SCHEMA,
            "st-omr-meter-v5-2n-frozen-feature-transfer-audit-v1",
        )
        self.assertEqual(
            m.REPORT_NAME,
            "v5_2n_frozen_feature_transfer_audit_v1.json",
        )


if __name__ == "__main__":
    unittest.main()
