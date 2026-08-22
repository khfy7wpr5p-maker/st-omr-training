import inspect
import unittest

from st_omr_training import meter_v5_2b_specialist_adaptation as m


class TestMeterV52BSourceGuards(unittest.TestCase):
    def test_training_function_never_mentions_val_or_holdout_paths(self):
        source = inspect.getsource(m.train_adapted_specialists_v1)
        self.assertNotIn("final_holdout", source.lower())
        self.assertNotIn("/val/", source.lower())
        self.assertNotIn("threshold =", source.lower())

    def test_diagnostic_gate_cannot_promote_production(self):
        source = inspect.getsource(m.evaluate_diagnostic_gate_v1)
        self.assertIn('"production_promotion_authorized": False', source)
        self.assertIn('"validation_opened": False', source)
        self.assertIn('"final_holdout_locked": True', source)


if __name__ == "__main__":
    unittest.main()
