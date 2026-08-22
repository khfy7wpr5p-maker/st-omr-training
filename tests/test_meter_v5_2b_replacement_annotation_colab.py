import inspect
import unittest

from st_omr_training import meter_v5_2b_replacement_annotation_colab as m


class _FakeSession:
    def sample_payload(self, index):
        expected = {
            63: ("150200092-1_1_1", "2/4"),
            125: ("150207112-1_1_1", "3/4"),
        }
        sample_id, meter = expected[index]
        return {
            "index": index,
            "sample_id": sample_id,
            "meter": meter,
            "locked_seed": False,
            "binding_token": "a" * 64,
        }


class TestMeterV52BReplacementAnnotationColab(unittest.TestCase):
    def test_exact_two_targets_only(self):
        self.assertEqual(
            m.TARGETS,
            (
                (63, "150200092-1_1_1", "2/4"),
                (125, "150207112-1_1_1", "3/4"),
            ),
        )

    def test_target_payload_maps_logical_positions_only(self):
        session = _FakeSession()
        first = m._target_payload(session, 0)
        second = m._target_payload(session, 1)
        self.assertEqual((first["index"], first["sample_id"]), (63, "150200092-1_1_1"))
        self.assertEqual((second["index"], second["sample_id"]), (125, "150207112-1_1_1"))
        self.assertEqual(first["replacement_position"], 0)
        self.assertEqual(second["replacement_position"], 1)
        with self.assertRaises(Exception):
            m._target_payload(session, 2)

    def test_ui_has_no_generic_300_sample_navigation(self):
        source = inspect.getsource(m.launch_replacement_annotation)
        self.assertNotIn("v52a-prev", source)
        self.assertNotIn("v52a-next", source)
        self.assertIn("approved target", source)
        self.assertIn("replacement_complete", source)

    def test_training_and_spatial_derivation_are_not_implemented_here(self):
        source = inspect.getsource(m).lower()
        self.assertNotIn("optimizer", source)
        self.assertNotIn("backward(", source)
        self.assertNotIn("midpoint", source)
        self.assertNotIn("tight-digit", source)


if __name__ == "__main__":
    unittest.main()
