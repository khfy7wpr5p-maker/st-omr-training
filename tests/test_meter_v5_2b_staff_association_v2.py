import inspect
import unittest
from types import SimpleNamespace

from st_omr_training import meter_v5_2b_staff_association_v2 as m


def staff(staff_id, y_min, y_max):
    return SimpleNamespace(
        staff_id=staff_id,
        staff_bbox=SimpleNamespace(y_min=float(y_min), y_max=float(y_max)),
    )


class TestMeterV52BStaffAssociationV2(unittest.TestCase):
    def test_unique_center_y_containment_selects_only_matching_staff(self):
        staffs = (staff("upper", 10, 50), staff("lower", 70, 110))
        selected = m.select_unique_containing_staff_v1(
            staffs, center_y_normalized=85.0, sample_id="sample"
        )
        self.assertEqual(selected.staff_id, "lower")

    def test_zero_containment_fails_closed_without_nearest_fallback(self):
        staffs = (staff("upper", 10, 50), staff("lower", 70, 110))
        with self.assertRaisesRegex(Exception, r"matches=0; accepted_staffs=2"):
            m.select_unique_containing_staff_v1(
                staffs, center_y_normalized=60.0, sample_id="sample"
            )

    def test_multiple_containment_fails_closed(self):
        staffs = (staff("a", 10, 60), staff("b", 60, 110))
        with self.assertRaisesRegex(Exception, r"matches=2; accepted_staffs=2"):
            m.select_unique_containing_staff_v1(
                staffs, center_y_normalized=60.0, sample_id="sample"
            )

    def test_non_finite_center_y_fails_closed(self):
        with self.assertRaisesRegex(Exception, "non-finite"):
            m.select_unique_containing_staff_v1(
                (staff("a", 10, 60),),
                center_y_normalized=float("nan"),
                sample_id="sample",
            )

    def test_preflight_contract_is_exact_containment_only(self):
        source = inspect.getsource(m.select_unique_containing_staff_v1)
        self.assertIn("staff.staff_bbox.y_min <= center_y <= staff.staff_bbox.y_max", source)
        self.assertIn("if len(matches) != 1", source)
        self.assertNotIn("min(staff", source)
        self.assertNotIn("sorted(staff", source)

    def test_derivation_requires_passed_preflight(self):
        source = inspect.getsource(m.derive_staff_relative_slots_v2)
        self.assertIn("verify_staff_association_preflight_v2", source)
        self.assertNotIn("detect_multistaff_geometry_v2", source)


if __name__ == "__main__":
    unittest.main()
