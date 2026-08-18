from __future__ import annotations

import math
import unittest

from st_omr_training.m4_e3i_geometry_preflight import (
    FROZEN_OFFSET_STAFF_SPACES,
    M4E3IGeometryPreflightError,
    audit_records,
)


def _record(sample: str, system: str, *, x: float, error: float, raw: float | None = None, robust: float | None = None, truth: str = "2/4"):
    result = {
        "sample_id": sample,
        "system_id": system,
        "selected_anchor_x": x,
        "anchor_error_staff_spaces": error,
        "truth": truth,
    }
    if raw is not None:
        result["raw_component_x_min"] = raw
    if robust is not None:
        result["robust_staff_x_min"] = robust
    return result


class M4E3IGeometryPreflightTests(unittest.TestCase):
    def test_oracle_best_of_two_is_an_optimistic_lower_bound(self) -> None:
        # Ground truth: target=100, truth spacing=10. Candidate 1 is at 130.
        # V6 model spacing is 8, so robust-left is chosen so frozen offset
        # produces selected x=130. Candidate 2 component-left=100 therefore
        # lands close to the target and the oracle improves the error.
        robust_left = 130.0 - FROZEN_OFFSET_STAFF_SPACES * 8.0
        v4 = {"records": [_record("s", "y", x=80.0, error=2.0)]}
        v5 = {"records": [_record("s", "y", x=110.0, error=1.0)]}
        v6 = {
            "records": [
                _record(
                    "s",
                    "y",
                    x=130.0,
                    error=3.0,
                    raw=100.0,
                    robust=robust_left,
                )
            ]
        }
        result = audit_records(v4, v5, v6)
        metrics = result["metrics"]
        self.assertTrue(math.isclose(metrics["candidate_1_anchor_p95_staff_spaces"], 3.0))
        self.assertLess(metrics["oracle_best_of_two_anchor_p95_staff_spaces"], 0.1)
        self.assertEqual(metrics["candidate_2_better_count"], 1)

    def test_surface_mismatch_fails_closed(self) -> None:
        v4 = {"records": [_record("a", "x", x=80.0, error=2.0)]}
        v5 = {"records": [_record("a", "x", x=90.0, error=1.0)]}
        v6 = {"records": [_record("b", "x", x=100.0, error=0.0, raw=100.0, robust=100.0)]}
        with self.assertRaises(M4E3IGeometryPreflightError):
            audit_records(v4, v5, v6)

    def test_inconsistent_persisted_errors_fail_closed(self) -> None:
        v4 = {"records": [_record("s", "y", x=80.0, error=1.0)]}
        v5 = {"records": [_record("s", "y", x=100.0, error=1.0)]}
        v6 = {"records": [_record("s", "y", x=130.0, error=99.0, raw=100.0, robust=130.5)]}
        with self.assertRaises(M4E3IGeometryPreflightError):
            audit_records(v4, v5, v6)


if __name__ == "__main__":
    unittest.main()
