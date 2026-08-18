from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from st_omr_training.system_geometry_train_topology_audit_v1 import (
    SystemGeometryTopologyAuditError,
    _horizontal_overlap_ratio,
    _summary,
    audit_d6_train_topology,
)


class SystemGeometryTrainTopologyAuditTests(unittest.TestCase):
    def test_summary_is_deterministic(self) -> None:
        values = [5.0, 1.0, 3.0, 2.0, 4.0]
        first = _summary(values)
        for _ in range(10):
            self.assertEqual(_summary(list(reversed(values))), first)
        self.assertEqual(first["count"], 5)
        self.assertEqual(first["min"], 1.0)
        self.assertEqual(first["median"], 3.0)
        self.assertEqual(first["max"], 5.0)

    def test_horizontal_overlap_uses_smaller_width_as_denominator(self) -> None:
        a = (0.0, 0.0, 100.0, 20.0)
        b = (20.0, 30.0, 80.0, 50.0)
        self.assertEqual(_horizontal_overlap_ratio(a, b), 1.0)
        c = (90.0, 60.0, 120.0, 80.0)
        self.assertAlmostEqual(_horizontal_overlap_ratio(a, c), 10.0 / 30.0)

    def test_wrong_manifest_fails_before_any_label_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text("{}", encoding="ascii")
            with self.assertRaisesRegex(SystemGeometryTopologyAuditError, "manifest SHA-256 mismatch"):
                audit_d6_train_topology(root)


if __name__ == "__main__":
    unittest.main()
