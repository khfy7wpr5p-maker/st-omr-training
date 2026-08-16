from __future__ import annotations

import unittest

from st_omr_training.stage7d13_r2_rest_audit import (
    R1_OUTPUT_STRIDE,
    Stage7D13R2RestAuditError,
    development_rows,
    summarize_rest_labels,
)


class TestStage7D13R2RestAudit(unittest.TestCase):
    @staticmethod
    def _label(
        *,
        split: str,
        record_hex: str,
        rest_rows: list[dict[str, object]],
        scale: float = 1.0,
    ) -> dict[str, object]:
        return {
            "record_id": record_hex * 64,
            "split": split,
            "transform": {"scale": scale},
            "targets": {"rest": rest_rows},
        }

    @staticmethod
    def _rest(
        class_name: str,
        *,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> dict[str, object]:
        return {
            "class": class_name,
            "bbox": {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1},
            "center": {"x": (x0 + x1) / 2.0, "y": (y0 + y1) / 2.0},
        }

    def test_development_rows_accepts_train_validation_only(self) -> None:
        rows = development_rows([{"split": "train"}, {"split": "validation"}])
        self.assertEqual(len(rows), 2)

    def test_development_rows_fails_closed_on_test(self) -> None:
        with self.assertRaisesRegex(Stage7D13R2RestAuditError, "sealed TEST"):
            development_rows([{"split": "test", "anything": "must-not-matter"}])

    def test_summarize_rest_geometry_and_zero_records(self) -> None:
        train = self._label(
            split="train",
            record_hex="a",
            scale=0.8,
            rest_rows=[
                self._rest("quarter", x0=10.0, y0=20.0, x1=14.0, y1=32.0),
                self._rest("eighth", x0=40.0, y0=40.0, x1=46.0, y1=56.0),
            ],
        )
        validation = self._label(
            split="validation",
            record_hex="b",
            scale=1.2,
            rest_rows=[],
        )
        result = summarize_rest_labels(
            [(train, "c" * 64), (validation, "d" * 64)]
        )

        self.assertEqual(result["record_split_counts"], {"train": 1, "validation": 1})
        self.assertEqual(result["rest_instance_counts"], {"train": 2})
        self.assertEqual(result["rest_positive_records"], {"train": 1, "validation": 0})
        self.assertEqual(result["rest_zero_records"], {"train": 0, "validation": 1})
        self.assertEqual(result["stride4_collision_records"], 0)

        class_counts = result["rest_class_counts"]
        assert isinstance(class_counts, dict)
        self.assertEqual(class_counts["train"]["quarter"], 1)
        self.assertEqual(class_counts["train"]["eighth"], 1)
        self.assertEqual(class_counts["train"]["half"], 0)

        geometry = result["geometry"]
        assert isinstance(geometry, dict)
        train_geometry = geometry["train"]
        self.assertEqual(train_geometry["width"]["count"], 2)
        self.assertEqual(train_geometry["width"]["min"], 4.0)
        self.assertEqual(train_geometry["width"]["max"], 6.0)
        self.assertEqual(train_geometry["transform_scale"]["p50"], 0.8)
        self.assertEqual(train_geometry["sub_stride_width_count"], 0)

    def test_sub_stride_geometry_is_reported_not_silently_normalized(self) -> None:
        label = self._label(
            split="train",
            record_hex="a",
            rest_rows=[
                self._rest(
                    "half",
                    x0=10.0,
                    y0=10.0,
                    x1=10.0 + R1_OUTPUT_STRIDE - 0.25,
                    y1=20.0,
                )
            ],
        )
        result = summarize_rest_labels([(label, "c" * 64)])
        geometry = result["geometry"]
        assert isinstance(geometry, dict)
        self.assertEqual(geometry["train"]["sub_stride_width_count"], 1)
        self.assertEqual(geometry["train"]["sub_stride_min_dimension_count"], 1)

    def test_stride4_collision_is_detected(self) -> None:
        label = self._label(
            split="train",
            record_hex="a",
            rest_rows=[
                self._rest("quarter", x0=8.0, y0=8.0, x1=10.0, y1=10.0),
                self._rest("eighth", x0=9.0, y0=9.0, x1=11.0, y1=11.0),
            ],
        )
        result = summarize_rest_labels([(label, "c" * 64)])
        self.assertEqual(result["stride4_collision_records"], 1)

    def test_invalid_rest_class_fails_closed(self) -> None:
        label = self._label(
            split="train",
            record_hex="a",
            rest_rows=[self._rest("whole", x0=10.0, y0=10.0, x1=20.0, y1=20.0)],
        )
        with self.assertRaisesRegex(Stage7D13R2RestAuditError, "outside half"):
            summarize_rest_labels([(label, "c" * 64)])

    def test_center_outside_bbox_fails_closed(self) -> None:
        row = self._rest("half", x0=10.0, y0=10.0, x1=20.0, y1=20.0)
        row["center"] = {"x": 30.0, "y": 15.0}
        label = self._label(
            split="validation",
            record_hex="b",
            rest_rows=[row],
        )
        with self.assertRaisesRegex(Stage7D13R2RestAuditError, "center lies outside bbox"):
            summarize_rest_labels([(label, "d" * 64)])


if __name__ == "__main__":
    unittest.main()
