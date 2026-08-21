from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import st_omr_training.meter_v4_4_bbox_annotation_colab as v44_colab


class MeterV44ColabCallbackJsonTests(unittest.TestCase):
    def test_callbacks_return_ipython_json_payloads(self):
        callbacks = {}

        class FakeOutput:
            @staticmethod
            def register_callback(name, callback):
                callbacks[name] = callback

        colab_module = types.ModuleType("google.colab")
        colab_module.output = FakeOutput()
        fake_session = mock.Mock()
        fake_session.resume_index.return_value = 0
        fake_session.sample_payload.return_value = {"index": 0, "total": 150}
        fake_session.save_from_preview.return_value = {
            "saved": True,
            "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
        }
        fake_session.set_review_flag.return_value = {"review_flag": True}

        with (
            mock.patch.object(v44_colab, "AnnotationSession", return_value=fake_session),
            mock.patch.dict(sys.modules, {"google.colab": colab_module}),
            mock.patch("IPython.display.display"),
        ):
            v44_colab.launch_colab_annotation(
                candidate_root="unused",
                manifest_path="unused",
            )

        get_result = callbacks[v44_colab._CALLBACK_GET](0)
        save_result = callbacks[v44_colab._CALLBACK_SAVE](
            {
                "token": "t",
                "x0": 1,
                "y0": 2,
                "x1": 3,
                "y1": 4,
                "preview_width": 100,
                "preview_height": 100,
            }
        )
        flag_result = callbacks[v44_colab._CALLBACK_FLAG](
            {"token": "t", "flagged": True}
        )

        for result in (get_result, save_result, flag_result):
            self.assertTrue(callable(getattr(result, "_repr_json_", None)))
        self.assertEqual(get_result.data["total"], 150)
        self.assertTrue(save_result.data["saved"])
        self.assertTrue(flag_result.data["review_flag"])


if __name__ == "__main__":
    unittest.main()
