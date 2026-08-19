from __future__ import annotations

from io import BytesIO
import unittest

from PIL import Image, ImageDraw

from st_omr_training.runtime_measure_system_boundaries_v2 import propose_measure_system_boundaries_v2
from test_runtime_measure_system_boundaries_v2 import _group


class MeasureSystemBoundariesV2StemRegression(unittest.TestCase):
    def test_high_coverage_vertical_stem_without_outer_line_anchors_is_not_measure_boundary(self) -> None:
        image = Image.new("L", (320, 180), 255)
        draw = ImageDraw.Draw(image)
        top = 40
        for offset in (0, 10, 20, 30, 40):
            draw.line((20, top + offset, 300, top + offset), fill=0, width=1)

        # 36/41 staff-span pixels are dark (>80% coverage) but the line does
        # not touch either outer staff line. A coverage-only detector would
        # create a false measure boundary here.
        draw.line((160, top + 3, 160, top + 38), fill=0, width=1)
        out = BytesIO()
        image.save(out, format="PNG", optimize=False, compress_level=9)
        png = out.getvalue()

        grouped = _group(png, policy="monostaff-v1", height=180)
        result = propose_measure_system_boundaries_v2(png, grouped)
        self.assertEqual(result.report.status, "accepted")
        self.assertEqual(len(result.report.logical_measures), 1)
        self.assertEqual(result.report.staff_evidence[0].raw_run_centers, ())


if __name__ == "__main__":
    unittest.main()
