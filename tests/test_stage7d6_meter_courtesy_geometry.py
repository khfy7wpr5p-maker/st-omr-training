from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from st_omr_training.stage7d5_geometry import (
    STAGE7D5_GEOMETRY_VERSION,
    Stage7D5GeometryError,
    _optional_object_bbox,
)


def _fixture(*, current_count: int, courtesy: bool) -> tuple[ET.Element, ET.Element, dict[ET.Element, ET.Element]]:
    current = "\n".join(
        f'''<g class="meterSig" id="meter-current-{index}">
              <g class="meterSig bounding-box" id="bbox-meter-current-{index}">
                <rect x="{150 + index * 60}" y="100" width="40" height="80" fill="transparent" stroke-width="0"/>
              </g>
            </g>'''
        for index in range(current_count)
    )
    courtesy_svg = (
        '''<g class="meterSig" id="meter-courtesy">
             <g class="meterSig bounding-box" id="bbox-meter-courtesy">
               <rect x="930" y="100" width="40" height="80" fill="transparent" stroke-width="0"/>
             </g>
           </g>'''
        if courtesy
        else ""
    )
    root = ET.fromstring(
        f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 400">
          <g class="measure" id="measure-1">
            {current}
            <g class="barLineAttr" id="barline-1">
              <path d="M 900 100 L 900 180" style="stroke-width: 3;"/>
            </g>
            {courtesy_svg}
          </g>
        </svg>'''
    )
    measure = next(element for element in root.iter() if element.attrib.get("id") == "measure-1")
    parent_map = {child: parent for parent in root.iter() for child in parent}
    return root, measure, parent_map


class Stage7D6CourtesyMeterGeometryTests(unittest.TestCase):
    def test_geometry_contract_is_versioned_after_courtesy_meter_fix(self) -> None:
        self.assertEqual(STAGE7D5_GEOMETRY_VERSION, "stage7d5-staff-structure-geometry-v2")

    def test_post_barline_courtesy_meter_is_not_bound_to_current_measure(self) -> None:
        root, measure, parent_map = _fixture(current_count=1, courtesy=True)
        bbox = _optional_object_bbox(measure, "meterSig", root, parent_map)
        self.assertIsNotNone(bbox)
        assert bbox is not None
        self.assertEqual((bbox.x_min, bbox.x_max), (150.0, 190.0))

    def test_courtesy_only_measure_has_no_current_meter_bbox(self) -> None:
        root, measure, parent_map = _fixture(current_count=0, courtesy=True)
        self.assertIsNone(_optional_object_bbox(measure, "meterSig", root, parent_map))

    def test_two_pre_barline_meter_groups_still_fail_closed(self) -> None:
        root, measure, parent_map = _fixture(current_count=2, courtesy=True)
        with self.assertRaisesRegex(Stage7D5GeometryError, "ambiguous current meterSig"):
            _optional_object_bbox(measure, "meterSig", root, parent_map)


if __name__ == "__main__":
    unittest.main()
