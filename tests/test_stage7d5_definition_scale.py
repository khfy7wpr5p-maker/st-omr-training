from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from st_omr_training.stage7d5_geometry import (
    Stage7D5GeometryError,
    _coordinate_root,
)


class Stage7D5DefinitionScaleTests(unittest.TestCase):
    def test_pinned_verovio_definition_scale_is_selected_by_class(self) -> None:
        root = ET.fromstring(
            b'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2100 2970">
              <svg class="definition-scale" viewBox="0 0 21000 29700">
                <g class="page-margin" transform="translate(500, 500)" />
              </svg>
            </svg>'''
        )
        coordinate_root, view_box = _coordinate_root(root)
        self.assertIn("definition-scale", coordinate_root.attrib["class"].split())
        self.assertEqual(view_box, (0.0, 0.0, 21000.0, 29700.0))

    def test_id_only_historical_shape_does_not_override_pinned_class_contract(self) -> None:
        root = ET.fromstring(
            b'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2100 2970">
              <svg id="definition-scale" viewBox="0 0 21000 29700" />
            </svg>'''
        )
        coordinate_root, view_box = _coordinate_root(root)
        self.assertIs(coordinate_root, root)
        self.assertEqual(view_box, (0.0, 0.0, 2100.0, 2970.0))

    def test_ambiguous_definition_scale_class_fails_closed(self) -> None:
        root = ET.fromstring(
            b'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2100 2970">
              <svg class="definition-scale" viewBox="0 0 21000 29700" />
              <svg class="definition-scale" viewBox="0 0 21000 29700" />
            </svg>'''
        )
        with self.assertRaises(Stage7D5GeometryError):
            _coordinate_root(root)


if __name__ == "__main__":
    unittest.main()
