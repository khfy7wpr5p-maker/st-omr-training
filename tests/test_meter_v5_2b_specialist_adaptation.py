import io
import unittest

from PIL import Image

from st_omr_training import meter_v5_2b_specialist_adaptation as m


class TestMeterV52BSpecialistAdaptation(unittest.TestCase):
    def test_frozen_training_configuration(self):
        config = m.FROZEN_TRAIN_CONFIG
        self.assertEqual(config.epochs, 12)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.learning_rate_micros, 100)
        self.assertEqual(config.weight_decay_micros, 100)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.checkpoint_selection, "fixed-final-epoch-no-sweep")
        self.assertEqual(
            m.FROZEN_THRESHOLDS,
            {"2": 0.48, "3": 0.60, "4": 0.47},
        )
        self.assertEqual(len(m.training_config_fingerprint_v1()), 64)
        with self.assertRaises(ValueError):
            m.SpecialistAdaptationConfigV1(epochs=13)
        with self.assertRaises(ValueError):
            m.SpecialistAdaptationConfigV1(device="cuda")

    def test_approved_staff_relative_geometry_is_frozen(self):
        self.assertEqual(m.NUMERATOR_LINE_INDEX, 1)
        self.assertEqual(m.DENOMINATOR_LINE_INDEX, 3)
        self.assertAlmostEqual(m.WIDTH_OVER_STAFF_SPACING, 1.5960569245912566)
        self.assertEqual(m.HEIGHT_OVER_STAFF_SPACING, 2.0)
        edges = m._slot_edges(100.0, 50.0, 10.0)
        self.assertAlmostEqual(edges[0], 100.0 - 1.5960569245912566 * 5.0)
        self.assertAlmostEqual(edges[2], 100.0 + 1.5960569245912566 * 5.0)
        self.assertEqual(edges[1], 40.0)
        self.assertEqual(edges[3], 60.0)
        box = m._integer_crop_box(edges, image_width=200, image_height=100)
        self.assertEqual(box[1], 40)
        self.assertEqual(box[3], 60)
        with self.assertRaises(m.MeterV5_2BError):
            m._integer_crop_box((-1.0, 1.0, 10.0, 10.0), image_width=20, image_height=20)

    def test_palette_adapter_is_deterministic_and_source_read_only(self):
        image = Image.new("P", (12, 9))
        palette = []
        for i in range(256):
            palette.extend((i, i, i))
        image.putpalette(palette)
        image.putpixel((3, 4), 27)
        raw = io.BytesIO()
        image.save(raw, format="PNG", optimize=False, compress_level=9)
        source = raw.getvalue()

        first = m._prepare_runtime_raster(source)
        second = m._prepare_runtime_raster(source)
        self.assertEqual(first, second)
        runtime, mode, width, height, adapter = first
        self.assertEqual(mode, "RGB")
        self.assertEqual((width, height), (12, 9))
        self.assertEqual(adapter, "P_TO_RGB_TRAINING_V1")
        self.assertNotEqual(runtime, source)
        with Image.open(io.BytesIO(runtime)) as converted:
            converted.load()
            self.assertEqual(converted.mode, "RGB")
            self.assertEqual(converted.getpixel((3, 4)), (27, 27, 27))

    def test_historical_canvas_never_upscales(self):
        crop = Image.new("L", (20, 30), 255)
        crop.putpixel((10, 15), 0)
        canvas = m._historical_canvas(crop)
        self.assertEqual(canvas.mode, "L")
        self.assertEqual(canvas.size, (64, 64))
        # 20x30 input is centered unchanged: its top-left starts at (22, 17).
        self.assertEqual(canvas.crop((22, 17, 42, 47)).size, (20, 30))
        self.assertEqual(canvas.getpixel((32, 32)), 0)

    def test_safety_surface_remains_closed(self):
        self.assertFalse(m.production_promotion_allowed())
        self.assertFalse(m.validation_opened_by_this_module())
        self.assertTrue(m.final_holdout_locked())
        self.assertEqual(m.DIAGNOSTIC_SEED_TOTAL, 30)
        self.assertEqual(m.ADAPTATION_TRAIN_TOTAL, 270)
        self.assertEqual(
            m.HUMAN_QA_CONFIRMATION,
            "V5_2A_300_CONTACT_SHEETS_15_OF_15_PASS",
        )


if __name__ == "__main__":
    unittest.main()
