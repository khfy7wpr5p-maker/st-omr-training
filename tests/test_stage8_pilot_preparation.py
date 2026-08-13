from __future__ import annotations

from io import BytesIO
import unittest

from PIL import Image, PngImagePlugin

from st_omr_training.stage8_pilot_preparation import (
    Stage8PilotPreparationError,
    inspect_primus_auxiliary_package,
    pilot_preparation_policy_fingerprint,
    prepare_training_png,
)


def _png_bytes(mode: str = "P", *, transparency: bool = False, metadata: bool = False) -> bytes:
    if mode == "P":
        image = Image.new("P", (12, 8), 0)
        palette = []
        for index in range(256):
            palette.extend((index, index, index))
        image.putpalette(palette)
        for x in range(12):
            for y in range(8):
                image.putpixel((x, y), 0 if (x + y) % 3 else 255)
    elif mode == "1":
        image = Image.new("1", (12, 8), 1)
        for x in range(2, 10):
            image.putpixel((x, 4), 0)
    elif mode == "RGBA":
        image = Image.new("RGBA", (12, 8), (255, 255, 255, 255))
    elif mode == "RGB":
        image = Image.new("RGB", (12, 8), "white")
        for x in range(2, 10):
            image.putpixel((x, 4), (0, 0, 0))
    else:
        image = Image.new(mode, (12, 8), 255)
        for x in range(2, 10):
            image.putpixel((x, 4), 0)

    info = PngImagePlugin.PngInfo() if metadata else None
    if info is not None:
        info.add_text("private-test-metadata", "must-not-survive")
    output = BytesIO()
    kwargs = {"pnginfo": info} if info is not None else {}
    if transparency:
        kwargs["transparency"] = 0
    image.save(output, format="PNG", **kwargs)
    return output.getvalue()


def _mei(
    *,
    clef_shape: str = "G",
    clef_line: str = "2",
    key_sig: str = "0",
    meter_count: str = "4",
    meter_unit: str = "4",
    body: str | None = None,
) -> bytes:
    events = body or """
      <note dur="4" oct="4" pname="c"/>
      <note dur="4" oct="4" pname="d"/>
      <note dur="4" oct="4" pname="e"/>
      <note dur="4" oct="4" pname="f"/>
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="4.0.0">
  <music><body><mdiv><score>
    <scoreDef key.sig="{key_sig}" meter.count="{meter_count}" meter.unit="{meter_unit}">
      <staffGrp><staffDef n="1" lines="5" clef.shape="{clef_shape}" clef.line="{clef_line}"/></staffGrp>
    </scoreDef>
    <section>
      <measure><staff n="1"><layer n="1">{events}</layer></staff></measure>
    </section>
  </score></mdiv></body></music>
</mei>""".encode("utf-8")


class Stage8PilotPreparationTests(unittest.TestCase):
    def test_policy_fingerprint_is_deterministic(self) -> None:
        first = pilot_preparation_policy_fingerprint()
        second = pilot_preparation_policy_fingerprint()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_prepare_training_png_is_deterministic_and_preserves_geometry(self) -> None:
        source = _png_bytes("P", metadata=True)
        first, first_evidence = prepare_training_png(source)
        second, second_evidence = prepare_training_png(source)
        self.assertEqual(first, second)
        self.assertEqual(first_evidence, second_evidence)
        self.assertEqual((first_evidence.width, first_evidence.height), (12, 8))
        self.assertEqual(first_evidence.source_mode, "P")

        with Image.open(BytesIO(first)) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.mode, "L")
            self.assertEqual(image.size, (12, 8))
            self.assertEqual(getattr(image, "n_frames", 1), 1)
            self.assertNotIn("private-test-metadata", image.info)

    def test_prepare_training_png_accepts_only_monochrome_palette_or_l(self) -> None:
        for mode in ("1", "L", "P"):
            with self.subTest(mode=mode):
                derived, evidence = prepare_training_png(_png_bytes(mode))
                self.assertEqual(evidence.source_mode, mode)
                with Image.open(BytesIO(derived)) as image:
                    self.assertEqual(image.mode, "L")
                    self.assertEqual(image.size, (12, 8))
        with self.assertRaises(Stage8PilotPreparationError):
            prepare_training_png(_png_bytes("RGB"))

    def test_prepare_training_png_rejects_non_png_alpha_and_transparency(self) -> None:
        with self.assertRaises(Stage8PilotPreparationError):
            prepare_training_png(b"not-a-png")
        with self.assertRaises(Stage8PilotPreparationError):
            prepare_training_png(_png_bytes("RGBA"))
        with self.assertRaises(Stage8PilotPreparationError):
            prepare_training_png(_png_bytes("P", transparency=True))

    def test_coherent_supported_auxiliary_package_is_only_triage_eligible(self) -> None:
        semantic = (
            b"clef-G2\tkeySignature-CM\ttimeSignature-4/4\t"
            b"note-C4_quarter\tnote-D4_quarter\tnote-E4_quarter\tnote-F4_quarter\tbarline\t"
        )
        inspection = inspect_primus_auxiliary_package(
            mei_bytes=_mei(),
            semantic_bytes=semantic,
            agnostic_bytes=b"clef.G-L2\tnote.quarter-L3\tbarline\t",
        )
        self.assertTrue(inspection.metadata_coherent)
        self.assertTrue(inspection.v1_eligible)
        self.assertEqual(inspection.v1_rejection_reasons, ())
        self.assertEqual(inspection.clef, "G2")
        self.assertEqual(inspection.key_signature_fifths, 0)
        self.assertEqual(inspection.meter, (4, 4))

    def test_auxiliary_header_mismatch_fails_v1_triage(self) -> None:
        inspection = inspect_primus_auxiliary_package(
            mei_bytes=_mei(),
            semantic_bytes=b"clef-C1\tkeySignature-CM\ttimeSignature-4/4\tnote-C4_quarter\t",
            agnostic_bytes=b"clef.C-L1\tnote.quarter-L3\t",
        )
        self.assertFalse(inspection.metadata_coherent)
        self.assertFalse(inspection.v1_eligible)
        self.assertIn("mei_semantic_header_mismatch", inspection.v1_rejection_reasons)

    def test_outside_v1_features_are_explicitly_rejected(self) -> None:
        body = """
          <multiRest num="4"/>
          <beam><note dur="16" oct="4" pname="c"/></beam>
        """
        inspection = inspect_primus_auxiliary_package(
            mei_bytes=_mei(
                clef_shape="C",
                clef_line="1",
                key_sig="3f",
                meter_count="6",
                meter_unit="8",
                body=body,
            ),
            semantic_bytes=(
                b"clef-C1\tkeySignature-EbM\ttimeSignature-6/8\t"
                b"multirest-4\tnote-C4_sixteenth\t"
            ),
            agnostic_bytes=b"clef.C-L1\tmultirest\tnote.sixteenth-L3\t",
        )
        self.assertTrue(inspection.metadata_coherent)
        self.assertFalse(inspection.v1_eligible)
        for reason in (
            "unsupported_clef",
            "unsupported_key_signature",
            "unsupported_meter_or_meter_symbol",
            "explicit_beam_notation_deferred",
            "unsupported_mei_structure",
            "unsupported_note_duration",
            "semantic_duration_outside_v1",
        ):
            self.assertIn(reason, inspection.v1_rejection_reasons)

    def test_empty_mei_score_and_unsupported_accidental_fail_triage(self) -> None:
        empty = _mei(body="<space dur=\"4\"/>")
        inspection = inspect_primus_auxiliary_package(
            mei_bytes=empty,
            semantic_bytes=b"clef-G2\tkeySignature-CM\ttimeSignature-4/4\t",
            agnostic_bytes=b"clef.G-L2\t",
        )
        self.assertFalse(inspection.v1_eligible)
        self.assertIn("empty_mei_event_stream", inspection.v1_rejection_reasons)
        self.assertIn("unsupported_mei_structure", inspection.v1_rejection_reasons)

        accidental = _mei(body='<note dur="4" oct="4" pname="c"><accid accid="ss"/></note>')
        inspection = inspect_primus_auxiliary_package(
            mei_bytes=accidental,
            semantic_bytes=b"clef-G2\tkeySignature-CM\ttimeSignature-4/4\tnote-C#4_quarter\t",
            agnostic_bytes=b"clef.G-L2\taccidental.sharp\tnote.quarter-L3\t",
        )
        self.assertFalse(inspection.v1_eligible)
        self.assertIn("unsupported_accidental", inspection.v1_rejection_reasons)

    def test_auxiliary_files_must_be_nonempty_and_mei_well_formed(self) -> None:
        with self.assertRaises(Stage8PilotPreparationError):
            inspect_primus_auxiliary_package(
                mei_bytes=b"<mei>",
                semantic_bytes=b"x",
                agnostic_bytes=b"y",
            )
        with self.assertRaises(Stage8PilotPreparationError):
            inspect_primus_auxiliary_package(
                mei_bytes=_mei(),
                semantic_bytes=b"",
                agnostic_bytes=b"y",
            )


if __name__ == "__main__":
    unittest.main()
