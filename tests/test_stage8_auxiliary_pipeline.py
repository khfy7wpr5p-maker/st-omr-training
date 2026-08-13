from __future__ import annotations

import unittest

from st_omr_training.stage8_auxiliary_pipeline import (
    Stage8AuxiliaryPipelineError,
    auxiliary_pipeline_policy_fingerprint,
    prepare_guarded_primus_v1_target,
)


def _mei(*, staff_n: str = "1", lines: str = "5", key_sig: str = "0") -> bytes:
    events = ''.join('<note dur="4" oct="4" pname="c"/>' for _ in range(4))
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="4.0.0">
  <music><body><mdiv><score>
    <scoreDef key.sig="{key_sig}" meter.count="4" meter.unit="4">
      <staffGrp><staffDef n="{staff_n}" lines="{lines}" clef.shape="G" clef.line="2"/></staffGrp>
    </scoreDef>
    <section><measure n="1"><staff n="1"><layer n="1">{events}</layer></staff></measure></section>
  </score></mdiv></body></music>
</mei>'''.encode("utf-8")


SEMANTIC = (
    b"clef-G2 keySignature-CM timeSignature-4/4 "
    b"note-C4_quarter note-C4_quarter note-C4_quarter note-C4_quarter"
)


class Stage8AuxiliaryPipelineTests(unittest.TestCase):
    def test_exact_v1_shape_produces_hash_bound_target(self) -> None:
        musicxml, evidence = prepare_guarded_primus_v1_target(
            mei_bytes=_mei(),
            semantic_bytes=SEMANTIC,
        )
        self.assertTrue(musicxml.startswith(b"<?xml"))
        self.assertEqual(evidence.measure_count, 1)
        self.assertEqual(evidence.note_count, 4)
        self.assertEqual(
            evidence.pipeline_policy_fingerprint,
            auxiliary_pipeline_policy_fingerprint(),
        )

    def test_nonfive_line_staff_fails_closed(self) -> None:
        with self.assertRaises(Stage8AuxiliaryPipelineError):
            prepare_guarded_primus_v1_target(
                mei_bytes=_mei(lines="4"),
                semantic_bytes=SEMANTIC,
            )

    def test_wrong_staff_identity_fails_closed(self) -> None:
        with self.assertRaises(Stage8AuxiliaryPipelineError):
            prepare_guarded_primus_v1_target(
                mei_bytes=_mei(staff_n="2"),
                semantic_bytes=SEMANTIC,
            )

    def test_nonzero_key_signature_fails_closed(self) -> None:
        with self.assertRaises(Stage8AuxiliaryPipelineError):
            prepare_guarded_primus_v1_target(
                mei_bytes=_mei(key_sig="1s"),
                semantic_bytes=SEMANTIC,
            )

    def test_missing_explicit_key_signature_fails_closed(self) -> None:
        mei = _mei().replace(b' key.sig="0"', b'', 1)
        with self.assertRaises(Stage8AuxiliaryPipelineError):
            prepare_guarded_primus_v1_target(
                mei_bytes=mei,
                semantic_bytes=SEMANTIC,
            )


if __name__ == "__main__":
    unittest.main()
