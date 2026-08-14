from __future__ import annotations

from fractions import Fraction
import unittest

from st_omr_training.core import DisplayAccidental
from st_omr_training.musicxml_roundtrip import (
    SemanticEventProjection,
    SemanticMeasureProjection,
    SemanticPartProjection,
    SemanticPitchProjection,
    SemanticScoreProjection,
    SemanticVoiceProjection,
)
from st_omr_training.training_tokens import encode_tokens, tokenize_projection
from st_omr_training.validation_diagnostics import (
    ValidationDiagnosticError,
    analyze_validation_sample,
    build_validation_diagnostic_report,
)


def _pitch(step: str, octave: int = 4) -> SemanticPitchProjection:
    return SemanticPitchProjection(
        step=step,
        alter=0,
        octave=octave,
        display_accidental=DisplayAccidental.NONE,
    )


def _projection(events: tuple[SemanticEventProjection, ...]) -> SemanticScoreProjection:
    return SemanticScoreProjection(
        parts=(
            SemanticPartProjection(
                part_id="P1",
                staff_count=1,
                measures=(
                    SemanticMeasureProjection(
                        number=1,
                        time_signature=(4, 4),
                        key_signature=0,
                        clef="treble",
                        voices=(SemanticVoiceProjection(voice_id=1, events=events),),
                    ),
                ),
            ),
        ),
    )


def _ids(projection: SemanticScoreProjection) -> tuple[int, ...]:
    return encode_tokens(tokenize_projection(projection))


class ValidationDiagnosticsTests(unittest.TestCase):
    def test_exact_sample_reports_perfect_semantic_metrics(self):
        target = _projection(
            (
                SemanticEventProjection("note", Fraction(0, 1), Fraction(1, 4), 1, (_pitch("C"),)),
                SemanticEventProjection("rest", Fraction(1, 4), Fraction(1, 4), 1, ()),
                SemanticEventProjection(
                    "chord",
                    Fraction(1, 2),
                    Fraction(1, 2),
                    1,
                    (_pitch("C"), _pitch("E")),
                ),
            )
        )
        token_ids = _ids(target)
        sample = analyze_validation_sample(
            sample_id="1" * 64,
            family_id="family-1",
            target_token_ids=token_ids,
            predicted_token_ids=token_ids,
            extra_feature_tags=("degradation:clean",),
        )
        self.assertTrue(sample.exact_sequence)
        self.assertEqual(sample.token_edits, 0)
        self.assertEqual(sample.exact_measures, 1)
        self.assertEqual(sample.meter_correct, 1)
        self.assertEqual(sample.event_edits, 0)
        self.assertEqual(sample.event_type_correct, 3)
        self.assertEqual(sample.duration_correct, 3)
        self.assertEqual(sample.pitch_identity_correct, 2)
        self.assertEqual(sample.rest_recognition_correct, 1)
        self.assertEqual(sample.chord_size_correct, 1)
        self.assertIn("degradation:clean", sample.feature_tags)
        self.assertIn("event:note", sample.feature_tags)
        self.assertIn("event:rest", sample.feature_tags)
        self.assertIn("event:chord", sample.feature_tags)

    def test_diagnostic_localizes_pitch_rest_and_chord_size_errors(self):
        target = _projection(
            (
                SemanticEventProjection("note", Fraction(0, 1), Fraction(1, 4), 1, (_pitch("C"),)),
                SemanticEventProjection("rest", Fraction(1, 4), Fraction(1, 4), 1, ()),
                SemanticEventProjection(
                    "chord",
                    Fraction(1, 2),
                    Fraction(1, 2),
                    1,
                    (_pitch("C"), _pitch("E")),
                ),
            )
        )
        predicted = _projection(
            (
                SemanticEventProjection("note", Fraction(0, 1), Fraction(1, 4), 1, (_pitch("D"),)),
                SemanticEventProjection("note", Fraction(1, 4), Fraction(1, 4), 1, (_pitch("E"),)),
                SemanticEventProjection(
                    "chord",
                    Fraction(1, 2),
                    Fraction(1, 2),
                    1,
                    (_pitch("C"), _pitch("E"), _pitch("G")),
                ),
            )
        )
        sample = analyze_validation_sample(
            sample_id="2" * 64,
            family_id="family-2",
            target_token_ids=_ids(target),
            predicted_token_ids=_ids(predicted),
        )
        self.assertFalse(sample.exact_sequence)
        self.assertEqual(sample.meter_correct, 1)
        self.assertEqual(sample.event_type_correct, 2)
        self.assertEqual(sample.duration_correct, 3)
        self.assertEqual(sample.rest_events, 1)
        self.assertEqual(sample.rest_recognition_correct, 0)
        self.assertEqual(sample.pitched_events, 2)
        self.assertEqual(sample.pitch_identity_correct, 0)
        self.assertEqual(sample.chord_events, 1)
        self.assertEqual(sample.chord_size_correct, 0)

    def test_report_is_canonical_order_independent_and_feature_bucketed(self):
        target = _projection(
            (
                SemanticEventProjection("note", Fraction(0, 1), Fraction(1, 2), 1, (_pitch("C"),)),
                SemanticEventProjection("note", Fraction(1, 2), Fraction(1, 2), 1, (_pitch("D"),)),
            )
        )
        ids = _ids(target)
        first = analyze_validation_sample(
            sample_id="a" * 64,
            family_id="family-a",
            target_token_ids=ids,
            predicted_token_ids=ids,
            extra_feature_tags=("degradation:light",),
        )
        second = analyze_validation_sample(
            sample_id="b" * 64,
            family_id="family-b",
            target_token_ids=ids,
            predicted_token_ids=ids,
            extra_feature_tags=("degradation:medium",),
        )
        report_a, raw_a, digest_a = build_validation_diagnostic_report(
            (second, first),
            repository_sha="c" * 40,
            checkpoint_sha256="d" * 64,
            checkpoint_state_sha256="e" * 64,
            source_run_id="f" * 64,
        )
        report_b, raw_b, digest_b = build_validation_diagnostic_report(
            (first, second),
            repository_sha="c" * 40,
            checkpoint_sha256="d" * 64,
            checkpoint_state_sha256="e" * 64,
            source_run_id="f" * 64,
        )
        self.assertEqual(raw_a, raw_b)
        self.assertEqual(digest_a, digest_b)
        self.assertEqual(report_a, report_b)
        self.assertEqual(report_a["aggregate"]["exact_sequence_accuracy"], 1.0)
        self.assertEqual(report_a["test_samples_exposed"], 0)
        self.assertEqual(report_a["feature_buckets"]["degradation:light"]["samples"], 1)
        self.assertEqual(report_a["feature_buckets"]["degradation:medium"]["samples"], 1)

    def test_duplicate_sample_ids_fail_closed(self):
        target = _projection(
            (
                SemanticEventProjection("note", Fraction(0, 1), Fraction(1, 1), 1, (_pitch("C"),)),
            )
        )
        ids = _ids(target)
        sample = analyze_validation_sample(
            sample_id="3" * 64,
            family_id="family-3",
            target_token_ids=ids,
            predicted_token_ids=ids,
        )
        with self.assertRaises(ValidationDiagnosticError):
            build_validation_diagnostic_report(
                (sample, sample),
                repository_sha="4" * 40,
                checkpoint_sha256="5" * 64,
                checkpoint_state_sha256="6" * 64,
                source_run_id="7" * 64,
            )

    def test_invalid_token_id_fails_closed(self):
        target = _projection(
            (
                SemanticEventProjection("note", Fraction(0, 1), Fraction(1, 1), 1, (_pitch("C"),)),
            )
        )
        with self.assertRaises(ValidationDiagnosticError):
            analyze_validation_sample(
                sample_id="8" * 64,
                family_id="family-8",
                target_token_ids=_ids(target),
                predicted_token_ids=(1, 999, 2),
            )


if __name__ == "__main__":
    unittest.main()
