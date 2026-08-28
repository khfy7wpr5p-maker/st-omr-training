from __future__ import annotations

import unittest

from st_omr_training.poly_evaluation_contract import (
    BenchmarkIdentity,
    BenchmarkSampleDescriptor,
    ErrorClass,
    ErrorCount,
    PolyEvaluationContractError,
    PolyphonicComplexityProfile,
    REQUIRED_ERROR_CLASSES,
    REQUIRED_ROBUSTNESS_BUCKETS,
    REQUIRED_VOICE_STRATA,
    RobustnessBucket,
    VoiceStratum,
    required_metric_ids,
    validate_comparison_benchmark,
    validate_error_counts,
    validate_required_metric_result,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _complexity(voice_count: int = 2) -> PolyphonicComplexityProfile:
    return PolyphonicComplexityProfile(
        voice_count=voice_count,
        staff_count=2,
        simultaneous_note_density=0.4,
        chord_density=0.3,
        overlap_density=0.2,
        tie_density=0.1,
        beam_complexity=0.5,
        rhythmic_complexity=0.6,
        tuplet_present=True,
        grace_present=False,
        cross_staff_present=False,
    )


class PolyEvaluationContractTests(unittest.TestCase):
    def test_error_taxonomy_is_exact_and_versioned_surface_is_complete(self) -> None:
        self.assertEqual(
            tuple(item.value for item in REQUIRED_ERROR_CLASSES),
            (
                "PITCH",
                "DURATION",
                "ONSET",
                "VOICE",
                "STAFF",
                "REST",
                "ACCIDENTAL",
                "TIE",
                "SLUR",
                "TUPLET",
                "BEAM",
                "STEM",
                "CHORD_GROUPING",
                "CROSS_STAFF",
                "METER",
                "MEASURE_BOUNDARY",
                "GRACE",
                "ORNAMENT",
                "OTHER",
                "AMBIGUOUS",
            ),
        )

    def test_required_metric_family_matches_tr_poly_02_contract(self) -> None:
        self.assertEqual(
            required_metric_ids(),
            (
                "parse_success",
                "musicxml_validity",
                "ter",
                "normalized_edit_distance",
                "exact_sequence_accuracy",
                "tedn",
                "pitch_accuracy",
                "duration_accuracy",
                "onset_accuracy",
                "voice_accuracy",
                "staff_accuracy",
                "notehead_stem_f1",
                "beam_relation_f1",
                "tie_relation_f1",
                "accidental_note_f1",
                "note_staff_f1",
            ),
        )

    def test_complexity_maps_to_frozen_voice_strata(self) -> None:
        self.assertEqual(_complexity(1).voice_stratum, VoiceStratum.VOICE_1)
        self.assertEqual(_complexity(2).voice_stratum, VoiceStratum.VOICE_2)
        self.assertEqual(_complexity(3).voice_stratum, VoiceStratum.VOICE_3)
        self.assertEqual(_complexity(4).voice_stratum, VoiceStratum.VOICE_4_PLUS)
        self.assertEqual(_complexity(8).voice_stratum, VoiceStratum.VOICE_4_PLUS)
        self.assertEqual(
            tuple(item.value for item in REQUIRED_VOICE_STRATA),
            ("1_voice", "2_voice", "3_voice", "4_plus_voice"),
        )

    def test_complexity_rejects_invalid_density_and_voice_count(self) -> None:
        with self.assertRaises(PolyEvaluationContractError):
            PolyphonicComplexityProfile(
                voice_count=0,
                staff_count=1,
                simultaneous_note_density=0.0,
                chord_density=0.0,
                overlap_density=0.0,
                tie_density=0.0,
                beam_complexity=0.0,
                rhythmic_complexity=0.0,
                tuplet_present=False,
                grace_present=False,
                cross_staff_present=False,
            )
        with self.assertRaises(PolyEvaluationContractError):
            PolyphonicComplexityProfile(
                voice_count=1,
                staff_count=1,
                simultaneous_note_density=1.1,
                chord_density=0.0,
                overlap_density=0.0,
                tie_density=0.0,
                beam_complexity=0.0,
                rhythmic_complexity=0.0,
                tuplet_present=False,
                grace_present=False,
                cross_staff_present=False,
            )

    def test_benchmark_identity_hash_is_deterministic(self) -> None:
        identity = BenchmarkIdentity(
            benchmark_id="teacher-gold-poly-v1",
            benchmark_version="1",
            dataset_manifest_sha256=SHA_A,
            split_manifest_sha256=SHA_B,
        )
        self.assertEqual(identity.canonical_sha256(), identity.canonical_sha256())
        self.assertEqual(len(identity.canonical_sha256()), 64)

    def test_comparison_requires_identical_benchmark_identity(self) -> None:
        first = BenchmarkIdentity(
            benchmark_id="teacher-gold-poly-v1",
            benchmark_version="1",
            dataset_manifest_sha256=SHA_A,
            split_manifest_sha256=SHA_B,
        )
        same = BenchmarkIdentity(
            benchmark_id="teacher-gold-poly-v1",
            benchmark_version="1",
            dataset_manifest_sha256=SHA_A,
            split_manifest_sha256=SHA_B,
        )
        changed = BenchmarkIdentity(
            benchmark_id="teacher-gold-poly-v1",
            benchmark_version="1",
            dataset_manifest_sha256=SHA_A,
            split_manifest_sha256=SHA_C,
        )
        self.assertIs(validate_comparison_benchmark((first, same)), first)
        with self.assertRaises(PolyEvaluationContractError):
            validate_comparison_benchmark((first, changed))

    def test_sample_descriptor_keeps_split_complexity_and_robustness(self) -> None:
        descriptor = BenchmarkSampleDescriptor(
            sample_id=SHA_C,
            family_id="family-001",
            split="validation",
            complexity=_complexity(3),
            robustness_bucket=RobustnessBucket.PHONE,
        )
        self.assertEqual(descriptor.complexity.voice_stratum, VoiceStratum.VOICE_3)
        self.assertEqual(descriptor.robustness_bucket, RobustnessBucket.PHONE)
        self.assertEqual(
            tuple(item.value for item in REQUIRED_ROBUSTNESS_BUCKETS),
            ("clean", "scan", "phone", "blur", "perspective", "low_contrast"),
        )

    def test_error_counts_are_unique_and_canonicalized(self) -> None:
        values = validate_error_counts(
            (
                ErrorCount(ErrorClass.VOICE, 3),
                ErrorCount(ErrorClass.PITCH, 2),
            )
        )
        self.assertEqual(
            tuple(item.error_class for item in values),
            (ErrorClass.PITCH, ErrorClass.VOICE),
        )
        with self.assertRaises(PolyEvaluationContractError):
            validate_error_counts(
                (
                    ErrorCount(ErrorClass.PITCH, 1),
                    ErrorCount(ErrorClass.PITCH, 2),
                )
            )

    def test_required_metric_validation_fails_closed(self) -> None:
        values = {
            metric_id: (0.5 if metric_id not in {"ter", "normalized_edit_distance", "tedn"} else 0.2)
            for metric_id in required_metric_ids()
        }
        validate_required_metric_result(values)

        missing = dict(values)
        missing.pop("voice_accuracy")
        with self.assertRaises(PolyEvaluationContractError):
            validate_required_metric_result(missing)

        unknown = dict(values)
        unknown["loss"] = 1.0
        with self.assertRaises(PolyEvaluationContractError):
            validate_required_metric_result(unknown)

        invalid = dict(values)
        invalid["pitch_accuracy"] = 1.1
        with self.assertRaises(PolyEvaluationContractError):
            validate_required_metric_result(invalid)

        invalid_distance = dict(values)
        invalid_distance["tedn"] = -0.1
        with self.assertRaises(PolyEvaluationContractError):
            validate_required_metric_result(invalid_distance)


if __name__ == "__main__":
    unittest.main()
