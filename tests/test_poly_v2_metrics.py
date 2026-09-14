from __future__ import annotations

import unittest

from st_omr_training.poly_2d_inference import (
    Poly2DInferenceIdentity,
    Poly2DInferenceResult,
    Poly2DInferenceStatus,
)
from st_omr_training.poly_evaluation_contract import (
    BenchmarkIdentity,
    BenchmarkSampleDescriptor,
    PolyphonicComplexityProfile,
    RobustnessBucket,
)
from st_omr_training.poly_v2_metrics import (
    MetricAvailability,
    POLY_V2_METRIC_ADAPTER_VERSION,
    PolyV2MetricError,
    UNSUPPORTED_METRIC_REASONS,
    evaluate_poly_v2_validation_sample,
)
from st_omr_training.polyphonic_representation import (
    Barline,
    BarlineLocation,
    BarlineStyle,
    BeamMark,
    BeamState,
    ClefAssignment,
    DisplayAccidentalV2,
    EventKind,
    ExactRational,
    KeySignature,
    NoteAtom,
    NoteType,
    PitchSpelling,
    PolyEvent,
    PolyMeasure,
    PolyPart,
    PolyScore,
    StemDirection,
    TieState,
    TimeSignature,
)
from st_omr_training.polyphonic_serialization import (
    BOS_TOKEN_ID,
    PAD_TOKEN_ID,
    tokenize_polyphonic_score,
    tokenizer_fingerprint,
)


def _score(
    *,
    step: str = "C",
    onset: tuple[int, int] = (0, 1),
    voice: int = 1,
    stem: StemDirection = StemDirection.UP,
    beam_state: BeamState = BeamState.BEGIN,
    ties: tuple[TieState, ...] = (TieState.START,),
    accidental: DisplayAccidentalV2 = DisplayAccidentalV2.SHARP,
    staff_override: int | None = 2,
) -> PolyScore:
    note = PolyEvent(
        event_id="e1",
        kind=EventKind.NOTE,
        onset=ExactRational(*onset),
        duration=ExactRational(1, 4),
        voice=voice,
        staff=1,
        note_type=NoteType.QUARTER,
        noteheads=(
            NoteAtom(
                atom_id="a1",
                pitch=PitchSpelling(
                    step=step,
                    alter=1 if accidental is DisplayAccidentalV2.SHARP else 0,
                    octave=4,
                    display_accidental=accidental,
                ),
                ties=ties,
                staff_override=staff_override,
            ),
        ),
        stem=stem,
        beams=(BeamMark(level=1, state=beam_state),),
    )
    measure = PolyMeasure(
        measure_index=1,
        source_number="1",
        time_signature=TimeSignature((4,), 4),
        key_signature=KeySignature(0, "major"),
        clefs=(
            ClefAssignment(staff=1, sign="G", line=2),
            ClefAssignment(staff=2, sign="F", line=4),
        ),
        events=(note,),
        barlines=(Barline(BarlineLocation.RIGHT, BarlineStyle.REGULAR),),
    )
    return PolyScore((PolyPart("P1", 2, (measure,)),))


def _identity() -> Poly2DInferenceIdentity:
    return Poly2DInferenceIdentity(
        model_state_sha256="1" * 64,
        model_profile_sha256="2" * 64,
        inference_profile_sha256="3" * 64,
        tokenizer_fingerprint_sha256=tokenizer_fingerprint(),
        max_decode_steps=1024,
    )


def _valid_result(score: PolyScore) -> Poly2DInferenceResult:
    target = tokenize_polyphonic_score(score)
    return Poly2DInferenceResult(
        token_ids=target.token_ids,
        status=Poly2DInferenceStatus.EOS_VALID,
        identity=_identity(),
        semantic_valid=True,
        prediction=score,
        prediction_sha256=score.canonical_sha256(),
        error_code=None,
    )


def _invalid_result() -> Poly2DInferenceResult:
    return Poly2DInferenceResult(
        token_ids=(BOS_TOKEN_ID, PAD_TOKEN_ID),
        status=Poly2DInferenceStatus.INVALID_CONTROL_TOKEN,
        identity=_identity(),
        semantic_valid=False,
        prediction=None,
        prediction_sha256=None,
        error_code="generated_pad",
    )


def _benchmark(*, suffix: str = "a") -> BenchmarkIdentity:
    return BenchmarkIdentity(
        benchmark_id="native-v2-validation",
        benchmark_version=f"v1-{suffix}",
        dataset_manifest_sha256="4" * 64,
        split_manifest_sha256="5" * 64,
    )


def _descriptor(*, split: str = "validation") -> BenchmarkSampleDescriptor:
    return BenchmarkSampleDescriptor(
        sample_id="6" * 64,
        family_id="family-1",
        split=split,
        complexity=PolyphonicComplexityProfile(
            voice_count=1,
            staff_count=2,
            simultaneous_note_density=0.0,
            chord_density=0.0,
            overlap_density=0.0,
            tie_density=1.0,
            beam_complexity=1.0,
            rhythmic_complexity=0.25,
            tuplet_present=False,
            grace_present=False,
            cross_staff_present=True,
        ),
        robustness_bucket=RobustnessBucket.CLEAN,
    )


class PolyV2MetricAdapterTests(unittest.TestCase):
    def test_exact_prediction_scores_all_exactly_representable_metrics_perfectly(self) -> None:
        score = _score()
        report = evaluate_poly_v2_validation_sample(
            reference=score,
            prediction=_valid_result(score),
            benchmark=_benchmark(),
            descriptor=_descriptor(),
        )
        self.assertEqual(report.adapter_version, POLY_V2_METRIC_ADAPTER_VERSION)
        self.assertEqual(report.metric("parse_success").value, 1.0)
        self.assertEqual(report.metric("ter").value, 0.0)
        self.assertEqual(report.metric("normalized_edit_distance").value, 0.0)
        self.assertEqual(report.metric("exact_sequence_accuracy").value, 1.0)
        for metric_id in (
            "pitch_accuracy",
            "duration_accuracy",
            "onset_accuracy",
            "voice_accuracy",
            "staff_accuracy",
            "accidental_note_f1",
            "note_staff_f1",
        ):
            self.assertEqual(report.metric(metric_id).value, 1.0, metric_id)

    def test_unadmitted_or_unrepresentable_metrics_are_explicitly_unsupported(self) -> None:
        score = _score()
        report = evaluate_poly_v2_validation_sample(
            reference=score,
            prediction=_valid_result(score),
            benchmark=_benchmark(),
            descriptor=_descriptor(),
        )
        expected = (
            "musicxml_validity",
            "tedn",
            "notehead_stem_f1",
            "beam_relation_f1",
            "tie_relation_f1",
        )
        self.assertEqual(report.unsupported_metric_ids, expected)
        for metric_id in expected:
            observation = report.metric(metric_id)
            self.assertEqual(observation.availability, MetricAvailability.UNSUPPORTED)
            self.assertIsNone(observation.value)
            self.assertEqual(observation.unsupported_reason, UNSUPPORTED_METRIC_REASONS[metric_id])
        self.assertFalse(report.full_contract_ready)
        with self.assertRaisesRegex(PolyV2MetricError, "unsupported metrics"):
            report.require_complete_required_metrics()

    def test_field_specific_semantic_errors_remain_visible(self) -> None:
        reference = _score()
        prediction_score = _score(step="D", onset=(1, 4), voice=2)
        report = evaluate_poly_v2_validation_sample(
            reference=reference,
            prediction=_valid_result(prediction_score),
            benchmark=_benchmark(),
            descriptor=_descriptor(),
        )
        self.assertEqual(report.metric("pitch_accuracy").value, 0.0)
        self.assertEqual(report.metric("onset_accuracy").value, 0.0)
        self.assertEqual(report.metric("voice_accuracy").value, 0.0)
        self.assertEqual(report.metric("duration_accuracy").value, 1.0)
        self.assertEqual(report.metric("staff_accuracy").value, 1.0)
        self.assertGreater(report.metric("ter").value, 0.0)
        self.assertEqual(report.metric("exact_sequence_accuracy").value, 0.0)

    def test_explicit_accidental_and_staff_relations_are_scored(self) -> None:
        reference = _score()
        prediction_score = _score(
            accidental=DisplayAccidentalV2.NONE,
            staff_override=1,
        )
        report = evaluate_poly_v2_validation_sample(
            reference=reference,
            prediction=_valid_result(prediction_score),
            benchmark=_benchmark(),
            descriptor=_descriptor(),
        )
        self.assertEqual(report.metric("accidental_note_f1").value, 0.0)
        self.assertEqual(report.metric("note_staff_f1").value, 0.0)

    def test_stem_beam_and_tie_state_do_not_create_proxy_relation_metrics(self) -> None:
        reference = _score()
        prediction_score = _score(
            stem=StemDirection.DOWN,
            beam_state=BeamState.END,
            ties=(),
        )
        report = evaluate_poly_v2_validation_sample(
            reference=reference,
            prediction=_valid_result(prediction_score),
            benchmark=_benchmark(),
            descriptor=_descriptor(),
        )
        for metric_id in ("notehead_stem_f1", "beam_relation_f1", "tie_relation_f1"):
            self.assertEqual(report.metric(metric_id).availability, MetricAvailability.UNSUPPORTED)
            self.assertIsNone(report.metric(metric_id).value)

    def test_invalid_free_running_output_is_not_excluded_from_available_metrics(self) -> None:
        reference = _score()
        report = evaluate_poly_v2_validation_sample(
            reference=reference,
            prediction=_invalid_result(),
            benchmark=_benchmark(),
            descriptor=_descriptor(),
        )
        self.assertEqual(report.metric("parse_success").value, 0.0)
        self.assertGreater(report.metric("ter").value, 0.0)
        for metric_id in (
            "pitch_accuracy",
            "duration_accuracy",
            "onset_accuracy",
            "voice_accuracy",
            "staff_accuracy",
            "accidental_note_f1",
            "note_staff_f1",
        ):
            self.assertEqual(report.metric(metric_id).value, 0.0, metric_id)
        for metric_id in ("notehead_stem_f1", "beam_relation_f1", "tie_relation_f1"):
            self.assertEqual(report.metric(metric_id).availability, MetricAvailability.UNSUPPORTED)

    def test_report_identity_is_deterministic_and_benchmark_bound(self) -> None:
        score = _score()
        kwargs = {
            "reference": score,
            "prediction": _valid_result(score),
            "descriptor": _descriptor(),
        }
        first = evaluate_poly_v2_validation_sample(benchmark=_benchmark(suffix="a"), **kwargs)
        second = evaluate_poly_v2_validation_sample(benchmark=_benchmark(suffix="a"), **kwargs)
        changed = evaluate_poly_v2_validation_sample(benchmark=_benchmark(suffix="b"), **kwargs)
        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertNotEqual(first.benchmark_identity_sha256, changed.benchmark_identity_sha256)
        self.assertNotEqual(first.fingerprint(), changed.fingerprint())

    def test_only_validation_descriptor_is_admitted(self) -> None:
        score = _score()
        for split in ("train", "test"):
            with self.subTest(split=split):
                with self.assertRaisesRegex(PolyV2MetricError, "VALIDATION"):
                    evaluate_poly_v2_validation_sample(
                        reference=score,
                        prediction=_valid_result(score),
                        benchmark=_benchmark(),
                        descriptor=_descriptor(split=split),
                    )


if __name__ == "__main__":
    unittest.main()
