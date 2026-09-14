from __future__ import annotations

from hashlib import sha256
import unittest

from st_omr_training.poly_evaluation_contract import required_metric_ids
from st_omr_training.poly_v2_metrics import (
    MetricAvailability,
    MetricObservation,
    POLY_V2_EVENT_ALIGNMENT_VERSION,
    POLY_V2_METRIC_ADAPTER_VERSION,
    POLY_V2_RELATION_METRIC_VERSION,
    PolyV2SampleMetricReport,
    UNSUPPORTED_METRIC_REASONS,
)
from st_omr_training.poly_v2_validation_aggregation import (
    POLY_V2_AGGREGATION_POLICY,
    POLY_V2_VALIDATION_AGGREGATION_VERSION,
    PolyV2ValidationAggregationError,
    aggregate_poly_v2_validation_reports,
    validate_common_candidate_reports,
)


def _sha(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _observations(*, score: float, parse_success: float) -> tuple[MetricObservation, ...]:
    values: list[MetricObservation] = []
    for metric_id in required_metric_ids():
        if metric_id in UNSUPPORTED_METRIC_REASONS:
            values.append(
                MetricObservation(
                    metric_id=metric_id,
                    availability=MetricAvailability.UNSUPPORTED,
                    value=None,
                    unsupported_reason=UNSUPPORTED_METRIC_REASONS[metric_id],
                )
            )
            continue
        if metric_id == "parse_success":
            value = parse_success
        elif metric_id in {"ter", "normalized_edit_distance"}:
            value = 1.0 - score
        else:
            value = score
        values.append(
            MetricObservation(
                metric_id=metric_id,
                availability=MetricAvailability.AVAILABLE,
                value=value,
            )
        )
    return tuple(values)


def _sample(
    label: str,
    *,
    voice: str,
    bucket: str = "clean",
    score: float = 0.8,
    parse_success: float = 1.0,
    candidate: str = "candidate-a",
    benchmark: str = "benchmark-a",
    checkpoint_bound: bool = True,
) -> PolyV2SampleMetricReport:
    return PolyV2SampleMetricReport(
        sample_id=_sha(f"sample:{label}"),
        family_id=f"family-{label}",
        split="validation",
        benchmark_identity_sha256=_sha(benchmark),
        candidate_identity_sha256=_sha(candidate),
        checkpoint_bound=checkpoint_bound,
        inference_evidence_sha256=_sha(f"inference:{label}"),
        reference_representation_sha256=_sha(f"reference:{label}"),
        prediction_representation_sha256=(
            _sha(f"prediction:{label}") if parse_success == 1.0 else None
        ),
        voice_stratum=voice,
        robustness_bucket=bucket,
        observations=_observations(score=score, parse_success=parse_success),
        adapter_version=POLY_V2_METRIC_ADAPTER_VERSION,
        alignment_version=POLY_V2_EVENT_ALIGNMENT_VERSION,
        relation_metric_version=POLY_V2_RELATION_METRIC_VERSION,
    )


def _all_voice_reports() -> tuple[PolyV2SampleMetricReport, ...]:
    return (
        _sample("a", voice="1_voice", score=1.0, bucket="clean"),
        _sample("b", voice="2_voice", score=0.8, bucket="scan"),
        _sample("c", voice="3_voice", score=0.6, bucket="phone"),
        _sample("d", voice="4_plus_voice", score=0.4, bucket="blur"),
    )


class PolyV2ValidationAggregationTests(unittest.TestCase):
    def test_aggregates_overall_voice_and_robustness_slices(self) -> None:
        report = aggregate_poly_v2_validation_reports(_all_voice_reports())

        self.assertEqual(report.aggregation_version, POLY_V2_VALIDATION_AGGREGATION_VERSION)
        self.assertEqual(report.aggregation_policy, POLY_V2_AGGREGATION_POLICY)
        self.assertEqual(report.sample_count, 4)
        self.assertEqual(report.family_count, 4)
        self.assertEqual(report.missing_voice_strata, ())
        self.assertEqual(len(report.voice_strata), 4)
        self.assertEqual(len(report.robustness_buckets), 4)
        self.assertAlmostEqual(report.overall.metric("pitch_accuracy").mean_value, 0.7)
        self.assertAlmostEqual(report.overall.metric("ter").mean_value, 0.3)
        self.assertEqual(report.overall.parse_success_count, 4)
        self.assertEqual(report.overall.parse_success_rate, 1.0)

    def test_invalid_output_remains_in_macro_denominator(self) -> None:
        reports = (
            _sample("a", voice="1_voice", score=1.0, parse_success=1.0),
            _sample("b", voice="1_voice", score=0.0, parse_success=0.0),
        )
        report = aggregate_poly_v2_validation_reports(reports)
        self.assertEqual(report.overall.parse_success_count, 1)
        self.assertEqual(report.overall.parse_success_rate, 0.5)
        self.assertEqual(report.overall.metric("pitch_accuracy").mean_value, 0.5)
        self.assertEqual(report.overall.metric("exact_sequence_accuracy").mean_value, 0.5)

    def test_unsupported_metrics_remain_explicit_and_close_comparison_gate(self) -> None:
        report = aggregate_poly_v2_validation_reports(_all_voice_reports())
        self.assertEqual(
            set(report.unsupported_metric_ids),
            set(UNSUPPORTED_METRIC_REASONS),
        )
        self.assertFalse(report.full_metric_contract_ready)
        self.assertFalse(report.common_comparison_ready)
        with self.assertRaisesRegex(PolyV2ValidationAggregationError, "unsupported metrics"):
            report.require_common_comparison_ready()
        tedn = report.overall.metric("tedn")
        self.assertEqual(tedn.availability, MetricAvailability.UNSUPPORTED)
        self.assertIsNone(tedn.mean_value)
        self.assertEqual(tedn.unsupported_count, 4)

    def test_missing_voice_strata_are_reported_not_hidden(self) -> None:
        report = aggregate_poly_v2_validation_reports(
            (_sample("a", voice="1_voice"), _sample("b", voice="2_voice"))
        )
        self.assertEqual(report.missing_voice_strata, ("3_voice", "4_plus_voice"))
        self.assertFalse(report.voice_coverage_ready)

    def test_mixed_candidate_identities_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            PolyV2ValidationAggregationError, "candidate/checkpoint identities"
        ):
            aggregate_poly_v2_validation_reports(
                (
                    _sample("a", voice="1_voice", candidate="candidate-a"),
                    _sample("b", voice="2_voice", candidate="candidate-b"),
                )
            )

    def test_mixed_benchmark_identities_are_rejected(self) -> None:
        with self.assertRaisesRegex(PolyV2ValidationAggregationError, "benchmark identities"):
            aggregate_poly_v2_validation_reports(
                (
                    _sample("a", voice="1_voice", benchmark="benchmark-a"),
                    _sample("b", voice="2_voice", benchmark="benchmark-b"),
                )
            )

    def test_duplicate_samples_are_rejected(self) -> None:
        sample = _sample("a", voice="1_voice")
        with self.assertRaisesRegex(PolyV2ValidationAggregationError, "duplicate sample_id"):
            aggregate_poly_v2_validation_reports((sample, sample))

    def test_unbound_checkpoint_evidence_is_rejected(self) -> None:
        with self.assertRaisesRegex(PolyV2ValidationAggregationError, "checkpoint-bound"):
            aggregate_poly_v2_validation_reports(
                (_sample("a", voice="1_voice", checkpoint_bound=False),)
            )

    def test_fingerprint_is_input_order_independent(self) -> None:
        reports = _all_voice_reports()
        first = aggregate_poly_v2_validation_reports(reports)
        second = aggregate_poly_v2_validation_reports(tuple(reversed(reports)))
        self.assertEqual(first.sample_ids, second.sample_ids)
        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_common_candidate_comparison_stays_closed_while_metrics_are_unsupported(self) -> None:
        first = aggregate_poly_v2_validation_reports(_all_voice_reports())
        second = aggregate_poly_v2_validation_reports(
            tuple(
                _sample(
                    label,
                    voice=voice,
                    score=score,
                    bucket=bucket,
                    candidate="candidate-b",
                )
                for label, voice, score, bucket in (
                    ("a", "1_voice", 0.9, "clean"),
                    ("b", "2_voice", 0.7, "scan"),
                    ("c", "3_voice", 0.5, "phone"),
                    ("d", "4_plus_voice", 0.3, "blur"),
                )
            )
        )
        with self.assertRaisesRegex(PolyV2ValidationAggregationError, "common comparison gate"):
            validate_common_candidate_reports((first, second))


if __name__ == "__main__":
    unittest.main()
