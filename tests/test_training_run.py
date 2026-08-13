from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from st_omr_training.dataset_builder import (
    SyntheticDatasetConfig,
    build_synthetic_dataset,
    write_synthetic_dataset,
)
from st_omr_training.dataset_manifest import DatasetSplit
from st_omr_training.musicxml_validator import validate_musicxml
from st_omr_training.musicxml_writer import write_musicxml
from st_omr_training.training_data import InputPreprocessConfig, TrainingSampleRef
from st_omr_training.training_model import BaselineModelConfig, TrainerConfig
from st_omr_training.training_run import (
    BaselineRunConfig,
    BaselineRunConfigError,
    BaselineRunError,
    PredictionMetrics,
    _evaluate_predictions,
    _levenshtein_distance,
    _score_from_projection,
    baseline_run_config_fingerprint,
    run_baseline_training,
)
from st_omr_training.training_tokens import (
    decode_token_ids,
    detokenize_tokens,
    encode_tokens,
)


def semantic_target_ids() -> tuple[int, ...]:
    return encode_tokens(
        (
            "BOS",
            "MEASURE_START",
            "TS_4_4",
            "REST",
            "DUR_HALF",
            "REST",
            "DUR_HALF",
            "MEASURE_END",
            "EOS",
        )
    )


class Stage7CRunContractTests(unittest.TestCase):
    def test_run_config_is_bounded_and_keeps_one_checkpoint(self) -> None:
        self.assertEqual(BaselineRunConfig().retained_checkpoints, 1)
        with self.assertRaises(BaselineRunConfigError):
            BaselineRunConfig(epochs=101)
        with self.assertRaises(BaselineRunConfigError):
            BaselineRunConfig(batch_size=True)
        with self.assertRaises(BaselineRunConfigError):
            BaselineRunConfig(retained_checkpoints=2)

    def test_run_fingerprint_is_deterministic(self) -> None:
        first = baseline_run_config_fingerprint(
            BaselineRunConfig(),
            BaselineModelConfig(),
            TrainerConfig(),
            InputPreprocessConfig(),
        )
        second = baseline_run_config_fingerprint(
            BaselineRunConfig(),
            BaselineModelConfig(),
            TrainerConfig(),
            InputPreprocessConfig(),
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_levenshtein_distance_counts_insert_delete_substitute(self) -> None:
        self.assertEqual(_levenshtein_distance((1, 2, 3), (1, 2, 3)), 0)
        self.assertEqual(_levenshtein_distance((1, 2, 3), (1, 4, 3)), 1)
        self.assertEqual(_levenshtein_distance((1, 2, 3), (1, 2)), 1)
        self.assertEqual(_levenshtein_distance((1, 2), (1, 2, 3)), 1)

    def test_semantic_projection_regenerates_valid_musicxml(self) -> None:
        ids = semantic_target_ids()
        projection = detokenize_tokens(decode_token_ids(ids))
        score = _score_from_projection(projection, score_id="stage7c-test")
        musicxml = write_musicxml(score)
        self.assertTrue(validate_musicxml(musicxml).is_valid)

    def test_perfect_greedy_prediction_reports_perfect_metrics(self) -> None:
        ids = semantic_target_ids()
        sample = TrainingSampleRef(
            sample_id="sample-1",
            family_id="family-1",
            split=DatasetSplit.VALIDATION,
            image_path=Path("unused.png"),
            image_sha256="a" * 64,
            target_path=Path("unused.musicxml"),
            target_sha256="b" * 64,
            target_token_ids=ids,
            source_width=512,
            source_height=64,
        )
        with patch(
            "st_omr_training.training_run._greedy_decode_sample",
            return_value=ids,
        ):
            metrics = _evaluate_predictions(
                object(),
                (sample,),
                preprocess_config=InputPreprocessConfig(),
                max_decode_tokens=64,
            )
        self.assertEqual(metrics.token_error_rate, 0.0)
        self.assertEqual(metrics.exact_sequence_accuracy, 1.0)
        self.assertEqual(metrics.detokenization_success_rate, 1.0)
        self.assertEqual(metrics.semantic_validity_rate, 1.0)
        self.assertEqual(metrics.musicxml_regeneration_validity_rate, 1.0)
        self.assertEqual(metrics.valid_semantic_predictions, 1)

    def test_prediction_metrics_reject_invalid_semantic_count(self) -> None:
        with self.assertRaises(BaselineRunError):
            PredictionMetrics(
                token_error_rate=0.0,
                exact_sequence_accuracy=0.0,
                detokenization_success_rate=0.0,
                semantic_validity_rate=0.0,
                musicxml_regeneration_validity_rate=0.0,
                validation_samples=1,
                valid_semantic_predictions=-1,
            )


class Stage7CBoundedOrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.dataset_root = Path(cls._temp.name) / "dataset"
        cls.build = build_synthetic_dataset(
            SyntheticDatasetConfig(
                family_count=3,
                seed_start=41_000,
                split_seed=29,
                measure_count=1,
                raster_width=512,
                family_profiles=("time-4-4", "note-only", "rest-only"),
                degradation_profiles=("clean",),
            )
        )
        write_synthetic_dataset(cls.build, cls.dataset_root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_bounded_run_writes_hash_addressed_checkpoint_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_root = Path(temp) / "runs"
            progress_events: list[dict[str, object]] = []
            metrics = PredictionMetrics(
                token_error_rate=0.0,
                exact_sequence_accuracy=1.0,
                detokenization_success_rate=1.0,
                semantic_validity_rate=1.0,
                musicxml_regeneration_validity_rate=1.0,
                validation_samples=1,
                valid_semantic_predictions=1,
            )
            with (
                patch(
                    "st_omr_training.training_run._mean_validation_loss",
                    side_effect=(2.0, 1.0),
                ),
                patch(
                    "st_omr_training.training_run._evaluate_predictions",
                    return_value=metrics,
                ),
            ):
                result = run_baseline_training(
                    self.build,
                    self.dataset_root,
                    run_root,
                    repository_sha="a" * 40,
                    run_config=BaselineRunConfig(
                        epochs=1,
                        batch_size=1,
                        max_train_samples=1,
                        max_validation_samples=1,
                        max_decode_tokens=64,
                    ),
                    trainer_config=TrainerConfig(master_seed=77, smoke_steps=1),
                    progress=progress_events.append,
                )

            self.assertTrue((result.run_directory / "COMPLETE").is_file())
            self.assertFalse((result.run_directory / "INCOMPLETE").exists())
            checkpoints = tuple(result.run_directory.glob("checkpoint-*.pt"))
            evidence = tuple(result.run_directory.glob("metrics-*.json"))
            self.assertEqual(len(checkpoints), 1)
            self.assertEqual(len(evidence), 1)
            self.assertEqual(checkpoints[0].stem.split("-", 1)[1], result.checkpoint_sha256)
            self.assertEqual(evidence[0].stem.split("-", 1)[1], result.metrics_sha256)
            self.assertEqual(
                [event["event"] for event in progress_events],
                [
                    "training_started",
                    "untrained_validation_started",
                    "untrained_validation_completed",
                    "epoch_started",
                    "training_step_completed",
                    "epoch_validation_started",
                    "epoch_completed",
                    "prediction_evaluation_started",
                    "prediction_evaluation_completed",
                    "training_completed",
                ],
            )
            self.assertEqual(progress_events[0]["epochs_total"], 1)
            self.assertEqual(progress_events[0]["training_steps_total"], 1)
            self.assertEqual(progress_events[4]["training_steps"], 1)
            self.assertEqual(progress_events[6]["selected_best"], True)
            self.assertEqual(progress_events[-1]["run_id"], result.run_id)

            with self.assertRaises(BaselineRunError):
                run_baseline_training(
                    self.build,
                    self.dataset_root,
                    run_root,
                    repository_sha="a" * 40,
                    run_config=BaselineRunConfig(
                        epochs=1,
                        batch_size=1,
                        max_train_samples=1,
                        max_validation_samples=1,
                        max_decode_tokens=64,
                    ),
                    trainer_config=TrainerConfig(master_seed=77, smoke_steps=1),
                )


if __name__ == "__main__":
    unittest.main()
