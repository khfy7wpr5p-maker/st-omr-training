import unittest
from dataclasses import FrozenInstanceError
from fractions import Fraction
from unittest.mock import patch

from st_omr_training.core import ChordEvent, DisplayAccidental, NoteEvent, RestEvent
from st_omr_training.generator import (
    DEFAULT_GENERATOR_VERSION,
    GenerationValidationError,
    GeneratorConfig,
    config_fingerprint,
    generate_score,
)
from st_omr_training.structure_validator import validate_score
from st_omr_training.validator import ValidationIssue, ValidationResult


def iter_notes(score):
    for measure in score.parts[0].measures:
        for event in measure.voices[0].events:
            if isinstance(event, NoteEvent):
                yield measure, event
            elif isinstance(event, ChordEvent):
                for note in event.notes:
                    yield measure, note


class GeneratorConfigTests(unittest.TestCase):
    def test_defaults_are_immutable_and_normalized(self):
        config = GeneratorConfig(steps=("c", "D"))
        self.assertEqual(config.steps, ("C", "D"))
        with self.assertRaises(FrozenInstanceError):
            config.measure_count = 3

    def test_measure_count_is_bounded(self):
        for value in (0, -1, 257, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    GeneratorConfig(measure_count=value)

    def test_time_signatures_are_v1_only_and_unique(self):
        with self.assertRaises(ValueError):
            GeneratorConfig(time_signatures=((6, 8),))
        with self.assertRaises(ValueError):
            GeneratorConfig(time_signatures=((4, 4), (4, 4)))

    def test_pitch_policy_is_restricted_and_unique(self):
        with self.assertRaises(ValueError):
            GeneratorConfig(steps=("H",))
        with self.assertRaises(ValueError):
            GeneratorConfig(steps=("C", "c"))
        with self.assertRaises(ValueError):
            GeneratorConfig(octaves=(2,))
        with self.assertRaises(ValueError):
            GeneratorConfig(octaves=(4, 4))

    def test_event_kinds_are_controlled(self):
        with self.assertRaises(ValueError):
            GeneratorConfig(event_kinds=())
        with self.assertRaises(ValueError):
            GeneratorConfig(event_kinds=("note", "beam"))
        with self.assertRaises(ValueError):
            GeneratorConfig(event_kinds=("note", "note"))

    def test_chord_generation_requires_two_pitch_positions(self):
        with self.assertRaises(ValueError):
            GeneratorConfig(steps=("C",), octaves=(4,), event_kinds=("chord",))

    def test_allow_accidentals_is_strict_bool(self):
        with self.assertRaises(TypeError):
            GeneratorConfig(allow_accidentals=1)

    def test_config_fingerprint_is_canonical(self):
        lower = GeneratorConfig(measure_count=2, steps=("c", "d"))
        upper = GeneratorConfig(measure_count=2, steps=("C", "D"))
        self.assertEqual(config_fingerprint(lower), config_fingerprint(upper))
        self.assertEqual(len(config_fingerprint(lower)), 64)


class DeterministicGeneratorTests(unittest.TestCase):
    def test_same_version_config_and_seed_produce_identical_score(self):
        config = GeneratorConfig(measure_count=12)
        first = generate_score(config, 12345)
        second = generate_score(config, 12345)
        self.assertEqual(first, second)
        self.assertEqual(first.score_id, second.score_id)

    def test_seed_and_generator_version_participate_in_identity(self):
        config = GeneratorConfig(measure_count=6)
        first = generate_score(config, 1)
        second = generate_score(config, 2)
        revised = generate_score(config, 1, generator_version="st-generator-v1-test")
        self.assertNotEqual(first.score_id, second.score_id)
        self.assertNotEqual(first.score_id, revised.score_id)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, revised)

    def test_invalid_seed_and_version_are_rejected(self):
        with self.assertRaises(TypeError):
            generate_score(GeneratorConfig(), True)
        with self.assertRaises(TypeError):
            generate_score(GeneratorConfig(), 1.0)
        with self.assertRaises(ValueError):
            generate_score(GeneratorConfig(), 1, generator_version="  ")

    def test_generated_score_passes_independent_validator(self):
        for seed in range(12):
            with self.subTest(seed=seed):
                score = generate_score(GeneratorConfig(measure_count=9), seed)
                self.assertTrue(validate_score(score).is_valid)

    def test_measure_timeline_exactly_fills_capacity(self):
        score = generate_score(GeneratorConfig(measure_count=24), 7788)
        for measure in score.parts[0].measures:
            cursor = Fraction(0, 1)
            for event in measure.voices[0].events:
                self.assertEqual(event.onset, cursor)
                cursor += event.duration.fraction
            self.assertEqual(cursor, measure.time_signature.capacity)
            self.assertEqual(measure.expected_duration, measure.time_signature.capacity)

    def test_v1_structure_policy_is_fixed(self):
        score = generate_score(GeneratorConfig(measure_count=10), 41)
        self.assertEqual(len(score.parts), 1)
        part = score.parts[0]
        self.assertEqual(part.staff_count, 1)
        self.assertEqual([measure.number for measure in part.measures], list(range(1, 11)))
        for measure in part.measures:
            self.assertEqual(measure.key_signature, 0)
            self.assertEqual(len(measure.voices), 1)
            self.assertEqual(measure.voices[0].voice_id, 1)
            for event in measure.voices[0].events:
                self.assertEqual(event.voice, 1)
                self.assertEqual(event.staff, 1)

    def test_rest_only_mode_never_creates_whole_measure_rest_symbol_proxy(self):
        config = GeneratorConfig(
            measure_count=20,
            time_signatures=((4, 4),),
            event_kinds=("rest",),
        )
        score = generate_score(config, 909)
        for measure in score.parts[0].measures:
            self.assertTrue(all(isinstance(event, RestEvent) for event in measure.voices[0].events))
            self.assertTrue(
                all(event.duration.fraction in {Fraction(1, 8), Fraction(1, 4), Fraction(1, 2)} for event in measure.voices[0].events)
            )

    def test_chord_only_mode_generates_valid_two_to_four_note_chords(self):
        config = GeneratorConfig(
            measure_count=20,
            event_kinds=("chord",),
            steps=("C", "D", "E", "F"),
            octaves=(4, 5),
        )
        score = generate_score(config, 2026)
        for measure in score.parts[0].measures:
            for chord in measure.voices[0].events:
                self.assertIsInstance(chord, ChordEvent)
                self.assertTrue(2 <= len(chord.notes) <= 4)
                positions = {(note.pitch.step, note.pitch.octave) for note in chord.notes}
                self.assertEqual(len(positions), len(chord.notes))
                self.assertTrue(all(note.onset == chord.onset for note in chord.notes))
                self.assertTrue(all(note.duration == chord.duration for note in chord.notes))

    def test_accidentals_can_be_completely_disabled(self):
        score = generate_score(
            GeneratorConfig(measure_count=12, allow_accidentals=False),
            77,
        )
        for _, note in iter_notes(score):
            self.assertEqual(note.pitch.alter, 0)
            self.assertIs(
                note.notation_intent.display_accidental,
                DisplayAccidental.NONE,
            )

    def test_natural_is_only_emitted_after_prior_alteration_in_same_measure(self):
        config = GeneratorConfig(
            measure_count=64,
            time_signatures=((4, 4),),
            steps=("F",),
            octaves=(4,),
            event_kinds=("note",),
            allow_accidentals=True,
        )
        score = generate_score(config, 314159)
        natural_count = 0
        for measure in score.parts[0].measures:
            state = {}
            for event in measure.voices[0].events:
                position = (event.pitch.step, event.pitch.octave)
                previous = state.get(position, 0)
                if event.notation_intent.display_accidental is DisplayAccidental.NATURAL:
                    natural_count += 1
                    self.assertNotEqual(previous, 0)
                    self.assertEqual(event.pitch.alter, 0)
                state[position] = event.pitch.alter
        self.assertGreater(natural_count, 0)

    def test_supported_whole_note_can_be_generated_only_in_four_four_capacity(self):
        found_whole = False
        config = GeneratorConfig(
            measure_count=128,
            time_signatures=((4, 4),),
            event_kinds=("note",),
            allow_accidentals=False,
        )
        score = generate_score(config, 424242)
        for measure in score.parts[0].measures:
            for event in measure.voices[0].events:
                if event.duration.fraction == Fraction(1, 1):
                    found_whole = True
                    self.assertEqual(event.onset, Fraction(0, 1))
                    self.assertEqual(measure.time_signature.capacity, Fraction(1, 1))
        self.assertTrue(found_whole)

    def test_provenance_contains_no_runtime_timestamp_and_is_stable(self):
        config = GeneratorConfig(measure_count=2)
        score = generate_score(config, -99)
        self.assertEqual(score.generator_version, DEFAULT_GENERATOR_VERSION)
        keys = [key for key, _ in score.provenance]
        self.assertEqual(
            keys,
            [
                "config_fingerprint",
                "created_by_pipeline",
                "generator_version",
                "seed",
                "source_id",
                "source_type",
            ],
        )
        self.assertNotIn("timestamp", keys)
        self.assertNotIn("created_at", keys)
        self.assertEqual(dict(score.provenance)["seed"], "-99")
        self.assertEqual(dict(score.provenance)["source_id"], score.score_id)

    def test_independent_validator_is_a_hard_generation_gate(self):
        invalid = ValidationResult((ValidationIssue("forced.invalid", "$", "forced"),))
        with patch("st_omr_training.generator.validate_score", return_value=invalid):
            with self.assertRaises(GenerationValidationError) as caught:
                generate_score(GeneratorConfig(measure_count=1), 1)
        self.assertEqual(caught.exception.result, invalid)


if __name__ == "__main__":
    unittest.main()
