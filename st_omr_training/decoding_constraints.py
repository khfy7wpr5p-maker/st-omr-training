"""Deterministic grammar constraints for greedy ST-OMR V1 token decoding."""

from __future__ import annotations

from fractions import Fraction

from .training_tokens import TOKEN_TO_ID


_EVENT_TOKENS = ("NOTE", "REST", "CHORD_2", "CHORD_3", "CHORD_4")
_TIME_CAPACITY = {
    "TS_2_4": Fraction(1, 2),
    "TS_3_4": Fraction(3, 4),
    "TS_4_4": Fraction(1, 1),
}
_DURATION_VALUE = {
    "DUR_WHOLE": Fraction(1, 1),
    "DUR_HALF": Fraction(1, 2),
    "DUR_QUARTER": Fraction(1, 4),
    "DUR_EIGHTH": Fraction(1, 8),
}
_STEP_TOKENS = tuple(f"STEP_{step}" for step in "ABCDEFG")
_ALTER_TOKENS = ("ALTER_M1", "ALTER_0", "ALTER_P1")
_OCTAVE_TOKENS = ("OCT_3", "OCT_4", "OCT_5", "OCT_6")


def _token_ids(tokens: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(TOKEN_TO_ID[token] for token in tokens)


class SemanticDecodeConstraint:
    """Track one valid V1 prefix after BOS and expose its allowed next tokens."""

    def __init__(self, *, measure_count: int) -> None:
        if isinstance(measure_count, bool) or not isinstance(measure_count, int):
            raise TypeError("measure_count must be a plain integer")
        if not 1 <= measure_count <= 64:
            raise ValueError("measure_count must be from 1 through 64")
        self._measure_count = measure_count
        self._phase = "measure_start"
        self._measures_completed = 0
        self._remaining = Fraction(0, 1)
        self._event_kind = ""
        self._pending_duration = Fraction(0, 1)
        self._pitches_remaining = 0
        self._chord_identities: set[tuple[str, str, str]] = set()
        self._current_step = ""
        self._current_alter = ""
        self._current_octave = ""

    @property
    def is_complete(self) -> bool:
        return self._phase == "complete"

    def allowed_token_ids(self) -> tuple[int, ...]:
        if self._phase == "measure_start":
            return _token_ids(("MEASURE_START",))
        if self._phase == "time":
            return _token_ids(tuple(_TIME_CAPACITY))
        if self._phase == "event":
            if self._remaining == 0:
                return _token_ids(("MEASURE_END",))
            return _token_ids(_EVENT_TOKENS)
        if self._phase == "duration":
            return _token_ids(
                tuple(
                    token
                    for token, duration in _DURATION_VALUE.items()
                    if duration <= self._remaining
                )
            )
        if self._phase == "pitch_step":
            return _token_ids(_STEP_TOKENS)
        if self._phase == "pitch_alter":
            return _token_ids(_ALTER_TOKENS)
        if self._phase == "pitch_octave":
            allowed = tuple(
                token
                for token in _OCTAVE_TOKENS
                if not (
                    self._event_kind.startswith("CHORD_")
                    and (self._current_step, self._current_alter, token)
                    in self._chord_identities
                )
            )
            if not allowed:
                raise RuntimeError("valid V1 prefix has no non-duplicate chord octave")
            return _token_ids(allowed)
        if self._phase == "pitch_accidental":
            accidental = {
                "ALTER_M1": "ACC_FLAT",
                "ALTER_0": "ACC_NATURAL",
                "ALTER_P1": "ACC_SHARP",
            }[self._current_alter]
            return _token_ids(("ACC_NONE", accidental))
        if self._phase == "sequence_boundary":
            if self._measures_completed >= self._measure_count:
                return _token_ids(("EOS",))
            return _token_ids(("MEASURE_START",))
        if self._phase == "complete":
            return ()
        raise RuntimeError("semantic decode constraint entered an invalid phase")

    def _finish_event(self) -> None:
        self._remaining -= self._pending_duration
        if self._remaining < 0:
            raise RuntimeError("semantic decode constraint overfilled a measure")
        self._event_kind = ""
        self._pending_duration = Fraction(0, 1)
        self._pitches_remaining = 0
        self._chord_identities.clear()
        self._current_step = ""
        self._current_alter = ""
        self._current_octave = ""
        self._phase = "event"

    def advance(self, token_id: int) -> None:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token_id must be a plain integer")
        allowed = self.allowed_token_ids()
        if token_id not in allowed:
            raise ValueError("token is outside the allowed V1 semantic prefix")
        token = next(name for name, value in TOKEN_TO_ID.items() if value == token_id)

        if self._phase == "measure_start":
            self._phase = "time"
            return
        if self._phase == "time":
            self._remaining = _TIME_CAPACITY[token]
            self._phase = "event"
            return
        if self._phase == "event":
            if token == "MEASURE_END":
                self._measures_completed += 1
                self._phase = "sequence_boundary"
            else:
                self._event_kind = token
                self._phase = "duration"
            return
        if self._phase == "duration":
            self._pending_duration = _DURATION_VALUE[token]
            if self._event_kind == "REST":
                self._finish_event()
            else:
                self._pitches_remaining = (
                    1 if self._event_kind == "NOTE" else int(self._event_kind[-1])
                )
                self._phase = "pitch_step"
            return
        if self._phase == "pitch_step":
            self._current_step = token
            self._phase = "pitch_alter"
            return
        if self._phase == "pitch_alter":
            self._current_alter = token
            self._phase = "pitch_octave"
            return
        if self._phase == "pitch_octave":
            self._current_octave = token
            self._phase = "pitch_accidental"
            return
        if self._phase == "pitch_accidental":
            if self._event_kind.startswith("CHORD_"):
                identity = (
                    self._current_step,
                    self._current_alter,
                    self._current_octave,
                )
                if identity in self._chord_identities:
                    raise RuntimeError("semantic decode constraint admitted a duplicate pitch")
                self._chord_identities.add(identity)
            self._pitches_remaining -= 1
            self._current_step = ""
            self._current_alter = ""
            self._current_octave = ""
            if self._pitches_remaining > 0:
                self._phase = "pitch_step"
            else:
                self._finish_event()
            return
        if self._phase == "sequence_boundary":
            self._phase = "complete" if token == "EOS" else "time"
            return
        raise RuntimeError("semantic decode constraint cannot advance after completion")
