"""Guarded Stage 8-3A entrypoint for PrIMuS-style auxiliary triage.

The underlying triage implementation predates the explicit declarative-XML
preflight. This wrapper is therefore the only Stage 8-3A triage entrypoint: it
rejects unsafe XML declaration surfaces before invoking bounded MEI/semantic/
agnostic format triage. Success remains triage evidence only.
"""

from __future__ import annotations

from typing import Final

from .stage8_pilot_preparation import (
    PrimusAuxiliaryInspection,
    Stage8PilotPreparationError,
    inspect_primus_auxiliary_package,
)


MAX_AUXILIARY_MEI_BYTES: Final[int] = 8 * 1024 * 1024
_XML_DECL_MARKERS: Final[tuple[bytes, ...]] = (
    b"<!" + b"DOC" + b"TYPE",
    b"<!" + b"ENT" + b"ITY",
)


def inspect_guarded_primus_auxiliary_package(
    *,
    mei_bytes: object,
    semantic_bytes: object,
    agnostic_bytes: object,
) -> PrimusAuxiliaryInspection:
    """Preflight MEI bytes, then run the existing fail-closed auxiliary triage."""

    if not isinstance(mei_bytes, bytes) or not mei_bytes:
        raise Stage8PilotPreparationError("MEI must be non-empty bytes")
    if len(mei_bytes) > MAX_AUXILIARY_MEI_BYTES:
        raise Stage8PilotPreparationError("MEI exceeds the Stage 8-3A byte limit")
    upper = mei_bytes.upper()
    if any(marker in upper for marker in _XML_DECL_MARKERS):
        raise Stage8PilotPreparationError(
            "declarative/external XML constructs are forbidden before auxiliary triage"
        )
    return inspect_primus_auxiliary_package(
        mei_bytes=mei_bytes,
        semantic_bytes=semantic_bytes,
        agnostic_bytes=agnostic_bytes,
    )
