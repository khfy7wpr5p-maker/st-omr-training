"""Stage 7-D10 repository-SHA compatibility loader.

The original D10 implementation was written with SHA-256 validators on fields
that actually carry Git object IDs and also treated the Stage 7-C repository
verifier as if it returned only HEAD. Stage 7-C intentionally returns
``(head, origin)`` and HEAD is a canonical 40-character lowercase Git SHA-1.

Keep the large deterministic D10 implementation byte-for-byte in the sibling
``stage7d10_local_roi_derivatives_impl`` module and adapt only that provenance
boundary here. SHA-256 artifact fields remain strict 64-character values.
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import stage7d10_local_roi_derivatives_impl as _impl

_HEX = frozenset("0123456789abcdef")
_REPOSITORY_SHA_FIELDS = frozenset(
    {
        "repository_sha",
        "manifest.repository_sha",
        "expected_repository_sha",
    }
)


def _git_sha40(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in _HEX for character in value)
    ):
        raise _impl.Stage7D10DerivativeError(
            f"{name} must be canonical lowercase 40-character Git SHA"
        )
    return value


# Keep explicit references so regression tests can verify the adapter contract
# without changing Stage 7-C itself.
_impl._stage7d10_original_hex64 = _impl._hex64
_impl._stage7d10_original_verify_authoritative_repository = (
    _impl.verify_authoritative_repository
)
_impl._stage7d10_repository_sha_fields = _REPOSITORY_SHA_FIELDS


def _d10_sha_validator(name: str, value: object) -> str:
    if name in _REPOSITORY_SHA_FIELDS:
        return _git_sha40(name, value)
    return _impl._stage7d10_original_hex64(name, value)


def _d10_verify_authoritative_repository(repository_root: str | Path) -> str:
    head, _origin = _impl._stage7d10_original_verify_authoritative_repository(
        repository_root
    )
    return _git_sha40("repository HEAD", head)


_impl._git_sha40 = _git_sha40
_impl._hex64 = _d10_sha_validator
_impl.verify_authoritative_repository = _d10_verify_authoritative_repository

# Make the canonical module name resolve to the patched implementation module.
# Existing imports and unittest.mock patch paths therefore continue to target
# the same function globals as before this compatibility correction.
sys.modules[__name__] = _impl
