# Verovio Runtime Evidence — Stage 3

This file records the exact local renderer binary used for Stage 3 pre-merge verification. The wheel itself is not stored in Git.

## Tested local runtime

- package: `verovio==6.2.1`
- wheel: `verovio-6.2.1-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`
- wheel size: `8912666` bytes
- wheel SHA-256: `00b9ab551de859fa61ac67d6a5f4f3d97a7f9389197644d9493b5e9c0b7b69ab`
- Python: CPython 3.13
- platform tag: manylinux2014 / manylinux_2_17 x86_64
- license metadata: LGPLv3

The locally supplied wheel SHA-256 was independently computed before installation and matched the SHA-256 published by PyPI for this exact Verovio 6.2.1 CPython 3.13 x86-64 wheel.

## Installation boundary

The wheel was installed only into the local verification environment with dependency resolution disabled. It is a dependency artifact, not repository content and not a training-data artifact.

## Local verification performed with the exact wheel

- all six Stage 2 golden MusicXML fixtures rendered successfully;
- repeated rendering of the same MusicXML produced byte-identical SVG and identical page SHA-256 values;
- 50 generated-score renders in the committed live-runtime suite passed;
- supplemental real-render stress rendered 120 generated scores spanning mixed, note-only, rest-only, chord-only, fixed 2/4, fixed 3/4, fixed 4/4, and accidentals-disabled configurations;
- 20 supplemental repeated-render cases produced byte-identical SVG output;
- full available regression suite on the committed Stage 3 branch state passed: 236 tests;
- Python compile validation passed for the committed Stage 3 source.

This section is specifically the local exact-wheel evidence and must not be rewritten as hosted evidence.

## GitHub-hosted CI evidence

GitHub-hosted CI became available after the repository was made public and the CI baseline was merged through PR #12.

The exact integrated `main` commit `5abbc9859a4a69bf9a17936bc41e722256f87472`, which contains the merged Stage 3 renderer implementation, passed GitHub Actions run `31647615123` on Python 3.13 / Ubuntu. That run successfully installed the pinned repository dependencies, independently asserted `verovio==6.2.1`, ran `pip check`, executed the complete unittest suite including the real Verovio runtime tests, and passed Python compile validation.

The hosted run establishes separate CI evidence for the current integrated Stage 3 code. It does not replace the local exact-wheel SHA-256 evidence above, and this document does not claim that GitHub Actions exposed or independently verified the same downloaded wheel file SHA-256.
