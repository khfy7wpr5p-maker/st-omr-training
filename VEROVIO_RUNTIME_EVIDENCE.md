# Verovio Runtime Evidence — Stage 3

This file records the exact local renderer binary used for Stage 3 verification. The wheel itself is not stored in Git.

## Tested runtime

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

## Verification performed with the exact runtime

- all six Stage 2 golden MusicXML fixtures rendered successfully;
- repeated rendering of the same MusicXML produced byte-identical SVG and identical page SHA-256 values;
- 50 generated-score renders in the committed live-runtime suite passed;
- supplemental real-render stress rendered 120 generated scores spanning mixed, note-only, rest-only, chord-only, fixed 2/4, fixed 3/4, fixed 4/4, and accidentals-disabled configurations;
- 20 supplemental repeated-render cases produced byte-identical SVG output;
- full available regression suite on the committed Stage 3 branch state passed: 236 tests;
- Python compile validation had already passed for the committed Stage 3 source before the real-runtime gate.

This evidence is local only. GitHub-hosted CI did not run and must not be inferred from this file.
