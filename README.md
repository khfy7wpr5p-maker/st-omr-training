# st-omr-training

Safe training and synthetic data laboratory for ST-OMR: canonical music generation, MusicXML serialization and validation, rendering, dataset creation, training, and evaluation.

This repository is isolated from the ScoreMosaic production runtime. Its purpose is to build traceable synthetic OMR training data, train candidate ST-OMR models, and evaluate them before any later integration decision.

## Project documents

- [Architecture](ARCHITECTURE.md)
- [Canonical data contract](DATA_CONTRACT.md)
- [MusicXML contract](MUSICXML_CONTRACT.md)
- [Safety and verification rules](SAFETY.md)
- [Current project status](STATUS.md)

## Current phase

Stage 1 ST Music Generator is implemented and locally verified. GitHub CI remains unavailable, so it is not CI-verified.

Stage 2-A freezes the MusicXML 4.0 boundary before serialization code begins. MusicXML writer implementation, renderer integration, dataset generation, model training, and ScoreMosaic integration remain later gated stages.
