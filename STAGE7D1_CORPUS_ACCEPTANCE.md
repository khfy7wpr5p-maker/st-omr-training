# Stage 7-D1 — Synthetic Corpus Byte / Manifest Acceptance

## Purpose

Stage 7-D1 is the storage-integrity gate between the frozen Synthetic Curriculum v1 export and any later Stage 7-D2 training package. It verifies bytes and lineage only. It does not train, load a checkpoint, choose a model, or expose the sealed test split to development logic.

## Accepted input

The verifier accepts exactly two external inputs:

1. the frozen transport archive `st-omr-synthetic-curriculum-v1-d9320e362f162cd2a.tar.gz`;
2. an extracted Stage 6 corpus root containing only:

```text
manifest.json
manifest.sha256
build.json
images/<png_sha256>.png
targets/<source_musicxml_sha256>.musicxml
```

The corpus and archive stay outside the Git repository.

## Frozen acceptance identities

```text
source_commit       adc8139539d3c8cd6a2e3ee4ce4de6db4dcfeb90
config_fingerprint  154bf1c3e6dfe4e6db096f8b668f29df0623cfd38352b89a04d295764c7458cb
build_id            d9320e362f162cd2ace2a830a7b93e0c21ceba2d51a4e95ef1c7a9b11a108352
manifest_sha256     44a963cd7dbc612fa29c2953ea8b2c8776d89ce470074e8f8b3fe25c6e165f34
transport_sha256    4a9f3bb337ef99386081dff29c4c1fc3047dc3ada4db13c93b6254e680918e2b
archive_size        494006801 bytes
```

## Fail-closed checks

The D1 verifier checks all of the following before emitting evidence:

- exact archive filename, byte length, and streaming SHA-256;
- exact top-level corpus layout with no extra entries;
- regular-file/directory boundaries with symlink rejection;
- canonical `manifest.json` bytes and frozen manifest SHA-256;
- exact `manifest.sha256` content;
- canonical `build.json` and frozen builder/build/config/manifest identities;
- exact sample/image/target and family/split counts;
- `family-exclusive-v1` split policy;
- exactly three derivatives per family;
- no duplicate sample ID or PNG hash;
- no MusicXML target alias across family/split boundaries;
- exact hash-addressed image and target filename sets;
- streaming SHA-256 of every one of the 1,536 PNG files and 512 MusicXML files;
- PNG signature on every image artifact;
- deterministic aggregate artifact-binding SHA-256 over artifact hashes and byte sizes.

The frozen manifest SHA binds the complete Stage 5 sample metadata. D1 independently re-checks the storage and family/split surfaces most relevant to copied-corpus acceptance rather than trusting filenames or Colab export claims.

## Test split rule

D1 is allowed to hash test-split files because byte-integrity verification must cover the complete frozen archive. That integrity read is not development access: D1 has no tokenizer, model, trainer, metric, decoding, checkpoint-selection, or batch interface and returns no sample bytes or paths.

D2 remains responsible for rejecting the test split before any train/validation loader or parameter-update boundary.

## Evidence

On PASS the verifier writes one small canonical JSON receipt with only frozen identities, counts, byte totals, and an aggregate artifact-binding hash. It never copies corpus data into Git.

Example execution in a Colab/local workspace after the archive has been copied locally and safely extracted:

```bash
python -m st_omr_training.synthetic_curriculum_corpus_gate \
  --corpus-root /content/d1/st-omr-synthetic-curriculum-v1 \
  --archive /content/drive/MyDrive/ST-OMR-SYNTHETIC/d9320e362f162cd2ace2a830a7b93e0c21ceba2d51a4e95ef1c7a9b11a108352/st-omr-synthetic-curriculum-v1-d9320e362f162cd2a.tar.gz \
  --evidence-output /content/st-omr-synthetic-curriculum-v1-d1.evidence.json
```

Before extracting, independently run `sha256sum` on the archive and require the frozen transport SHA above. Extract only that verified archive into a fresh external directory. The Python gate intentionally re-hashes the archive again before accepting the extracted corpus.

## Closure rule

Stage 7-D1 is not closed merely because unit tests or GitHub CI pass. Closure requires both:

1. exact PR-head regression/CI evidence for the verifier package; and
2. a PASS receipt produced by running that verifier against the actual frozen archive and corpus bytes in Colab/local workspace.

Only after both are present may merge approval be requested and Stage 7-D2 remain eligible for a separate later package.
