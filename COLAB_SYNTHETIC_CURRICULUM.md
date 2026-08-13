# ST-OMR Synthetic Curriculum v1 — Colab Runbook

Notebook: `notebooks/st_omr_synthetic_curriculum_v1_colab.ipynb`

## Frozen source

The notebook checks out exact commit:

`adc8139539d3c8cd6a2e3ee4ce4de6db4dcfeb90`

It installs and verifies:

```text
lxml==6.1.1
verovio==6.2.1
CairoSVG==2.8.2
Pillow==12.3.0
```

## Run order

Run every cell from top to bottom without changing the frozen constants. The notebook verifies the repository SHA, package versions, 512-family plan, 410/51/51 family split, config fingerprint, persisted manifest hash, target/image counts, and copied archive hash.

Corpus construction occurs first on `/content` local storage. After validation, one normalized archive and small evidence files are copied to:

```text
MyDrive/ST-OMR-SYNTHETIC/<build_id>/
```

The export directory contains the corpus archive, evidence JSON, `manifest.sha256`, and `build.json`.

## Acceptance evidence

A completed export must print `EXPORT PASS` and record:

- `build_id`
- `manifest_sha256`
- `config_fingerprint` in evidence JSON
- `transport_sha256`

`transport_sha256` protects the transfer archive. Dataset identity remains the build/config/manifest identities.

Python/platform/Cairo/package versions are captured in the evidence because raster bytes can depend on host runtime details.
