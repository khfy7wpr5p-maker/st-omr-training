# TR-POLY-03 — External Dataset / License Registry

Status: **metadata and admission contract only; no external dataset bytes downloaded, committed, opened, trained on, or evaluated**.

## Purpose

TR-POLY-03 creates the fail-closed external-dataset governance layer required
before OLiMPiC, GrandStaff, Muse OMR Benchmark, DoReMi, DeepScoresV2, MUSCIMA++
or another external corpus can enter ST-OMR research.

The core rule is:

> Public availability is not training permission, and a verified license is not
> an admitted installation.

Dataset licensing is independent from source-code licensing.

## Starting authority

- protected `main` base: `2118482a37b4569286d7a5f2deabdf1717a31cc1`
- TR-POLY-02 evaluation contract: merged / post-merge CI verified
- sealed TEST policy unchanged
- Stage 8 rights/provenance/admission policy unchanged

## Two independent gates

Every external dataset has two distinct states.

### 1. Legal-use classification

One of:

- `COMMERCIAL_CLEAN`
- `RESEARCH_ONLY`
- `EVALUATION_ONLY`
- `LICENSE_REVIEW_REQUIRED`

`LICENSE_REVIEW_REQUIRED` is fail-closed: the registry may not guess whether
redistribution, commercial use, training or evaluation is allowed.

### 2. Installation admission

One of:

- `CANDIDATE`
- `LICENSE_VERIFIED`
- `INSTALL_PINNED`

A legally verified dataset is still not training/evaluation-ready until the
exact locally installed artifact is pinned with an independently verified
SHA-256.

No candidate in TR-POLY-03 is installation-pinned.

## Required registry fields

Each record carries:

- dataset name
- dataset component
- authoritative source
- version
- license identity
- license-evidence source
- redistribution permission
- commercial-use permission
- training permission
- evaluation permission
- derivative restrictions
- legal-use class
- registry state
- exact artifact SHA-256 when installation-pinned
- notes

A single dataset name may have separate records when licensing differs by
component. This is required for cases such as GrandStaff-LMX, where the added
LMX/MusicXML material has a stated license but the underlying GrandStaff data
must be reviewed independently.

## Initial reviewed catalog

| Dataset / component | Evidence at TR-POLY-03 review | Classification | Registry state | Current ST-OMR use |
| --- | --- | --- | --- | --- |
| Muse OMR Benchmark | Hugging Face dataset card states dataset content is CC0-1.0; 1077 symbolic-score + augmented-PDF pairs; underlying works public domain | `COMMERCIAL_CLEAN` | `LICENSE_VERIFIED` | BLOCKED until SHA-256 install pin |
| DeepScoresV2 2.0 | ZHAW research-data record states CC BY 4.0; Zenodo record identifies v2.0 | `COMMERCIAL_CLEAN` | `LICENSE_VERIFIED` | BLOCKED until SHA-256 install pin |
| MUSCIMA++ annotations 2.1 | Repository `LICENSE.txt` is CC BY-NC-SA 4.0 | `RESEARCH_ONLY` | `LICENSE_VERIFIED` | BLOCKED until SHA-256 install pin; never commercial-candidate clean |
| OLiMPiC synthetic 1.0 | OLiMPiC README says datasets are CC BY-SA but does not state the exact CC version in the reviewed text; built on OpenScore Lieder | `LICENSE_REVIEW_REQUIRED` | `CANDIDATE` | BLOCKED |
| OLiMPiC scanned 1.0 | Same CC BY-SA statement; scanned-source/upstream obligations require exact review | `LICENSE_REVIEW_REQUIRED` | `CANDIDATE` | BLOCKED |
| GrandStaff-LMX added `.lmx`/`.musicxml` | OLiMPiC README explicitly limits its CC BY-SA statement to the added files | `LICENSE_REVIEW_REQUIRED` | `CANDIDATE` | BLOCKED pending exact CC version and original GrandStaff terms |
| Original GrandStaff | Publicly available for replication; repository code is MIT, but this is not proof of dataset license | `LICENSE_REVIEW_REQUIRED` | `CANDIDATE` | BLOCKED |
| DoReMi published subset | README says final published data include openly distributable/copyright-free scores; reviewed repository root exposes no explicit dataset license grant | `LICENSE_REVIEW_REQUIRED` | `CANDIDATE` | BLOCKED |

### Evidence locations

Muse OMR Benchmark:

`https://huggingface.co/datasets/musegroup/omr_benchmark`

DeepScoresV2:

`https://zenodo.org/records/4012193`

`https://digitalcollection.zhaw.ch/items/1376f518-211d-49c8-a658-dbc7f085d2b1`

MUSCIMA++:

`https://github.com/OMR-Research/muscima-pp/blob/master/LICENSE.txt`

OLiMPiC / GrandStaff-LMX:

`https://github.com/ufal/olimpic-icdar24/blob/master/README.md`

Original GrandStaff reference implementation:

`https://github.com/multiscore/e2e-pianoform`

DoReMi:

`https://github.com/steinbergmedia/DoReMi/blob/main/README.md`

## Why OLiMPiC is not yet marked commercial-clean

The reviewed OLiMPiC README states `CC BY-SA` for OLiMPiC synthetic, OLiMPiC
scanned and the *added* GrandStaff-LMX files, but it does not specify the exact
Creative Commons version in that statement. OLiMPiC is also built on the
OpenScore Lieder Corpus.

TR-POLY-03 therefore records the published statement but does not infer the
exact derivative/commercial obligations. Exact version and upstream obligations
must be reviewed before bytes are admitted.

The repository's MIT `LICENSE.txt` applies to source code and must not be used as
the dataset license.

## Why GrandStaff-LMX is split from GrandStaff

The OLiMPiC documentation says the CC BY-SA dataset statement applies to
GrandStaff-LMX **only for the added `.lmx` and `.musicxml` files**.

Therefore:

- added linearized/symbolic files are one licensing component;
- original GrandStaff images/data are a separate component;
- combining the two requires both components to pass their own license gate.

A code repository being MIT-licensed does not make the accompanying dataset MIT.

## Why DoReMi remains blocked

The DoReMi lifecycle document states that copyrighted works were excluded from
the final published dataset and only openly distributable scores are included.
That is useful provenance information, but it is not itself a complete dataset
license grant covering redistribution, commercial ML training, derivatives and
model-weight implications.

Until an explicit authoritative dataset license is found and reviewed, DoReMi
remains `LICENSE_REVIEW_REQUIRED`.

## DeepScoresV2 boundary

DeepScoresV2 2.0 has an authoritative ZHAW record identifying CC BY 4.0. That
permits commercial use subject to attribution, so the dataset-license layer can
classify it as `COMMERCIAL_CLEAN`.

This does **not** make it installed or admitted. Zenodo publishes MD5 identities
for its archives, while ST-OMR requires an exact independently verified SHA-256
for the locally admitted artifact.

## MUSCIMA++ boundary

MUSCIMA++ annotations are CC BY-NC-SA 4.0 and therefore `RESEARCH_ONLY`.

The registry makes it impossible for such a record to pass the
commercial-candidate training gate even after installation is pinned.

Underlying CVC-MUSCIMA images remain a separate licensing/acquisition component;
the annotation license must not silently authorize unrelated image bytes.

## Commercial-safety rule

A checkpoint intended to remain eligible for a future commercial candidate may
train only on records that are all:

1. `COMMERCIAL_CLEAN`;
2. legally training-permitted;
3. legally commercial-use permitted;
4. `INSTALL_PINNED` with exact SHA-256.

A `RESEARCH_ONLY` dataset may support a research checkpoint after installation
admission, but that checkpoint must remain research-only and cannot be promoted
into a commercial/production candidate lane.

## Evaluation-only rule

The contract supports `EVALUATION_ONLY` datasets. Such a record can become
`INSTALL_PINNED` for evaluation but is rejected by every training-admission
path.

No initial catalog entry is currently classified `EVALUATION_ONLY`; this class
exists for future external benchmark terms that explicitly prohibit training.

## Checksum policy

TR-POLY-03 commits no external dataset bytes and does not invent checksums.

A record can move to `INSTALL_PINNED` only after a later controlled installation
step independently computes and records the exact SHA-256 of the admitted
artifact/package. MD5 or a webpage file size is not sufficient for ST-OMR
admission identity.

## Model-weight implications

A permissive dataset license does not automatically answer every legal question
about resulting model weights. `COMMERCIAL_CLEAN` in this registry means the
reviewed dataset license itself does not impose a non-commercial/evaluation-only
restriction and permits the recorded uses under its terms.

Future checkpoint/model-card governance must still record all training data
classes and license obligations.

## Safety boundaries

TR-POLY-03:

- downloads no external dataset;
- stores no external dataset bytes;
- opens no external benchmark sample;
- performs no training or inference;
- opens no sealed TEST data;
- changes no existing split;
- changes no Stage 8 real-data rights gate;
- adds no dependency;
- produces no checkpoint;
- changes no ScoreMosaic behavior;
- does not treat public availability as permission;
- does not treat code licenses as dataset licenses.

## Next package boundary

TR-POLY-04 may build an external evaluation harness only for datasets whose
required legal component(s) have passed this registry and whose exact local
artifacts have been installation-pinned.

Because OLiMPiC/GrandStaff remain `LICENSE_REVIEW_REQUIRED`, TR-POLY-04 must not
download or execute them until that blocker is resolved. Harness interfaces may
be designed without bytes, but external execution remains closed.
