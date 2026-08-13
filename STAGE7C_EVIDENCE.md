# Stage 7-C Accepted Evidence

Status: **closed — exact-main CI verified**.

This file is the durable, small provenance record permitted by `TRAINING_CONTRACT.md`. It records hashes and acceptance facts only. It does not commit the generated dataset, checkpoint bytes, real/user data, or a production model.

## Source and CI identity

- pull request: #23;
- accepted PR source SHA: `7a993304218fa19609ea512665148dac3eea503a`;
- Stage 7-C merge commit on `main`: `2c2c478eb361fa90a3bccd819b623680eb12de0b`;
- exact-head GitHub Actions run: #67 (`31691794239`), success;
- post-merge exact-main GitHub Actions run: #68 (`31692849892`), success;
- complete regression on both runs: 361/361 tests;
- exact runtime: Ubuntu 24.04, CPython 3.13.14, x86_64 CPU;
- dependency identities: `torch==2.13.0+cpu`, `lxml==6.1.1`, `verovio==6.2.1`, `CairoSVG==2.8.2`, `Pillow==12.3.0`;
- `pip check` and Python `compileall`: passed.

The accepted run evidence binds to the PR source SHA because training occurred before squash merge. The squash merge changed Git history identity but not the accepted source tree. Post-merge run #68 independently verified the resulting exact `main` commit.

## Guarded runtime benchmark

- train samples: 153;
- validation samples: 21;
- dataset build/persist/load: `142.938192505` seconds;
- projected runtime: `694.2713895100333` seconds;
- safety factor: `2.0`;
- safety-adjusted runtime: `1388.5427790200665` seconds;
- hard budget: `14400.0` seconds;
- decision: within budget;
- benchmark artifact ID: `9177766170`;
- benchmark archive SHA-256: `597a699a038660e7204373c5323ddae9d8ba37ea8f0eea5e30bdd02e84559961`.

## Accepted authoritative run

- run ID: `4dcb0bc15af83fdc6c7b671c60999de4172dfab29dc5e6811e41db8a0945fcfe`;
- epochs completed: 40;
- training steps: 1560;
- trainable parameters: 36,835;
- untrained validation loss: `3.560342441426505`;
- best validation loss: `0.9992435761603257`;
- selected epoch: 36;
- validation samples: 21;
- valid semantic predictions: 21;
- detokenization success rate: `1.0`;
- semantic validity rate: `1.0`;
- MusicXML regeneration validity rate: `1.0`;
- exact sequence accuracy: `0.0`;
- token error rate: `0.8049901510177282`;
- sealed test split opened: `false`;
- clean source before and after execution: `true`;
- strict checkpoint reload: `true`.

## Dataset and configuration identity

- dataset build ID: `b9ccea4a87c7971b2209f26c2b28fc981ccad3655e02d0a0a03458b1811b8667`;
- dataset configuration fingerprint: `a692cb6aa04a1f358f5a6fd1d38da92078e4a6f50b632ffe65b17c11b3008cee`;
- manifest SHA-256: `b8934221aaddf64f62eaa06033f88ef3ec0cd7d19b27fe270b805bafc9483e2d`;
- frozen run fingerprint: `f0029f402caf8c9d541a2e54b3a1c0e714f41aba6d37f2a518019f01c2154af7`;
- tokenizer fingerprint: `a83c7e713a0442c73acc7e935843e7e2b9c356ab6c4d29b74bfdcb84c439a501`;
- preprocessing fingerprint: `be8ffe2fa2cdc7d5dd346b864883f58bc78d1c290b319b208fbf04c40b91c65b`;
- model fingerprint: `bda616f0a4ada721950a74add229bd7d53c1c87e337ef399a7759beb16930adf`;
- trainer fingerprint: `f2e737ddac4d7bb4e15b1b4bcc0bc6148a769d15fb806f3556c2bbee22f6e954`.

## Artifact identity

- authoritative artifact ID: `9177923796`;
- authoritative archive size: 57,461,358 bytes;
- authoritative archive SHA-256: `69b41888dc52e46c7bff1d278ce55ffddcaf0b24b2ee5c22cbb2e73f6f2d31cf`;
- checkpoint SHA-256: `75c33cefeb970305f5f9171b3274dbe8b785cbcf1e8d6851de7848e66d24efa4`;
- checkpoint model-state SHA-256: `79d354f2582f3f7cc106564b40f07a6027b62b2a74c9efe13a7b2437a6c3f7a0`;
- canonical metrics SHA-256: `c715f49be804bc89b157a7c1b8d8b72209c7083636470ff4e47e5c16833073fe`;
- authoritative verification SHA-256: `afce43cc4c3e11fc1ee13492b15b61f619147d6a52577ffc60765dab1ddc1f84`.

The downloaded authoritative ZIP independently matched GitHub's archive digest. The checkpoint, canonical metrics, and `VERIFIED` bytes independently matched their recorded hashes and hash-addressed filenames. `COMPLETE` bound exactly to the metrics hash.

## Fail-closed diagnostic history

Run #64 (`31690139847`) at source `1c1f1796a4bacc5fd11f6fcf4daa6076217ffef9` completed optimization but produced 0/21 semantically valid unconstrained predictions. The gate retained `INCOMPLETE` evidence and emitted no accepted metrics, `COMPLETE`, or `VERIFIED`. Grammar-constrained incremental inference corrected that failure without validation-target access, an external teacher, optimizer changes, or test-split access. The corrected code passed its focused regressions and the accepted exact-head run.

## Retention and interpretation boundary

GitHub scheduled the benchmark and authoritative Actions artifacts to expire on 2026-09-12. Their hashes remain durable in this file, but the large checkpoint bytes are intentionally not Git content. Reuse of those exact model bytes after artifact expiry requires a separately approved artifact-storage boundary; otherwise only a new provenance-bound run may recreate a checkpoint.

This result proves bounded synthetic trainability, evidence integrity, deterministic execution controls, and grammar-valid reconstruction. It does **not** prove recognition quality or production candidacy: exact sequence accuracy is 0% and token error rate is approximately 80.5%. Stage 8, Stage 9, the sealed test split, and ScoreMosaic integration remain locked pending separate scope and approval.
