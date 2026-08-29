# TR-POLY-08C — Exact Stage 6 Dataset Execution

## Purpose

TR-POLY-08C connects the already-verified Stage 6 synthetic dataset artifacts to the TR-POLY-08A bounded trainer and TR-POLY-08B checkpoint contract.

This stage proves an exact data path. It does **not** prove polyphonic recognition quality.

## Exact admission path

The execution path is:

`persisted Stage 6 manifest/build` → `selected TRAIN/VALIDATION MusicXML + PNG bytes` → `hash verification` → `supported V1 → V2 bridge` → `V2 lossless tokenizer` → `deterministic fit/pad grayscale tensor` → `teacher-forced V2 batch` → `bounded trainer` → `hash-bound checkpoint/reload verification`.

The Stage 6 source class remains `synthetic`.

## TEST remains sealed

TR-POLY-08C rejects `DatasetSplit.TEST` before dataset-root verification or artifact reads. Only manifest/build identity is inspected globally; target/image artifact bytes are read only for explicitly selected TRAIN or VALIDATION samples.

The integration suite corrupts the persisted TEST target and image while proving TRAIN/VALIDATION materialization still succeeds. This is a direct regression against accidental TEST artifact access.

## V1 → V2 bridge boundary

Stage 6 currently persists the frozen V1 MusicXML target surface. It is not a polyphonic V2 corpus.

TR-POLY-08C therefore introduces an explicit bridge profile:

`single_voice_v1_bridge`

The bridge:

- independently validates MusicXML through the existing Stage-2-C validator;
- accepts exactly part `P1`;
- accepts only source voice 1 / staff 1;
- preserves exact onset and duration as rational values;
- preserves note/rest/chord distinction;
- preserves pitch spelling and visible accidental intent;
- preserves visible whole/half/quarter/eighth note type;
- preserves active time, key fifths and clef state;
- groups MusicXML chord continuations deterministically;
- creates deterministic V2 event/notehead identities;
- requires lossless V2 JSON/token roundtrip;
- rejects unsupported dot/tie/beam/tuplet/grace/notation surfaces rather than inventing semantics.

No voice 2+, cross-staff, tie, tuplet or true polyphonic evidence is claimed from this bridge.

## Image materialization

The selected PNG must match the manifest SHA-256 and dimensions. The existing deterministic grayscale fit-inside/center-pad/no-crop preprocessing is reused with the 2D model input dimensions.

The TR-POLY-08C materialization fingerprint binds:

- execution version;
- V1→V2 bridge version/profile;
- Stage 6 source class;
- preprocessing fingerprint;
- model profile fingerprint;
- V2 tokenizer fingerprint;
- sealed TEST policy;
- semantic truncation policy.

Semantic target truncation is forbidden. A target longer than `max_target_tokens` fails closed.

## Split and batch boundary

- families remain exclusive under the Stage 5 manifest policy;
- TRAIN and VALIDATION families are checked again at execution time;
- a batch may not mix splits;
- teacher forcing uses `BOS...` as decoder input and the one-token-shifted target ending in `EOS` as labels;
- padding is V2 `PAD` only;
- maximum materialized batch size is inherited from the bounded TR-POLY-08A trainer.

## Checkpoint and claim boundary

The execution can produce a TR-POLY-08B checkpoint from actual persisted Stage 6 TRAIN/VALIDATION artifact bytes. The checkpoint remains research-only and retains:

- `authoritative_dataset_execution = false`;
- `test_split_accessed = false`;
- `benchmark_evidence = false`;
- `production_authority = false`.

TR-POLY-08C additionally records `controlled_dataset_execution = true`, meaning actual hash-verified Stage 6 artifact bytes were consumed by the bounded path. This does not upgrade registry authority.

`candidate.poly-2d-transformer.v1` therefore remains `architecture_only / none` until a later stage has a genuinely polyphonic V2 TRAIN corpus and an authorized artifact contract.

## Explicit non-claims

TR-POLY-08C does not claim:

- real-world scan/photo training;
- external dataset training;
- independent voice 2/3/4+ training;
- polyphonic accuracy improvement;
- benchmark superiority;
- TEST performance;
- ScoreMosaic shadow readiness;
- production authority.

## Next gate

The next safe gate is a native polyphonic V2 dataset/materialization source with explicit 2/3/4+ voice strata. Only after that exists should a model artifact be considered for the TR-POLY-09 common benchmark as polyphonic evidence.
