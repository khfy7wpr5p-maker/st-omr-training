# ST-OMR Training — Polyphonic V2 Current Architecture and Roadmap

Updated: 2026-09-14

This document separates implemented capability from measured quality and preserves the dependency order toward a defensible Polyphonic V2 model.

## 1. Current chain

Protected `main` baseline for the active package:

`c1d6f3b9e1419839def00e64a41625d258ef2b47`

```text
TR-POLY-02 evaluation taxonomy / benchmark identity       ✅
TR-POLY-03 external dataset + license registry            ✅
TR-POLY-04 deterministic external benchmark harness       ✅
TR-POLY-05 Polyphonic Representation V2                   ✅
TR-POLY-06 V2 parser/tokenizer/lossless roundtrip         ✅
TR-POLY-07 research model registry                         ✅
TR-POLY-08 tiny 2D Transformer                            ✅ RESEARCH
TR-POLY-08A bounded trainer                               ✅ SMOKE ONLY (≤2 steps)
TR-POLY-08B bounded research checkpoint                    ✅ SMOKE ONLY
TR-POLY-08C exact Stage 6 V1→V2 execution                 ✅
TR-POLY-09A native explicit V2 materialization            ✅
TR-POLY-09B1 free-running greedy inference                ✅ MERGED
TR-POLY-09B2 deterministic metric/adaptor layer           ✅ MERGED
TR-POLY-09B3 deterministic VALIDATION aggregation         ✅ MERGED
TR-POLY-09B4 deterministic quality-training regime        ✅ IMPLEMENTED / MERGE GATE
quality-checkpoint artifact                               🔒 NEXT
first native-V2 quality candidate                         🔒
quality VALIDATION B1→B2→B3                              🔒
P09C evidence-driven refinement                           🔒
missing metric admission                                  🔒 5 surfaces
P09D final candidate freeze                               🔒
Stage 9 sealed TEST decision                              🔒
Stage 10 ScoreMosaic shadow integration                   🔒
```

## 2. Measurement architecture

The model-quality measurement path is structurally available once a real quality checkpoint exists:

```text
native V2 VALIDATION image + canonical target
        ↓
exact verified quality checkpoint
        ↓
B1 free-running greedy prediction
        ↓
strict V2 parse OR explicit invalid/abstain
        ↓
B2 deterministic per-sample metrics
        ↓
B3 deterministic candidate aggregation
        ├─ overall
        ├─ 1 / 2 / 3 / 4+ voice
        └─ robustness buckets
```

B1 removes teacher forcing, B2 measures each prediction, and B3 prevents failure cases from disappearing in aggregation.

## 3. Why the earlier training path is not enough

TR-POLY-08A and TR-POLY-08B were intentionally designed to prove bounded deterministic training/checkpoint infrastructure. Both enforce a maximum of two optimizer steps.

That is sufficient to verify:

- gradient updates;
- TRAIN-only mutation;
- read-only VALIDATION;
- finite-state guards;
- checkpoint serialization/reload;
- provenance and hash binding.

It is not sufficient to estimate the achievable recognition quality of the 2D Transformer.

The smoke path therefore remains frozen as historical infrastructure evidence and must never be silently widened or relabeled as a quality-training run.

## 4. TR-POLY-09B4 quality-training regime

B4 creates a separate multi-epoch control plane while reusing the already-hardened one-step optimization primitive.

```text
quality config + quality provenance
        ↓
exact TRAIN tuple + exact VALIDATION tuple
        ↓
frozen seed model initialization
        ↓
Epoch 1
  TRAIN batch 1 → update
  TRAIN batch 2 → update
  ...
  full VALIDATION → read-only mean loss
        ↓
Epoch 2 ... N
        ↓
fixed budget completes
        ↓
minimum mean VALIDATION loss
        ↓
earliest exact-loss tie
        ↓
reload selected epoch state
        ↓
selected-state SHA-256 verification
```

### Frozen v1 policies

- optimizer: AdamW via the existing TR-POLY-08A primitive;
- loss: V2 PAD-aware cross-entropy;
- scheduler: none;
- batch order: exact caller-supplied tuple order;
- shuffle/sampling: none;
- validation interval: every epoch;
- validation coverage: every supplied VALIDATION batch;
- selection: minimum arithmetic-mean VALIDATION loss;
- tie: earliest epoch;
- stopping: fixed configured epoch budget;
- TEST: prohibited;
- production authority: prohibited.

A future shuffled, scheduled or early-stopped regime must receive a new version and cannot reuse the v1 trainer fingerprint.

## 5. Quality recipe identity

The quality trainer has an independent fingerprint that binds:

- quality trainer version;
- quality plan version;
- exact quality config;
- model profile;
- tokenizer and representation versions;
- pinned PyTorch runtime;
- underlying one-step primitive profile.

The run additionally fingerprints the **ordered batch plan**. The plan includes split, ordered sample IDs, dataset identity and tensor shapes for each supplied TRAIN/VALIDATION batch.

Therefore changing batch order changes the training-plan SHA even when the underlying sample set is unchanged.

## 6. Data separation

B4 accepts only existing `Poly2DTrainingBatch` values, so the earlier batch-level bounds remain active.

Additional B4 guards require:

- every TRAIN entry is actually TRAIN;
- every VALIDATION entry is actually VALIDATION;
- one exact dataset manifest across all batches and provenance;
- no duplicate sample ID anywhere across the supplied TRAIN/VALIDATION plan;
- the planned optimizer-step count fits the configured bound before training begins.

TEST cannot be constructed through the accepted batch boundary and receives no escape hatch in B4.

## 7. B4 resource ceilings

The v1 safety ceilings are deliberately larger than the smoke harness while still finite:

- maximum epochs: 128;
- maximum optimizer steps: 100,000;
- maximum supplied batches per split: 100,000;
- inherited per-batch image, batch-size, target-length, vocabulary and finite-value limits.

These values are ceilings, not a claim that the default recipe is optimal.

The default v1 quality config uses eight epochs and an 8,192-step ceiling. Real experiment parameters remain fingerprinted and must be frozen before execution.

## 8. Epoch and run evidence

Each epoch emits:

- epoch number;
- cumulative optimizer steps;
- mean TRAIN loss;
- mean read-only VALIDATION loss;
- exact model-state SHA-256.

The run result binds:

- initial model-state SHA;
- selected model-state SHA;
- selected epoch;
- selected validation loss;
- every epoch record;
- exact training-plan SHA;
- model profile;
- quality-trainer profile;
- underlying one-step primitive profile;
- quality provenance;
- dataset identity;
- tokenizer fingerprint;
- repository SHA;
- pinned runtime;
- explicit `test_split_accessed=false`;
- explicit `production_authority=false`.

The returned model is reloaded to the selected epoch state and its exact state SHA is checked against the result.

## 9. Checkpoint dependency

B4 deliberately stops before artifact persistence.

The old TR-POLY-08B checkpoint cannot be reused because its schema semantically means “bounded ≤2-step research checkpoint” and validates that bound on reload.

The next package must provide a separate quality-checkpoint schema with:

- selected B4 state only;
- exact B4 config and provenance;
- B4 result fingerprint / selected epoch evidence;
- non-overwriting artifact directory;
- checkpoint/metadata/receipt hashes;
- hash verification before deserialization;
- `torch.load(..., weights_only=True)`;
- strict model-state reload;
- dedicated research registry binding;
- `benchmark_evidence=false`;
- `production_authority=false`;
- `test_split_accessed=false`.

A verified quality-checkpoint loader then needs a B1 wrapper that creates a checkpoint-bound inference identity without weakening the existing smoke-checkpoint wrapper.

## 10. B1–B3 measurement rules remain unchanged

Invalid free-running predictions remain in the metric population. B3 continues to require one exact benchmark identity and one exact checkpoint/candidate identity.

Required voice strata remain:

- `1_voice`;
- `2_voice`;
- `3_voice`;
- `4_plus_voice`.

Observed robustness buckets remain separate:

- clean;
- scan;
- phone;
- blur;
- perspective;
- low contrast.

## 11. Current metric coverage

Numeric:

- `parse_success`;
- `ter`;
- `normalized_edit_distance`;
- `exact_sequence_accuracy`;
- `pitch_accuracy`;
- `duration_accuracy`;
- `onset_accuracy`;
- `voice_accuracy`;
- `staff_accuracy`;
- `accidental_note_f1`;
- `note_staff_f1`.

Still unsupported:

- `musicxml_validity`;
- `tedn`;
- `notehead_stem_f1`;
- `beam_relation_f1`;
- `tie_relation_f1`.

Partial VALIDATION evidence may guide P09C, but complete winner/promotion claims remain closed while required metrics are unsupported.

## 12. First quality experiment order

```text
B4 exact-head green merge
        ↓
quality-checkpoint contract + reload
        ↓
freeze exact native-V2 TRAIN/VALIDATION manifest/build
        ↓
freeze exact B4 training recipe
        ↓
train on TRAIN only
        ↓
select/freeze best VALIDATION-loss checkpoint
        ↓
run checkpoint on VALIDATION via B1
        ↓
B2 per-sample metrics
        ↓
B3 voice/robustness aggregation
        ↓
P09C failure-driven refinement
```

The first quality run must be treated as a baseline, not as the final architecture.

## 13. P09C decision policy

Use measured quality evidence rather than model-size intuition:

```text
high parse failure       → decoder/search diagnosis
high TER, valid parses   → sequence modeling/search diagnosis
weak pitch               → visual/pitch evidence diagnosis
weak duration/onset      → rhythm/data diagnosis
weak 2+/3+/4+ voice      → polyphonic separation/data diagnosis
weak note_staff_f1       → staff/cross-staff diagnosis
strong clean, weak scan  → robustness/domain coverage diagnosis
```

Only demonstrated failure families should justify model/data/search expansion.

## 14. Safety invariants

- TEST remains sealed until Stage 9.
- TRAIN is the only parameter-updating split.
- VALIDATION is read-only and may select a candidate but never update it.
- Smoke contracts remain immutable.
- Quality training receives separate version/provenance identities.
- Unsupported benchmark metrics never receive proxy numbers.
- Invalid/abstain outputs remain in benchmark denominators.
- Different benchmark/candidate identities cannot be mixed.
- External data requires rights/license/install-pin admission.
- Teacher corrections and ScoreMosaic uploads are not automatic training data.
- No quality-training or benchmark artifact grants production authority.

## Immediate next action

Finish TR-POLY-09B4 exact-head CI/merge. Then implement the separate quality-checkpoint artifact/reload contract and checkpoint-bound B1 inference bridge. Only after those gates are green should the first real native-V2 quality training/VALIDATION experiment run. TEST remains sealed.
