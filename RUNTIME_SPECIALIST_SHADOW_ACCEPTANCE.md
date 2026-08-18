# ST-OMR Real Specialist Shadow Acceptance v1

Status: **shadow-only; no real specialist is wired to Deterministic Resolver**.

This gate examines persisted real specialist artifacts before runtime integration. Checkpoint binaries remain external to GitHub. Sealed TEST is not opened and no optimizer/training step is run.

## Common runtime evidence output

A future adapter may emit only the existing model-agnostic `SpecialistObservation` boundary:

- `task`
- `measure_id`
- `staff_id`
- `status = accepted|ambiguous|rejected`
- `confidence_milli`
- allowed `class_label`
- localized `bbox`
- fail-closed `reasons`

The shadow package itself always returns `resolver_connection_allowed() == false`.

## NoteHead

Real artifact tested: D13 NoteHead epoch-10 checkpoint.

Input contract:

- float32 tensor `[B,1,128,512]`;
- finite values only;
- grayscale measure representation, dark ink mapped toward 1 and white background toward 0;
- exact class vocabulary `open|filled`.

Raw output contract:

- `heatmap_logits`: `[B,2,32,128]`;
- `bbox_size`: `[B,2,32,128]`, finite and positive;
- `center_offset`: `[B,2,32,128]`, finite and sigmoid-bounded.

Frozen decoder/evaluation thresholds:

- local max kernel: 3;
- score threshold: 0.25;
- top-k: 256 per measure;
- center tolerance: 4 px;
- bbox IoU threshold: 0.50;
- acceptance: center F1 >= 0.85, bbox F1 >= 0.75, macro F1 >= 0.90.

Persisted epoch-10 evidence:

- center F1: `0.9882845985`;
- bbox F1: `0.9882845985`;
- macro F1: `0.9855884316`;
- completed epochs: `10/10`.

Real-checkpoint smoke evidence:

- checkpoint SHA-256: `f34de0a6a3627421ea2c7e0f23d007de94c67576e16366183b6d60b96c14a106`;
- same fixed synthetic input: `10/10` identical output digest;
- output digest: `1d78abb3923ef49a565b9b14b38919f6e604648b07915ff8251a4cf160176968`;
- wrong height, wrong channel count, float64 input and NaN input: `4/4` rejected.

Decision: **PASS for shadow adapter development**. This is not production promotion and not Resolver wiring.

## Accidental

Real artifact tested: latest persisted D13 Accidental epoch-05 checkpoint. The original run did not complete the frozen 10-epoch profile.

Input contract:

- float32 `[B,1,128,512]`, finite only;
- exact classes `sharp|flat|natural`.

Raw output contract:

- `heatmap_logits`: `[B,3,32,128]`;
- `bbox_size`: `[B,2,32,128]`;
- `center_offset`: `[B,2,32,128]`.

Thresholds:

- decoder score >= 0.25, local max 3, top-k 256;
- center tolerance 4 px, bbox IoU 0.50;
- acceptance center F1 >= 0.80, bbox F1 >= 0.70, macro F1 >= 0.85.

Persisted epoch-05 evidence:

- center F1: `0.9037118` — pass;
- bbox F1: `0.9034379` — pass;
- macro F1: `0.7938513` — **fail**;
- completed epochs: `5/10` — **incomplete**.

Real-checkpoint smoke evidence:

- checkpoint SHA-256: `39d8c812ab793aaf318bb92881f94d4d286bbe6783a6319e4c326759994e3bf9`;
- deterministic output: `10/10` identical;
- digest: `d5f7c37bd0ca578fee0d5309b1c1918b734fc5e36d9ceb0dcf1402f4aefbc954`;
- malformed/non-finite inputs: `4/4` fail closed.

Decision: **HOLD**. Do not build a Resolver connection from this checkpoint.

## Rest R2

Rest R2 is no longer safely represented by one three-class checkpoint. The later architecture is:

`high-recall Rest proposals -> value-specific half/quarter/eighth verifier -> deterministic arbitration -> SpecialistObservation`.

The generic R2 proposal checkpoint was smoke-tested as an external artifact:

- SHA-256: `89dfe890961a42f13a8a2b29df4808649cefc2b20a9edf42b6107f6b75f3f35a`;
- `10/10` deterministic output;
- digest: `6f27991728fcdf16f62cdaee3bb3db0eee5ca7fb3d2ccad92b299fdd83a55fb0`;
- `4/4` malformed/non-finite inputs rejected;
- artifact metadata explicitly says it is not a production checkpoint.

Value-specific frozen evidence:

| Class | Proposal threshold | Verifier threshold | Recall | FP reduction | Gate | TEST | Promotion |
|---|---:|---:|---:|---:|---|---|---|
| half | 0.50 | 0.0706893578 | 1.0000 | 0.8125 | PASS | closed | false |
| quarter | 0.20 | 0.3782260716 | 0.989071 | 0.752369 | PASS | closed | false |
| eighth | 0.50 | 0.5620679259 | 0.985612 | 0.792714 | PASS | closed | false |

Quarter and eighth evidence is bound to frozen checkpoint SHAs. Half final closure is bound to its recorded closure fingerprint; the final closure package does not expose a copied checkpoint binary, so this contract deliberately does not mislabel the closure fingerprint as a checkpoint SHA.

Rest fail-closed adapter rule:

- proposal below its class threshold -> `ambiguous`, not a Rest label;
- verifier score below its frozen class threshold -> `ambiguous`;
- NaN/Inf or invalid bbox -> `rejected`;
- two Rest classes accepted for the same proposal without a frozen tie-break -> `ambiguous`;
- no class may be invented from musical duration context at this layer.

Decision: **HOLD**. The three value-specific gates pass, but a deterministic integrated half/quarter/eighth arbitration adapter is not yet frozen or tested, and all promotion flags remain false.

## Shadow acceptance matrix

| Specialist | Real artifact smoke | Quality/final gate | Complete/integrated | Shadow decision | Resolver connected |
|---|---|---|---|---|---|
| NoteHead | PASS 10/10 | PASS | 10/10 epochs | **PASS** | no |
| Accidental | PASS 10/10 | macro FAIL | 5/10 epochs | **HOLD** | no |
| Rest R2 | PASS 10/10 proposal + class gates PASS | PASS per class | arbitration missing | **HOLD** | no |

## CI boundary

GitHub CI validates the frozen identities, thresholds, decisions, uniqueness, finite-value checks, 10/10 evidence records, and that this package cannot import/connect the deterministic Resolver. CI does **not** download private Drive checkpoints. Real checkpoint execution is a separate external shadow evidence step, keeping checkpoint binaries out of the repository.
