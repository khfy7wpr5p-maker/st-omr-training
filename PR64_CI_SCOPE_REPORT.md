# PR #64 — CI Scope Report

Status: **SHADOW ACCEPTANCE GREEN / REAL MODEL NOT YET ACCEPTED / RESOLVER NOT YET VALIDATED**

This report exists to prevent the PR #64 CI result from being misread as proof that the real Accidental model or a production Resolver integration is accepted.

## 1. Provenance

- PR: `#64` — `Runtime: shadow-accept real specialists before resolver wiring`
- Base: `main` at `92f0249d1722cbe0ac75d88b6997a8b7e7747095`
- CI-validated head before this report commit: `d9f3ff0f28ce8df1d5c5950e2ec7100bebf4e2cf`
- CI run: `32140691233`
- CI job: `95722533896` (`Python 3.13 / Ubuntu`)
- Full suite result: `696 tests`, `OK (expected failures=1)`
- Python compile step: PASS
- Sealed TEST access: **not used**

The CI runner does not download private Drive checkpoints. Therefore CI can validate frozen evidence contracts and deterministic shadow behavior, but it cannot by itself establish a new real-checkpoint acceptance result.

## 2. Gate A — What CI/shadow acceptance DID validate

### A1. Frozen Accidental R2 key/local contract

The following 22 design-frozen cases are represented and compared against deterministic expected outputs:

- Key-signature cases: `K01`–`K06` — 6/6 PASS
- Local/mixed accidental cases: `L01`–`L10` — 10/10 PASS
- Fail-closed ambiguity/rejection cases: `A01`–`A06` — 6/6 PASS
- Total: **22/22 PASS**

### A2. Deterministic invariants validated

CI verifies that the shadow evaluator distinguishes:

- two separate sharps `# #` from one double-sharp glyph;
- two separate flats `b b` from one double-flat glyph;
- staff-start key-signature evidence from measure-local accidental evidence according to the frozen case inputs;
- key-signature output from mode inference (`mode=UNKNOWN` is preserved);
- ambiguous or invalid evidence from accepted musical mutation.

### A3. Fail-closed behavior validated

For the frozen ambiguity/rejection cases, CI verifies that no key signature or local note alteration is created when the expected status is:

- `AMBIGUOUS`
- `AMBIGUOUS_TARGET`
- `REJECTED`

### A4. Isolation validated

CI verifies that the Accidental R2 shadow package:

- does not import/use Torch;
- does not construct/use an optimizer;
- does not connect to `runtime_deterministic_resolver`;
- does not authorize Resolver connection;
- does not open sealed TEST;
- does not change D10/D13 training code or checkpoints.

## 3. Gate B — What CI/shadow acceptance DID NOT validate

The green `22/22` result is **not** evidence for any of the following:

1. The real Accidental neural model recognizing `sharp`, `flat`, `natural`, `double_sharp`, or `double_flat` from raster images with the required quality.
2. The current real Accidental checkpoint reaching the frozen D13 quality gates after epoch 10.
3. Per-class real-model recall/precision/F1 for sharp, flat, natural, double-sharp, or double-flat.
4. Real-image separation of staff-start key signatures versus measure-local accidentals.
5. Real geometry robustness for clef-relative key-signature zones, canonical staff positions, notehead association, staff overlap, or degraded/scanned pages.
6. A real specialist adapter feeding actual Accidental predictions into the runtime Resolver.
7. End-to-end raster -> Accidental specialist -> key/local router -> deterministic composer -> Resolver behavior.
8. Production promotion, Resolver wiring, or merge authorization.

The 22 cases are deterministic contract fixtures. They prove that the frozen rules behave as specified for those inputs; they do **not** prove that a neural detector will produce the correct inputs from real images.

## 4. Current real Accidental model status

The last accepted real-checkpoint evidence frozen in PR #64 remains the epoch-05 Accidental checkpoint:

- completed epochs: `5/10`
- Center F1: `0.9037118` — passes gate `>= 0.80`
- BBox F1: `0.9034379` — passes gate `>= 0.70`
- Macro F1: `0.7938513` — fails gate `>= 0.85`
- shadow decision: **HOLD**

A separate authoritative resume is currently running outside GitHub CI on the exact historical training SHA `cf82ecbc0ef8df3d635e6e1923b4c4000c40da5b`, with sealed TEST still closed.

Latest external Drive log observed for this report:

- preflight records: `11064`
- TRAIN: `9840`
- VALIDATION: `1224`
- TEST: `0`
- latest observed preflight progress: `8500/11064` (`76.8%`)
- no verified epoch-06-or-later training metric is yet available in this report

Therefore the real Accidental model remains **NOT ACCEPTED / HOLD** at report time.

## 5. Current Resolver status

PR #64 intentionally contains no real Accidental-to-Resolver wiring.

Validated:

- shadow package cannot authorize a Resolver connection;
- deterministic case evaluator can produce frozen key/local outputs from explicit synthetic evidence groups.

Not validated:

- real Accidental model adapter;
- real confidence/bbox/staff evidence ingestion;
- real key-signature-zone routing from image geometry;
- real local-note target binding;
- real conflict arbitration across symbol specialists;
- end-to-end Resolver behavior with real Accidental predictions.

Resolver status for Accidental remains: **NOT WIRED / NOT END-TO-END VALIDATED**.

## 6. Known limitations

1. The current legacy Accidental D13 model vocabulary is only `sharp`, `flat`, `natural`; `double_sharp` and `double_flat` are design targets for Accidental R2, not validated classes of the current real model.
2. The 22-case acceptance pack uses explicit evidence groups; it does not measure detector/localizer error.
3. Key-signature recognition is currently validated as deterministic rule behavior, not as full image-derived geometry behavior.
4. Major/minor mode inference is deliberately out of scope; key signature alone keeps `mode=UNKNOWN`.
5. Tie, octave, voice, cautionary accidental, key-change, cancellation, and other advanced accidental-scope cases are deferred to later deterministic acceptance packs.
6. Sealed TEST remains unopened; all development decisions must continue to use TRAIN/VALIDATION-only evidence.
7. Rest R2 remains separately on HOLD because integrated deterministic Rest class arbitration is not yet frozen/accepted.

## 7. Remaining controls before PR #64 merge

These controls apply to merging PR #64 as a **shadow-acceptance/infrastructure PR**. A merge must not be represented as production Accidental acceptance.

1. Re-run CI on the exact final PR head after this report commit and require green full suite + compile.
2. Fresh-read the final PR diff and confirm no forbidden scope drift:
   - no Resolver wiring;
   - no D10/D13 training mutation;
   - no checkpoint binary;
   - no TEST access;
   - no production promotion flag.
3. Confirm the PR remains draft until explicit merge approval is given.
4. Confirm the report and PR body continue to label the real Accidental model as `HOLD` unless new externally verified checkpoint evidence is deliberately frozen in a later commit.

The real model does **not** need to be promoted merely to merge a shadow-only HOLD contract; however a PR #64 merge must not unlock Resolver wiring or production use.

## 8. Additional gates required before any real Accidental Resolver wiring/promotion

These are **post-shadow / pre-production** gates and are distinct from PR #64's deterministic CI pass:

1. Complete the authoritative Accidental epoch `6 -> 10` resume with TEST closed.
2. Verify persisted checkpoint/metadata hashes, exact repository SHA, optimizer-step count, finite state, and authoritative completion receipt.
3. Evaluate the selected final/best checkpoint on VALIDATION and require:
   - Center F1 `>= 0.80`
   - BBox F1 `>= 0.70`
   - Macro F1 `>= 0.85`
4. If Macro F1 remains below gate, run class-specific TRAIN/VALIDATION diagnostics before deciding Accidental R2 architecture.
5. For Accidental R2, validate real specialists/verifiers for `sharp`, `flat`, `natural`, and—when trained with sufficient data—`double_sharp`, `double_flat`.
6. Run real-image key-signature/local-accidental separation acceptance, not only explicit evidence fixtures.
7. Freeze and test deterministic context routing using clef-relative key zone, canonical staff position/order, structural boundaries, staff identity, and note-target association.
8. Add an isolated real-model adapter and run end-to-end shadow validation through the Resolver without production promotion.
9. Require fail-closed behavior for low confidence, invalid bbox, staff ambiguity, target ambiguity, class conflict, NaN/Inf, and unsupported symbol/state.
10. Run focused tests plus full CI on the wiring branch.
11. Obtain explicit approval before merge/promotion of any Resolver-wiring PR.

## 9. Decision summary

| Surface | Current status | Meaning |
|---|---|---|
| Accidental R2 deterministic 22-case contract | **PASS 22/22** | Rules match frozen expected outputs |
| PR #64 full CI at `d9f3ff0...` | **PASS** | Shadow code/tests compile and full suite is green |
| Real Accidental epoch-05 checkpoint | **HOLD** | Macro F1 below gate; run incomplete |
| Resumed epoch 6–10 training | **IN PROGRESS / preflight observed** | No new accepted model metric yet |
| Real double-sharp/double-flat model recognition | **NOT VALIDATED** | R2 design target only |
| Real key/local image separation | **NOT VALIDATED** | Deterministic fixture behavior only |
| Accidental -> real Resolver wiring | **NOT WIRED** | Explicitly forbidden in PR #64 |
| Production promotion | **NOT AUTHORIZED** | Requires later gates + explicit approval |
| Sealed TEST | **CLOSED** | Not used |

**Bottom line:** PR #64 currently proves a green shadow/deterministic acceptance contract. It does not prove that the real Accidental model has passed quality gates, and it does not prove real Resolver integration. Those remain separate, later acceptance surfaces.
