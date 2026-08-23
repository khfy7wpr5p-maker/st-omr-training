# Meter V5-2M — Historical Retention Contract Correction V3

## Why this correction is required

V5-2C V2 corrected the historical M4A validation oracle to the exact SHA-bound frozen checkpoint results:

- 2-AI: TP=185, FP=30, FN=1, TN=3156
- 3-AI: TP=203, FP=1, FN=1, TN=3167
- 4-AI: TP=788, FP=46, FN=4, TN=2534

However, the V2 evaluator inherited V1's absolute candidate precision floor `precision >= 0.98`. That floor was preregistered against V1's invalid oracle, where 2-AI had been incorrectly represented with only 4 false positives. Under the corrected V2 oracle, frozen 2-AI precision is `185 / (185 + 30) = 0.8604651162790697`.

Therefore `candidate precision >= 0.98` is not a retention criterion for 2-AI; it requires a candidate to materially outperform the frozen baseline. Absolute quality targets, if desired later, must be defined separately from retention.

## Corrected retention-only gate

For 2-AI and 3-AI, compare each candidate directly to its exact corrected frozen V2 baseline at the frozen threshold. PASS requires all of:

- F1 drop <= `0.005`
- recall drop <= `0.005`
- precision drop <= `0.005`

No absolute precision or recall floor is used in this retention-only gate.

This is intentionally symmetric and baseline-relative. It answers only: "did adaptation materially degrade the frozen historical behavior?"

## Non-retroactivity

This correction does not reinterpret V5-2L as PASS. V5-2L remains HOLD because it independently exceeds the relative F1 and recall degradation limits and fails to learn V5 adaptation TRAIN. The V5-2L first-30 diagnostic remains NOT RUN.

V3 applies prospectively to future repair candidates and to explicit read-only re-audits. It does not alter historical JSON evidence files.

## Safety boundary

V5-2M authorizes no training, optimizer step, checkpoint write, threshold tuning, BBox/crop/spatial change, reserve TRAIN opening, V5 validation opening, FINAL_HOLDOUT opening, 4-AI mutation, Resolver wiring, or production promotion.
