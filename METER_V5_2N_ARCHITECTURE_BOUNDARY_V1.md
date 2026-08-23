# Meter V5-2N — Adapter Architecture Boundary V1

This boundary records what may and may not be inferred from the existing adapter code before any new repair training.

## Current target lane

The current Meter digit lane remains two independent binary specialists, 2-AI and 3-AI, operating on the exact approved 64x64 staff-relative digit-slot crops. 4-AI remains frozen.

The current specialist representation immediately before the binary linear head is 64-dimensional.

## Existing D11 adapter code

The repository already contains older real-domain adapters (`meter_real_domain_adaptation_v2`, `v3_a1`, `v3_a2`). They demonstrate a useful generic pattern: freeze a base model and place a zero-initialized residual adapter above it so initialization preserves base behavior.

They are **not** implementation templates for the current digit lane. They belong to the older D11 four-class full-meter model and include 96x96/glyph-window spatial semantics. Those spatial rules are not imported, copied, adapted, or experimentally reused by V5-2N.

## Only conclusion authorized by this stage

A future frozen-base additive residual is an architectural **candidate class**, not a selected implementation. V5-2N first measures whether the exact frozen 64D digit-specialist features transfer to V5. No residual topology, hidden width, loss coefficient, optimizer, learning rate, epoch count, source penalty or training recipe is selected here.

Any future repair must continue to use the existing approved slot pixels unless a new spatial rule receives separate explicit user approval.
