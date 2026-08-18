# Meter V2 upstream topology gate

Status: **SHADOW-ONLY / PREFLIGHT REQUIRED / NOT PRODUCTION-ACCEPTED**

## Why this gate exists

Meter Slot Geometry Adapter v1 is intentionally downstream of the shared runtime geometry lane. It must consume the same accepted staff/system/measure identities that other specialists consume; it must not silently invent a private Meter-only page topology.

Current `runtime_geometry_engine_v2.py` can emit multiple detected five-line staffs, but when accepted it places every detected staff in one runtime system (`system-1`). `runtime_measure_geometry_v1.py` then checks cross-staff measure boundaries within each declared system.

Therefore a source page that truly contains more than one musical system cannot be called a faithful end-to-end upstream replay until the shared runtime lane can preserve that system topology. Treating all staffs on such a page as one system can turn a page/system-layout limitation into an apparent Meter failure.

## Required preflight

`tools/meter_v2_upstream_topology_preflight_v1.py` reads only frozen D6/D10 **VALIDATION metadata/labels** and reports:

- number of systems per source page;
- number of staff instances per page;
- staffs per system;
- measures per system;
- count of pages with more than one system;
- exact D10 eight-measure linkage per D6 validation sample.

It does not read TEST, load checkpoints, run models, tune thresholds, mutate Drive, wire Resolver, or promote production code.

## Decision rule

- `READY_FOR_PIXEL_REPLAY`: all 153 D6 validation pages can be represented by the current one-system runtime topology and metadata is valid.
- `HOLD_SYSTEM_GROUPING_REQUIRED`: at least one validation page contains more than one true system. Do **not** force the Meter replay through a false one-system topology. Freeze a separate deterministic shared staff-to-system grouping contract first.
- `HOLD_INVALID_TOPOLOGY_METADATA`: D6/D10 linkage or system references are malformed; fail closed and investigate provenance.

No measure-index-specific Meter workaround is permitted for a shared page/system topology problem.

## Scope boundary

This gate does **not** authorize a new System Geometry implementation. A shared system-grouping layer changes upstream runtime architecture and requires its own explicit design/acceptance scope before implementation. Meter V2 remains shadow-only until that boundary is resolved and a real pixel replay succeeds.
