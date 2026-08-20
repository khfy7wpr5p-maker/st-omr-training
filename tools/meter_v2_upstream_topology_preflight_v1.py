"""Read-only Meter V2 upstream system-topology preflight.

This diagnostic intentionally stops before raster/model replay. It checks whether
D6 VALIDATION topology can be represented faithfully by the current runtime
Geometry v2 -> Measure Geometry v1 lane before Meter consumes that lane.

No training, threshold tuning, TEST access, checkpoint access, Resolver wiring,
Drive writes, production promotion, or merge authority.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

DRIVE = Path("/content/gdrive_r2/MyDrive")
SYNTH = DRIVE / "ST-OMR-SYNTHETIC" / "d9320e362f162cd2ace2a830a7b93e0c21ceba2d51a4e95ef1c7a9b11a108352"
D6 = SYNTH / "stage7d6-staff-structure-derivatives-f33e70ec24a60ebab547ed7d4a395902129b0e23"
D10 = DRIVE / "ST-OMR-D10" / "stage7d10-authoritative-562c8fcfabf1b41573f1ef591d88ae65335ce16a"
OUT = Path("/content/meter-v2-upstream-topology-preflight-v1")

EXPECTED_D6_MANIFEST_SHA256 = "e8e415eb6ba9d91a1a880709c3f31d559aa20bf5149734f45b5f84ced16afee9"
EXPECTED_D10_MANIFEST_SHA256 = "6927e1bcc5251257a983a306e2f1875c9515f97c6724a8fe9f24382c6ff30db4"
EXPECTED_D6_VALIDATION = 153
EXPECTED_D10_METER_VALIDATION = 1224
EXPECTED_MEASURES_PER_SAMPLE = 8


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _counter_json(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def main() -> None:
    d6_manifest_path = D6 / "manifest.json"
    d10_manifest_path = D10 / "manifest.json"
    if not d6_manifest_path.is_file() or not d10_manifest_path.is_file():
        raise RuntimeError("required D6/D10 manifests are not available on mounted Drive")
    if _sha(d6_manifest_path) != EXPECTED_D6_MANIFEST_SHA256:
        raise RuntimeError("D6 manifest SHA mismatch")
    if _sha(d10_manifest_path) != EXPECTED_D10_MANIFEST_SHA256:
        raise RuntimeError("D10 manifest SHA mismatch")

    d6 = _load(d6_manifest_path)
    d10 = _load(d10_manifest_path)
    if d10.get("test_records") != 0:
        raise RuntimeError("D10 TEST surface is not sealed")
    if any(row.get("split") == "test" for row in d6.get("records", ())):
        raise RuntimeError("D6 manifest unexpectedly exposes TEST")
    if any(row.get("split") == "test" for row in d10.get("records", ())):
        raise RuntimeError("D10 manifest unexpectedly exposes TEST")

    d6_val = [row for row in d6["records"] if row.get("split") == "validation"]
    meter_val = [
        row for row in d10["records"]
        if row.get("split") == "validation" and row.get("kind") == "meter"
    ]
    if len(d6_val) != EXPECTED_D6_VALIDATION:
        raise RuntimeError(f"expected {EXPECTED_D6_VALIDATION} D6 validation samples, got {len(d6_val)}")
    if len(meter_val) != EXPECTED_D10_METER_VALIDATION:
        raise RuntimeError(
            f"expected {EXPECTED_D10_METER_VALIDATION} D10 validation Meter records, got {len(meter_val)}"
        )

    meter_by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in meter_val:
        meter_by_sample[str(row["source_sample_id"])].append(row)
    d6_ids = {str(row["sample_id"]) for row in d6_val}
    if set(meter_by_sample) != d6_ids:
        raise RuntimeError("D10 validation source_sample_id inventory does not match D6 validation")

    system_counts = Counter()
    staff_counts = Counter()
    staff_per_system = Counter()
    measures_per_system = Counter()
    topology_pairs = Counter()
    multi_system_pages: list[dict[str, object]] = []
    invalid: list[dict[str, object]] = []

    for index, row in enumerate(sorted(d6_val, key=lambda item: str(item["sample_id"])), start=1):
        sample_id = str(row["sample_id"])
        label_sha = str(row["label_sha256"])
        label_path = D6 / "labels" / f"{label_sha}.json"
        if not label_path.is_file():
            raise RuntimeError(f"missing D6 validation label: {label_path}")
        raw = label_path.read_bytes()
        if sha256(raw).hexdigest() != label_sha:
            raise RuntimeError(f"D6 label SHA mismatch for {sample_id}")
        label = json.loads(raw.decode("ascii"))
        if label.get("split") != "validation" or label.get("sample_id") != sample_id:
            raise RuntimeError(f"D6 label identity mismatch for {sample_id}")
        geometry = label.get("geometry")
        if not isinstance(geometry, dict):
            raise RuntimeError(f"missing D6 geometry for {sample_id}")

        systems = geometry.get("systems")
        staffs = geometry.get("staff_instances")
        measures = geometry.get("measures")
        if not isinstance(systems, list) or not isinstance(staffs, list) or not isinstance(measures, list):
            raise RuntimeError(f"malformed D6 topology for {sample_id}")

        system_ids = [str(item.get("system_id")) for item in systems if isinstance(item, dict)]
        if len(system_ids) != len(systems) or len(set(system_ids)) != len(system_ids):
            invalid.append({"sample_id": sample_id, "reason": "SYSTEM_ID_INVENTORY"})
            continue
        system_set = set(system_ids)
        staff_systems = [str(item.get("system_id")) for item in staffs if isinstance(item, dict)]
        measure_systems = [str(item.get("system_id")) for item in measures if isinstance(item, dict)]
        if any(value not in system_set for value in staff_systems + measure_systems):
            invalid.append({"sample_id": sample_id, "reason": "UNKNOWN_SYSTEM_REFERENCE"})
            continue

        d10_rows = meter_by_sample[sample_id]
        numbers = sorted(int(item["measure_number"]) for item in d10_rows)
        if len(d10_rows) != EXPECTED_MEASURES_PER_SAMPLE or numbers != list(range(1, 9)):
            invalid.append({"sample_id": sample_id, "reason": "D10_MEASURE_INVENTORY"})
            continue

        system_counts[len(systems)] += 1
        staff_counts[len(staffs)] += 1
        topology_pairs[(len(systems), len(staffs))] += 1
        for system_id in system_ids:
            staff_per_system[sum(1 for value in staff_systems if value == system_id)] += 1
            measures_per_system[sum(1 for value in measure_systems if value == system_id)] += 1

        if len(systems) > 1:
            multi_system_pages.append(
                {
                    "sample_id": sample_id,
                    "family_id": row.get("family_id"),
                    "system_count": len(systems),
                    "staff_count": len(staffs),
                    "measures_by_system": [
                        sum(1 for value in measure_systems if value == system_id) for system_id in system_ids
                    ],
                    "staffs_by_system": [
                        sum(1 for value in staff_systems if value == system_id) for system_id in system_ids
                    ],
                }
            )

    if invalid:
        state = "HOLD_INVALID_TOPOLOGY_METADATA"
    elif multi_system_pages:
        # Runtime Geometry v2 currently emits one system containing every accepted staff.
        # A multi-system D6 page therefore cannot be replayed faithfully through the
        # current Geometry v2 -> Measure Geometry v1 lane without first adding a
        # deterministic staff-to-system grouping boundary.
        state = "HOLD_SYSTEM_GROUPING_REQUIRED"
    else:
        state = "READY_FOR_PIXEL_REPLAY"

    report = {
        "stage": "meter-v2-upstream-topology-preflight-v1",
        "state": state,
        "claim_boundary": "D6_D10_VALIDATION_TOPOLOGY_READONLY_NO_PIXEL_OR_MODEL_REPLAY",
        "d6_validation_samples": len(d6_val),
        "d10_meter_validation_records": len(meter_val),
        "system_count_distribution": _counter_json(system_counts),
        "staff_count_distribution": _counter_json(staff_counts),
        "topology_system_staff_distribution": {
            f"systems={systems},staffs={staffs}": count
            for (systems, staffs), count in sorted(topology_pairs.items())
        },
        "staffs_per_system_distribution": _counter_json(staff_per_system),
        "measures_per_system_distribution": _counter_json(measures_per_system),
        "multi_system_page_count": len(multi_system_pages),
        "multi_system_examples_first_20": multi_system_pages[:20],
        "invalid_count": len(invalid),
        "invalid_examples_first_20": invalid[:20],
        "current_runtime_assumption": {
            "geometry_v2_system_count_when_accepted": 1,
            "all_detected_staffs_share_system_id": "system-1",
            "measure_geometry_cross_staff_boundary_check_is_system_scoped": True,
        },
        "safety": {
            "training_started": False,
            "optimizer_steps_added": 0,
            "threshold_tuning": False,
            "test_opened": False,
            "checkpoint_access": False,
            "pixel_replay_started": False,
            "drive_write": False,
            "resolver_wiring": False,
            "production_promotion": False,
        },
        "source_sha256": {
            "d6_manifest": EXPECTED_D6_MANIFEST_SHA256,
            "d10_manifest": EXPECTED_D10_MANIFEST_SHA256,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("METER V2 UPSTREAM TOPOLOGY PREFLIGHT COMPLETE")
    print("STATE        :", state)
    print("D6 VALIDATION:", len(d6_val))
    print("D10 METER VAL:", len(meter_val))
    print("SYSTEM COUNTS:", dict(sorted(system_counts.items())))
    print("STAFF COUNTS :", dict(sorted(staff_counts.items())))
    print("MULTI-SYSTEM :", len(multi_system_pages))
    print("STAFF/SYSTEM :", dict(sorted(staff_per_system.items())))
    print("MEAS/SYSTEM  :", dict(sorted(measures_per_system.items())))
    print("INVALID      :", len(invalid))
    print("TEST         : CLOSED")
    print("PIXEL REPLAY : NOT STARTED")
    print("TRAINING     : NONE")
    print("RESOLVER     : NOT WIRED")
    print("REPORT       :", output)


if __name__ == "__main__":
    main()
