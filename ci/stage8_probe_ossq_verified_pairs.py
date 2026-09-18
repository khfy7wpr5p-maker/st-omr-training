"""Live, non-authoritative Stage 8 probe for the exact 14 B8Q VERIFIED OSSQ pairs.

The runner re-materializes the frozen B8P bytes in ephemeral CI storage, verifies
B8P/B8Q/B8M/B8N identities, then probes each exact VERIFIED pair through the
existing Stage 8-1 semantic and quarantine byte validator. It emits hash-only
evidence. Passing the probe is not Stage 8 admission and grants no split,
training, TEST, production, redistribution, or commercial authority.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

from b8p_materialize_ossq_system_pairs import materialize, prepare_sources
from st_omr_training.poly_v2_ossq_pairing_preflight import B8O_PAIRING_SOURCE_SPECS
from st_omr_training.poly_v2_ossq_stage8_probe import (
    B8S_STAGE8_PROBE_VERSION,
    build_stage8_probe_candidates,
    semantic_gate_diagnostic_from_error,
)
from st_omr_training.poly_v2_ossq_system_pair_materialization import (
    B8P_EXPECTED_B8N_RECEIPT_SHA256,
    B8P_EXPECTED_B8O_RECEIPT_SHA256,
    build_b8p_materialization_receipt,
    receipt_to_json as b8p_receipt_to_json,
)
from st_omr_training.real_data_contract import (
    AdmissionState,
    PairingState,
    RealDataOrigin,
    RealDataSample,
    RealDataSplit,
    ReviewState,
    RightsBasis,
    real_data_sample_id,
)
from st_omr_training.real_data_intake import (
    RealDataIntakeError,
    semantic_fingerprint_from_musicxml,
    validate_quarantined_sample_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
B8N_RECEIPT = ROOT / "evidence" / "ossq_b8n_source_byte_pins.json"
B8P_RECEIPT = ROOT / "evidence" / "ossq_b8p_system_pair_materialization.json"
B8R_RECEIPT = ROOT / "evidence" / "ossq_b8r_cross_render_audit.json"
B8Q_RECEIPT = ROOT / "evidence" / "ossq_b8q_pair_review_admission.json"
B8S_RECEIPT = ROOT / "evidence" / "ossq_stage8_verified_pair_probe.json"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path.name} must contain a JSON object")
    return payload


def _pair_paths(dataset_root: Path, score_id: str, segment_id: str) -> tuple[Path, Path, Path]:
    specs = {item.score_id: item for item in B8O_PAIRING_SOURCE_SPECS}
    spec = specs[score_id]
    score_dir = dataset_root / "scores" / spec.work_path
    return (
        score_dir / f"sq{score_id}_scanned.pdf",
        score_dir / "images" / "scanned" / "systemwise" / f"{segment_id}.png",
        score_dir / "musicxml" / "scanned" / "systemwise" / f"{segment_id}.musicxml",
    )


def _probe_candidate(candidate, dataset_root: Path) -> dict[str, object]:
    source_path, image_path, xml_path = _pair_paths(
        dataset_root,
        candidate.score_id,
        candidate.segment_id,
    )
    source_bytes = source_path.read_bytes()
    image_bytes = image_path.read_bytes()
    xml_bytes = xml_path.read_bytes()

    if sha256(source_bytes).hexdigest() != candidate.source_document_sha256:
        raise RuntimeError(f"source byte identity drift for {candidate.segment_id}")
    if sha256(image_bytes).hexdigest() != candidate.image_sha256:
        raise RuntimeError(f"image byte identity drift for {candidate.segment_id}")
    if sha256(xml_bytes).hexdigest() != candidate.musicxml_sha256:
        raise RuntimeError(f"MusicXML byte identity drift for {candidate.segment_id}")

    base = {
        "score_id": candidate.score_id,
        "segment_id": candidate.segment_id,
        "page_number": candidate.page_number,
        "imslp_id": candidate.imslp_id,
        "family_id": candidate.family_id,
        "source_document_sha256": candidate.source_document_sha256,
        "image_sha256": candidate.image_sha256,
        "musicxml_sha256": candidate.musicxml_sha256,
        "provenance_evidence_sha256": candidate.provenance_evidence_sha256,
        "rights_evidence_sha256": candidate.rights_evidence_sha256,
        "pairing_evidence_sha256": candidate.pairing_evidence_sha256,
        "pairing_method": candidate.pairing_method,
        "probe_split": "train-non-authoritative",
    }

    try:
        semantic = semantic_fingerprint_from_musicxml(xml_bytes)
    except RealDataIntakeError as exc:
        diagnostic = semantic_gate_diagnostic_from_error(exc)
        return {
            **base,
            "semantic_fingerprint": None,
            "semantic_issue_code": diagnostic.issue_code,
            "semantic_issue_path": diagnostic.issue_path,
            "semantic_issue_source": diagnostic.source,
            "stage8_probe_decision": "rejected",
            "reason_code": "stage8-semantic-token-gate-rejected",
            "byte_receipt_sha256": None,
        }

    sample_id = real_data_sample_id(
        family_id=candidate.family_id,
        page_number=candidate.page_number,
        source_document_sha256=candidate.source_document_sha256,
        image_sha256=candidate.image_sha256,
        musicxml_sha256=candidate.musicxml_sha256,
        semantic_fingerprint=semantic,
    )
    quarantine = RealDataSample(
        sample_id=sample_id,
        family_id=candidate.family_id,
        split=RealDataSplit.TRAIN,
        page_number=candidate.page_number,
        origin=RealDataOrigin.CURATED,
        rights_basis=RightsBasis.PUBLIC_DOMAIN,
        source_document_sha256=candidate.source_document_sha256,
        image_sha256=candidate.image_sha256,
        musicxml_sha256=candidate.musicxml_sha256,
        semantic_fingerprint=semantic,
        provenance_evidence_sha256=candidate.provenance_evidence_sha256,
        rights_evidence_sha256=candidate.rights_evidence_sha256,
        pairing_evidence_sha256=candidate.pairing_evidence_sha256,
        explicit_training_permission_sha256=None,
        privacy_review_evidence_sha256=None,
        rights_review=ReviewState.APPROVED,
        pairing_review=PairingState.VERIFIED,
        admission_state=AdmissionState.QUARANTINED,
    )

    try:
        receipt = validate_quarantined_sample_bytes(
            quarantine,
            source_document_bytes=source_bytes,
            training_image_png_bytes=image_bytes,
            musicxml_bytes=xml_bytes,
        )
    except RealDataIntakeError:
        return {
            **base,
            "semantic_fingerprint": semantic,
            "semantic_issue_code": None,
            "semantic_issue_path": None,
            "semantic_issue_source": None,
            "stage8_probe_decision": "rejected",
            "reason_code": "stage8-quarantine-intake-rejected",
            "byte_receipt_sha256": None,
        }

    return {
        **base,
        "semantic_fingerprint": semantic,
        "semantic_issue_code": None,
        "semantic_issue_path": None,
        "semantic_issue_source": None,
        "stage8_probe_decision": "probe-passed",
        "reason_code": "stage8-quarantine-intake-probe-passed",
        "byte_receipt_sha256": receipt.receipt_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--preprocessor-root", required=True, type=Path)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    preprocessor_root = args.preprocessor_root.resolve()

    b8n = _load(B8N_RECEIPT)
    b8p = _load(B8P_RECEIPT)
    b8r = _load(B8R_RECEIPT)
    b8q = _load(B8Q_RECEIPT)
    candidates = build_stage8_probe_candidates(
        b8p_payload=b8p,
        b8r_payload=b8r,
        b8q_payload=b8q,
        b8n_payload=b8n,
    )

    source_hashes, task_lines = prepare_sources(dataset_root)
    task_file = dataset_root / "b8s_ready_scores.txt"
    task_file.write_text("\n".join(task_lines) + "\n", encoding="utf-8")
    materialize(dataset_root, preprocessor_root, task_file)

    live_b8p = build_b8p_materialization_receipt(
        dataset_root=dataset_root,
        b8n_source_sha256_by_score=source_hashes,
        b8n_receipt_sha256=B8P_EXPECTED_B8N_RECEIPT_SHA256,
        b8o_receipt_sha256=B8P_EXPECTED_B8O_RECEIPT_SHA256,
    )
    committed_b8p = B8P_RECEIPT.read_text(encoding="ascii").strip()
    if b8p_receipt_to_json(live_b8p) != committed_b8p:
        raise RuntimeError("live B8P materialization differs from committed receipt before Stage 8 probe")

    observations = tuple(
        sorted(
            (_probe_candidate(candidate, dataset_root) for candidate in candidates),
            key=lambda item: str(item["segment_id"]),
        )
    )
    passed = sum(item["stage8_probe_decision"] == "probe-passed" for item in observations)
    semantic_rejected = sum(
        item["reason_code"] == "stage8-semantic-token-gate-rejected"
        for item in observations
    )
    intake_rejected = sum(
        item["reason_code"] == "stage8-quarantine-intake-rejected"
        for item in observations
    )
    passed_families = len(
        {
            str(item["family_id"])
            for item in observations
            if item["stage8_probe_decision"] == "probe-passed"
        }
    )
    semantic_issue_counts: dict[str, int] = {}
    for item in observations:
        issue_code = item.get("semantic_issue_code")
        if isinstance(issue_code, str):
            semantic_issue_counts[issue_code] = semantic_issue_counts.get(issue_code, 0) + 1


    payload = {
        "version": B8S_STAGE8_PROBE_VERSION,
        "b8n_receipt_sha256": b8n["receipt_sha256"],
        "b8p_receipt_sha256": b8p["receipt_sha256"],
        "b8r_receipt_sha256": b8r["receipt_sha256"],
        "b8q_receipt_sha256": b8q["receipt_sha256"],
        "candidate_count": len(candidates),
        "candidate_family_count": len({item.family_id for item in candidates}),
        "observations": list(observations),
        "decision_counts": [
            ["probe-passed", passed],
            ["semantic-rejected", semantic_rejected],
            ["intake-rejected", intake_rejected],
        ],
        "passing_family_count": passed_families,
        "semantic_issue_counts": sorted(semantic_issue_counts.items()),
        "rights_scope": "research-training-candidate-only",
        "raw_source_bytes_persisted_as_evidence": False,
        "raw_pair_bytes_persisted_as_evidence": False,
        "stage8_admission_authority": False,
        "train_validation_assignment_authority": False,
        "test_artifact_bytes_accessed": False,
        "production_authority": False,
        "commercial_use_authority": False,
        "redistribution_authority": False,
    }
    receipt_sha = sha256(_canonical_bytes(payload)).hexdigest()
    payload["receipt_sha256"] = receipt_sha
    canonical = _canonical_bytes(payload).decode("ascii")

    if B8S_RECEIPT.exists():
        committed = B8S_RECEIPT.read_text(encoding="ascii").strip()
        if committed != canonical:
            raise RuntimeError("live Stage 8 probe differs from committed receipt")
        print("B8S_MODE=VERIFIED")
        print("B8S_LIVE_PROBE_MATCHES_COMMITTED_RECEIPT=true")
    else:
        print("B8S_MODE=DISCOVERY")
        print("B8S_RECEIPT_JSON=" + canonical)

    print(f"B8S_RECEIPT_SHA256={receipt_sha}")
    print(f"B8S_CANDIDATE_COUNT={len(candidates)}")
    print(f"B8S_CANDIDATE_FAMILY_COUNT={len({item.family_id for item in candidates})}")
    print(
        "B8S_DECISION_COUNTS="
        f"probe-passed:{passed},semantic-rejected:{semantic_rejected},intake-rejected:{intake_rejected}"
    )
    print(f"B8S_PASSING_FAMILY_COUNT={passed_families}")
    print("B8S_SEMANTIC_ISSUE_COUNTS=" + json.dumps(sorted(semantic_issue_counts.items()), separators=(",", ":")))
    print("B8S_RIGHTS_SCOPE=research-training-candidate-only")
    print("B8S_STAGE8_ADMISSION_AUTHORITY=false")
    print("B8S_TRAIN_VALIDATION_ASSIGNMENT_AUTHORITY=false")
    print("B8S_TEST_ARTIFACT_BYTES_ACCESSED=false")
    print("B8S_PRODUCTION_AUTHORITY=false")
    print("B8S_COMMERCIAL_USE_AUTHORITY=false")
    print("B8S_REDISTRIBUTION_AUTHORITY=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
