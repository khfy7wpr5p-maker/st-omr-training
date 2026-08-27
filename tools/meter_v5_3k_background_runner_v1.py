"""V5-3K stdlib-only detached Colab forensics runner.

This runner is launched by a later exact-SHA notebook wrapper. It prepares an
isolated pinned runtime, checks out the exact CI-green V5-3K TRAIN-only
feature/domain-shift implementation, and writes one descriptive forensic
report. It never trains, tunes thresholds, or opens protected validation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import traceback


FORENSICS_IMPLEMENTATION_HEAD = "0efa6b3ba315d671e5a449789ff6de103c735439"
FORENSICS_MODULE_BLOB = "c719990b5f949759e643b5cdcefc5fe2a7a650f6"
V53G_HEAD = "b36a9d2f5daade2c3568cac8cbc736ca75ca435f"
EXPECTED_V53G_REPORT_SHA256 = (
    "682c2d405287051fef18b803e2597777cb7fc55c6ba0814ea3b2d4df0fa35b9d"
)
EXPECTED_V53H_ENVELOPE_SHA256 = (
    "f41b0fddb9d139018e0ddd16c9765d9415031e6308efd67e16aef3a05d205bf7"
)
EXPECTED_V53J_REPORT_SHA256 = (
    "7a49d29e0d7257be7c59d499ab3d9ab575d369a7473b0b5298ea62aa80c7d37f"
)
EXPECTED_RESCUE_ARTIFACT_SHA256 = {
    "2": "a27cef8d4ff89565cfe4a15e0e429a21e60daa2656324ed0380fde8674a022e6",
    "3": "b8a4f379c33d3aa0df77b54821996a799251abb0e7cbd8de9764b09c5efd3d65",
}
REPOSITORY = "khfy7wpr5p-maker/st-omr-training"

MYDRIVE = Path("/content/drive/MyDrive")
DATA_ROOT = MYDRIVE / "TEST" / "METER_V2_1500_PACKAGE_AB_CLEAN"
ANN = DATA_ROOT / "annotations"
CHECKPOINT_ROOT = MYDRIVE / "ST-OMR-METER-SPECIALISTS"
M4A_ROOT = CHECKPOINT_ROOT / "m4a-234-digit-specialist-dataset-freeze-v2"
D10_ROOT = (
    MYDRIVE
    / "ST-OMR-D10"
    / "stage7d10-authoritative-562c8fcfabf1b41573f1ef591d88ae65335ce16a"
)
SOURCE_REPORT = ANN / "v5_3g_authoritative_rescue_training_report.json"
SOURCE_ENVELOPE = ANN / f"v5_3g_execution_envelope_{V53G_HEAD}.json"
V53J_REPORT = ANN / "v5_3j_rescue_failure_forensics_v1.json"
RESCUE_DIR = ANN / "v5_3g_authoritative_rescue_artifacts"
RESULT = ANN / "v5_3k_feature_domain_shift_forensics_v1.json"

CONTROL_DIR = ANN / "v5_3k_background_control"
LOCK = CONTROL_DIR / f"launch_{FORENSICS_IMPLEMENTATION_HEAD}.json"
HEARTBEAT = CONTROL_DIR / f"heartbeat_{FORENSICS_IMPLEMENTATION_HEAD}.json"
PROGRESS = CONTROL_DIR / f"progress_{FORENSICS_IMPLEMENTATION_HEAD}.json"

REPO = Path("/content/st-omr-v5-3k-forensics")
VENV = Path("/content/st-omr-v5-3k-venv")
VENV_PYTHON = VENV / "bin" / "python"
WORKER = Path("/content/v5_3k_forensics_worker.py")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def sha_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected regular evidence file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_sources() -> dict[str, Path]:
    return {
        "DATA_ROOT": DATA_ROOT,
        "CHECKPOINT_ROOT": CHECKPOINT_ROOT,
        "M4A_ROOT": M4A_ROOT,
        "D10_ROOT": D10_ROOT,
        "SOURCE_REPORT": SOURCE_REPORT,
        "SOURCE_ENVELOPE": SOURCE_ENVELOPE,
        "V53J_REPORT": V53J_REPORT,
        "RESCUE_DIR": RESCUE_DIR,
    }


def verify_source_surface() -> None:
    directory_names = {"DATA_ROOT", "CHECKPOINT_ROOT", "M4A_ROOT", "D10_ROOT", "RESCUE_DIR"}
    for name, path in _required_sources().items():
        if name in directory_names:
            ok = path.is_dir() and not path.is_symlink()
        else:
            ok = path.is_file() and not path.is_symlink()
        if not ok:
            raise RuntimeError(f"required V5-3K source missing/non-regular: {name}={path}")
    if RESULT.exists():
        raise RuntimeError(f"V5-3K result already exists: {RESULT}")
    if sha_file(SOURCE_REPORT) != EXPECTED_V53G_REPORT_SHA256:
        raise RuntimeError("V5-3G report SHA mismatch")
    if sha_file(SOURCE_ENVELOPE) != EXPECTED_V53H_ENVELOPE_SHA256:
        raise RuntimeError("V5-3H envelope SHA mismatch")
    if sha_file(V53J_REPORT) != EXPECTED_V53J_REPORT_SHA256:
        raise RuntimeError("V5-3J report SHA mismatch")
    for digit in ("2", "3"):
        artifact = RESCUE_DIR / f"digit_{digit}_rescue.pt"
        if sha_file(artifact) != EXPECTED_RESCUE_ARTIFACT_SHA256[digit]:
            raise RuntimeError(f"{digit}-AI rescue artifact SHA mismatch")


def checkout_exact_forensics_source() -> None:
    repo_url = f"https://github.com/{REPOSITORY}.git"
    if REPO.exists():
        shutil.rmtree(REPO)
    subprocess.check_call(["git", "clone", "--no-checkout", repo_url, str(REPO)])
    subprocess.check_call(
        ["git", "-C", str(REPO), "fetch", "origin", FORENSICS_IMPLEMENTATION_HEAD, "--depth", "1"]
    )
    fetched = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "FETCH_HEAD"], text=True
    ).strip()
    if fetched != FORENSICS_IMPLEMENTATION_HEAD:
        raise RuntimeError(f"V5-3K FETCH_HEAD mismatch: {fetched}")
    subprocess.check_call(
        ["git", "-C", str(REPO), "checkout", "--detach", FORENSICS_IMPLEMENTATION_HEAD]
    )
    actual_head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_head != FORENSICS_IMPLEMENTATION_HEAD:
        raise RuntimeError(f"V5-3K HEAD mismatch: {actual_head}")
    if subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("V5-3K repository worktree dirty")
    module_rel = "st_omr_training/meter_v5_3k_feature_domain_shift_forensics_v1.py"
    actual_blob = subprocess.check_output(
        ["git", "-C", str(REPO), "hash-object", module_rel], text=True
    ).strip()
    if actual_blob != FORENSICS_MODULE_BLOB:
        raise RuntimeError(f"V5-3K module blob mismatch: {actual_blob}")


def prepare_isolated_runtime() -> dict[str, str]:
    if VENV.exists():
        shutil.rmtree(VENV)
    subprocess.check_call([sys.executable, "-m", "venv", "--without-pip", str(VENV)])
    if not VENV_PYTHON.is_file():
        raise RuntimeError(f"isolated Python not created: {VENV_PYTHON}")
    pip_target = [sys.executable, "-m", "pip", "--python", str(VENV_PYTHON)]
    subprocess.check_call(pip_target + ["install", "-r", str(REPO / "requirements.txt")])
    subprocess.check_call(
        pip_target
        + [
            "install",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
            "-r",
            str(REPO / "requirements-training.txt"),
        ]
    )
    subprocess.check_call(pip_target + ["check"])

    expected = {
        "lxml": "6.1.1",
        "verovio": "6.2.1",
        "CairoSVG": "2.8.2",
        "Pillow": "12.3.0",
        "scipy": "1.18.0",
        "torch": "2.13.0+cpu",
    }
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["V53K_EXPECTED_RUNTIME"] = json.dumps(expected, sort_keys=True)
    runtime_check = r'''
from importlib import metadata
import json, os, sys
if sys.prefix == sys.base_prefix:
    raise RuntimeError("not isolated")
expected = json.loads(os.environ["V53K_EXPECTED_RUNTIME"])
actual = {name: metadata.version(name) for name in expected}
bad = {name: (expected[name], actual[name]) for name in expected if expected[name] != actual[name]}
if bad:
    raise RuntimeError(f"runtime mismatch: {bad}")
print(json.dumps(actual, sort_keys=True))
'''
    actual_raw = subprocess.check_output(
        [str(VENV_PYTHON), "-c", runtime_check], env=env, text=True
    ).strip()
    actual = json.loads(actual_raw)
    if actual != expected:
        raise RuntimeError(f"isolated runtime changed: {actual}")
    return actual


WORKER_SOURCE = r'''from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, sys

FORENSICS_IMPLEMENTATION_HEAD = "0efa6b3ba315d671e5a449789ff6de103c735439"
V53G_HEAD = "b36a9d2f5daade2c3568cac8cbc736ca75ca435f"
REPO = Path("/content/st-omr-v5-3k-forensics")
MYDRIVE = Path("/content/drive/MyDrive")
DATA_ROOT = MYDRIVE / "TEST" / "METER_V2_1500_PACKAGE_AB_CLEAN"
ANN = DATA_ROOT / "annotations"
CHECKPOINT_ROOT = MYDRIVE / "ST-OMR-METER-SPECIALISTS"
M4A_ROOT = CHECKPOINT_ROOT / "m4a-234-digit-specialist-dataset-freeze-v2"
D10_ROOT = MYDRIVE / "ST-OMR-D10" / "stage7d10-authoritative-562c8fcfabf1b41573f1ef591d88ae65335ce16a"
SOURCE_REPORT = ANN / "v5_3g_authoritative_rescue_training_report.json"
SOURCE_ENVELOPE = ANN / f"v5_3g_execution_envelope_{V53G_HEAD}.json"
V53J_REPORT = ANN / "v5_3j_rescue_failure_forensics_v1.json"
RESCUE_DIR = ANN / "v5_3g_authoritative_rescue_artifacts"
RESULT = ANN / "v5_3k_feature_domain_shift_forensics_v1.json"
PROGRESS = ANN / "v5_3k_background_control" / f"progress_{FORENSICS_IMPLEMENTATION_HEAD}.json"

if sys.prefix == sys.base_prefix or os.environ.get("PYTHONNOUSERSITE") != "1":
    raise RuntimeError("isolated V5-3K worker boundary missing")
sys.path.insert(0, str(REPO))

from st_omr_training import meter_v5_2b_specialist_adaptation as v52b
from st_omr_training import meter_v5_3k_feature_domain_shift_forensics_v1 as forensics

if forensics.EXPECTED_V53J_REPORT_SHA256 != "7a49d29e0d7257be7c59d499ab3d9ab575d369a7473b0b5298ea62aa80c7d37f":
    raise RuntimeError("V5-3K V5-3J binding changed")
boundary = forensics.safety_boundary()
for key, expected in {
    "training": False,
    "fitting": False,
    "backward": False,
    "optimizer_steps": 0,
    "checkpoint_write": False,
    "rescue_artifact_write": False,
    "threshold_tuning": False,
    "threshold_sweep": False,
    "repair_recipe_selected": False,
    "retraining_authorized": False,
    "digit4_loaded": False,
    "historical_validation_opened": False,
    "first30_opened": False,
    "v5_reserve_opened": False,
    "v5_validation_opened": False,
    "final_holdout_locked": True,
}.items():
    if boundary.get(key) != expected:
        raise RuntimeError(f"V5-3K safety boundary changed: {key}")

d2 = v52b.locate_checkpoint_by_sha_v1(CHECKPOINT_ROOT, v52b.DIGIT2_SHA256)
d3 = v52b.locate_checkpoint_by_sha_v1(CHECKPOINT_ROOT, v52b.DIGIT3_SHA256)
print("MODULE/CHECKPOINT/SAFETY = PASS", flush=True)

def progress(done, total, phase):
    payload = {
        "schema": "st-omr-meter-v5-3k-background-progress-v1",
        "forensics_implementation_head": FORENSICS_IMPLEMENTATION_HEAD,
        "status": "FORENSICS_EVALUATING",
        "phase": phase,
        "done": int(done),
        "total": int(total),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    tmp = PROGRESS.with_suffix(PROGRESS.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, PROGRESS)

result = forensics.run_feature_domain_shift_forensics_v1(
    DATA_ROOT,
    m4a_root=M4A_ROOT,
    d10_root=D10_ROOT,
    digit2_frozen=d2,
    digit3_frozen=d3,
    v53g_report=SOURCE_REPORT,
    v53h_envelope=SOURCE_ENVELOPE,
    v53j_report=V53J_REPORT,
    rescue_artifact_dir=RESCUE_DIR,
    progress=progress,
)

if RESULT.exists():
    raise RuntimeError(f"refusing to overwrite V5-3K result: {RESULT}")
for key, expected in {
    "frozen_state_bit_identical": True,
    "rescue_state_bit_identical_during_forensics": True,
    "repair_recipe_selected": False,
    "retraining_authorized": False,
    "historical_validation_opened": False,
    "first30_opened": False,
    "v5_reserve_opened": False,
    "v5_validation_opened": False,
    "final_holdout_locked": True,
}.items():
    if result.get(key) != expected:
        raise RuntimeError(f"V5-3K result boundary changed: {key}")

tmp = RESULT.with_suffix(RESULT.suffix + ".tmp")
tmp.write_text(
    json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
os.replace(tmp, RESULT)
digest = hashlib.sha256(RESULT.read_bytes()).hexdigest()
print("V5-3K FEATURE/DOMAIN FORENSICS = COMPLETED", flush=True)
for digit in ("2", "3"):
    item = result["per_specialist"][digit]
    witness = item["fixed_threshold_witness"]
    critical = item["critical_v5_positive_vs_historical_tn"]
    hard = item["historical_hard_tn_subpopulation"]
    print(
        f"{digit}-AI SIGNATURE={item['v5_3j_failure_signature']} "
        f"V5_RECOVERY={witness['v5_positive_recovery_fraction']} "
        f"HARD_HIST_TN={witness['historical_hard_tn_count']}",
        flush=True,
    )
    print(
        f"{digit}-AI CRITICAL_64D_SHIFT_MAX="
        f"{critical['feature_64d']['standardized_mean_shift']['max_absolute_standardized_mean_shift']} "
        f"CRITICAL_8D_LOGIT_GAP="
        f"{critical['output_logit_gap_decomposition']['mean_logit_gap_a_minus_b']}",
        flush=True,
    )
    print(
        f"{digit}-AI HARD_NEG_64D_SHIFT_MAX="
        f"{hard['feature_64d']['standardized_mean_shift']['max_absolute_standardized_mean_shift']}",
        flush=True,
    )
print("FROZEN/RESCUE STATE = BIT IDENTICAL", flush=True)
print("REPAIR/RETRAINING = NOT AUTHORIZED", flush=True)
print("HISTORICAL VALIDATION/FIRST-30/V5 VAL = CLOSED", flush=True)
print("FINAL_HOLDOUT = LOCKED", flush=True)
print("V5-3K REPORT SHA256 =", digest, flush=True)
'''


def run_worker() -> None:
    compile(WORKER_SOURCE, str(WORKER), "exec")
    WORKER.write_text(WORKER_SOURCE, encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    subprocess.check_call([str(VENV_PYTHON), "-u", str(WORKER)], env=env)


def main() -> int:
    if not LOCK.is_file() or LOCK.is_symlink():
        raise RuntimeError("V5-3K launch lock missing/non-regular")
    state = json.loads(LOCK.read_text(encoding="utf-8"))
    if state.get("forensics_implementation_head") != FORENSICS_IMPLEMENTATION_HEAD:
        raise RuntimeError("V5-3K launch lock head mismatch")
    if state.get("status") != "ALLOCATED":
        raise RuntimeError(f"V5-3K launch lock is not ALLOCATED: {state.get('status')}")

    state.update({"pid": os.getpid(), "status": "BOOTSTRAPPING", "started_at_utc": utc_now()})
    atomic_json(LOCK, state)
    stop = threading.Event()

    def heartbeat_loop() -> None:
        while not stop.is_set():
            atomic_json(
                HEARTBEAT,
                {
                    "schema": "st-omr-meter-v5-3k-background-heartbeat-v1",
                    "forensics_implementation_head": FORENSICS_IMPLEMENTATION_HEAD,
                    "pid": os.getpid(),
                    "status": state.get("status"),
                    "utc": utc_now(),
                },
            )
            stop.wait(20)

    thread = threading.Thread(target=heartbeat_loop, daemon=True)
    thread.start()
    try:
        verify_source_surface()
        checkout_exact_forensics_source()
        print("EXACT FORENSICS SOURCE = PASS", flush=True)
        actual_runtime = prepare_isolated_runtime()
        print("ISOLATED VENV BOOTSTRAP = PASS", flush=True)
        print("ISOLATED RUNTIME =", json.dumps(actual_runtime, sort_keys=True), flush=True)
        print("PINNED RUNTIME = PASS", flush=True)

        state["status"] = "FORENSICS_EVALUATING"
        atomic_json(LOCK, state)
        run_worker()

        if not RESULT.is_file():
            raise RuntimeError("V5-3K result not written")
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        state.update(
            {
                "status": "COMPLETED",
                "result_sha256": sha_file(RESULT),
                "completed_at_utc": utc_now(),
            }
        )
        atomic_json(LOCK, state)
        return 0
    except BaseException as exc:
        state.update(
            {
                "status": "FAILED",
                "failed_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        atomic_json(LOCK, state)
        traceback.print_exc()
        return 1
    finally:
        stop.set()
        thread.join(timeout=5)
        atomic_json(
            HEARTBEAT,
            {
                "schema": "st-omr-meter-v5-3k-background-heartbeat-v1",
                "forensics_implementation_head": FORENSICS_IMPLEMENTATION_HEAD,
                "pid": os.getpid(),
                "status": state.get("status"),
                "utc": utc_now(),
                "final": True,
            },
        )


if __name__ == "__main__":
    raise SystemExit(main())
