"""V5-3I stdlib-only detached Colab runner.

This runner is launched by a later exact-SHA notebook wrapper. It prepares an
isolated pinned runtime, checks out the exact CI-green V5-3I gate
implementation, evaluates TRAIN-only acceptance, and writes one gate report.
It never trains and never opens protected validation surfaces.
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


GATE_IMPLEMENTATION_HEAD = "844c6673f03635177a39b1ab20ab62e9392d922a"
GATE_MODULE_BLOB = "abb5f1ae4c42b0c5f3ae26b80f2a467f47582197"
V53G_HEAD = "b36a9d2f5daade2c3568cac8cbc736ca75ca435f"
V53H_WRAPPER_HEAD = "aa426442efdef97e3323906096087dabffa1171b"
EXPECTED_V53G_REPORT_SHA256 = (
    "682c2d405287051fef18b803e2597777cb7fc55c6ba0814ea3b2d4df0fa35b9d"
)
EXPECTED_V53H_ENVELOPE_SHA256 = (
    "f41b0fddb9d139018e0ddd16c9765d9415031e6308efd67e16aef3a05d205bf7"
)
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
RESCUE_DIR = ANN / "v5_3g_authoritative_rescue_artifacts"
RESULT = ANN / "v5_3i_train_acceptance_gate_v1.json"

CONTROL_DIR = ANN / "v5_3i_background_control"
LOCK = CONTROL_DIR / f"launch_{GATE_IMPLEMENTATION_HEAD}.json"
HEARTBEAT = CONTROL_DIR / f"heartbeat_{GATE_IMPLEMENTATION_HEAD}.json"
PROGRESS = CONTROL_DIR / f"progress_{GATE_IMPLEMENTATION_HEAD}.json"

REPO = Path("/content/st-omr-v5-3i-gate")
VENV = Path("/content/st-omr-v5-3i-venv")
VENV_PYTHON = VENV / "bin" / "python"
WORKER = Path("/content/v5_3i_gate_worker.py")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
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
        "RESCUE_DIR": RESCUE_DIR,
    }


def verify_source_surface() -> None:
    for name, path in _required_sources().items():
        if name in {"DATA_ROOT", "CHECKPOINT_ROOT", "M4A_ROOT", "D10_ROOT", "RESCUE_DIR"}:
            ok = path.is_dir() and not path.is_symlink()
        else:
            ok = path.is_file() and not path.is_symlink()
        if not ok:
            raise RuntimeError(f"required V5-3I source missing/non-regular: {name}={path}")
    if RESULT.exists():
        raise RuntimeError(f"V5-3I result already exists: {RESULT}")
    if sha_file(SOURCE_REPORT) != EXPECTED_V53G_REPORT_SHA256:
        raise RuntimeError("V5-3G report SHA mismatch")
    if sha_file(SOURCE_ENVELOPE) != EXPECTED_V53H_ENVELOPE_SHA256:
        raise RuntimeError("V5-3H envelope SHA mismatch")


def checkout_exact_gate_source() -> None:
    repo_url = f"https://github.com/{REPOSITORY}.git"
    if REPO.exists():
        shutil.rmtree(REPO)
    subprocess.check_call(["git", "clone", "--no-checkout", repo_url, str(REPO)])
    subprocess.check_call(
        ["git", "-C", str(REPO), "fetch", "origin", GATE_IMPLEMENTATION_HEAD, "--depth", "1"]
    )
    fetched = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "FETCH_HEAD"], text=True
    ).strip()
    if fetched != GATE_IMPLEMENTATION_HEAD:
        raise RuntimeError(f"V5-3I FETCH_HEAD mismatch: {fetched}")
    subprocess.check_call(
        ["git", "-C", str(REPO), "checkout", "--detach", GATE_IMPLEMENTATION_HEAD]
    )
    actual_head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_head != GATE_IMPLEMENTATION_HEAD:
        raise RuntimeError(f"V5-3I HEAD mismatch: {actual_head}")
    if subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("V5-3I repository worktree dirty")
    module_rel = "st_omr_training/meter_v5_3i_train_acceptance_gate_v1.py"
    actual_blob = subprocess.check_output(
        ["git", "-C", str(REPO), "hash-object", module_rel], text=True
    ).strip()
    if actual_blob != GATE_MODULE_BLOB:
        raise RuntimeError(f"V5-3I module blob mismatch: {actual_blob}")


def prepare_isolated_runtime() -> dict[str, str]:
    if VENV.exists():
        shutil.rmtree(VENV)
    subprocess.check_call(
        [sys.executable, "-m", "venv", "--without-pip", str(VENV)]
    )
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
    env["V53I_EXPECTED_RUNTIME"] = json.dumps(expected, sort_keys=True)
    runtime_check = r'''
from importlib import metadata
import json, os, sys
if sys.prefix == sys.base_prefix:
    raise RuntimeError("not isolated")
expected = json.loads(os.environ["V53I_EXPECTED_RUNTIME"])
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

GATE_IMPLEMENTATION_HEAD = "844c6673f03635177a39b1ab20ab62e9392d922a"
V53G_HEAD = "b36a9d2f5daade2c3568cac8cbc736ca75ca435f"
REPO = Path("/content/st-omr-v5-3i-gate")
MYDRIVE = Path("/content/drive/MyDrive")
DATA_ROOT = MYDRIVE / "TEST" / "METER_V2_1500_PACKAGE_AB_CLEAN"
ANN = DATA_ROOT / "annotations"
CHECKPOINT_ROOT = MYDRIVE / "ST-OMR-METER-SPECIALISTS"
M4A_ROOT = CHECKPOINT_ROOT / "m4a-234-digit-specialist-dataset-freeze-v2"
D10_ROOT = MYDRIVE / "ST-OMR-D10" / "stage7d10-authoritative-562c8fcfabf1b41573f1ef591d88ae65335ce16a"
SOURCE_REPORT = ANN / "v5_3g_authoritative_rescue_training_report.json"
SOURCE_ENVELOPE = ANN / f"v5_3g_execution_envelope_{V53G_HEAD}.json"
RESCUE_DIR = ANN / "v5_3g_authoritative_rescue_artifacts"
RESULT = ANN / "v5_3i_train_acceptance_gate_v1.json"
PROGRESS = ANN / "v5_3i_background_control" / f"progress_{GATE_IMPLEMENTATION_HEAD}.json"

if sys.prefix == sys.base_prefix or os.environ.get("PYTHONNOUSERSITE") != "1":
    raise RuntimeError("isolated V5-3I worker boundary missing")
sys.path.insert(0, str(REPO))

from st_omr_training import meter_v5_2b_specialist_adaptation as v52b
from st_omr_training import meter_v5_3i_train_acceptance_gate_v1 as gate

if gate.V53G_HEAD_SHA != V53G_HEAD:
    raise RuntimeError("V5-3I V5-3G binding changed")
if gate.V53H_WRAPPER_HEAD_SHA != "aa426442efdef97e3323906096087dabffa1171b":
    raise RuntimeError("V5-3I V5-3H binding changed")
boundary = gate.safety_boundary()
for key, expected in {
    "training": False,
    "backward": False,
    "optimizer_steps": 0,
    "checkpoint_write": False,
    "rescue_artifact_write": False,
    "digit4_loaded": False,
    "historical_validation_opened": False,
    "first30_opened": False,
    "v5_reserve_opened": False,
    "v5_validation_opened": False,
    "final_holdout_locked": True,
    "retraining_authorized_on_hold": False,
}.items():
    if boundary.get(key) != expected:
        raise RuntimeError(f"V5-3I safety boundary changed: {key}")

d2 = v52b.locate_checkpoint_by_sha_v1(CHECKPOINT_ROOT, v52b.DIGIT2_SHA256)
d3 = v52b.locate_checkpoint_by_sha_v1(CHECKPOINT_ROOT, v52b.DIGIT3_SHA256)
print("MODULE/CHECKPOINT/SAFETY = PASS", flush=True)

def progress(done, total, phase):
    payload = {
        "schema": "st-omr-meter-v5-3i-background-progress-v1",
        "gate_implementation_head": GATE_IMPLEMENTATION_HEAD,
        "status": "GATE_EVALUATING",
        "phase": phase,
        "done": int(done),
        "total": int(total),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    tmp = PROGRESS.with_suffix(PROGRESS.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, PROGRESS)

result = gate.run_train_acceptance_gate_v1(
    DATA_ROOT,
    m4a_root=M4A_ROOT,
    d10_root=D10_ROOT,
    digit2_frozen=d2,
    digit3_frozen=d3,
    v53g_report=SOURCE_REPORT,
    v53h_envelope=SOURCE_ENVELOPE,
    rescue_artifact_dir=RESCUE_DIR,
    progress=progress,
)

if RESULT.exists():
    raise RuntimeError(f"refusing to overwrite V5-3I result: {RESULT}")
if result.get("decision") not in {"PASS", "HOLD"}:
    raise RuntimeError("V5-3I returned invalid decision")
for key, expected in {
    "historical_validation_retention_executed": False,
    "first30_opened": False,
    "v5_reserve_opened": False,
    "v5_validation_opened": False,
    "final_holdout_locked": True,
    "retraining_authorized": False,
}.items():
    if result.get(key) != expected:
        raise RuntimeError(f"V5-3I result boundary changed: {key}")

tmp = RESULT.with_suffix(RESULT.suffix + ".tmp")
tmp.write_text(
    json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
os.replace(tmp, RESULT)
digest = hashlib.sha256(RESULT.read_bytes()).hexdigest()
print("V5-3I TRAIN ACCEPTANCE DECISION =", result["decision"], flush=True)
for digit in ("2", "3"):
    item = result["per_specialist"][digit]
    v5 = item["v5_train"]
    hist = item["historical_train"]
    print(
        f"{digit}-AI V5 F1={v5['combined_metrics']['f1']} "
        f"FP={v5['combined_metrics']['fp']} FN={v5['combined_metrics']['fn']} "
        f"V5_REGRESSIONS={v5['frozen_correct_regression_count']} "
        f"HIST_TRAIN_REGRESSIONS={hist['frozen_correct_regression_count']}",
        flush=True,
    )
print("FROZEN STATE BIT IDENTICAL =", result["frozen_state_bit_identical"], flush=True)
print("ONLY RESCUE PARAMETERS CHANGED =", result["only_rescue_parameters_changed"], flush=True)
print("HISTORICAL VALIDATION/FIRST-30/V5 VAL = CLOSED", flush=True)
print("FINAL_HOLDOUT = LOCKED", flush=True)
print("V5-3I REPORT SHA256 =", digest, flush=True)
'''


def run_worker() -> None:
    compile(WORKER_SOURCE, str(WORKER), "exec")
    WORKER.write_text(WORKER_SOURCE, encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    subprocess.check_call([str(VENV_PYTHON), "-u", str(WORKER)], env=env)


def main() -> int:
    if not LOCK.is_file() or LOCK.is_symlink():
        raise RuntimeError("V5-3I launch lock missing/non-regular")
    state = json.loads(LOCK.read_text(encoding="utf-8"))
    if state.get("gate_implementation_head") != GATE_IMPLEMENTATION_HEAD:
        raise RuntimeError("V5-3I launch lock head mismatch")
    if state.get("status") != "ALLOCATED":
        raise RuntimeError(f"V5-3I launch lock is not ALLOCATED: {state.get('status')}")

    state.update({"pid": os.getpid(), "status": "BOOTSTRAPPING", "started_at_utc": utc_now()})
    atomic_json(LOCK, state)
    stop = threading.Event()

    def heartbeat_loop() -> None:
        while not stop.is_set():
            atomic_json(
                HEARTBEAT,
                {
                    "schema": "st-omr-meter-v5-3i-background-heartbeat-v1",
                    "gate_implementation_head": GATE_IMPLEMENTATION_HEAD,
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
        checkout_exact_gate_source()
        print("EXACT GATE SOURCE = PASS", flush=True)
        actual_runtime = prepare_isolated_runtime()
        print("ISOLATED VENV BOOTSTRAP = PASS", flush=True)
        print("ISOLATED RUNTIME =", json.dumps(actual_runtime, sort_keys=True), flush=True)
        print("PINNED RUNTIME = PASS", flush=True)

        state["status"] = "GATE_EVALUATING"
        atomic_json(LOCK, state)
        run_worker()

        if not RESULT.is_file():
            raise RuntimeError("V5-3I result not written")
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        state.update(
            {
                "status": "COMPLETED",
                "decision": result.get("decision"),
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
                "schema": "st-omr-meter-v5-3i-background-heartbeat-v1",
                "gate_implementation_head": GATE_IMPLEMENTATION_HEAD,
                "pid": os.getpid(),
                "status": state.get("status"),
                "decision": state.get("decision"),
                "utc": utc_now(),
                "final": True,
            },
        )


if __name__ == "__main__":
    raise SystemExit(main())
