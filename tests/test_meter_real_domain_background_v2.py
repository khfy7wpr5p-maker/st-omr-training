from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import tools.meter_real_domain_background_runner_v1 as runner


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "st_omr_meter_real_domain_background_v2_colab.ipynb"
)


class _StatusRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def update(self, event: str, payload: dict[str, object]) -> None:
        self.events.append((event, payload))


class MeterRealDomainBackgroundV2Tests(unittest.TestCase):
    def test_durable_status_persists_progress_and_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            status = runner.DurableStatus(path)
            status.update(
                "training_batch",
                {"phase_index": 8, "phase_total": 9, "epoch": 3, "epochs_total": 8, "batch": 12, "batches_total": 30},
            )
            observed = json.loads(path.read_text("ascii"))
            self.assertEqual((observed["phase_index"], observed["phase_total"]), (8, 9))
            self.assertEqual((observed["epoch"], observed["epochs_total"]), (3, 8))
            self.assertEqual((observed["batch"], observed["batches_total"]), (12, 30))
            status.finish(state="COMPLETE", payload={"result": "HOLD_NO_ACCEPTED_CANDIDATE"})
            terminal = json.loads(path.read_text("ascii"))
            self.assertEqual(terminal["state"], "COMPLETE")
            self.assertEqual(terminal["result"], "HOLD_NO_ACCEPTED_CANDIDATE")

    def test_notebook_is_clean_and_exposes_durable_progress(self) -> None:
        document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", ())) for cell in document["cells"])
        self.assertEqual(document["nbformat"], 4)
        for cell in document["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])
        for required in (
            "meter_real_domain_background_runner_v1.py",
            "start_new_session=True",
            "STATUS_PATH",
            "phase_index",
            "files_completed",
            "epochs_total",
            "heartbeat_age_seconds",
            "resume",
        ):
            self.assertIn(required, source)
        self.assertNotIn("/01_REVIEW/test", source)
        self.assertIn("metrics['test_opened'] is False", source)
        self.assertIn("metrics['runtime_connected'] is False", source)
        self.assertIn("metrics['production_promotion_authorized'] is False", source)

    def test_local_cache_copy_reports_exact_numerator_and_denominator(self) -> None:
        old_expected = runner.EXPECTED_D10_RECORDS
        runner.EXPECTED_D10_RECORDS = 2
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "source"
                cache = root / "cache"
                source.mkdir()
                rows = []
                for index, split in enumerate(("train", "validation")):
                    image = f"images/{index}.png"
                    label = f"labels/{index}.json"
                    (source / image).parent.mkdir(parents=True, exist_ok=True)
                    (source / label).parent.mkdir(parents=True, exist_ok=True)
                    (source / image).write_bytes(f"image-{index}".encode("ascii"))
                    (source / label).write_bytes(f"label-{index}".encode("ascii"))
                    rows.append({"split": split, "image_path": image, "label_path": label})
                manifest = json.dumps({"records": rows}, separators=(",", ":"), sort_keys=True).encode("ascii")
                (source / "manifest.json").write_bytes(manifest)
                (source / "receipt.json").write_bytes(b"receipt")
                (source / "COMPLETE").write_bytes(b"complete")
                status = _StatusRecorder()
                result = runner.materialize_d10_cache(
                    source_root=source,
                    cache_root=cache,
                    expected_manifest_sha256=sha256(manifest).hexdigest(),
                    status=status,  # type: ignore[arg-type]
                )
                self.assertEqual(result, cache)
                self.assertTrue((cache / "CACHE_COMPLETE.json").is_file())
                final = status.events[-1]
                self.assertEqual(final[0], "d10_cache_complete")
                self.assertEqual(final[1]["files_completed"], 7)
                self.assertEqual(final[1]["files_total"], 7)
        finally:
            runner.EXPECTED_D10_RECORDS = old_expected

    def test_cache_planning_rejects_test_before_copy(self) -> None:
        old_expected = runner.EXPECTED_D10_RECORDS
        runner.EXPECTED_D10_RECORDS = 1
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "source"
                source.mkdir()
                manifest = json.dumps(
                    {"records": [{"split": "test", "image_path": "never", "label_path": "never"}]},
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
                (source / "manifest.json").write_bytes(manifest)
                with self.assertRaisesRegex(RuntimeError, "sealed TEST"):
                    runner.materialize_d10_cache(
                        source_root=source,
                        cache_root=root / "cache",
                        expected_manifest_sha256=sha256(manifest).hexdigest(),
                        status=_StatusRecorder(),  # type: ignore[arg-type]
                    )
        finally:
            runner.EXPECTED_D10_RECORDS = old_expected


if __name__ == "__main__":
    unittest.main()
