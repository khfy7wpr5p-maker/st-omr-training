import copy
import inspect
import unittest

from st_omr_training import meter_v5_2b_post_replacement_qa_v1 as m


def _annotation(sample_id, meter, *, sha=None):
    value = sha or (sample_id[:1] * 64 if sample_id else "a" * 64)
    return {
        "sample_id": sample_id,
        "meter": meter,
        "split": "train",
        "x": "10",
        "y": "5",
        "w": "20",
        "h": "30",
        "status": "PASS",
        "image_sha256": value,
        "image_width": "100",
        "image_height": "50",
        "updated_utc": "2026-08-22T00:00:00Z",
    }


def _fixture():
    selection = []
    current = []
    archived = []
    seed_ids = []
    seed_rows = []

    replacements = {
        63: ("150200092-1_1_1", "150201200-1_1_1", "2/4"),
        125: ("150207112-1_1_1", "110003725-1_1_1", "3/4"),
    }

    for index in range(300):
        if index < 100:
            meter = "2/4"
        elif index < 200:
            meter = "3/4"
        else:
            meter = "4/4"

        new_id = f"sample-{index:03d}"
        old_id = new_id
        if index in replacements:
            new_id, old_id, meter = replacements[index]

        sha = (f"{index:064x}")[-64:]
        row = {
            "index": str(index),
            "sample_id": new_id,
            "family_id": f"family-{new_id}",
            "meter": meter,
            "split": "train",
            "folder": new_id,
            "image_relpath": f"train/{meter}/{new_id}/image.png",
            "image_sha256": sha,
            "image_width": "100",
            "image_height": "50",
            "seed_annotation": "1" if index < 30 else "0",
            "selection_rank": f"rank-{index:03d}",
        }
        selection.append(row)
        current_row = _annotation(new_id, meter, sha=sha)
        current.append(current_row)

        archived_row = _annotation(old_id, meter, sha=sha)
        archived.append(archived_row)

        if index < 30:
            seed_ids.append(new_id)
            seed_rows.append(copy.deepcopy(current_row))

    return selection, current, archived, seed_ids, seed_rows


class TestMeterV52BPostReplacementQA(unittest.TestCase):
    def test_exact_target_contract(self):
        self.assertEqual(
            m.TARGETS,
            (
                (63, "150200092-1_1_1", "2/4", "150201200-1_1_1"),
                (125, "150207112-1_1_1", "3/4", "110003725-1_1_1"),
            ),
        )

    def test_valid_298_preserved_plus_two_new_state_passes(self):
        selection, current, archived, seed_ids, seed_rows = _fixture()
        result = m.validate_post_replacement_rows_v1(
            selection_rows=selection,
            annotation_rows=current,
            archived_annotation_rows=archived,
            seed_sample_ids=seed_ids,
            seed_annotation_rows=seed_rows,
        )
        self.assertEqual(result["preserved_annotation_count"], 298)
        self.assertEqual(result["replacement_annotation_count"], 2)
        self.assertEqual(result["seed_mutation_count"], 0)
        self.assertEqual(result["per_class"]["2/4"], {"PASS": 100, "REVIEW": 0})
        self.assertEqual(result["per_class"]["3/4"], {"PASS": 100, "REVIEW": 0})
        self.assertEqual(result["per_class"]["4/4"], {"PASS": 100, "REVIEW": 0})

    def test_preserved_annotation_mutation_fails_closed(self):
        selection, current, archived, seed_ids, seed_rows = _fixture()
        current[200]["x"] = "11"
        with self.assertRaises(Exception):
            m.validate_post_replacement_rows_v1(
                selection_rows=selection,
                annotation_rows=current,
                archived_annotation_rows=archived,
                seed_sample_ids=seed_ids,
                seed_annotation_rows=seed_rows,
            )

    def test_replacement_binding_change_fails_closed(self):
        selection, current, archived, seed_ids, seed_rows = _fixture()
        selection[63]["sample_id"] = "wrong-replacement"
        current[63]["sample_id"] = "wrong-replacement"
        with self.assertRaises(Exception):
            m.validate_post_replacement_rows_v1(
                selection_rows=selection,
                annotation_rows=current,
                archived_annotation_rows=archived,
                seed_sample_ids=seed_ids,
                seed_annotation_rows=seed_rows,
            )

    def test_seed_annotation_mutation_fails_closed(self):
        selection, current, archived, seed_ids, seed_rows = _fixture()
        current[0]["w"] = "21"
        archived[0]["w"] = "21"
        with self.assertRaises(Exception):
            m.validate_post_replacement_rows_v1(
                selection_rows=selection,
                annotation_rows=current,
                archived_annotation_rows=archived,
                seed_sample_ids=seed_ids,
                seed_annotation_rows=seed_rows,
            )

    def test_replacement_review_status_fails_closed(self):
        selection, current, archived, seed_ids, seed_rows = _fixture()
        current[125]["status"] = "REVIEW"
        current[125]["x"] = current[125]["y"] = current[125]["w"] = current[125]["h"] = ""
        with self.assertRaises(Exception):
            m.validate_post_replacement_rows_v1(
                selection_rows=selection,
                annotation_rows=current,
                archived_annotation_rows=archived,
                seed_sample_ids=seed_ids,
                seed_annotation_rows=seed_rows,
            )

    def test_module_has_no_training_or_slot_derivation_execution(self):
        source = inspect.getsource(m).lower()
        self.assertNotIn("optimizer =", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("derive_staff_relative_slots_v1(", source)
        self.assertNotIn("derive_staff_relative_slots_v2(", source)
        self.assertNotIn("train_adapted_specialists_v1(", source)
        self.assertNotIn("evaluate_diagnostic_gate_v1(", source)


if __name__ == "__main__":
    unittest.main()
