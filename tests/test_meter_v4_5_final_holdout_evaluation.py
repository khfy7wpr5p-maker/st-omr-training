from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image
import torch

import st_omr_training.meter_v4_5_final_holdout_evaluation as v45
from st_omr_training.meter_v4_4_bbox_contract import BBox
from st_omr_training.meter_v4_1_numerator_specialist import NumeratorSpecialistV4_1
from st_omr_training.training_model import model_state_sha256


class _FakePerfectModel:
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        logits = torch.full((batch.shape[0], 3), -10.0, dtype=torch.float32)
        encoded = batch[:, 0, 0, 0].round().to(torch.int64)
        for i, class_index in enumerate(encoded.tolist()):
            logits[i, class_index] = 10.0
        return logits


def _prereg() -> dict[str, object]:
    return {
        "schema": v45.PREREG_SCHEMA, "stage": v45.V4_5_STAGE, "status": "PREREGISTERED_NO_FINAL_INFERENCE",
        "parents": {
            "v4_4_selection_sha256": v45.EXPECTED_SELECTION_SHA256,
            "v4_4_image_binding_sha256": v45.EXPECTED_IMAGE_BINDING_SHA256,
            "v4_4_bbox_manifest_sha256": v45.EXPECTED_BBOX_MANIFEST_SHA256,
            "v4_4_completion_receipt_file_sha256": v45.EXPECTED_COMPLETION_RECEIPT_SHA256,
            "v4_2_result_file_sha256": v45.EXPECTED_V4_2_RESULT_SHA256,
            "v4_2_checkpoint_file_sha256": v45.EXPECTED_CHECKPOINT_SHA256,
            "v4_2_model_state_sha256": v45.EXPECTED_MODEL_STATE_SHA256,
            "v4_2_configuration_fingerprint": v45.EXPECTED_CONFIG_FINGERPRINT,
        },
        "surface": {"records": 150, "families": 150, "classes": {"2": 50, "3": 50, "4": 50}, "holdout_read_only": True, "sealed_test_opened": False},
        "representation": {"source_coordinates": "original_image_integer_pixels", "horizontal_padding_milli": 150, "vertical_padding_milli": 50, "numerator_fraction_milli": 500, "output_size": 64, "grayscale_mode": "L", "resample": "BILINEAR", "ink_normalization": "(255-gray)/255"},
        "evaluation": {"checkpoint_deserializations_max": 1, "inference_records_exact": 150, "optimizer_steps": 0, "training": False, "tuning": False, "calibration": False, "threshold_search": False, "test_time_augmentation": False, "ensemble": False, "rerun_after_lock": False},
        "gate": {"accuracy_min": 0.9, "macro_f1_min": 0.9, "per_class_recall_min": {"2": 0.9, "3": 0.9, "4": 0.9}, "pass_name": "FINAL_HOLDOUT_PASS", "fail_name": "FINAL_HOLDOUT_FAIL"},
        "downstream": {"runtime_connected": False, "resolver_connected": False, "production_promotion_authorized": False},
    }


def _human() -> dict[str, object]:
    return {
        "schema": v45.HUMAN_REVIEW_SCHEMA,
        "stage": "st-omr-meter-v4-4-final-holdout-bbox-annotation-v1",
        "review_status": "PASS", "reviewed_count": 150, "contact_sheet_count": 6,
        "contact_sheet_files": [f"contact_sheet_{i:02d}.png" for i in range(1, 7)],
        "human_confirmation": {"explicit": True, "statement": "contact sheet'ler temiz, onayliyorum", "recorded_at": "2026-08-21T23:25:00+03:00"},
        "mechanical_receipt": {"schema": "st-omr-meter-v4-4-bbox-complete-v1", "sha256": v45.EXPECTED_COMPLETION_RECEIPT_SHA256, "selection_sha256": v45.EXPECTED_SELECTION_SHA256, "image_binding_sha256": v45.EXPECTED_IMAGE_BINDING_SHA256, "bbox_manifest_sha256": v45.EXPECTED_BBOX_MANIFEST_SHA256, "annotated_count": 150, "missing_bbox": 0, "invalid_bbox": 0, "unique_family_count": 150, "class_counts": {"2": 50, "3": 50, "4": 50}},
        "review_assertions": {"both_meter_digits_contained": True, "digits_not_cut": True, "correct_meter_target": True, "no_symbol_drift": True, "visible_meter_sign": True},
        "downstream_gates": {"model_evaluated": False, "inference_count": 0, "candidate_checkpoint_opened": False, "test_opened": False, "runtime_connected": False, "production_promotion_authorized": False},
        "decision": {"v4_4_complete": True, "v4_5_one_time_independent_evaluation_may_be_prepared": True, "production_promotion_authorized": False},
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="ascii")


class MeterV45Tests(unittest.TestCase):
    def test_frozen_parent_hashes_are_exact(self):
        self.assertEqual(v45.EXPECTED_SELECTION_SHA256, "4335a48a091912ba422c16d8fcbaaa7bbf5f7a0a43f088146a50a3e02e3ed7dc")
        self.assertEqual(v45.EXPECTED_CHECKPOINT_SHA256, "2dc820bc0cbadf5db90a7ddee7f5a9daba06e546dcae1da560d1ac9718e3692a")
        self.assertEqual(v45.PER_CLASS_RECALL_MIN, {"2": 0.9, "3": 0.9, "4": 0.9})

    def test_preregistration_pass_and_threshold_mutation_fails(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pre.json"; value = _prereg(); _write_json(path, value)
            self.assertEqual(v45.validate_preregistration(path)["status"], "PREREGISTERED_NO_FINAL_INFERENCE")
            value["gate"]["accuracy_min"] = 0.89; _write_json(path, value)
            with self.assertRaises(v45.MeterV4_5Error): v45.validate_preregistration(path)

    def test_human_review_pass_and_prior_inference_fails(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "human.json"; value = _human(); _write_json(path, value)
            self.assertEqual(v45.validate_human_review_evidence(path)["review_status"], "PASS")
            value["downstream_gates"]["inference_count"] = 1; _write_json(path, value)
            with self.assertRaises(v45.MeterV4_5Error): v45.validate_human_review_evidence(path)

    def test_original_numerator_bounds_match_frozen_geometry(self):
        self.assertEqual(v45.numerator_bounds_original_v4_5(BBox(100,20,40,80), image_width=300, image_height=150), (94,16,146,64))
        self.assertEqual(v45.numerator_bounds_original_v4_5(BBox(1,1,20,40), image_width=100, image_height=100), (0,0,24,23))

    def test_crop_is_deterministic_gray64(self):
        image = Image.new("RGB", (120,80), "white")
        for y in range(10,50):
            for x in range(20,40): image.putpixel((x,y), (0,0,0))
        bbox = BBox(20,10,20,40)
        first = v45.render_numerator_crop_original_v4_5(image,bbox); second = v45.render_numerator_crop_original_v4_5(image,bbox)
        self.assertEqual(first.mode, "L"); self.assertEqual(first.size, (64,64)); self.assertEqual(first.tobytes(), second.tobytes())
        tensor = v45.crop_tensor_v4_5(image,bbox); self.assertEqual(tuple(tensor.shape),(1,64,64)); self.assertEqual(tensor.dtype,torch.float32)

    def test_invalid_bbox_is_rejected(self):
        with self.assertRaises(v45.MeterV4_5Error): v45.numerator_bounds_original_v4_5(BBox(-1,0,5,5), image_width=10, image_height=10)
        with self.assertRaises(v45.MeterV4_5Error): v45.numerator_bounds_original_v4_5(BBox(0,0,11,5), image_width=10, image_height=10)

    def test_metrics_perfect_and_gate_pass(self):
        truth = ["2"]*50+["3"]*50+["4"]*50; summary = v45.summarize_predictions_v4_5(truth,truth)
        self.assertEqual(summary.accuracy,1.0); self.assertEqual(summary.macro_f1,1.0); self.assertEqual(summary.confusion,((50,0,0),(0,50,0),(0,0,50)))
        decision = v45.final_decision_v4_5(summary); self.assertEqual(decision["name"],"FINAL_HOLDOUT_PASS"); self.assertFalse(decision["production_promotion_authorized"])

    def test_gate_fails_when_one_class_recall_below_90(self):
        truth=["2"]*50+["3"]*50+["4"]*50; pred=truth.copy()
        for i in range(6): pred[100+i]="3"
        summary=v45.summarize_predictions_v4_5(truth,pred); self.assertEqual(summary.per_class_recall["4"],0.88)
        decision=v45.final_decision_v4_5(summary); self.assertEqual(decision["name"],"FINAL_HOLDOUT_FAIL"); self.assertIn("FINAL_4_RECALL_BELOW_90_PERCENT",decision["reasons"])

    def test_metrics_reject_wrong_cardinality(self):
        with self.assertRaises(v45.MeterV4_5Error): v45.summarize_predictions_v4_5(["2"]*149,["2"]*149)

    def test_checkpoint_hash_rejects_wrong_bytes_and_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); cp=root/"candidate.pt"; cp.write_bytes(b"not-the-candidate")
            with self.assertRaises(v45.MeterV4_5Error): v45.validate_checkpoint_file_hash(cp)
            link=root/"link.pt"; link.symlink_to(cp)
            with self.assertRaises(v45.MeterV4_5Error): v45.validate_checkpoint_file_hash(link)

    def test_exact_candidate_loader_accepts_synthetic_trusted_state(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"candidate.pt"; torch.manual_seed(123); model=NumeratorSpecialistV4_1().cpu(); state_sha=model_state_sha256(model)
            torch.save({"schema":v45.CHECKPOINT_SCHEMA,"model_state_sha256":state_sha,"config_fingerprint_v4_1":v45.config_fingerprint_v4_1(),"state_dict":{n:t.detach().cpu() for n,t in model.state_dict().items()}},path)
            with mock.patch.object(v45,"EXPECTED_MODEL_STATE_SHA256",state_sha): loaded=v45._load_exact_candidate_after_lock(path)
            self.assertEqual(model_state_sha256(loaded),state_sha)

    def test_exclusive_lock_cannot_be_acquired_twice(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"lock.json"; first=v45._exclusive_lock(path,{"schema":"x","rerun":False}); self.assertEqual(len(first),64)
            with self.assertRaises(v45.MeterV4_5Error): v45._exclusive_lock(path,{"schema":"x","rerun":False})

    def test_result_writer_is_fresh_and_immutable(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/"out"; result_sha=v45._write_fresh_result(output,{"schema":"test","value":1}); self.assertEqual(len(result_sha),64); self.assertTrue((output/"COMPLETE").is_file())
            with self.assertRaises(v45.MeterV4_5Error): v45._write_fresh_result(output,{"schema":"test","value":2})

    def test_one_shot_run_creates_lock_before_model_open_and_refuses_second_run(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); output=root/"v45"; checkpoint=root/"candidate.pt"; checkpoint.write_bytes(b"fixture"); prepared=[]
            for class_index,name in enumerate(("2","3","4")):
                for offset in range(50):
                    tensor=torch.zeros((1,64,64),dtype=torch.float32); tensor[0,0,0]=float(class_index)
                    prepared.append(v45.PreparedRecord(index=len(prepared),family_id=f"f-{name}-{offset}",folder_name=f"folder-{name}-{offset}",true_class=name,image_sha256=f"{len(prepared):064x}"[-64:],bbox_file_sha256=f"{len(prepared)+1:064x}"[-64:],bbox=BBox(1,1,2,4),tensor=tensor))
            prereg=_prereg(); human=_human(); v42={"decision":{"name":"FULL_TRAIN_DEV_SCREEN_PASS"}}; lock_path=output.with_name(output.name+".ONE_SHOT_LOCK.json")
            def load_after_lock(_path): self.assertTrue(lock_path.is_file()); return _FakePerfectModel()
            with mock.patch.object(v45,"validate_preregistration",return_value=prereg), mock.patch.object(v45,"validate_human_review_evidence",return_value=human), mock.patch.object(v45,"prepare_final_holdout_v4_5",return_value=tuple(prepared)), mock.patch.object(v45,"validate_v4_2_result",return_value=v42), mock.patch.object(v45,"validate_checkpoint_file_hash",return_value=checkpoint), mock.patch.object(v45,"_load_exact_candidate_after_lock",side_effect=load_after_lock):
                result=v45.run_meter_v4_5_one_time_final_holdout_evaluation(candidate_root=root,manifest_path=root/"manifest",completion_receipt_path=root/"receipt",human_review_evidence_path=root/"human",preregistration_path=root/"prereg",v4_2_result_path=root/"v42",checkpoint_path=checkpoint,output_root=output,git_commit_sha="a"*40)
                self.assertEqual(result["decision"]["name"],"FINAL_HOLDOUT_PASS"); self.assertEqual(result["safety"]["inference_count"],150); self.assertEqual(result["safety"]["checkpoint_deserializations"],1); self.assertFalse(result["safety"]["production_promotion_authorized"])
                with self.assertRaises(v45.MeterV4_5Error): v45.run_meter_v4_5_one_time_final_holdout_evaluation(candidate_root=root,manifest_path=root/"manifest",completion_receipt_path=root/"receipt",human_review_evidence_path=root/"human",preregistration_path=root/"prereg",v4_2_result_path=root/"v42",checkpoint_path=checkpoint,output_root=output,git_commit_sha="a"*40)

    def test_existing_partial_or_lock_fails_before_preflight(self):
        for kind in ("part","lock"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as td:
                root=Path(td); output=root/"v45"; target=output.with_name("."+output.name+".part") if kind=="part" else output.with_name(output.name+".ONE_SHOT_LOCK.json")
                target.mkdir() if kind=="part" else target.write_text("locked",encoding="ascii")
                with mock.patch.object(v45,"validate_preregistration") as pre:
                    with self.assertRaises(v45.MeterV4_5Error): v45.run_meter_v4_5_one_time_final_holdout_evaluation(candidate_root=root,manifest_path=root/"m",completion_receipt_path=root/"r",human_review_evidence_path=root/"h",preregistration_path=root/"p",v4_2_result_path=root/"v",checkpoint_path=root/"c",output_root=output,git_commit_sha="a"*40)
                    pre.assert_not_called()


if __name__ == "__main__": unittest.main()
