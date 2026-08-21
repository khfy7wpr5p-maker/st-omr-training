"""Crash-safe annotation session state for Meter V4-4."""
from __future__ import annotations

import base64
import io
import math
from hashlib import sha256
from pathlib import Path

from PIL import Image

from st_omr_training.meter_v4_4_bbox_contract import (
    BBox, ImageInfo, MeterV4_4AnnotationError, SelectedSample,
    EXPECTED_CLASS_COUNTS, EXPECTED_SELECTED_COUNT, EXPECTED_SELECTION_SHA256,
    IMAGE_BINDING_NAME, IMAGE_BINDING_SCHEMA, PROGRESS_NAME, PROGRESS_SCHEMA, V4_4_SCHEMA,
    atomic_write_json, bounded_string, canonical_json, discover_selected_samples, fail,
    load_and_validate_selection_manifest, preview_rect_to_original, read_bbox_contract,
    read_json, read_png_info, sha256_bytes, validate_bbox, write_bbox_atomic,
)

def _sample_binding_record(sample: SelectedSample, info: ImageInfo) -> dict[str, object]:
    return {
        "index": sample.index,
        "numerator_class": sample.numerator_class,
        "meter_class": sample.meter_class,
        "folder_name": sample.folder_name,
        "family_id": sample.family_id,
        "image_sha256": info.sha256,
        "image_size_bytes": info.size_bytes,
        "image_width": info.width,
        "image_height": info.height,
    }


def build_image_binding(samples: tuple[SelectedSample, ...]) -> dict[str, object]:
    if len(samples) != EXPECTED_SELECTED_COUNT:
        fail("image binding requires exactly 150 selected samples")
    records = []
    for sample in samples:
        info = read_png_info(sample.image_path)
        contract = read_bbox_contract(
            sample.bbox_path,
            expected_meter=sample.meter_class,
            image_width=info.width,
            image_height=info.height,
        )
        if contract.bbox is not None:
            fail("cannot create initial image binding after bbox annotation has started")
        records.append(_sample_binding_record(sample, info))
    binding: dict[str, object] = {
        "schema": IMAGE_BINDING_SCHEMA,
        "stage": V4_4_SCHEMA,
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "selected_count": EXPECTED_SELECTED_COUNT,
        "class_counts": EXPECTED_CLASS_COUNTS,
        "records": records,
        "model_evaluated": False,
        "candidate_checkpoint_opened": False,
        "test_opened": False,
        "runtime_connected": False,
        "production_promotion_authorized": False,
    }
    binding["image_binding_sha256"] = sha256_bytes(canonical_json(records))
    return binding


def validate_image_binding(
    binding: dict[str, object],
    samples: tuple[SelectedSample, ...],
) -> tuple[ImageInfo, ...]:
    if binding.get("schema") != IMAGE_BINDING_SCHEMA:
        fail("image binding schema mismatch")
    if binding.get("selection_sha256") != EXPECTED_SELECTION_SHA256:
        fail("image binding selection SHA mismatch")
    if binding.get("selected_count") != EXPECTED_SELECTED_COUNT:
        fail("image binding selected_count mismatch")
    if binding.get("class_counts") != EXPECTED_CLASS_COUNTS:
        fail("image binding class counts mismatch")
    for key in (
        "model_evaluated",
        "candidate_checkpoint_opened",
        "test_opened",
        "runtime_connected",
        "production_promotion_authorized",
    ):
        if binding.get(key) is not False:
            fail(f"image binding safety flag must be false: {key}")
    records = binding.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_SELECTED_COUNT:
        fail("image binding records must contain exactly 150 rows")
    expected_sha = sha256_bytes(canonical_json(records))
    if binding.get("image_binding_sha256") != expected_sha:
        fail("image binding payload SHA mismatch")

    infos: list[ImageInfo] = []
    for sample, row in zip(samples, records, strict=True):
        if not isinstance(row, dict):
            fail("image binding row must be an object")
        for key, expected in (
            ("index", sample.index),
            ("numerator_class", sample.numerator_class),
            ("meter_class", sample.meter_class),
            ("folder_name", sample.folder_name),
            ("family_id", sample.family_id),
        ):
            if row.get(key) != expected:
                fail(f"image binding sample mismatch: {key}")
        current = read_png_info(sample.image_path)
        info = ImageInfo(
            bounded_string("image_sha256", row.get("image_sha256"), max_len=64),
            row.get("image_size_bytes") if type(row.get("image_size_bytes")) is int else -1,
            row.get("image_width") if type(row.get("image_width")) is int else -1,
            row.get("image_height") if type(row.get("image_height")) is int else -1,
        )
        if current != info:
            fail(f"selected image changed after V4-4 binding: {sample.folder_name}")
        infos.append(info)
    return tuple(infos)


def load_or_create_image_binding(
    candidate_root: str | Path,
    samples: tuple[SelectedSample, ...],
) -> tuple[dict[str, object], tuple[ImageInfo, ...]]:
    path = Path(candidate_root) / IMAGE_BINDING_NAME
    if path.exists():
        binding = read_json(path)
    else:
        binding = build_image_binding(samples)
        atomic_write_json(path, binding)
    infos = validate_image_binding(binding, samples)
    return binding, infos


def _read_all_bboxes(
    samples: tuple[SelectedSample, ...],
    infos: tuple[ImageInfo, ...],
) -> tuple[BBox | None, ...]:
    bboxes = []
    for sample, info in zip(samples, infos, strict=True):
        contract = read_bbox_contract(
            sample.bbox_path,
            expected_meter=sample.meter_class,
            image_width=info.width,
            image_height=info.height,
        )
        bboxes.append(contract.bbox)
    return tuple(bboxes)


def _fresh_progress(binding: dict[str, object]) -> dict[str, object]:
    return {
        "schema": PROGRESS_SCHEMA,
        "stage": V4_4_SCHEMA,
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "image_binding_sha256": binding["image_binding_sha256"],
        "selected_count": EXPECTED_SELECTED_COUNT,
        "current_index": 0,
        "review_flags": [],
        "annotations": {},
        "model_evaluated": False,
        "candidate_checkpoint_opened": False,
        "test_opened": False,
        "runtime_connected": False,
        "production_promotion_authorized": False,
    }


def reconcile_progress(
    progress: dict[str, object],
    *,
    binding: dict[str, object],
    samples: tuple[SelectedSample, ...],
    infos: tuple[ImageInfo, ...],
) -> dict[str, object]:
    if progress.get("schema") != PROGRESS_SCHEMA:
        fail("progress schema mismatch")
    if progress.get("selection_sha256") != EXPECTED_SELECTION_SHA256:
        fail("progress selection SHA mismatch")
    if progress.get("image_binding_sha256") != binding.get("image_binding_sha256"):
        fail("progress image binding SHA mismatch")
    if progress.get("selected_count") != EXPECTED_SELECTED_COUNT:
        fail("progress selected_count mismatch")
    for key in (
        "model_evaluated",
        "candidate_checkpoint_opened",
        "test_opened",
        "runtime_connected",
        "production_promotion_authorized",
    ):
        if progress.get(key) is not False:
            fail(f"progress safety flag must be false: {key}")
    current_index = progress.get("current_index")
    if type(current_index) is not int or not 0 <= current_index < EXPECTED_SELECTED_COUNT:
        fail("progress current_index outside bounds")
    flags = progress.get("review_flags")
    if not isinstance(flags, list) or any(not isinstance(value, str) for value in flags):
        fail("progress review_flags malformed")
    valid_folders = {sample.folder_name for sample in samples}
    if len(flags) != len(set(flags)) or not set(flags).issubset(valid_folders):
        fail("progress review_flags contain invalid/duplicate samples")
    annotations = progress.get("annotations")
    if not isinstance(annotations, dict):
        fail("progress annotations must be an object")
    if not set(annotations).issubset(valid_folders):
        fail("progress contains annotation outside frozen selection")

    disk_bboxes = _read_all_bboxes(samples, infos)
    updated = dict(progress)
    updated_annotations = dict(annotations)
    for sample, info, disk_bbox in zip(samples, infos, disk_bboxes, strict=True):
        saved = updated_annotations.get(sample.folder_name)
        if saved is not None:
            if not isinstance(saved, dict):
                fail("progress annotation row malformed")
            bbox_raw = saved.get("bbox")
            if not isinstance(bbox_raw, dict):
                fail("progress bbox missing")
            try:
                progress_bbox = BBox(
                    bbox_raw["x"], bbox_raw["y"], bbox_raw["w"], bbox_raw["h"]
                )
            except KeyError as exc:
                raise MeterV4_4AnnotationError("progress bbox incomplete") from exc
            validate_bbox(progress_bbox, image_width=info.width, image_height=info.height)
            if saved.get("family_id") != sample.family_id or saved.get("image_sha256") != info.sha256:
                fail("progress annotation binding mismatch")
            if disk_bbox is None:
                fail("progress says annotated but bbox_meter.txt is blank")
            if disk_bbox != progress_bbox:
                fail("progress bbox disagrees with bbox_meter.txt")
        elif disk_bbox is not None:
            updated_annotations[sample.folder_name] = {
                "family_id": sample.family_id,
                "image_sha256": info.sha256,
                "bbox": disk_bbox.as_dict(),
            }
    updated["annotations"] = updated_annotations
    return updated


def load_or_create_progress(
    candidate_root: str | Path,
    *,
    binding: dict[str, object],
    samples: tuple[SelectedSample, ...],
    infos: tuple[ImageInfo, ...],
) -> dict[str, object]:
    path = Path(candidate_root) / PROGRESS_NAME
    if path.exists():
        progress = read_json(path)
    else:
        progress = _fresh_progress(binding)
    reconciled = reconcile_progress(
        progress, binding=binding, samples=samples, infos=infos
    )
    if reconciled != progress or not path.exists():
        atomic_write_json(path, reconciled)
    return reconciled


def binding_token(
    *,
    sample: SelectedSample,
    info: ImageInfo,
    binding_sha: str,
) -> str:
    raw = (
        f"{EXPECTED_SELECTION_SHA256}\n{binding_sha}\n{sample.index}\n"
        f"{sample.folder_name}\n{sample.family_id}\n{info.sha256}\n"
    ).encode("ascii")
    return sha256(raw).hexdigest()


def preview_png(
    path: Path,
    *,
    max_width: int = 1000,
    max_height: int = 720,
) -> tuple[str, int, int]:
    if type(max_width) is not int or type(max_height) is not int or max_width <= 0 or max_height <= 0:
        fail("preview bounds must be positive integers")
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        scale = min(1.0, max_width / width, max_height / height)
        preview_width = max(1, int(math.floor(width * scale)))
        preview_height = max(1, int(math.floor(height * scale)))
        if (preview_width, preview_height) != (width, height):
            image = image.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
        if image.mode not in {"L", "RGB", "RGBA"}:
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return data_uri, preview_width, preview_height


class AnnotationSession:
    """Stateful, fail-closed facade used by the Colab UI callbacks."""

    def __init__(
        self,
        *,
        candidate_root: str | Path,
        manifest_path: str | Path,
    ) -> None:
        self.candidate_root = Path(candidate_root)
        self.manifest_path = Path(manifest_path)
        _, rows = load_and_validate_selection_manifest(self.manifest_path)
        self.samples = discover_selected_samples(self.candidate_root, rows)
        self.binding, self.infos = load_or_create_image_binding(self.candidate_root, self.samples)
        self.progress = load_or_create_progress(
            self.candidate_root,
            binding=self.binding,
            samples=self.samples,
            infos=self.infos,
        )

    @property
    def progress_path(self) -> Path:
        return self.candidate_root / PROGRESS_NAME

    @property
    def annotated_count(self) -> int:
        annotations = self.progress["annotations"]
        assert isinstance(annotations, dict)
        return len(annotations)

    def resume_index(self) -> int:
        annotations = self.progress["annotations"]
        assert isinstance(annotations, dict)
        for sample in self.samples:
            if sample.folder_name not in annotations:
                return sample.index
        current = self.progress["current_index"]
        assert type(current) is int
        return current

    def _persist_progress(self) -> None:
        self.progress = reconcile_progress(
            self.progress,
            binding=self.binding,
            samples=self.samples,
            infos=self.infos,
        )
        atomic_write_json(self.progress_path, self.progress)

    def set_current_index(self, index: int) -> None:
        if type(index) is not int or not 0 <= index < len(self.samples):
            fail("sample index outside frozen selection")
        self.progress["current_index"] = index
        self._persist_progress()

    def set_review_flag(self, *, token: str, flagged: bool) -> dict[str, object]:
        sample, info = self._resolve_token(token)
        if type(flagged) is not bool:
            fail("review flag must be boolean")
        flags = self.progress["review_flags"]
        assert isinstance(flags, list)
        flag_set = set(flags)
        if flagged:
            flag_set.add(sample.folder_name)
        else:
            flag_set.discard(sample.folder_name)
        self.progress["review_flags"] = sorted(flag_set)
        self.progress["current_index"] = sample.index
        self._persist_progress()
        return self.sample_payload(sample.index)

    def _resolve_token(self, token: str) -> tuple[SelectedSample, ImageInfo]:
        if not isinstance(token, str) or len(token) != 64:
            fail("sample binding token malformed")
        binding_sha = self.binding.get("image_binding_sha256")
        if not isinstance(binding_sha, str):
            fail("image binding SHA missing")
        for sample, info in zip(self.samples, self.infos, strict=True):
            if binding_token(sample=sample, info=info, binding_sha=binding_sha) == token:
                return sample, info
        fail("sample binding token does not match frozen selection")

    def save_from_preview(
        self,
        *,
        token: str,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        preview_width: int,
        preview_height: int,
    ) -> dict[str, object]:
        sample, info = self._resolve_token(token)
        bbox = preview_rect_to_original(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            preview_width=preview_width,
            preview_height=preview_height,
            image_width=info.width,
            image_height=info.height,
        )
        write_bbox_atomic(sample, bbox, expected_image=info)
        annotations = self.progress["annotations"]
        assert isinstance(annotations, dict)
        annotations[sample.folder_name] = {
            "family_id": sample.family_id,
            "image_sha256": info.sha256,
            "bbox": bbox.as_dict(),
        }
        self.progress["current_index"] = sample.index
        self._persist_progress()
        return {
            "saved": True,
            "index": sample.index,
            "annotated_count": self.annotated_count,
            "bbox": bbox.as_dict(),
        }

    def sample_payload(self, index: int) -> dict[str, object]:
        if type(index) is not int or not 0 <= index < len(self.samples):
            fail("sample index outside frozen selection")
        sample = self.samples[index]
        info = self.infos[index]
        current_info = read_png_info(sample.image_path)
        if current_info != info:
            fail("image changed after binding")
        contract = read_bbox_contract(
            sample.bbox_path,
            expected_meter=sample.meter_class,
            image_width=info.width,
            image_height=info.height,
        )
        data_uri, pw, ph = preview_png(sample.image_path)
        binding_sha = self.binding["image_binding_sha256"]
        assert isinstance(binding_sha, str)
        flags = self.progress["review_flags"]
        assert isinstance(flags, list)
        self.progress["current_index"] = index
        self._persist_progress()
        return {
            "index": index,
            "total": len(self.samples),
            "annotated_count": self.annotated_count,
            "meter_class": sample.meter_class,
            "folder_name": sample.folder_name,
            "family_id": sample.family_id,
            "image_width": info.width,
            "image_height": info.height,
            "preview_width": pw,
            "preview_height": ph,
            "preview_data_uri": data_uri,
            "bbox": contract.bbox.as_dict() if contract.bbox else None,
            "review_flag": sample.folder_name in set(flags),
            "binding_token": binding_token(sample=sample, info=info, binding_sha=binding_sha),
        }
