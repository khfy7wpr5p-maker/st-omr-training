"""Mechanical completion receipt and human-review contact sheets for Meter V4-4."""
from __future__ import annotations

from collections import Counter
import io
import math
from pathlib import Path

from PIL import Image, ImageDraw

from st_omr_training.meter_v4_4_bbox_contract import (
    BBox, COMPLETE_NAME, COMPLETE_SCHEMA, EXPECTED_CLASS_COUNTS, EXPECTED_SELECTED_COUNT,
    EXPECTED_SELECTION_SHA256, IMAGE_BINDING_NAME, PROGRESS_NAME, REVIEW_DIR_NAME,
    V4_4_SCHEMA, atomic_write_bytes, atomic_write_json, canonical_json, discover_selected_samples, fail,
    load_and_validate_selection_manifest, read_bbox_contract, read_json, read_png_info,
    sha256_bytes, sha256_file,
)
from st_omr_training.meter_v4_4_bbox_state import reconcile_progress, validate_image_binding

def build_completion_receipt(
    *,
    candidate_root: str | Path,
    manifest_path: str | Path,
) -> dict[str, object]:
    root = Path(candidate_root)
    _, rows = load_and_validate_selection_manifest(manifest_path)
    samples = discover_selected_samples(root, rows)
    binding_path = root / IMAGE_BINDING_NAME
    if not binding_path.is_file():
        fail("V4-4 image binding is missing")
    binding = read_json(binding_path)
    infos = validate_image_binding(binding, samples)

    progress_path = root / PROGRESS_NAME
    if not progress_path.is_file():
        fail("V4-4 progress file is missing")
    progress = reconcile_progress(
        read_json(progress_path),
        binding=binding,
        samples=samples,
        infos=infos,
    )
    annotations = progress["annotations"]
    assert isinstance(annotations, dict)
    if len(annotations) != EXPECTED_SELECTED_COUNT:
        fail(f"annotation incomplete: {len(annotations)}/150")
    flags = progress["review_flags"]
    assert isinstance(flags, list)
    if flags:
        fail(f"review flags remain unresolved: {len(flags)}")

    records: list[dict[str, object]] = []
    class_counts: Counter[str] = Counter()
    for sample, info in zip(samples, infos, strict=True):
        contract = read_bbox_contract(
            sample.bbox_path,
            expected_meter=sample.meter_class,
            image_width=info.width,
            image_height=info.height,
        )
        if contract.bbox is None:
            fail("completion gate found missing bbox")
        class_counts[sample.numerator_class] += 1
        bbox_raw_sha = sha256_file(sample.bbox_path)
        records.append(
            {
                "index": sample.index,
                "numerator_class": sample.numerator_class,
                "meter_class": sample.meter_class,
                "folder_name": sample.folder_name,
                "family_id": sample.family_id,
                "image_sha256": info.sha256,
                "image_width": info.width,
                "image_height": info.height,
                "bbox": contract.bbox.as_dict(),
                "bbox_file_sha256": bbox_raw_sha,
            }
        )
    if dict(class_counts) != EXPECTED_CLASS_COUNTS:
        fail("completion class counts are not 50/50/50")

    bbox_manifest_sha = sha256_bytes(canonical_json(records))
    receipt: dict[str, object] = {
        "schema": COMPLETE_SCHEMA,
        "stage": V4_4_SCHEMA,
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "image_binding_sha256": binding["image_binding_sha256"],
        "annotated_count": EXPECTED_SELECTED_COUNT,
        "missing_bbox": 0,
        "invalid_bbox": 0,
        "unique_family_count": EXPECTED_SELECTED_COUNT,
        "class_counts": EXPECTED_CLASS_COUNTS,
        "records": records,
        "bbox_manifest_sha256": bbox_manifest_sha,
        "human_visual_review_passed": False,
        "model_evaluated": False,
        "inference_count": 0,
        "candidate_checkpoint_opened": False,
        "test_opened": False,
        "runtime_connected": False,
        "production_promotion_authorized": False,
    }
    return receipt


def write_completion_receipt(
    *,
    candidate_root: str | Path,
    manifest_path: str | Path,
) -> Path:
    root = Path(candidate_root)
    receipt = build_completion_receipt(candidate_root=root, manifest_path=manifest_path)
    path = root / COMPLETE_NAME
    if path.exists():
        existing = read_json(path)
        if existing != receipt:
            fail("existing completion receipt differs from deterministic recomputation")
        return path
    atomic_write_json(path, receipt)
    return path


def generate_review_contact_sheets(
    *,
    candidate_root: str | Path,
    manifest_path: str | Path,
    per_sheet: int = 25,
    columns: int = 5,
    tile_width: int = 320,
    tile_height: int = 220,
) -> tuple[Path, ...]:
    if any(type(value) is not int or value <= 0 for value in (per_sheet, columns, tile_width, tile_height)):
        fail("contact sheet dimensions must be positive integers")
    root = Path(candidate_root)
    receipt = build_completion_receipt(candidate_root=root, manifest_path=manifest_path)
    records = receipt["records"]
    assert isinstance(records, list)
    _, rows = load_and_validate_selection_manifest(manifest_path)
    samples = discover_selected_samples(root, rows)
    output_dir = root / REVIEW_DIR_NAME
    if output_dir.exists() and output_dir.is_symlink():
        fail("review output directory cannot be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_per_sheet = math.ceil(per_sheet / columns)
    outputs: list[Path] = []
    for sheet_index, start in enumerate(range(0, len(samples), per_sheet), start=1):
        chunk = samples[start : start + per_sheet]
        canvas = Image.new("RGB", (columns * tile_width, rows_per_sheet * tile_height), "white")
        for offset, sample in enumerate(chunk):
            record = records[start + offset]
            assert isinstance(record, dict)
            bbox_raw = record["bbox"]
            assert isinstance(bbox_raw, dict)
            bbox = BBox(bbox_raw["x"], bbox_raw["y"], bbox_raw["w"], bbox_raw["h"])
            with Image.open(sample.image_path) as image:
                image = image.convert("RGB")
                iw, ih = image.size
                draw_h = tile_height - 32
                scale = min(tile_width / iw, draw_h / ih)
                pw = max(1, int(math.floor(iw * scale)))
                ph = max(1, int(math.floor(ih * scale)))
                preview = image.resize((pw, ph), Image.Resampling.LANCZOS)
                xoff = (tile_width - pw) // 2
                yoff = (draw_h - ph) // 2
                draw = ImageDraw.Draw(preview)
                x0 = int(math.floor(bbox.x * pw / iw))
                y0 = int(math.floor(bbox.y * ph / ih))
                x1 = int(math.ceil((bbox.x + bbox.w) * pw / iw))
                y1 = int(math.ceil((bbox.y + bbox.h) * ph / ih))
                draw.rectangle((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)), outline="red", width=3)
                col = offset % columns
                row = offset // columns
                tile_x = col * tile_width
                tile_y = row * tile_height
                canvas.paste(preview, (tile_x + xoff, tile_y + yoff))
                label_draw = ImageDraw.Draw(canvas)
                label = f"{start + offset + 1:03d}/150  {sample.meter_class}  {sample.family_id}"
                label_draw.text((tile_x + 4, tile_y + draw_h + 8), label, fill="black")
        output = output_dir / f"contact_sheet_{sheet_index:02d}.png"
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        atomic_write_bytes(output, buffer.getvalue())
        outputs.append(output)
    return tuple(outputs)
