"""Security and persistence primitives for Meter V4-4 bbox annotation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Final

from PIL import Image, UnidentifiedImageError

V4_4_SCHEMA: Final[str] = "st-omr-meter-v4-4-final-holdout-bbox-annotation-v1"
V4_3_MANIFEST_SCHEMA: Final[str] = "st-omr-meter-v4-3-final-holdout-admission-manifest-v1"
EXPECTED_SELECTION_SHA256: Final[str] = "4335a48a091912ba422c16d8fcbaaa7bbf5f7a0a43f088146a50a3e02e3ed7dc"
EXPECTED_SELECTED_COUNT: Final[int] = 150
EXPECTED_CLASS_COUNTS: Final[dict[str, int]] = {"2": 50, "3": 50, "4": 50}
METER_BY_NUMERATOR: Final[dict[str, str]] = {"2": "2/4", "3": "3/4", "4": "4/4"}
IMAGE_BINDING_NAME: Final[str] = "FINAL_HOLDOUT_150_V4_4_IMAGE_BINDING.json"
PROGRESS_NAME: Final[str] = "FINAL_HOLDOUT_150_BBOX_PROGRESS.json"
COMPLETE_NAME: Final[str] = "FINAL_HOLDOUT_150_BBOX_COMPLETE.json"
REVIEW_DIR_NAME: Final[str] = "FINAL_HOLDOUT_150_BBOX_REVIEW"
IMAGE_BINDING_SCHEMA: Final[str] = "st-omr-meter-v4-4-image-binding-v1"
PROGRESS_SCHEMA: Final[str] = "st-omr-meter-v4-4-bbox-progress-v1"
COMPLETE_SCHEMA: Final[str] = "st-omr-meter-v4-4-bbox-complete-v1"

_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_BBOX_BYTES = 4096
_MAX_IMAGE_BYTES = 64 * 1024 * 1024
_MAX_IMAGE_DIMENSION = 20000
_MAX_IMAGE_PIXELS = 100_000_000
_FOLDER_RE = re.compile(r"^(?P<num>[234])_4_[0-9a-f]{12}_(?P<family>(?:aa|ab)_\d+)-")
_FIELD_RE = re.compile(r"(?<!\S)([A-Za-z0-9_]+)=([^\s]*)(?!\S)")
_INT_RE = re.compile(r"^(0|[1-9][0-9]*)$")
_REQUIRED = ("id","meter","split","bbox_x","bbox_y","bbox_w","bbox_h","admit","notes")


class MeterV4_4AnnotationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise MeterV4_4AnnotationError(message)


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        fail(f"expected regular file: {path}")
    h = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        fail(f"JSON must be a regular file: {path}")
    if not 0 < path.stat().st_size <= _MAX_JSON_BYTES:
        fail(f"JSON size outside bounds: {path}")
    try:
        value = json.loads(path.read_bytes().decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeterV4_4AnnotationError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        fail("JSON root must be an object")
    return value


def atomic_write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        fail(f"refusing to replace symlink: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=str(path.parent))
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
    raw = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    atomic_write_bytes(path, raw)


def bounded_string(name: str, value: object, max_len: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > max_len:
        fail(f"{name} must be a bounded non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class ManifestSample:
    numerator_class: str
    meter_class: str
    folder_name: str
    family_id: str


@dataclass(frozen=True, slots=True)
class SelectedSample:
    index: int
    numerator_class: str
    meter_class: str
    folder_name: str
    family_id: str
    folder_path: Path
    image_path: Path
    bbox_path: Path


@dataclass(frozen=True, slots=True)
class ImageInfo:
    sha256: str
    size_bytes: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class BBox:
    x: int
    y: int
    w: int
    h: int

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass(frozen=True, slots=True)
class BBoxContract:
    raw_text: str
    fields: dict[str, str]
    bbox: BBox | None


def load_and_validate_selection_manifest(
    manifest_path: str | Path, *, expected_selection_sha256: str | None = None
) -> tuple[dict[str, object], tuple[ManifestSample, ...]]:
    expected = EXPECTED_SELECTION_SHA256 if expected_selection_sha256 is None else expected_selection_sha256
    manifest = read_json(Path(manifest_path))
    if manifest.get("schema") != V4_3_MANIFEST_SCHEMA:
        fail("V4-3 manifest schema mismatch")
    if manifest.get("selection_sha256") != expected or manifest.get("selected_count") != 150:
        fail("V4-3 frozen selection identity mismatch")
    for key in ("bbox_annotation_complete","model_evaluated","candidate_checkpoint_opened","test_opened","runtime_connected","production_promotion_authorized"):
        if manifest.get(key) is not False:
            fail(f"V4-3 safety flag must be false: {key}")
    selected = manifest.get("selected")
    if not isinstance(selected, list) or len(selected) != 150:
        fail("V4-3 selected list must contain exactly 150 rows")
    if sha256_bytes(canonical_json(selected)) != expected:
        fail("V4-3 selected payload does not reproduce frozen selection SHA")

    rows: list[ManifestSample] = []
    families: set[str] = set()
    folders: set[str] = set()
    classes: Counter[str] = Counter()
    for raw in selected:
        if not isinstance(raw, dict):
            fail("selected row must be an object")
        num = bounded_string("numerator_class", raw.get("numerator_class"), 1)
        meter = bounded_string("meter_class", raw.get("meter_class"), 3)
        folder = bounded_string("folder_name", raw.get("folder_name"), 256)
        family = bounded_string("family_id", raw.get("family_id"), 128)
        if num not in METER_BY_NUMERATOR or meter != METER_BY_NUMERATOR[num]:
            fail("selected meter/numerator mismatch")
        if folder in {".",".."} or "/" in folder or "\\" in folder or "\x00" in folder:
            fail("folder_name contains path traversal")
        match = _FOLDER_RE.match(folder)
        if not match or match.group("num") != num or match.group("family") != family:
            fail("folder/family binding outside frozen grammar")
        if family in families or folder in folders:
            fail("selected family/folder is not unique")
        families.add(family); folders.add(folder); classes[num] += 1
        rows.append(ManifestSample(num, meter, folder, family))
    if dict(classes) != EXPECTED_CLASS_COUNTS or manifest.get("selected_classes") != EXPECTED_CLASS_COUNTS:
        fail("selected class counts must remain 50/50/50")
    return manifest, tuple(rows)


def _path_has_symlink(root: Path, path: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    cursor = root
    if cursor.is_symlink():
        return True
    for part in parts:
        cursor /= part
        if cursor.is_symlink():
            return True
    return False


def discover_selected_samples(candidate_root: str | Path, rows: tuple[ManifestSample, ...]) -> tuple[SelectedSample, ...]:
    root = Path(candidate_root)
    if not root.is_dir() or root.is_symlink():
        fail("candidate root must be an existing non-symlink directory")
    root_resolved = root.resolve()
    wanted = {row.folder_name: row for row in rows}
    found: dict[str, Path] = {}
    for folder in sorted(root.rglob("*"), key=lambda p: str(p)):
        if not folder.is_dir() or folder.is_symlink() or folder.name not in wanted:
            continue
        rel = folder.relative_to(root)
        if len(rel.parts) > 3:
            continue
        if _path_has_symlink(root, folder):
            fail("selected path contains symlink")
        resolved = folder.resolve()
        if resolved == root_resolved or root_resolved not in resolved.parents:
            fail("selected path escaped candidate root")
        if folder.name in found:
            fail("selected folder occurs more than once")
        found[folder.name] = folder
    if set(found) != set(wanted):
        fail(f"selected sample folders missing: {sorted(set(wanted)-set(found))[:5]}")
    result = []
    for index, row in enumerate(rows):
        folder = found[row.folder_name]
        image, bbox = folder/"image.png", folder/"bbox_meter.txt"
        for path in (image, bbox):
            if not path.is_file() or path.is_symlink() or _path_has_symlink(root, path):
                fail(f"selected file is not a safe regular file: {path}")
        result.append(SelectedSample(index,row.numerator_class,row.meter_class,row.folder_name,row.family_id,folder,image,bbox))
    return tuple(result)


def read_png_info(path: Path) -> ImageInfo:
    if not path.is_file() or path.is_symlink():
        fail("image must be a regular file")
    size = path.stat().st_size
    if not 8 < size <= _MAX_IMAGE_BYTES:
        fail("image size outside bounds")
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            fail("unexpected image format; PNG required")
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                fail("unexpected decoder format")
            width, height = image.size
            if width <= 0 or height <= 0 or width > _MAX_IMAGE_DIMENSION or height > _MAX_IMAGE_DIMENSION or width*height > _MAX_IMAGE_PIXELS:
                fail("image dimensions outside bounds")
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise MeterV4_4AnnotationError(f"PNG decode failed: {path}") from exc
    return ImageInfo(sha256_file(path), size, width, height)


def validate_bbox(bbox: BBox, *, image_width: int, image_height: int) -> None:
    if any(type(v) is not int for v in (bbox.x,bbox.y,bbox.w,bbox.h)):
        fail("bbox values must be integers")
    if bbox.x < 0 or bbox.y < 0 or bbox.w <= 0 or bbox.h <= 0:
        fail("bbox geometry is invalid")
    if bbox.x+bbox.w > image_width or bbox.y+bbox.h > image_height:
        fail("bbox exceeds original image bounds")


def read_bbox_contract(path: Path, *, expected_meter: str, image_width: int, image_height: int) -> BBoxContract:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= _MAX_BBOX_BYTES:
        fail("bbox_meter.txt is not a bounded regular file")
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MeterV4_4AnnotationError("bbox_meter.txt must be UTF-8") from exc
    if "\x00" in text:
        fail("bbox_meter.txt contains NUL")
    fields: dict[str,str] = {}
    for match in _FIELD_RE.finditer(text):
        key, value = match.group(1), match.group(2)
        if key in fields:
            fail(f"duplicate bbox field: {key}")
        fields[key] = value
    if not set(_REQUIRED).issubset(fields) or not fields["id"] or not fields["split"]:
        fail("bbox_meter.txt provenance/required fields missing")
    if fields["meter"] != expected_meter:
        fail("bbox_meter.txt meter mismatch")
    values = [fields[k] for k in ("bbox_x","bbox_y","bbox_w","bbox_h")]
    if all(v == "" for v in values):
        bbox = None
    elif any(v == "" for v in values):
        fail("partial bbox is forbidden")
    else:
        if any(not _INT_RE.fullmatch(v) for v in values):
            fail("bbox values must be integer pixel coordinates")
        bbox = BBox(*(int(v) for v in values))
        validate_bbox(bbox, image_width=image_width, image_height=image_height)
    return BBoxContract(text, fields, bbox)


def _replace_bbox_fields(text: str, bbox: BBox) -> str:
    for key, value in (("bbox_x",bbox.x),("bbox_y",bbox.y),("bbox_w",bbox.w),("bbox_h",bbox.h)):
        pattern = re.compile(rf"(?<!\S){key}=[^\s]*(?!\S)")
        text, count = pattern.subn(f"{key}={value}", text, count=1)
        if count != 1:
            fail(f"cannot safely replace {key}")
    return text


def write_bbox_atomic(sample: SelectedSample, bbox: BBox, *, expected_image: ImageInfo) -> None:
    if read_png_info(sample.image_path) != expected_image:
        fail("image binding changed before bbox write")
    current = read_bbox_contract(sample.bbox_path, expected_meter=sample.meter_class, image_width=expected_image.width, image_height=expected_image.height)
    validate_bbox(bbox, image_width=expected_image.width, image_height=expected_image.height)
    before = sample.bbox_path.read_bytes()
    updated = _replace_bbox_fields(current.raw_text, bbox).encode("utf-8")
    if sha256_file(sample.bbox_path) != sha256_bytes(before):
        fail("bbox_meter.txt changed concurrently")
    atomic_write_bytes(sample.bbox_path, updated)
    check = read_bbox_contract(sample.bbox_path, expected_meter=sample.meter_class, image_width=expected_image.width, image_height=expected_image.height)
    if check.bbox != bbox:
        fail("bbox atomic write verification failed")
    for key in ("id","meter","split","admit","notes"):
        if check.fields[key] != current.fields[key]:
            fail(f"bbox write mutated protected field: {key}")


def preview_rect_to_original(*, x0:int,y0:int,x1:int,y1:int,preview_width:int,preview_height:int,image_width:int,image_height:int) -> BBox:
    values=(x0,y0,x1,y1,preview_width,preview_height,image_width,image_height)
    if any(type(v) is not int for v in values):
        fail("preview coordinates/dimensions must be integers")
    if min(preview_width,preview_height,image_width,image_height) <= 0:
        fail("preview/image dimensions must be positive")
    if not (0 <= x0 <= preview_width and 0 <= x1 <= preview_width and 0 <= y0 <= preview_height and 0 <= y1 <= preview_height):
        fail("preview coordinate outside bounds")
    left,right=sorted((x0,x1)); top,bottom=sorted((y0,y1))
    if left==right or top==bottom:
        fail("drawn bbox must have positive area")
    ox0=(left*image_width)//preview_width
    oy0=(top*image_height)//preview_height
    ox1=min(image_width,(right*image_width+preview_width-1)//preview_width)
    oy1=min(image_height,(bottom*image_height+preview_height-1)//preview_height)
    bbox=BBox(ox0,oy0,ox1-ox0,oy1-oy0)
    validate_bbox(bbox,image_width=image_width,image_height=image_height)
    return bbox
