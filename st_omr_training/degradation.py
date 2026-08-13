"""Deterministic, bounded rasterization and controlled degradation for ST-OMR V1.

Stage 4 consumes only validated, self-contained SVG page bytes emitted by the
Stage 3 renderer boundary. It creates a clean grayscale PNG raster and an
appearance-degraded derivative while preserving symbolic lineage and family
identity. Random choices are derived deterministically from an explicit seed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256, shake_256
from importlib import metadata
from io import BytesIO
import json
import platform
import re
import xml.etree.ElementTree as ET
from typing import Final


DEGRADATION_VERSION: Final[str] = "st-controlled-degradation-v1"
CAIROSVG_PINNED_VERSION: Final[str] = "2.8.2"
PILLOW_PINNED_VERSION: Final[str] = "12.3.0"
MAX_SOURCE_SVG_BYTES: Final[int] = 16 * 1024 * 1024
MAX_OUTPUT_PIXELS: Final[int] = 16_000_000
MIN_RASTER_WIDTH: Final[int] = 512
MAX_RASTER_WIDTH: Final[int] = 2400
MAX_ROTATION_MDEG: Final[int] = 3000
MAX_BLUR_MILLI: Final[int] = 2000
MAX_NOISE_LEVEL: Final[int] = 20

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_FAMILY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class DegradationInputError(ValueError):
    """Raised when a Stage 4 source violates the frozen input contract."""


class DegradationRuntimeError(RuntimeError):
    """Raised when the pinned raster/degradation runtime is unavailable."""


class DegradationExecutionError(RuntimeError):
    """Raised when rasterization/degradation fails closed."""


def _require_hex64(name: str, value: object) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise DegradationInputError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True, slots=True)
class DegradationSource:
    """Validated lineage inputs for one clean Stage 3 SVG page."""

    family_id: str
    page_number: int
    source_musicxml_sha256: str
    renderer_config_fingerprint: str
    svg: bytes
    svg_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.family_id, str) or _FAMILY_RE.fullmatch(self.family_id) is None:
            raise DegradationInputError("family_id must match the bounded family identifier contract")
        if not isinstance(self.page_number, int) or isinstance(self.page_number, bool) or self.page_number < 1:
            raise DegradationInputError("page_number must be a positive integer")
        _require_hex64("source_musicxml_sha256", self.source_musicxml_sha256)
        _require_hex64("renderer_config_fingerprint", self.renderer_config_fingerprint)
        _require_hex64("svg_sha256", self.svg_sha256)
        if not isinstance(self.svg, bytes) or not self.svg:
            raise DegradationInputError("svg must be non-empty bytes")
        if len(self.svg) > MAX_SOURCE_SVG_BYTES:
            raise DegradationInputError("svg exceeds the Stage 4 input byte limit")
        if sha256(self.svg).hexdigest() != self.svg_sha256:
            raise DegradationInputError("svg_sha256 does not match the supplied SVG bytes")
        _validate_svg_surface(self.svg)


@dataclass(frozen=True, slots=True)
class DegradationConfig:
    """Explicit deterministic Stage 4 V1 image transformation parameters."""

    seed: int = 0
    raster_width: int = 1400
    rotation_mdeg: int = 0
    blur_milli: int = 0
    noise_level: int = 0
    brightness_milli: int = 1000
    contrast_milli: int = 1000
    jpeg_quality: int = 0

    def __post_init__(self) -> None:
        integer_fields = {
            "seed": self.seed,
            "raster_width": self.raster_width,
            "rotation_mdeg": self.rotation_mdeg,
            "blur_milli": self.blur_milli,
            "noise_level": self.noise_level,
            "brightness_milli": self.brightness_milli,
            "contrast_milli": self.contrast_milli,
            "jpeg_quality": self.jpeg_quality,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if not 0 <= self.seed <= (2**63 - 1):
            raise ValueError("seed must be between 0 and 2^63-1")
        if not MIN_RASTER_WIDTH <= self.raster_width <= MAX_RASTER_WIDTH:
            raise ValueError(f"raster_width must be between {MIN_RASTER_WIDTH} and {MAX_RASTER_WIDTH}")
        if not -MAX_ROTATION_MDEG <= self.rotation_mdeg <= MAX_ROTATION_MDEG:
            raise ValueError(f"rotation_mdeg must be between {-MAX_ROTATION_MDEG} and {MAX_ROTATION_MDEG}")
        if not 0 <= self.blur_milli <= MAX_BLUR_MILLI:
            raise ValueError(f"blur_milli must be between 0 and {MAX_BLUR_MILLI}")
        if not 0 <= self.noise_level <= MAX_NOISE_LEVEL:
            raise ValueError(f"noise_level must be between 0 and {MAX_NOISE_LEVEL}")
        if not 800 <= self.brightness_milli <= 1200:
            raise ValueError("brightness_milli must be between 800 and 1200")
        if not 750 <= self.contrast_milli <= 1250:
            raise ValueError("contrast_milli must be between 750 and 1250")
        if self.jpeg_quality != 0 and not 65 <= self.jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be 0 (disabled) or between 65 and 95")


@dataclass(frozen=True, slots=True)
class DegradedPage:
    family_id: str
    page_number: int
    source_musicxml_sha256: str
    renderer_config_fingerprint: str
    source_svg_sha256: str
    clean_raster_sha256: str
    degradation_config_fingerprint: str
    config: DegradationConfig
    derivative_id: str
    png_sha256: str
    degradation_version: str
    cairosvg_version: str
    pillow_version: str
    cairo_runtime_version: str
    python_version: str
    platform_system: str
    platform_machine: str
    clean_width: int
    clean_height: int
    width: int
    height: int
    mode: str
    png: bytes


def _contains_external_url(value: str) -> bool:
    for match in re.finditer(r"url\(([^)]*)\)", value, flags=re.IGNORECASE):
        target = match.group(1).strip().strip("\"\'")
        if not target.startswith("#"):
            return True
    return False


def _validate_svg_surface(data: bytes) -> None:
    lowered = data.lower()
    if b"<!doctype" in lowered:
        raise DegradationInputError("DOCTYPE is forbidden at the Stage 4 SVG boundary")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise DegradationInputError("svg is malformed") from exc
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise DegradationInputError("Stage 4 input root must be SVG")
    view_box = root.attrib.get("viewBox")
    if not isinstance(view_box, str):
        raise DegradationInputError("Stage 4 SVG must carry an explicit viewBox")
    parts = view_box.replace(",", " ").split()
    if len(parts) != 4:
        raise DegradationInputError("SVG viewBox must contain four numeric values")
    try:
        x0, y0, vb_width, vb_height = (float(part) for part in parts)
    except ValueError as exc:
        raise DegradationInputError("SVG viewBox must be numeric") from exc
    import math
    if not all(math.isfinite(value) for value in (x0, y0, vb_width, vb_height)):
        raise DegradationInputError("SVG viewBox values must be finite")
    if vb_width <= 0 or vb_height <= 0 or vb_width > 100_000 or vb_height > 100_000:
        raise DegradationInputError("SVG viewBox dimensions are outside Stage 4 bounds")
    aspect = vb_width / vb_height
    if not 0.2 <= aspect <= 5.0:
        raise DegradationInputError("SVG viewBox aspect ratio is outside Stage 4 bounds")

    forbidden = {"script", "foreignObject", "iframe", "object", "embed", "image", "audio", "video"}
    element_count = 0
    for element in root.iter():
        element_count += 1
        if element_count > 500_000:
            raise DegradationInputError("svg exceeds the Stage 4 element-count limit")
        local_name = element.tag.rsplit("}", 1)[-1]
        if local_name in forbidden:
            raise DegradationInputError("active or externally loaded SVG content is forbidden")
        if local_name == "style" and isinstance(element.text, str):
            text = element.text
            if "@import" in text.lower() or _contains_external_url(text):
                raise DegradationInputError("external SVG stylesheet/resource reference is forbidden")
        for attr_name, attr_value in element.attrib.items():
            local_attr = attr_name.rsplit("}", 1)[-1]
            value = attr_value.strip()
            if local_attr == "href" and not value.startswith("#"):
                raise DegradationInputError("external SVG href is forbidden")
            if _contains_external_url(value):
                raise DegradationInputError("external SVG URL reference is forbidden")


def _load_image_runtime():
    try:
        import cairosvg  # type: ignore
        import cairocffi  # type: ignore
        from PIL import Image, ImageChops, ImageEnhance, ImageFilter  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure environment-specific
        raise DegradationRuntimeError("Stage 4 raster/degradation dependencies are unavailable") from exc

    try:
        cairosvg_version = metadata.version("CairoSVG")
        pillow_version = metadata.version("Pillow")
    except metadata.PackageNotFoundError as exc:  # pragma: no cover
        raise DegradationRuntimeError("Stage 4 dependency metadata is unavailable") from exc
    if cairosvg_version != CAIROSVG_PINNED_VERSION:
        raise DegradationRuntimeError(
            f"expected CairoSVG=={CAIROSVG_PINNED_VERSION}, got {cairosvg_version}"
        )
    if pillow_version != PILLOW_PINNED_VERSION:
        raise DegradationRuntimeError(
            f"expected Pillow=={PILLOW_PINNED_VERSION}, got {pillow_version}"
        )
    try:
        cairo_runtime_version = str(cairocffi.cairo_version_string())
    except Exception as exc:  # pragma: no cover
        raise DegradationRuntimeError("Cairo runtime version is unavailable") from exc
    if not cairo_runtime_version:
        raise DegradationRuntimeError("Cairo runtime version is empty")
    return (
        cairosvg, Image, ImageChops, ImageEnhance, ImageFilter,
        cairosvg_version, pillow_version, cairo_runtime_version,
    )


def degradation_config_fingerprint(config: DegradationConfig) -> str:
    if not isinstance(config, DegradationConfig):
        raise TypeError("config must be DegradationConfig")
    payload = {
        "degradation_version": DEGRADATION_VERSION,
        "cairosvg_version": CAIROSVG_PINNED_VERSION,
        "pillow_version": PILLOW_PINNED_VERSION,
        "config": asdict(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return sha256(encoded).hexdigest()


def _map_u16(raw: bytes, offset: int, minimum: int, maximum: int) -> int:
    value = int.from_bytes(raw[offset:offset + 2], "big")
    return minimum + (value * (maximum - minimum + 1) // 65536)


def sample_degradation_config(seed: int, profile: str = "medium", *, raster_width: int = 1400) -> DegradationConfig:
    """Derive replayable V1 parameters from a seed without global RNG state."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= (2**63 - 1):
        raise ValueError("seed must be between 0 and 2^63-1")
    if profile not in {"clean", "light", "medium"}:
        raise ValueError("profile must be clean, light, or medium")
    if not isinstance(raster_width, int) or isinstance(raster_width, bool):
        raise TypeError("raster_width must be an integer")
    if not MIN_RASTER_WIDTH <= raster_width <= MAX_RASTER_WIDTH:
        raise ValueError(f"raster_width must be between {MIN_RASTER_WIDTH} and {MAX_RASTER_WIDTH}")
    if profile == "clean":
        return DegradationConfig(seed=seed, raster_width=raster_width)

    raw = sha256(f"{DEGRADATION_VERSION}|{profile}|{seed}".encode("ascii")).digest()
    if profile == "light":
        bounds = {
            "rotation": (-1000, 1000),
            "blur": (0, 800),
            "noise": (0, 6),
            "brightness": (950, 1050),
            "contrast": (950, 1050),
            "jpeg": (88, 95),
        }
    else:
        bounds = {
            "rotation": (-2500, 2500),
            "blur": (200, 1500),
            "noise": (2, 12),
            "brightness": (880, 1120),
            "contrast": (850, 1150),
            "jpeg": (72, 92),
        }
    return DegradationConfig(
        seed=seed,
        raster_width=raster_width,
        rotation_mdeg=_map_u16(raw, 0, *bounds["rotation"]),
        blur_milli=_map_u16(raw, 2, *bounds["blur"]),
        noise_level=_map_u16(raw, 4, *bounds["noise"]),
        brightness_milli=_map_u16(raw, 6, *bounds["brightness"]),
        contrast_milli=_map_u16(raw, 8, *bounds["contrast"]),
        jpeg_quality=_map_u16(raw, 10, *bounds["jpeg"]),
    )


def _encode_png(image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _check_image_bounds(image, label: str) -> None:
    width, height = image.size
    if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
        raise DegradationExecutionError(f"{label} has invalid dimensions")
    if width * height > MAX_OUTPUT_PIXELS:
        raise DegradationExecutionError(f"{label} exceeds the Stage 4 pixel limit")
    histogram = image.histogram()
    total = width * height
    ink = sum(histogram[:245])
    light_pixels = sum(histogram[180:])
    if ink < max(16, total // 200_000):
        raise DegradationExecutionError(f"{label} is effectively blank")
    if light_pixels < total // 10:
        raise DegradationExecutionError(f"{label} is implausibly dark")


def _ink_count(image) -> int:
    return sum(image.histogram()[:240])


def _apply_uniform_noise(image, Image, ImageChops, *, level: int, key: bytes):
    if level == 0:
        return image
    byte_count = image.size[0] * image.size[1]
    raw = shake_256(key).digest(byte_count)
    # Map each byte deterministically into [-level, +level], encoded around 128.
    lut = [128 + ((value * (2 * level + 1)) // 256) - level for value in range(256)]
    noise = Image.frombytes("L", image.size, raw).point(lut)
    return ImageChops.add(image, noise, scale=1.0, offset=-128)


def degrade_page(source: DegradationSource, config: DegradationConfig) -> DegradedPage:
    """Rasterize one validated SVG page and produce a deterministic V1 derivative."""

    if not isinstance(source, DegradationSource):
        raise TypeError("source must be DegradationSource")
    if not isinstance(config, DegradationConfig):
        raise TypeError("config must be DegradationConfig")

    (
        cairosvg, Image, ImageChops, ImageEnhance, ImageFilter,
        cairosvg_version, pillow_version, cairo_runtime_version,
    ) = _load_image_runtime()
    try:
        raster_png = cairosvg.svg2png(
            bytestring=source.svg,
            output_width=config.raster_width,
            background_color="#ffffff",
            unsafe=False,
        )
        if not isinstance(raster_png, bytes) or not raster_png:
            raise DegradationExecutionError("CairoSVG returned empty/non-byte PNG output")
        with Image.open(BytesIO(raster_png)) as opened:
            opened.load()
            clean = opened.convert("L")
    except DegradationExecutionError:
        raise
    except Exception as exc:
        raise DegradationExecutionError(f"SVG rasterization failed: {type(exc).__name__}") from exc

    _check_image_bounds(clean, "clean raster")
    clean_width, clean_height = clean.size
    clean_png = _encode_png(clean)
    clean_hash = sha256(clean_png).hexdigest()
    config_hash = degradation_config_fingerprint(config)

    image = clean
    clean_ink = _ink_count(clean)
    if config.rotation_mdeg:
        image = image.rotate(
            config.rotation_mdeg / 1000.0,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=255,
        )
        _check_image_bounds(image, "geometric derivative")
        rotated_ink = _ink_count(image)
        if clean_ink <= 0 or not (0.65 * clean_ink <= rotated_ink <= 1.45 * clean_ink):
            raise DegradationExecutionError("geometric transform failed the ink-retention gate")

    if config.blur_milli:
        image = image.filter(ImageFilter.GaussianBlur(radius=config.blur_milli / 1000.0))
    if config.contrast_milli != 1000:
        image = ImageEnhance.Contrast(image).enhance(config.contrast_milli / 1000.0)
    if config.brightness_milli != 1000:
        image = ImageEnhance.Brightness(image).enhance(config.brightness_milli / 1000.0)
    if config.noise_level:
        noise_key = (
            f"{DEGRADATION_VERSION}|{source.svg_sha256}|{config_hash}|{config.seed}".encode("ascii")
        )
        image = _apply_uniform_noise(image, Image, ImageChops, level=config.noise_level, key=noise_key)
    if config.jpeg_quality:
        jpeg_buffer = BytesIO()
        image.save(
            jpeg_buffer,
            format="JPEG",
            quality=config.jpeg_quality,
            optimize=False,
            progressive=False,
        )
        try:
            with Image.open(BytesIO(jpeg_buffer.getvalue())) as reopened:
                reopened.load()
                image = reopened.convert("L")
        except Exception as exc:
            raise DegradationExecutionError("JPEG round-trip failed") from exc

    _check_image_bounds(image, "degraded raster")
    output_png = _encode_png(image)
    output_hash = sha256(output_png).hexdigest()
    lineage_payload = {
        "degradation_version": DEGRADATION_VERSION,
        "family_id": source.family_id,
        "page_number": source.page_number,
        "source_musicxml_sha256": source.source_musicxml_sha256,
        "renderer_config_fingerprint": source.renderer_config_fingerprint,
        "source_svg_sha256": source.svg_sha256,
        "clean_raster_sha256": clean_hash,
        "degradation_config_fingerprint": config_hash,
        "png_sha256": output_hash,
    }
    derivative_id = sha256(
        json.dumps(lineage_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()

    return DegradedPage(
        family_id=source.family_id,
        page_number=source.page_number,
        source_musicxml_sha256=source.source_musicxml_sha256,
        renderer_config_fingerprint=source.renderer_config_fingerprint,
        source_svg_sha256=source.svg_sha256,
        clean_raster_sha256=clean_hash,
        degradation_config_fingerprint=config_hash,
        config=config,
        derivative_id=derivative_id,
        png_sha256=output_hash,
        degradation_version=DEGRADATION_VERSION,
        cairosvg_version=cairosvg_version,
        pillow_version=pillow_version,
        cairo_runtime_version=cairo_runtime_version,
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        clean_width=clean_width,
        clean_height=clean_height,
        width=image.size[0],
        height=image.size[1],
        mode=image.mode,
        png=output_png,
    )


def source_from_render_result(render_result: object, *, family_id: str, page_number: int = 1) -> DegradationSource:
    """Build a Stage 4 source from a Stage 3 RenderResult-like object.

    The helper deliberately validates the copied hashes instead of trusting object
    provenance claims. It avoids a hard renderer import so Stage 4 remains a
    separate layer.
    """

    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise DegradationInputError("page_number must be a positive integer")
    try:
        musicxml_hash = render_result.source_musicxml_sha256
        renderer_fingerprint = render_result.config_fingerprint
        pages = render_result.pages
    except AttributeError as exc:
        raise DegradationInputError("render_result is missing required Stage 3 provenance") from exc
    if not isinstance(pages, tuple):
        raise DegradationInputError("render_result.pages must be an immutable tuple")
    matches = [page for page in pages if getattr(page, "page_number", None) == page_number]
    if len(matches) != 1:
        raise DegradationInputError("requested page_number is not uniquely present in render_result")
    page = matches[0]
    try:
        svg = page.svg
        svg_hash = page.sha256
    except AttributeError as exc:
        raise DegradationInputError("rendered page is missing SVG bytes or hash") from exc
    return DegradationSource(
        family_id=family_id,
        page_number=page_number,
        source_musicxml_sha256=musicxml_hash,
        renderer_config_fingerprint=renderer_fingerprint,
        svg=svg,
        svg_sha256=svg_hash,
    )


def degrade_render_result_page(
    render_result: object,
    *,
    family_id: str,
    page_number: int = 1,
    config: DegradationConfig | None = None,
) -> DegradedPage:
    """Convenience composition of the Stage 3 result and Stage 4 page boundary."""

    effective_config = DegradationConfig() if config is None else config
    if not isinstance(effective_config, DegradationConfig):
        raise TypeError("config must be DegradationConfig")
    return degrade_page(
        source_from_render_result(render_result, family_id=family_id, page_number=page_number),
        effective_config,
    )
