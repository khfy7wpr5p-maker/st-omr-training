"""Deterministic, fail-closed Verovio renderer adapter for ST-OMR V1.

Stage 3 keeps rendering behind an adapter boundary. Only Stage-2-C-valid,
canonical ST-OMR V1 MusicXML is accepted. Verovio is imported lazily and its
Python package version is pinned so renderer drift cannot be accepted silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
from typing import Final
import json
import xml.etree.ElementTree as ET

from .musicxml_validator import validate_musicxml
from .musicxml_writer import musicxml_sha256
from .validator import ValidationResult


VEROVIO_PINNED_VERSION: Final[str] = "6.2.1"
RENDERER_ADAPTER_VERSION: Final[str] = "st-verovio-renderer-v1"
RENDERER_NAME: Final[str] = "verovio"
MAX_RENDER_PAGES: Final[int] = 64


class RendererUnavailableError(RuntimeError):
    """Raised when the pinned Verovio runtime is unavailable or mismatched."""


class RenderInputError(ValueError):
    """Raised when input MusicXML cannot enter the renderer boundary."""

    def __init__(self, message: str, validation: ValidationResult) -> None:
        super().__init__(message)
        self.validation = validation


class RenderExecutionError(RuntimeError):
    """Raised when the renderer fails closed during configuration or rendering."""


@dataclass(frozen=True, slots=True)
class RendererConfig:
    """Frozen V1 renderer configuration with explicit deterministic options."""

    page_height: int = 2970
    page_width: int = 2100
    page_margin_top: int = 50
    page_margin_right: int = 50
    page_margin_bottom: int = 50
    page_margin_left: int = 50
    scale: int = 100
    breaks: str = "auto"
    font: str = "Leipzig"

    def __post_init__(self) -> None:
        integer_fields = {
            "page_height": (self.page_height, 100, 60000),
            "page_width": (self.page_width, 100, 100000),
            "page_margin_top": (self.page_margin_top, 0, 500),
            "page_margin_right": (self.page_margin_right, 0, 500),
            "page_margin_bottom": (self.page_margin_bottom, 0, 500),
            "page_margin_left": (self.page_margin_left, 0, 500),
            "scale": (self.scale, 1, 1000),
        }
        for name, (value, minimum, maximum) in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
        if self.breaks not in {"auto", "none", "line", "smart", "encoded"}:
            raise ValueError("breaks must be a supported Verovio break mode")
        if self.font != "Leipzig":
            raise ValueError("Stage 3 V1 pins the Leipzig music font")

    def verovio_options(self) -> dict[str, object]:
        return {
            "pageHeight": self.page_height,
            "pageWidth": self.page_width,
            "pageMarginTop": self.page_margin_top,
            "pageMarginRight": self.page_margin_right,
            "pageMarginBottom": self.page_margin_bottom,
            "pageMarginLeft": self.page_margin_left,
            "scale": self.scale,
            "breaks": self.breaks,
            "font": self.font,
            "fontFallback": "Leipzig",
            "smuflTextFont": "embedded",
            "xmlIdChecksum": True,
            "svgFormatRaw": True,
            "svgViewBox": True,
            "svgHtml5": False,
            "svgRemoveXlink": True,
            "adjustPageHeight": False,
            "adjustPageWidth": False,
            "landscape": False,
            "mmOutput": False,
            "scaleToPageSize": False,
        }


@dataclass(frozen=True, slots=True)
class RenderedPage:
    page_number: int
    svg: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class RenderResult:
    source_musicxml_sha256: str
    renderer_name: str
    renderer_package_version: str
    renderer_runtime_version: str
    adapter_version: str
    config_fingerprint: str
    pages: tuple[RenderedPage, ...]


def renderer_config_fingerprint(config: RendererConfig) -> str:
    if not isinstance(config, RendererConfig):
        raise TypeError("config must be RendererConfig")
    payload = {
        "adapter_version": RENDERER_ADAPTER_VERSION,
        "renderer": RENDERER_NAME,
        "renderer_package_version": VEROVIO_PINNED_VERSION,
        "options": config.verovio_options(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return sha256(encoded).hexdigest()


def _load_verovio_runtime():
    try:
        import verovio  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised when dependency is absent
        raise RendererUnavailableError(
            f"verovio=={VEROVIO_PINNED_VERSION} is required for Stage 3 rendering"
        ) from exc
    try:
        package_version = metadata.version("verovio")
    except metadata.PackageNotFoundError as exc:  # pragma: no cover
        raise RendererUnavailableError("Verovio package metadata is unavailable") from exc
    if package_version != VEROVIO_PINNED_VERSION:
        raise RendererUnavailableError(
            f"expected verovio=={VEROVIO_PINNED_VERSION}, got {package_version}"
        )
    return verovio, package_version


def _validate_svg(svg: object, page_number: int) -> bytes:
    if not isinstance(svg, str) or not svg.strip():
        raise RenderExecutionError(f"Verovio returned empty/non-text SVG for page {page_number}")
    try:
        data = svg.encode("utf-8", errors="strict")
        root = ET.fromstring(data)
    except (UnicodeError, ET.ParseError) as exc:
        raise RenderExecutionError(f"Verovio returned malformed SVG for page {page_number}") from exc

    local_name = root.tag.rsplit("}", 1)[-1]
    if local_name != "svg":
        raise RenderExecutionError(f"renderer output page {page_number} is not SVG")

    forbidden = {"script", "foreignObject", "iframe", "object", "embed", "image"}
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] in forbidden:
            raise RenderExecutionError(f"forbidden active SVG element on page {page_number}")
        if element.tag.rsplit("}", 1)[-1] == "style" and isinstance(element.text, str):
            lowered = element.text.lower().replace(" ", "")
            if "@import" in lowered or "url(http:" in lowered or "url(https:" in lowered or "url(file:" in lowered:
                raise RenderExecutionError(f"external SVG stylesheet reference rejected on page {page_number}")
        for attr_name, attr_value in element.attrib.items():
            if attr_name.rsplit("}", 1)[-1] == "href" and not attr_value.startswith("#"):
                raise RenderExecutionError(f"external SVG reference rejected on page {page_number}")
    return data


def render_musicxml_svg(data: object, config: RendererConfig | None = None) -> RenderResult:
    """Validate canonical V1 MusicXML and render all pages to self-contained SVG.

    Determinism is scoped to the same pinned Verovio package/runtime, renderer
    configuration, platform/resource bundle, and MusicXML bytes. Page hashes are
    derived artifact hashes and do not replace canonical musical identity.
    """

    validation = validate_musicxml(data)
    if not validation.is_valid:
        raise RenderInputError("MusicXML failed Stage 2-C validation", validation)
    assert isinstance(data, bytes)

    render_config = RendererConfig() if config is None else config
    if not isinstance(render_config, RendererConfig):
        raise TypeError("config must be RendererConfig")

    verovio, package_version = _load_verovio_runtime()
    try:
        toolkit = verovio.toolkit()
        runtime_version = str(toolkit.getVersion())
        if not runtime_version.startswith(VEROVIO_PINNED_VERSION):
            raise RendererUnavailableError(
                f"Verovio runtime version mismatch: {runtime_version!r}"
            )
        if toolkit.setInputFrom("xml") is False:
            raise RenderExecutionError("Verovio rejected explicit MusicXML input mode")
        if toolkit.setOptions(render_config.verovio_options()) is False:
            raise RenderExecutionError("Verovio rejected the pinned renderer options")
        try:
            xml_text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RenderInputError("validated MusicXML is not UTF-8", validation) from exc
        if toolkit.loadData(xml_text) is False:
            raise RenderExecutionError("Verovio rejected validated MusicXML input")
        page_count = toolkit.getPageCount()
    except (RendererUnavailableError, RenderInputError, RenderExecutionError):
        raise
    except Exception as exc:
        raise RenderExecutionError(f"Verovio setup/load failed: {type(exc).__name__}") from exc

    if not isinstance(page_count, int) or isinstance(page_count, bool):
        raise RenderExecutionError("Verovio page count is not an integer")
    if not 1 <= page_count <= MAX_RENDER_PAGES:
        raise RenderExecutionError(
            f"Verovio page count must be between 1 and {MAX_RENDER_PAGES}, got {page_count}"
        )

    pages: list[RenderedPage] = []
    for page_number in range(1, page_count + 1):
        try:
            svg_text = toolkit.renderToSVG(page_number, True)
        except Exception as exc:
            raise RenderExecutionError(
                f"Verovio failed while rendering page {page_number}: {type(exc).__name__}"
            ) from exc
        svg_bytes = _validate_svg(svg_text, page_number)
        pages.append(
            RenderedPage(
                page_number=page_number,
                svg=svg_bytes,
                sha256=sha256(svg_bytes).hexdigest(),
            )
        )

    return RenderResult(
        source_musicxml_sha256=musicxml_sha256(data),
        renderer_name=RENDERER_NAME,
        renderer_package_version=package_version,
        renderer_runtime_version=runtime_version,
        adapter_version=RENDERER_ADAPTER_VERSION,
        config_fingerprint=renderer_config_fingerprint(render_config),
        pages=tuple(pages),
    )
