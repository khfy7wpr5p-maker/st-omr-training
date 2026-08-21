"""Google Colab mouse UI for Meter V4-4 full-meter bbox annotation."""

from __future__ import annotations

import json

from st_omr_training.meter_v4_4_final_holdout_bbox_annotation import (
    AnnotationSession,
    MeterV4_4AnnotationError,
)


_CALLBACK_GET = "st_omr_meter_v44.get"
_CALLBACK_SAVE = "st_omr_meter_v44.save"
_CALLBACK_FLAG = "st_omr_meter_v44.flag"


def _require_dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MeterV4_4AnnotationError(f"{name} payload must be an object")
    return value


def launch_colab_annotation(*, candidate_root: str, manifest_path: str) -> AnnotationSession:
    """Launch the bounded browser canvas and return the live session object.

    The browser never writes files directly. All writes pass through Python
    callbacks that revalidate the frozen sample binding token, original image
    dimensions, image SHA, bbox contract and target path.
    """
    try:
        from google.colab import output
        from IPython.display import HTML, display
    except ImportError as exc:
        raise MeterV4_4AnnotationError("V4-4 mouse UI requires Google Colab") from exc

    session = AnnotationSession(candidate_root=candidate_root, manifest_path=manifest_path)

    def get_callback(index: object) -> dict[str, object]:
        if type(index) is not int:
            raise MeterV4_4AnnotationError("sample index must be integer")
        return session.sample_payload(index)

    def save_callback(payload: object) -> dict[str, object]:
        body = _require_dict(payload, "save")
        required = ("token", "x0", "y0", "x1", "y1", "preview_width", "preview_height")
        if not all(key in body for key in required):
            raise MeterV4_4AnnotationError("save payload is incomplete")
        return session.save_from_preview(
            token=body["token"],
            x0=body["x0"],
            y0=body["y0"],
            x1=body["x1"],
            y1=body["y1"],
            preview_width=body["preview_width"],
            preview_height=body["preview_height"],
        )

    def flag_callback(payload: object) -> dict[str, object]:
        body = _require_dict(payload, "review flag")
        if set(body) != {"token", "flagged"}:
            raise MeterV4_4AnnotationError("review flag payload malformed")
        return session.set_review_flag(
            token=body["token"],
            flagged=body["flagged"],
        )

    output.register_callback(_CALLBACK_GET, get_callback)
    output.register_callback(_CALLBACK_SAVE, save_callback)
    output.register_callback(_CALLBACK_FLAG, flag_callback)

    initial = session.resume_index()
    html = f"""
<div id="v44-root" style="font-family:Arial,sans-serif;max-width:1100px">
  <h3>ST-OMR Meter V4-4 — Final Holdout BBox Annotation</h3>
  <div style="padding:10px;border:1px solid #bbb;margin-bottom:10px">
    <b>Kural:</b> BBox yalnız üst rakamı değil, <b>tam meter işaretini</b> kapsamalı:
    üst rakam + alt rakam birlikte. SAVE yapılmadan dosyaya yazılmaz.
  </div>
  <div id="v44-status" style="margin:6px 0;font-weight:bold"></div>
  <div id="v44-meta" style="margin:6px 0"></div>
  <div style="overflow:auto;border:1px solid #bbb;display:inline-block;max-width:100%">
    <canvas id="v44-canvas" style="display:block;max-width:100%;touch-action:none;cursor:crosshair"></canvas>
  </div>
  <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
    <button id="v44-prev">Previous</button>
    <button id="v44-save"><b>SAVE BBOX</b></button>
    <button id="v44-review">Flag review / skip</button>
    <button id="v44-next">Next</button>
  </div>
  <div id="v44-message" style="margin-top:8px;white-space:pre-wrap"></div>
</div>
<script>
(() => {{
  const CB_GET = {json.dumps(_CALLBACK_GET)};
  const CB_SAVE = {json.dumps(_CALLBACK_SAVE)};
  const CB_FLAG = {json.dumps(_CALLBACK_FLAG)};
  const canvas = document.getElementById('v44-canvas');
  const ctx = canvas.getContext('2d');
  const statusEl = document.getElementById('v44-status');
  const metaEl = document.getElementById('v44-meta');
  const messageEl = document.getElementById('v44-message');
  const reviewBtn = document.getElementById('v44-review');
  const img = new Image();
  let sample = null;
  let rect = null;
  let dragging = false;
  let start = null;

  function unwrap(result) {{
    if (!result || !result.data) throw new Error('Colab callback returned no data');
    if (result.data['application/json'] !== undefined) return result.data['application/json'];
    throw new Error('Colab callback did not return JSON');
  }}

  async function invoke(name, args) {{
    const result = await google.colab.kernel.invokeFunction(name, args, {{}});
    return unwrap(result);
  }}

  function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}

  function pointer(ev) {{
    const r = canvas.getBoundingClientRect();
    const x = Math.round((ev.clientX - r.left) * canvas.width / r.width);
    const y = Math.round((ev.clientY - r.top) * canvas.height / r.height);
    return {{x: clamp(x, 0, canvas.width), y: clamp(y, 0, canvas.height)}};
  }}

  function existingRect(s) {{
    if (!s.bbox) return null;
    const x0 = Math.floor(s.bbox.x * s.preview_width / s.image_width);
    const y0 = Math.floor(s.bbox.y * s.preview_height / s.image_height);
    const x1 = Math.ceil((s.bbox.x + s.bbox.w) * s.preview_width / s.image_width);
    const y1 = Math.ceil((s.bbox.y + s.bbox.h) * s.preview_height / s.image_height);
    return {{x0, y0, x1, y1}};
  }}

  function draw() {{
    if (!sample || !img.complete) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    if (rect) {{
      const left = Math.min(rect.x0, rect.x1);
      const top = Math.min(rect.y0, rect.y1);
      const w = Math.abs(rect.x1 - rect.x0);
      const h = Math.abs(rect.y1 - rect.y0);
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#ff0000';
      ctx.strokeRect(left, top, w, h);
    }}
  }}

  async function loadIndex(index) {{
    messageEl.textContent = 'Loading...';
    try {{
      sample = await invoke(CB_GET, [index]);
      canvas.width = sample.preview_width;
      canvas.height = sample.preview_height;
      rect = existingRect(sample);
      statusEl.textContent = `Tamamlanan: ${{sample.annotated_count}} / ${{sample.total}} — Örnek ${{sample.index + 1}} / ${{sample.total}}`;
      metaEl.textContent = `Meter: ${{sample.meter_class}} | Family: ${{sample.family_id}} | Original: ${{sample.image_width}}×${{sample.image_height}} px`;
      reviewBtn.textContent = sample.review_flag ? 'Unflag review' : 'Flag review / skip';
      img.onload = () => {{ draw(); messageEl.textContent = sample.bbox ? 'Mevcut bbox gösteriliyor; yeniden çizip SAVE ile düzeltebilirsin.' : 'Mouse ile tam meter işaretini çerçevele.'; }};
      img.src = sample.preview_data_uri;
    }} catch (err) {{
      messageEl.textContent = 'BLOCKED: ' + err;
      throw err;
    }}
  }}

  canvas.addEventListener('pointerdown', (ev) => {{
    if (!sample) return;
    dragging = true;
    start = pointer(ev);
    rect = {{x0:start.x, y0:start.y, x1:start.x, y1:start.y}};
    canvas.setPointerCapture(ev.pointerId);
    draw();
  }});
  canvas.addEventListener('pointermove', (ev) => {{
    if (!dragging || !rect) return;
    const p = pointer(ev);
    rect.x1 = p.x; rect.y1 = p.y;
    draw();
  }});
  canvas.addEventListener('pointerup', (ev) => {{
    if (!dragging || !rect) return;
    const p = pointer(ev);
    rect.x1 = p.x; rect.y1 = p.y;
    dragging = false;
    draw();
  }});

  document.getElementById('v44-prev').onclick = async () => {{
    if (!sample) return;
    await loadIndex(Math.max(0, sample.index - 1));
  }};
  document.getElementById('v44-next').onclick = async () => {{
    if (!sample) return;
    await loadIndex(Math.min(sample.total - 1, sample.index + 1));
  }};
  document.getElementById('v44-save').onclick = async () => {{
    if (!sample || !rect) {{
      messageEl.textContent = 'Önce bbox çiz.';
      return;
    }}
    messageEl.textContent = 'Saving...';
    try {{
      const result = await invoke(CB_SAVE, [{{
        token: sample.binding_token,
        x0: Math.round(rect.x0), y0: Math.round(rect.y0),
        x1: Math.round(rect.x1), y1: Math.round(rect.y1),
        preview_width: sample.preview_width,
        preview_height: sample.preview_height
      }}]);
      messageEl.textContent = `Kaydedildi. Original bbox: x=${{result.bbox.x}}, y=${{result.bbox.y}}, w=${{result.bbox.w}}, h=${{result.bbox.h}}`;
      const next = Math.min(sample.total - 1, sample.index + 1);
      await loadIndex(next);
    }} catch (err) {{
      messageEl.textContent = 'SAVE BLOCKED: ' + err;
      throw err;
    }}
  }};
  reviewBtn.onclick = async () => {{
    if (!sample) return;
    try {{
      sample = await invoke(CB_FLAG, [{{token: sample.binding_token, flagged: !sample.review_flag}}]);
      reviewBtn.textContent = sample.review_flag ? 'Unflag review' : 'Flag review / skip';
      messageEl.textContent = sample.review_flag ? 'Review flag eklendi.' : 'Review flag kaldırıldı.';
    }} catch (err) {{
      messageEl.textContent = 'FLAG BLOCKED: ' + err;
      throw err;
    }}
  }};

  loadIndex({initial});
}})();
</script>
"""
    display(HTML(html))
    return session
