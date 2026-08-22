"""Google Colab canvas UI for METER V5-1 train-only bbox pilot."""
from __future__ import annotations

import json

from st_omr_training.meter_v5_1_bbox_pilot import (
    AnnotationSession,
    MeterV5_1PilotError,
)

_CALLBACK_GET = "st_omr_meter_v51_pilot.get"
_CALLBACK_SAVE = "st_omr_meter_v51_pilot.save"
_CALLBACK_REVIEW = "st_omr_meter_v51_pilot.review"


def _require_dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MeterV5_1PilotError(f"{name} payload must be an object")
    return value


def launch_colab_pilot(*, data_root: str) -> AnnotationSession:
    """Launch one-image-at-a-time train-only annotation UI."""
    try:
        from google.colab import output
        from IPython.display import HTML, JSON, display
    except ImportError as exc:
        raise MeterV5_1PilotError("V5-1 bbox pilot UI requires Google Colab") from exc

    session = AnnotationSession(data_root=data_root)

    def get_callback(index: object) -> object:
        if type(index) is not int:
            raise MeterV5_1PilotError("sample index must be integer")
        return JSON(session.sample_payload(index))

    def save_callback(payload: object) -> object:
        body = _require_dict(payload, "save")
        required = {"token", "x0", "y0", "x1", "y1", "preview_width", "preview_height"}
        if set(body) != required:
            raise MeterV5_1PilotError("save payload malformed")
        return JSON(session.save_from_preview(
            token=body["token"],
            x0=body["x0"], y0=body["y0"],
            x1=body["x1"], y1=body["y1"],
            preview_width=body["preview_width"],
            preview_height=body["preview_height"],
        ))

    def review_callback(payload: object) -> object:
        body = _require_dict(payload, "review")
        if set(body) != {"token"}:
            raise MeterV5_1PilotError("review payload malformed")
        return JSON(session.mark_review(token=body["token"]))

    output.register_callback(_CALLBACK_GET, get_callback)
    output.register_callback(_CALLBACK_SAVE, save_callback)
    output.register_callback(_CALLBACK_REVIEW, review_callback)

    initial = session.resume_index()
    html = f"""
<div id="v51-root" style="font-family:Arial,sans-serif;max-width:1180px">
  <h3>ST-OMR Meter V5-1 — 30 TRAIN BBox Pilot</h3>
  <div style="padding:10px;border:1px solid #bbb;margin-bottom:10px;background:#fafafa">
    <b>Kural:</b> Tek dikdörtgen içinde üst + alt meter rakamını tamamen kapsa.
    Clef, key signature ve sağdaki ilk nota mümkün olduğunca dışarıda kalsın.
    Görüntü dosyası değiştirilmez; kırmızı kutu yalnız bu canvas preview üzerindedir.
    <br><b>FINAL HOLDOUT KİLİTLİDİR.</b>
  </div>
  <div id="v51-status" style="margin:6px 0;font-weight:bold"></div>
  <div id="v51-meta" style="margin:6px 0"></div>
  <div style="overflow:auto;border:1px solid #bbb;display:inline-block;max-width:100%">
    <canvas id="v51-canvas" style="display:block;max-width:100%;touch-action:none;cursor:crosshair"></canvas>
  </div>
  <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
    <button id="v51-prev">ÖNCEKİ</button>
    <button id="v51-save"><b>KAYDET VE SONRAKİ</b></button>
    <button id="v51-review">REVIEW / ATLA</button>
    <button id="v51-next">SONRAKİ</button>
  </div>
  <div id="v51-message" style="margin-top:8px;white-space:pre-wrap"></div>
</div>
<script>
(() => {{
  const CB_GET = {json.dumps(_CALLBACK_GET)};
  const CB_SAVE = {json.dumps(_CALLBACK_SAVE)};
  const CB_REVIEW = {json.dumps(_CALLBACK_REVIEW)};
  const canvas = document.getElementById('v51-canvas');
  const ctx = canvas.getContext('2d');
  const statusEl = document.getElementById('v51-status');
  const metaEl = document.getElementById('v51-meta');
  const messageEl = document.getElementById('v51-message');
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
    messageEl.textContent = 'Yükleniyor...';
    try {{
      sample = await invoke(CB_GET, [index]);
      canvas.width = sample.preview_width;
      canvas.height = sample.preview_height;
      rect = existingRect(sample);
      statusEl.textContent = `İşlenen: ${{sample.handled_count}} / 30 | PASS: ${{sample.pass_count}} | REVIEW: ${{sample.review_count}} | Örnek: ${{sample.index + 1}} / 30`;
      metaEl.textContent = `Meter: ${{sample.meter}} | Split: TRAIN | Sample: ${{sample.sample_id}} | Original: ${{sample.image_width}}×${{sample.image_height}} px`;
      img.onload = () => {{
        draw();
        if (sample.status === 'PASS') {{
          messageEl.textContent = 'Mevcut PASS bbox gösteriliyor. İstersen yeniden çizip KAYDET ile tek satırı güncelleyebilirsin.';
        }} else if (sample.status === 'REVIEW') {{
          messageEl.textContent = 'Bu örnek REVIEW olarak işaretli. Bbox çizebilir ve PASS olarak kaydedebilirsin.';
        }} else {{
          messageEl.textContent = 'Mouse ile yalnız tam meter çiftini çerçevele.';
        }}
      }};
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

  document.getElementById('v51-prev').onclick = async () => {{
    if (!sample) return;
    await loadIndex(Math.max(0, sample.index - 1));
  }};
  document.getElementById('v51-next').onclick = async () => {{
    if (!sample) return;
    await loadIndex(Math.min(29, sample.index + 1));
  }};
  document.getElementById('v51-save').onclick = async () => {{
    if (!sample || !rect) {{ messageEl.textContent = 'Önce bbox çiz.'; return; }}
    messageEl.textContent = 'Kaydediliyor...';
    try {{
      const result = await invoke(CB_SAVE, [{{
        token: sample.binding_token,
        x0: Math.round(rect.x0), y0: Math.round(rect.y0),
        x1: Math.round(rect.x1), y1: Math.round(rect.y1),
        preview_width: sample.preview_width,
        preview_height: sample.preview_height
      }}]);
      const next = Math.min(29, sample.index + 1);
      if (sample.index === 29) {{
        messageEl.textContent = `Kaydedildi. Pilot son örnek. PASS=${{result.pass_count}}, REVIEW=${{result.review_count}}. Audit hücresini çalıştır.`;
        await loadIndex(29);
      }} else {{
        await loadIndex(next);
      }}
    }} catch (err) {{
      messageEl.textContent = 'SAVE BLOCKED: ' + err;
      throw err;
    }}
  }};
  document.getElementById('v51-review').onclick = async () => {{
    if (!sample) return;
    messageEl.textContent = 'REVIEW kaydediliyor...';
    try {{
      const result = await invoke(CB_REVIEW, [{{token: sample.binding_token}}]);
      const next = Math.min(29, sample.index + 1);
      if (sample.index === 29) {{
        messageEl.textContent = `REVIEW kaydedildi. Pilot son örnek. PASS=${{result.pass_count}}, REVIEW=${{result.review_count}}. Audit hücresini çalıştır.`;
        await loadIndex(29);
      }} else {{
        await loadIndex(next);
      }}
    }} catch (err) {{
      messageEl.textContent = 'REVIEW BLOCKED: ' + err;
      throw err;
    }}
  }};

  loadIndex({initial});
}})();
</script>
"""
    display(HTML(html))
    return session
