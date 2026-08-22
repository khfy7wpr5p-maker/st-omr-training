"""Google Colab canvas UI for METER V5-2 1,200 TRAIN full-meter BBoxes."""
from __future__ import annotations

import json
from pathlib import Path

from st_omr_training.meter_v5_2_train_bbox_scale import (
    ScaleAnnotationSession,
    MeterV5_2ScaleError,
)

_CALLBACK_GET = "st_omr_meter_v52_scale.get"
_CALLBACK_SAVE = "st_omr_meter_v52_scale.save"
_CALLBACK_REVIEW = "st_omr_meter_v52_scale.review"


def _require_dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MeterV5_2ScaleError(f"{name} payload must be an object")
    return value


def launch_colab_scale(*, data_root: str, session: ScaleAnnotationSession | None = None) -> ScaleAnnotationSession:
    """Launch the TRAIN-only one-full-meter-box-per-image annotation UI."""
    try:
        from google.colab import output
        from IPython.display import HTML, JSON, display
    except ImportError as exc:
        raise MeterV5_2ScaleError("V5-2 bbox scale UI requires Google Colab") from exc

    if session is None:
        session = ScaleAnnotationSession(data_root=data_root)
    elif session.data_root.resolve() != Path(data_root).resolve():
        raise MeterV5_2ScaleError("provided V5-2 session belongs to a different data root")

    def get_callback(index: object) -> object:
        if type(index) is not int:
            raise MeterV5_2ScaleError("sample index must be integer")
        return JSON(session.sample_payload(index))

    def save_callback(payload: object) -> object:
        body = _require_dict(payload, "save")
        required = {"token", "x0", "y0", "x1", "y1", "preview_width", "preview_height"}
        if set(body) != required:
            raise MeterV5_2ScaleError("save payload malformed")
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
            raise MeterV5_2ScaleError("review payload malformed")
        return JSON(session.mark_review(token=body["token"]))

    output.register_callback(_CALLBACK_GET, get_callback)
    output.register_callback(_CALLBACK_SAVE, save_callback)
    output.register_callback(_CALLBACK_REVIEW, review_callback)

    initial = session.resume_index()
    html = f"""
<div id="v52-root" style="font-family:Arial,sans-serif;max-width:1180px">
  <h3>ST-OMR Meter V5-2 — 1200 TRAIN Full-Meter BBox</h3>
  <div style="padding:10px;border:1px solid #bbb;margin-bottom:10px;background:#fafafa">
    <b>Kural:</b> Tek kırmızı dikdörtgen üst + alt meter rakamının tamamını birlikte kapsar.
    Clef, key signature ve sağdaki ilk nota mümkün olduğunca dışarıda kalır.
    <br><b>BBox ikiye bölünmez. Numerator/denominator kutusu üretilmez.</b>
    <br>İlk 30 V5-1 kutusu kilitli ve değiştirilemez. VAL ve FINAL HOLDOUT kapalıdır.
  </div>
  <div id="v52-status" style="margin:6px 0;font-weight:bold"></div>
  <div id="v52-meta" style="margin:6px 0"></div>
  <div style="overflow:auto;border:1px solid #bbb;display:inline-block;max-width:100%">
    <canvas id="v52-canvas" style="display:block;max-width:100%;touch-action:none;cursor:crosshair"></canvas>
  </div>
  <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
    <button id="v52-prev">ÖNCEKİ</button>
    <button id="v52-save"><b>KAYDET VE SONRAKİ</b></button>
    <button id="v52-review">REVIEW / ATLA</button>
    <button id="v52-next">SONRAKİ</button>
  </div>
  <div id="v52-message" style="margin-top:8px;white-space:pre-wrap"></div>
</div>
<script>
(() => {{
  const CB_GET = {json.dumps(_CALLBACK_GET)};
  const CB_SAVE = {json.dumps(_CALLBACK_SAVE)};
  const CB_REVIEW = {json.dumps(_CALLBACK_REVIEW)};
  const canvas = document.getElementById('v52-canvas');
  const ctx = canvas.getContext('2d');
  const statusEl = document.getElementById('v52-status');
  const metaEl = document.getElementById('v52-meta');
  const messageEl = document.getElementById('v52-message');
  const saveBtn = document.getElementById('v52-save');
  const reviewBtn = document.getElementById('v52-review');
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
    sample = await invoke(CB_GET, [index]);
    canvas.width = sample.preview_width;
    canvas.height = sample.preview_height;
    rect = existingRect(sample);
    const locked = sample.locked_seed === true;
    saveBtn.disabled = locked;
    reviewBtn.disabled = locked;
    canvas.style.cursor = locked ? 'not-allowed' : 'crosshair';
    statusEl.textContent =
      `PASS: ${{sample.pass_count}} / 1200 | REVIEW: ${{sample.review_count}} | ` +
      `Kalan işlenmemiş: ${{sample.remaining_unhandled}} | Örnek: ${{sample.index + 1}} / 1200`;
    metaEl.textContent =
      `Meter: ${{sample.meter}} | Split: TRAIN | Sample: ${{sample.sample_id}} | ` +
      `Original: ${{sample.image_width}}×${{sample.image_height}} px` +
      (locked ? ' | V5-1 SEED: KİLİTLİ' : '');
    img.onload = () => {{
      draw();
      if (locked) {{
        messageEl.textContent = 'Bu V5-1 kutusu kabul edilmiş seed kaydıdır; değiştirilemez.';
      }} else if (sample.status === 'PASS') {{
        messageEl.textContent = 'Mevcut PASS bbox gösteriliyor. Yeniden çizip kaydedebilirsin.';
      }} else if (sample.status === 'REVIEW') {{
        messageEl.textContent = 'Bu örnek REVIEW. Tek full-meter bbox çizip PASS olarak kaydedebilirsin.';
      }} else {{
        messageEl.textContent = 'Tek kutu ile üst ve alt meter rakamlarını birlikte çerçevele.';
      }}
    }};
    img.src = sample.preview_data_uri;
  }}

  canvas.addEventListener('pointerdown', (ev) => {{
    if (!sample || sample.locked_seed) return;
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

  document.getElementById('v52-prev').onclick = async () => {{
    if (!sample) return;
    await loadIndex(Math.max(0, sample.index - 1));
  }};
  document.getElementById('v52-next').onclick = async () => {{
    if (!sample) return;
    await loadIndex(Math.min(1199, sample.index + 1));
  }};
  saveBtn.onclick = async () => {{
    if (!sample || sample.locked_seed) return;
    if (!rect) {{ messageEl.textContent = 'Önce full-meter bbox çiz.'; return; }}
    messageEl.textContent = 'Kaydediliyor...';
    const result = await invoke(CB_SAVE, [{{
      token: sample.binding_token,
      x0: Math.round(rect.x0), y0: Math.round(rect.y0),
      x1: Math.round(rect.x1), y1: Math.round(rect.y1),
      preview_width: sample.preview_width,
      preview_height: sample.preview_height
    }}]);
    if (sample.index < 1199) {{
      await loadIndex(sample.index + 1);
    }} else {{
      messageEl.textContent =
        `Son örnek kaydedildi. PASS=${{result.pass_count}}, REVIEW=${{result.review_count}}. Audit hücresini çalıştır.`;
    }}
  }};
  reviewBtn.onclick = async () => {{
    if (!sample || sample.locked_seed) return;
    messageEl.textContent = 'REVIEW kaydediliyor...';
    const result = await invoke(CB_REVIEW, [{{token: sample.binding_token}}]);
    if (sample.index < 1199) {{
      await loadIndex(sample.index + 1);
    }} else {{
      messageEl.textContent =
        `Son örnek REVIEW. PASS=${{result.pass_count}}, REVIEW=${{result.review_count}}.`;
    }}
  }};

  loadIndex({initial});
}})();
</script>
"""
    display(HTML(html))
    return session
