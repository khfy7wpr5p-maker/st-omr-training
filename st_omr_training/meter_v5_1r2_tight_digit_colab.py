"""Google Colab UI for the model-blind METER V5-1R2 tight-digit pilot."""
from __future__ import annotations

import json

from .meter_v5_1r2_tight_digit_pilot import (
    MeterV5_1R2PilotError,
    TightDigitAnnotationSession,
)

_CALLBACK_GET = "st_omr_meter_v51r2.get"
_CALLBACK_SAVE = "st_omr_meter_v51r2.save"
_CALLBACK_REVIEW = "st_omr_meter_v51r2.review"


def _dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MeterV5_1R2PilotError(f"{name} payload must be object")
    return value


def launch_tight_digit_colab(*, data_root: str) -> TightDigitAnnotationSession:
    """Launch the 9-sample numerator+denominator human annotation UI."""
    try:
        from google.colab import output
        from IPython.display import HTML, JSON, display
    except ImportError as exc:
        raise MeterV5_1R2PilotError("V5-1R2 tight-digit UI requires Google Colab") from exc

    session = TightDigitAnnotationSession(data_root=data_root)

    def get_callback(index: object) -> object:
        if type(index) is not int:
            raise MeterV5_1R2PilotError("index must be integer")
        return JSON(session.sample_payload(index))

    def save_callback(payload: object) -> object:
        body = _dict(payload, "save")
        if set(body) != {"token", "numerator", "denominator", "preview_width", "preview_height"}:
            raise MeterV5_1R2PilotError("save payload malformed")
        return JSON(session.save_from_preview(
            token=body["token"],
            numerator=_dict(body["numerator"], "numerator"),
            denominator=_dict(body["denominator"], "denominator"),
            preview_width=body["preview_width"],
            preview_height=body["preview_height"],
        ))

    def review_callback(payload: object) -> object:
        body = _dict(payload, "review")
        if set(body) != {"token"}:
            raise MeterV5_1R2PilotError("review payload malformed")
        return JSON(session.mark_review(token=body["token"]))

    output.register_callback(_CALLBACK_GET, get_callback)
    output.register_callback(_CALLBACK_SAVE, save_callback)
    output.register_callback(_CALLBACK_REVIEW, review_callback)

    initial = session.resume_index()
    html = f"""
<div id="r2-root" style="font-family:Arial,sans-serif;max-width:1180px">
  <h3>ST-OMR Meter V5-1R2 — 9 TRAIN Tight-Digit Pilot</h3>
  <div style="padding:10px;border:1px solid #bbb;margin-bottom:10px;background:#fafafa">
    <b>Amaç:</b> Bu aşamada model çalışmaz. Sadece gerçek insan GT üretiriz.<br>
    <b>Kırmızı kesik kutu:</b> önceki onaylı full-meter bbox, sadece referans.<br>
    <b>Mavi:</b> NUMERATOR. <b>Yeşil:</b> DENOMINATOR.<br>
    Her kutu yalnız kendi rakamını sıkı biçimde kapsasın; diğer rakamı, clef/key signature/notayı mümkün olduğunca dışarıda bırak.<br>
    <b>İki kutunun dikey olarak çakışması serbesttir.</b> Midpoint zorlaması yoktur.<br>
    <b>VALIDATION / FINAL HOLDOUT KAPALI. MODEL / TRAINING KAPALI.</b>
  </div>
  <div id="r2-status" style="margin:6px 0;font-weight:bold"></div>
  <div id="r2-meta" style="margin:6px 0"></div>
  <div style="margin:8px 0;display:flex;gap:8px;flex-wrap:wrap">
    <button id="r2-num" style="font-weight:bold">NUMERATOR ÇİZ</button>
    <button id="r2-den" style="font-weight:bold">DENOMINATOR ÇİZ</button>
    <span id="r2-role" style="padding:4px 8px;border:1px solid #aaa">Aktif: NUMERATOR</span>
  </div>
  <div style="overflow:auto;border:1px solid #bbb;display:inline-block;max-width:100%">
    <canvas id="r2-canvas" style="display:block;max-width:100%;touch-action:none;cursor:crosshair"></canvas>
  </div>
  <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
    <button id="r2-prev">ÖNCEKİ</button>
    <button id="r2-save"><b>İKİ KUTUYU KAYDET VE SONRAKİ</b></button>
    <button id="r2-review">REVIEW / ATLA</button>
    <button id="r2-next">SONRAKİ</button>
  </div>
  <div id="r2-msg" style="margin-top:8px;white-space:pre-wrap"></div>
</div>
<script>
(() => {{
  const CB_GET = {json.dumps(_CALLBACK_GET)};
  const CB_SAVE = {json.dumps(_CALLBACK_SAVE)};
  const CB_REVIEW = {json.dumps(_CALLBACK_REVIEW)};
  const canvas = document.getElementById('r2-canvas');
  const ctx = canvas.getContext('2d');
  const statusEl = document.getElementById('r2-status');
  const metaEl = document.getElementById('r2-meta');
  const msgEl = document.getElementById('r2-msg');
  const roleEl = document.getElementById('r2-role');
  const img = new Image();
  let sample = null;
  let active = 'numerator';
  let boxes = {{numerator:null, denominator:null}};
  let dragging = false;
  let start = null;

  function unwrap(result) {{
    if (!result || !result.data || result.data['application/json'] === undefined) throw new Error('Colab callback JSON missing');
    return result.data['application/json'];
  }}
  async function invoke(name,args) {{ return unwrap(await google.colab.kernel.invokeFunction(name,args,{{}})); }}
  function clamp(v,lo,hi) {{ return Math.max(lo,Math.min(hi,v)); }}
  function pointer(ev) {{
    const r=canvas.getBoundingClientRect();
    return {{
      x:clamp(Math.round((ev.clientX-r.left)*canvas.width/r.width),0,canvas.width),
      y:clamp(Math.round((ev.clientY-r.top)*canvas.height/r.height),0,canvas.height)
    }};
  }}
  function originalToPreview(b) {{
    if (!b || b.x === null) return null;
    return {{
      x0:Math.floor(b.x*sample.preview_width/sample.image_width),
      y0:Math.floor(b.y*sample.preview_height/sample.image_height),
      x1:Math.ceil((b.x+b.w)*sample.preview_width/sample.image_width),
      y1:Math.ceil((b.y+b.h)*sample.preview_height/sample.image_height)
    }};
  }}
  function stroke(box,color,width,dash=[]) {{
    if (!box) return;
    const left=Math.min(box.x0,box.x1), top=Math.min(box.y0,box.y1);
    const w=Math.abs(box.x1-box.x0), h=Math.abs(box.y1-box.y0);
    ctx.save(); ctx.strokeStyle=color; ctx.lineWidth=width; ctx.setLineDash(dash); ctx.strokeRect(left,top,w,h); ctx.restore();
  }}
  function draw() {{
    if (!sample || !img.complete) return;
    ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(img,0,0,canvas.width,canvas.height);
    stroke(originalToPreview(sample.full_bbox),'#e31a1c',2,[7,5]);
    stroke(boxes.numerator,'#1565c0',3); stroke(boxes.denominator,'#1b8f3a',3);
  }}
  function setActive(role) {{ active=role; roleEl.textContent='Aktif: '+role.toUpperCase(); }}
  async function loadIndex(index) {{
    msgEl.textContent='Yükleniyor...';
    sample=await invoke(CB_GET,[index]);
    canvas.width=sample.preview_width; canvas.height=sample.preview_height;
    boxes.numerator=originalToPreview(sample.roles.numerator);
    boxes.denominator=originalToPreview(sample.roles.denominator);
    statusEl.textContent=`İşlenen ${{sample.handled_count}}/9 | PASS sample ${{sample.pass_sample_count}} | REVIEW sample ${{sample.review_sample_count}} | Örnek ${{sample.index+1}}/9`;
    metaEl.textContent=`Meter: ${{sample.meter}} | TRAIN | Sample: ${{sample.sample_id}} | Original: ${{sample.image_width}}×${{sample.image_height}}`;
    img.onload=()=>{{draw(); msgEl.textContent='Önce NUMERATOR sonra DENOMINATOR kutusunu çiz. Kutular çakışabilir.';}};
    img.src=sample.preview_data_uri;
  }}
  document.getElementById('r2-num').onclick=()=>setActive('numerator');
  document.getElementById('r2-den').onclick=()=>setActive('denominator');
  canvas.addEventListener('pointerdown',ev=>{{ if(!sample)return; dragging=true; start=pointer(ev); boxes[active]={{x0:start.x,y0:start.y,x1:start.x,y1:start.y}}; canvas.setPointerCapture(ev.pointerId); draw(); }});
  canvas.addEventListener('pointermove',ev=>{{ if(!dragging||!boxes[active])return; const p=pointer(ev); boxes[active].x1=p.x; boxes[active].y1=p.y; draw(); }});
  canvas.addEventListener('pointerup',ev=>{{ if(!dragging||!boxes[active])return; const p=pointer(ev); boxes[active].x1=p.x; boxes[active].y1=p.y; dragging=false; draw(); }});
  document.getElementById('r2-prev').onclick=async()=>{{if(sample) await loadIndex(Math.max(0,sample.index-1));}};
  document.getElementById('r2-next').onclick=async()=>{{if(sample) await loadIndex(Math.min(8,sample.index+1));}};
  document.getElementById('r2-save').onclick=async()=>{{
    if(!sample||!boxes.numerator||!boxes.denominator){{msgEl.textContent='İki kutu da gerekli.';return;}}
    msgEl.textContent='Kaydediliyor...';
    try {{
      await invoke(CB_SAVE,[{{token:sample.binding_token,numerator:boxes.numerator,denominator:boxes.denominator,preview_width:sample.preview_width,preview_height:sample.preview_height}}]);
      if(sample.index===8){{await loadIndex(8);msgEl.textContent='9. örnek kaydedildi. Audit hücresini çalıştır.';}}
      else await loadIndex(sample.index+1);
    }} catch(err) {{msgEl.textContent='SAVE BLOCKED: '+err; throw err;}}
  }};
  document.getElementById('r2-review').onclick=async()=>{{
    if(!sample)return; msgEl.textContent='REVIEW kaydediliyor...';
    try {{await invoke(CB_REVIEW,[{{token:sample.binding_token}}]); if(sample.index===8) await loadIndex(8); else await loadIndex(sample.index+1);}}
    catch(err){{msgEl.textContent='REVIEW BLOCKED: '+err;throw err;}}
  }};
  setActive('numerator'); loadIndex({initial});
}})();
</script>
"""
    display(HTML(html))
    return session
