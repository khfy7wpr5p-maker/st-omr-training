"""Exact two-sample Colab UI for approved V5-2B TRAIN replacements.

This UI is intentionally narrower than the V5-2A 300-sample annotator. It can
only save human full-meter BBoxes for the two approved replacement samples at
indices 63 and 125 after the bounded replacement apply evidence has been
verified. It cannot navigate to or mutate the other 298 annotations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from .meter_v5_2a_specialist_adaptation import AdaptationAnnotationSession, MeterV5_2AError
from . import meter_v5_2b_train_replacement_v1 as repl

TARGETS: Final[tuple[tuple[int, str, str], ...]] = (
    (63, "150200092-1_1_1", "2/4"),
    (125, "150207112-1_1_1", "3/4"),
)
_CALLBACK_GET: Final[str] = "st_omr_meter_v52b_repl.get"
_CALLBACK_SAVE: Final[str] = "st_omr_meter_v52b_repl.save"


def _fail(message: str) -> None:
    raise MeterV5_2AError(message)


def _verified_session(data_root: str | Path) -> AdaptationAnnotationSession:
    root = Path(data_root)
    evidence = repl.verify_applied_replacements_v1(root)
    if evidence.get("selection_count") != 300 or evidence.get("annotation_count") not in {298, 299, 300}:
        _fail("replacement annotation UI requires applied 300-selection evidence")
    expected_ids = {sample_id for _index, sample_id, _meter in TARGETS}
    if set(evidence.get("new_sample_ids_unannotated", [])) != expected_ids:
        _fail("replacement apply identities differ from approved two-sample target set")

    session = AdaptationAnnotationSession(data_root=root)
    for index, sample_id, meter in TARGETS:
        sample = session.samples[index]
        if sample.sample_id != sample_id or sample.meter != meter or sample.seed_annotation:
            _fail(f"replacement target binding changed at index {index}")
    return session


def _target_payload(session: AdaptationAnnotationSession, position: int) -> dict[str, object]:
    if type(position) is not int or not 0 <= position < len(TARGETS):
        _fail("replacement logical position outside 0..1")
    index, sample_id, meter = TARGETS[position]
    payload = session.sample_payload(index)
    if payload["sample_id"] != sample_id or payload["meter"] != meter or payload["locked_seed"] is not False:
        _fail("replacement payload binding changed")
    payload = dict(payload)
    payload["replacement_position"] = position
    payload["replacement_total"] = len(TARGETS)
    return payload


def launch_replacement_annotation(*, data_root: str) -> AdaptationAnnotationSession:
    """Launch a two-target-only human full-meter BBox UI in Google Colab."""
    try:
        from google.colab import output
        from IPython.display import HTML, JSON, display
    except ImportError as exc:
        raise MeterV5_2AError("V5-2B replacement annotation UI requires Google Colab") from exc

    session = _verified_session(data_root)

    def get_callback(position: object) -> object:
        if type(position) is not int:
            _fail("replacement position must be integer")
        return JSON(_target_payload(session, position))

    def save_callback(payload: object) -> object:
        if not isinstance(payload, dict):
            _fail("replacement save payload must be an object")
        required = {
            "position", "token", "x0", "y0", "x1", "y1",
            "preview_width", "preview_height",
        }
        if set(payload) != required:
            _fail("replacement save payload malformed")
        position = payload["position"]
        if type(position) is not int or not 0 <= position < len(TARGETS):
            _fail("replacement save position outside 0..1")
        expected = _target_payload(session, position)
        if payload["token"] != expected["binding_token"]:
            _fail("replacement save token does not match approved target")
        result = session.save_from_preview(
            token=payload["token"],
            x0=payload["x0"], y0=payload["y0"],
            x1=payload["x1"], y1=payload["y1"],
            preview_width=payload["preview_width"],
            preview_height=payload["preview_height"],
        )
        if result["sample_id"] != expected["sample_id"]:
            _fail("replacement save returned unexpected sample identity")
        return JSON({
            **result,
            "position": position,
            "replacement_complete": all(
                sample_id in session.annotations and session.annotations[sample_id]["status"] == "PASS"
                for _index, sample_id, _meter in TARGETS
            ),
        })

    output.register_callback(_CALLBACK_GET, get_callback)
    output.register_callback(_CALLBACK_SAVE, save_callback)

    unhandled_positions = [
        position
        for position, (_index, sample_id, _meter) in enumerate(TARGETS)
        if sample_id not in session.annotations or session.annotations[sample_id]["status"] != "PASS"
    ]
    initial = unhandled_positions[0] if unhandled_positions else 0
    complete = not unhandled_positions

    html = f"""
<div id="v52b-repl-root" style="font-family:Arial,sans-serif;max-width:1180px">
  <h3>ST-OMR Meter V5-2B — Yalnız 2 Replacement Full-Meter BBox</h3>
  <div style="padding:10px;border:1px solid #bbb;margin-bottom:10px;background:#fafafa">
    <b>Yetki sınırı:</b> yalnız index 63 / 150200092-1_1_1 (2/4) ve index 125 / 150207112-1_1_1 (3/4).<br>
    <b>Kural:</b> Tek kırmızı dikdörtgen üst + alt meter rakamının tamamını birlikte kapsar; clef/key/ilk nota mümkün olduğunca dışarıda kalır.<br>
    <b>Yasak:</b> başka örnek düzenleme, midpoint, tight-digit kutu, otomatik BBox, model GT.<br>
    TRAINING=CLOSED | VAL=CLOSED | FINAL_HOLDOUT=LOCKED | 4-AI=FROZEN
  </div>
  <div id="v52b-repl-status" style="margin:6px 0;font-weight:bold"></div>
  <div id="v52b-repl-meta" style="margin:6px 0"></div>
  <div style="overflow:auto;border:1px solid #bbb;display:inline-block;max-width:100%">
    <canvas id="v52b-repl-canvas" style="display:block;max-width:100%;touch-action:none;cursor:crosshair"></canvas>
  </div>
  <div style="margin-top:10px">
    <button id="v52b-repl-save"><b>KAYDET VE DİĞER REPLACEMENT'A GEÇ</b></button>
  </div>
  <div id="v52b-repl-message" style="margin-top:8px;white-space:pre-wrap"></div>
</div>
<script>
(() => {{
  const CB_GET={json.dumps(_CALLBACK_GET)};
  const CB_SAVE={json.dumps(_CALLBACK_SAVE)};
  const TOTAL=2;
  let position={initial};
  let sample=null, rect=null, dragging=false, start=null;
  const canvas=document.getElementById('v52b-repl-canvas');
  const ctx=canvas.getContext('2d');
  const statusEl=document.getElementById('v52b-repl-status');
  const metaEl=document.getElementById('v52b-repl-meta');
  const msg=document.getElementById('v52b-repl-message');
  const saveBtn=document.getElementById('v52b-repl-save');
  const img=new Image();
  function unwrap(result) {{
    if (!result || !result.data || result.data['application/json']===undefined) throw new Error('Colab callback JSON missing');
    return result.data['application/json'];
  }}
  async function invoke(name,args) {{ return unwrap(await google.colab.kernel.invokeFunction(name,args,{{}})); }}
  function clamp(v,lo,hi) {{ return Math.max(lo,Math.min(hi,v)); }}
  function pointer(ev) {{
    const r=canvas.getBoundingClientRect();
    return {{x:clamp(Math.round((ev.clientX-r.left)*canvas.width/r.width),0,canvas.width),y:clamp(Math.round((ev.clientY-r.top)*canvas.height/r.height),0,canvas.height)}};
  }}
  function draw() {{
    if(!sample||!img.complete)return;
    ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(img,0,0,canvas.width,canvas.height);
    if(rect){{const left=Math.min(rect.x0,rect.x1),top=Math.min(rect.y0,rect.y1);ctx.lineWidth=3;ctx.strokeStyle='#ff0000';ctx.strokeRect(left,top,Math.abs(rect.x1-rect.x0),Math.abs(rect.y1-rect.y0));}}
  }}
  async function load(pos) {{
    position=pos; rect=null; msg.textContent='Yükleniyor...';
    sample=await invoke(CB_GET,[position]);
    canvas.width=sample.preview_width; canvas.height=sample.preview_height;
    statusEl.textContent=`Replacement ${{position+1}} / 2 | Global PASS: ${{sample.pass_count}} / 300`;
    metaEl.textContent=`Index: ${{sample.index}} | Meter: ${{sample.meter}} | Sample: ${{sample.sample_id}} | Original: ${{sample.image_width}}×${{sample.image_height}} px`;
    img.onload=()=>{{draw();msg.textContent=sample.status==='PASS'?'Bu replacement zaten PASS. Gerekmedikçe yeniden çizme.':'Üst ve alt meter rakamlarını tek kırmızı kutuda birlikte çerçevele.';}};
    img.src=sample.preview_data_uri;
  }}
  canvas.addEventListener('pointerdown',ev=>{{dragging=true;start=pointer(ev);rect={{x0:start.x,y0:start.y,x1:start.x,y1:start.y}};canvas.setPointerCapture(ev.pointerId);draw();}});
  canvas.addEventListener('pointermove',ev=>{{if(!dragging||!rect)return;const p=pointer(ev);rect.x1=p.x;rect.y1=p.y;draw();}});
  canvas.addEventListener('pointerup',ev=>{{if(!dragging||!rect)return;const p=pointer(ev);rect.x1=p.x;rect.y1=p.y;dragging=false;draw();}});
  saveBtn.onclick=async()=>{{
    if(!sample||!rect){{msg.textContent='Önce full-meter BBox çiz.';return;}}
    saveBtn.disabled=true; msg.textContent='Kaydediliyor...';
    try{{
      const result=await invoke(CB_SAVE,[{{position:position,token:sample.binding_token,x0:Math.round(rect.x0),y0:Math.round(rect.y0),x1:Math.round(rect.x1),y1:Math.round(rect.y1),preview_width:sample.preview_width,preview_height:sample.preview_height}}]);
      if(result.replacement_complete){{statusEl.textContent='2 / 2 REPLACEMENT BBOX PASS';msg.textContent='Tamamlandı. Bu hücrede başka örnek düzenlenemez.';saveBtn.disabled=true;return;}}
      await load(position===0?1:0); saveBtn.disabled=false;
    }}catch(err){{msg.textContent='FAIL-CLOSED: '+err;saveBtn.disabled=false;}}
  }};
  if({str(complete).lower()}){{statusEl.textContent='2 / 2 REPLACEMENT BBOX ZATEN PASS';msg.textContent='Yeni çizim gerekmiyor.';saveBtn.disabled=true;}}
  else{{load(position);}}
}})();
</script>
"""
    display(HTML(html))
    return session
