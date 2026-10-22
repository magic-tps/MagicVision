"""Export local review packages and apply explicit, hash-bound human decisions.

This utility never approves a download or prediction automatically. It does not
split datasets, train models, upload images, or alter the original labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import uuid
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import RLock
from urllib.parse import urlsplit

from PIL import Image, ImageDraw
from data_pipeline import CLASSES, read_labels, read_manifest, save_json, sha, write_manifest
import data_pipeline as legacy_schema
import campus_schema
import campus22_schema


def schema_for(value='legacy11'):
    if isinstance(value, dict):
        names = value.get('classes')
        if names == campus22_schema.CLASSES: return campus22_schema
        if names == campus_schema.CLASSES: return campus_schema
        if names == CLASSES: return legacy_schema
        raise ValueError('Unknown package class contract')
    if value == 'legacy11': return legacy_schema
    if value == 'campus-v4': return campus_schema
    if value == 'campus22': return campus22_schema
    raise ValueError('Unknown taxonomy')


def now():
    return datetime.now(timezone.utc).isoformat()


def unique_rows(manifests):
    rows = [row for path in manifests for row in read_manifest(path)]
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate record IDs: combine manifests without repeating source records')
    return rows


def inside(package, relative):
    path = (Path(package) / relative).resolve()
    if not path.is_relative_to(Path(package).resolve()):
        raise ValueError('Review file escapes package')
    return path


def package_data(package):
    package = Path(package)
    info = json.loads((package / 'package.json').read_text(encoding='utf-8'))
    if sha(package / 'source_manifest.jsonl') != info['source_manifest_sha256']:
        raise ValueError('Source manifest snapshot changed; export a new package')
    return info


def render(package, *, regenerate_previews=True, write_page=True):
    """Regenerate previews after editing labels; decisions keep their old hashes."""
    package = Path(package)
    info = package_data(package)
    schema = schema_for(info); names = schema.CLASSES
    figures = []
    decisions = {d['id']: d for d in read_manifest(package / 'decisions.jsonl')}
    (package / 'previews').mkdir(exist_ok=True)
    for item in info['items']:
        image = inside(package, item['image'])
        labels = inside(package, item['label'])
        boxes = schema.read_labels(labels)
        preview = f'previews/{item["key"]}.jpg'
        if regenerate_previews:
            with Image.open(image) as im:
                im = im.convert('RGB')
                im.thumbnail((1200, 1200))
                draw = ImageDraw.Draw(im)
                for c, x, y, w, h in boxes:
                    rect = ((x-w/2)*im.width, (y-h/2)*im.height,
                            (x+w/2)*im.width, (y+h/2)*im.height)
                    draw.rectangle(rect, outline='yellow', width=3)
                    draw.text(rect[:2], f'{c}: {names[c]}', fill='yellow', stroke_width=1, stroke_fill='black')
                im.save(package / preview, quality=90)
        details = html.escape(json.dumps(item['provenance'], ensure_ascii=False, indent=2))
        problem = html.escape(str(item.get('problem_detected', 'review_all_classes')))
        source = html.escape(str(item.get('source', item['provenance'].get('source', 'unknown'))))
        current_classes = sorted({names[box[0]] for box in boxes} |
                                 set(item['provenance'].get('candidate_classes', [])))
        classes_text = html.escape(', '.join(current_classes) if current_classes else 'sin cajas actuales')
        priority = html.escape(str(item.get('review_priority', 'unranked')))
        label_sha = sha(labels)
        decision = decisions.get(item['id'])
        status = 'needs_review'
        decision_text = 'Pendiente de revisión.'
        if decision:
            valid = (decision.get('package_id') == info['package_id']
                     and decision.get('reviewed_label_sha256') == label_sha
                     and decision.get('reviewed_image_sha256') == sha(image))
            if valid:
                status = {'approve': 'approved', 'reject': 'rejected', 'defer': 'needs_review'}[decision['action']]
                decision_text = 'Decisión registrada: ' + decision['action']
            else:
                decision_text = 'Las etiquetas o la imagen cambiaron desde la revisión. Revisa y registra la decisión de nuevo.'
        reviewer = html.escape(str((decision or {}).get('reviewer', '')), quote=True)
        note = html.escape(str((decision or {}).get('note', '')), quote=True)
        recommended = html.escape(str(item.get('recommended_split', item.get('original_split'))))
        figures.append(
            f'<article><h2>{html.escape(item["id"])}</h2>'
            f'<p><strong>Prioridad {priority}</strong> · clases: {classes_text} · fuente: {source} · estado: {status}</p>'
            f'<p><strong>Problema:</strong> {problem}</p>'
            f'<p>Grupo: {html.escape(item["group"])}; partición original: '
            f'{html.escape(str(item.get("original_split")))}; partición recomendada: {recommended}</p>'
            f'<a href="{item["image"]}">Imagen completa</a> · '
            f'<a href="{item["label"]}">Etiquetas editables YOLO</a>'
            f'<div class="editor" data-image="{item["image"]}" data-label="{html.escape(Path(item["label"]).name)}" '
            f'data-boxes="{html.escape(json.dumps(boxes), quote=True)}">'
            f'<p>Clase para dibujar: <select class="new-class"></select> '
            f'<button class="save">Guardar etiquetas corregidas</button></p>'
            f'<canvas aria-label="Editor de cajas; arrastra para añadir una caja"></canvas>'
            f'<div class="box-list"></div><p class="status"></p></div>'
            f'<details><summary>Vista guardada al ejecutar export/refresh</summary><img src="{preview}" alt="Cajas guardadas"></details>'
            f'<p>SHA256 de las etiquetas mostradas: <code>{label_sha}</code></p>'
            f'<div class="review-controls" data-id="{html.escape(item["id"], quote=True)}">'
            f'<label>Revisor humano <input class="reviewer" autocomplete="name" value="{reviewer}"></label> '
            f'<label>Nota <input class="note" size="55" value="{note}"></label><br>'
            '<button data-action="approve">Aprobar cajas y cobertura</button>'
            '<button data-action="negative">Confirmar negativo</button>'
            '<button data-action="reject">Rechazar</button>'
            '<button data-action="defer">Dejar pendiente</button>'
            f'<p class="decision-status">{html.escape(decision_text)}</p></div>'
            f'<details><summary>Procedencia y licencia</summary><pre>{details}</pre></details></article>')
    page = ('<!doctype html><html lang="es"><meta charset="utf-8">'
            '<title>Revisión local MagicVision</title><style>'
            'body{font-family:system-ui;max-width:1200px;margin:auto;padding:24px;overflow-wrap:anywhere}'
            'article{border-top:2px solid #888;margin-top:36px}img,canvas{display:block;max-width:100%;margin-top:12px}'
            'canvas{touch-action:none;cursor:crosshair}button,select{padding:6px;margin:4px}.status{font-weight:bold}'
            'pre,code{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
            f'<h1>Revisión de las {len(names)} clases</h1><p>Las cajas son propuestas. '
            'Revisar objetos faltantes, clase y geometría en la imagen completa. '
            'Una etiqueta vacía no acredita ausencia de objetos. En el servidor local, pulsa Guardar etiquetas corregidas, '
            'completa Revisor humano y Nota, y registra tu decisión. Al volver a abrir la página se cargan los datos guardados. '
            'El editor permite dibujar y eliminar cajas; para mover una caja, elimínala y dibújala de nuevo. '
            'Si abres el HTML como archivo, debes reemplazar el TXT en labels/ y ejecutar refresh; '
            'una descarga del navegador no modifica ese archivo automáticamente.</p>'
            '<p>Clases: '+html.escape(', '.join(f'{i}={c}' for i, c in enumerate(names)))+'</p>'
            + ''.join(figures) + EDITOR_SCRIPT.replace('__CLASS_NAMES__', json.dumps(names)).replace(
                '__PACKAGE_ID__', json.dumps(info['package_id'])) + '</html>')
    if write_page:
        (package / 'index.html').write_text(page, encoding='utf-8')
    return page


EDITOR_SCRIPT = r'''<script>
const names = __CLASS_NAMES__;
const packageId = __PACKAGE_ID__;
for (const editor of document.querySelectorAll('.editor')) {
  let boxes = JSON.parse(editor.dataset.boxes), start = null;
  const canvas = editor.querySelector('canvas'), ctx = canvas.getContext('2d');
  const image = new Image(), selector = editor.querySelector('.new-class');
  const status = editor.querySelector('.status'), list = editor.querySelector('.box-list');
  const options = select => names.forEach((name, id) => select.add(new Option(id+': '+name, id)));
  options(selector);
  function paint() {
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2; ctx.strokeStyle = '#ffff00'; ctx.font = '14px sans-serif';
    for (const [c,x,y,w,h] of boxes) {
      const left=(x-w/2)*canvas.width, top=(y-h/2)*canvas.height;
      ctx.strokeRect(left,top,w*canvas.width,h*canvas.height);
      ctx.fillStyle='#000';ctx.fillRect(left,Math.max(0,top-17),145,17);
      ctx.fillStyle='#fff';ctx.fillText(names[c],left+2,Math.max(14,top-3));
    }
  }
  function redraw() {
    paint();list.replaceChildren();
    boxes.forEach((box, index) => {
      const row=document.createElement('div'), select=document.createElement('select'), remove=document.createElement('button');
      options(select);select.value=box[0]; select.onchange=()=>{box[0]=Number(select.value);dirty();paint();};
      remove.textContent='Eliminar caja '+(index+1);remove.onclick=()=>{boxes.splice(index,1);dirty();redraw();};
      row.append(select,remove);list.append(row);
    });
  }
  function dirty(){editor.dataset.dirty='true';status.textContent='Cambios en el navegador sin guardar. Guarda sobre labels/'+editor.dataset.label;}
  function point(event) { const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(1,(event.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(event.clientY-r.top)/r.height))]; }
  image.onload=()=>{const scale=Math.min(1,1100/image.width);canvas.width=Math.round(image.width*scale);canvas.height=Math.round(image.height*scale);redraw();};
  image.src=editor.dataset.image;
  canvas.onpointerdown=event=>{start=point(event);canvas.setPointerCapture(event.pointerId);};
  canvas.onpointermove=event=>{if(!start)return;const end=point(event);paint();ctx.strokeStyle='#00ffdd';ctx.strokeRect(start[0]*canvas.width,start[1]*canvas.height,(end[0]-start[0])*canvas.width,(end[1]-start[1])*canvas.height);};
  canvas.onpointercancel=()=>{start=null;paint();};
  canvas.onpointerup=event=>{if(!start)return;const end=point(event);const w=Math.abs(end[0]-start[0]),h=Math.abs(end[1]-start[1]);if(w*canvas.width>=3&&h*canvas.height>=3){boxes.push([Number(selector.value),(start[0]+end[0])/2,(start[1]+end[1])/2,w,h]);dirty();}start=null;redraw();};
  editor.querySelector('.save').onclick=async()=>{
    const text=boxes.map(b=>b[0]+' '+b.slice(1).map(n=>n.toFixed(8)).join(' ')).join('\n')+(boxes.length?'\n':'');
    try {
      if (location.protocol.startsWith('http')) {
        const response=await fetch('/api/labels',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({package_id:packageId,id:editor.closest('article').querySelector('.review-controls').dataset.id,boxes})});
        const result=await response.json();if(!response.ok)throw new Error(result.error||response.statusText);
        editor.dataset.dirty='false';
        editor.closest('article').querySelector('code').textContent=result.sha256;
        editor.closest('article').querySelector('.decision-status').textContent='Etiquetas guardadas. Registra la decisión sobre estas cajas.';
        status.textContent='Etiquetas guardadas en el paquete. Registra ahora la decisión humana.';
      } else if ('showSaveFilePicker' in window) {
        const handle=await showSaveFilePicker({suggestedName:editor.dataset.label,types:[{description:'YOLO labels',accept:{'text/plain':['.txt']}}]});
        const writable=await handle.createWritable();await writable.write(text);await writable.close();
        status.textContent='Guardado. Confirma que reemplazaste labels/'+editor.dataset.label+' y ejecuta refresh antes de decide.';
      } else {
        const url=URL.createObjectURL(new Blob([text],{type:'text/plain'})),a=document.createElement('a');a.href=url;a.download=editor.dataset.label;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
        status.textContent='Descargado: mueve el TXT a labels/'+editor.dataset.label+' reemplazando el archivo y ejecuta refresh antes de decide.';
      }
    } catch(error) {status.textContent='No se guardó: '+error.message;}
  };
}
for (const controls of document.querySelectorAll('.review-controls')) {
  for (const button of controls.querySelectorAll('button')) button.onclick=async()=>{
    const status=controls.querySelector('.decision-status');
    if (controls.closest('article').querySelector('.editor').dataset.dirty==='true') {
      status.textContent='Guarda las etiquetas corregidas antes de registrar la decisión.';return;
    }
    if (!location.protocol.startsWith('http')) {
      status.textContent='Inicia el servidor local indicado en README.md para registrar decisiones.';return;
    }
    try {
      const response=await fetch('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({package_id:packageId,id:controls.dataset.id,action:button.dataset.action,
          reviewer:controls.querySelector('.reviewer').value,note:controls.querySelector('.note').value})});
      const result=await response.json();if(!response.ok)throw new Error(result.error||response.statusText);
      status.textContent='Decisión registrada: '+result.action;
    } catch(error) {status.textContent='No se registró: '+error.message;}
  };
}
</script>'''


def export(manifests, output, limit=120, classes=None, selection='rare-first', taxonomy='legacy11',
           image_mode='copy'):
    out = Path(output)
    if limit < 1:
        raise ValueError('Limit must be positive')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty package directory')
    rows = unique_rows(manifests)
    schema = schema_for(taxonomy); names = schema.CLASSES
    counts = Counter()
    annotated = {}
    for row in rows:
        ids = {box[0] for box in schema.read_labels(row['label'])}
        # Candidate categories route unannotated images to a review queue only.
        # They are never converted into boxes or verified negatives.
        ids.update(names.index(name) for name in row.get('candidate_classes', []))
        annotated[row['id']] = ids
        counts.update(ids)
    wanted = set(names.index(name) for name in classes) if classes else None
    pending = [r for r in rows if r.get('review_status') != 'approved' or not r.get('coverage_complete')]
    if wanted is not None:
        pending = [r for r in pending if annotated[r['id']] & wanted]
    # Rare annotated/candidate classes first; unknown negatives also get a queue.
    if selection != 'manifest-order':
        pending.sort(key=lambda r: (-max((1/counts[c] for c in annotated[r['id']]), default=0),
                                    not r.get('negative_candidate', False), r['id']))
    if selection not in {'rare-first', 'balanced', 'manifest-order'}:
        raise ValueError('Unknown review selection')
    if selection == 'balanced':
        queues = {c: iter([r for r in pending if c in annotated[r['id']]])
                  for c in sorted(wanted if wanted is not None else range(len(names)), key=lambda c: counts[c])}
        selected = []; used = set()
        while len(selected) < limit:
            added = False
            for queue in queues.values():
                row = next((r for r in queue if r['sha256'] not in used), None)
                if row is not None:
                    selected.append(row); used.add(row['sha256']); added = True
                    if len(selected) == limit: break
            if not added: break
        for row in pending:
            if len(selected) == limit: break
            if row['sha256'] not in used:
                selected.append(row); used.add(row['sha256'])
    elif selection == 'rare-first':
        selected = pending[:limit]
    else:
        selected = pending[:limit]
    for row in selected:
        if sha(row['image']) != row['sha256']:
            raise ValueError(f'Image changed since import: {row["id"]}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'images').mkdir()
    (out / 'labels').mkdir()
    write_manifest(out / 'source_manifest.jsonl', rows)
    items = []
    for i, row in enumerate(selected):
        key = f'{i:05d}_{row["sha256"][:16]}'
        image = f'images/{key}{Path(row["image"]).suffix.lower()}'
        label = f'labels/{key}.txt'
        if image_mode == 'hardlink':
            try:
                os.link(row['image'], out / image)
            except OSError:
                shutil.copyfile(row['image'], out / image)
        elif image_mode == 'copy':
            shutil.copyfile(row['image'], out / image)
        else:
            raise ValueError('Unknown image mode')
        shutil.copyfile(row['label'], out / label)
        items.append({'id': row['id'], 'key': key, 'image': image, 'label': label,
                      'source_image_sha256': row['sha256'], 'source_label_sha256': sha(row['label']),
                      'group': row['group'], 'original_split': row.get('original_split'),
                      'source': row.get('source'), 'review_status': row.get('review_status'),
                      'review_priority': row.get('review_priority'),
                      'problem_detected': row.get('problem_detected'),
                      'recommended_split': row.get('recommended_split'),
                      'provenance': {k: v for k, v in row.items() if k in
                         {'synthetic', 'annotation_status', 'candidate_classes', 'label_origin', 'mapping', 'attributes'}
                         or any(term in k.lower() for term in
                         ('source', 'license', 'author', 'consent', 'url', 'title', 'review_reason'))}})
    info = {'schema_version': 1, 'package_id': str(uuid.uuid4()), 'created_at': now(),
            'source_manifest_sha256': sha(out / 'source_manifest.jsonl'), 'classes': names, 'taxonomy': taxonomy,
            'rows_total': len(rows), 'rows_exported': len(items), 'selection': selection,
            'image_mode': image_mode, 'items': items}
    save_json(out / 'package.json', info)
    write_manifest(out / 'decisions.jsonl', [])
    (out / 'classes.txt').write_text('\n'.join(names)+'\n', encoding='utf-8')
    render(out)
    return info


def decide(package, record_id, action, reviewer, note, *, complete=False,
           provenance_reviewed=False, verified_negative=False):
    """An explicit human command, not an inference from labels or model confidence."""
    package = Path(package)
    info = package_data(package)
    if action not in {'approve', 'reject', 'defer'}:
        raise ValueError('Unknown review action')
    if not reviewer.strip() or not note.strip():
        raise ValueError('Human reviewer identity and review note are required')
    item = next((i for i in info['items'] if i['id'] == record_id), None)
    if item is None:
        raise ValueError('Record is not in this package')
    image = inside(package, item['image']); label = inside(package, item['label'])
    if sha(image) != item['source_image_sha256']:
        raise ValueError('Packaged image changed')
    boxes = schema_for(info).read_labels(label)
    if action == 'approve':
        if not complete or not provenance_reviewed:
            raise ValueError('Approval requires explicit all-class coverage and provenance review')
        if not boxes and not verified_negative:
            raise ValueError('Empty labels require explicit verified-negative decision')
        if boxes and verified_negative:
            raise ValueError('Positive boxes cannot be a verified negative')
    decision = {'id': record_id, 'package_id': info['package_id'], 'action': action,
                'reviewer': reviewer.strip(), 'reviewed_at': now(), 'note': note.strip(),
                'coverage_complete': complete, 'provenance_reviewed': provenance_reviewed,
                'verified_negative': verified_negative,
                'reviewed_image_sha256': sha(image), 'reviewed_label_sha256': sha(label)}
    decisions = [d for d in read_manifest(package / 'decisions.jsonl') if d['id'] != record_id]
    write_manifest(package / 'decisions.jsonl', decisions + [decision])
    with (package / 'decisions_audit.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(decision, ensure_ascii=False)+'\n')
    return decision


def apply(package, output):
    package = Path(package); out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty reviewed dataset directory')
    info = package_data(package)
    decisions = read_manifest(package / 'decisions.jsonl')
    if len({d['id'] for d in decisions}) != len(decisions):
        raise ValueError('Duplicate decisions')
    items = {i['id']: i for i in info['items']}
    rows = read_manifest(package / 'source_manifest.jsonl')
    by_id = {r['id']: r for r in rows}
    # Validate the entire batch before creating any approved output.
    for decision in decisions:
        if decision['id'] not in items or decision.get('package_id') != info['package_id']:
            raise ValueError('Decision belongs to another package or unknown record')
        item = items[decision['id']]; row = by_id[decision['id']]
        if sha(row['image']) != item['source_image_sha256'] or sha(row['label']) != item['source_label_sha256']:
            raise ValueError(f'Source image/labels changed: {decision["id"]}; export a new package')
        if sha(inside(package, item['image'])) != decision.get('reviewed_image_sha256') or decision.get('reviewed_image_sha256') != item['source_image_sha256']:
            raise ValueError('Reviewed image changed after decision')
        label = inside(package, item['label'])
        if sha(label) != decision.get('reviewed_label_sha256'):
            raise ValueError('Reviewed labels changed after decision; refresh and decide again')
        boxes = schema_for(info).read_labels(label)
        if decision.get('action') not in {'approve', 'reject', 'defer'}:
            raise ValueError('Unknown decision action')
        if not str(decision.get('reviewer', '')).strip() or not str(decision.get('note', '')).strip():
            raise ValueError('Reviewer and note are required')
        reviewed_at = datetime.fromisoformat(decision['reviewed_at'])
        if reviewed_at.tzinfo is None:
            raise ValueError('Review timestamp must include timezone')
        if decision['action'] == 'approve':
            if decision.get('coverage_complete') is not True or decision.get('provenance_reviewed') is not True:
                raise ValueError('Approval missing complete coverage or provenance review')
            if not boxes and decision.get('verified_negative') is not True:
                raise ValueError('Unverified negative')
            if boxes and decision.get('verified_negative'):
                raise ValueError('Negative decision contains boxes')
            if len({tuple(box) for box in boxes}) != len(boxes):
                raise ValueError('Duplicate boxes: correct labels before approving')
    out.mkdir(parents=True, exist_ok=True)
    applied = Counter()
    for decision in decisions:
        row = by_id[decision['id']]; item = items[decision['id']]
        corrected = out / 'labels' / f'{item["key"]}.txt'
        corrected.parent.mkdir(exist_ok=True)
        shutil.copyfile(inside(package, item['label']), corrected)
        row.update(label=str(corrected.resolve()),
                   review_status={'approve': 'approved', 'reject': 'rejected', 'defer': 'needs_review'}[decision['action']],
                   coverage_complete=decision['action'] == 'approve',
                   verified_negative=decision['action'] == 'approve' and decision.get('verified_negative', False),
                   provenance_reviewed=decision.get('provenance_reviewed', False),
                   reviewer=decision['reviewer'], reviewed_at=decision['reviewed_at'],
                   review_reason=decision['note'], reviewed_image_sha256=decision['reviewed_image_sha256'],
                   reviewed_label_sha256=decision['reviewed_label_sha256'],
                   review_package_id=info['package_id'])
        applied[decision['action']] += 1
    write_manifest(out / 'manifest.jsonl', rows)
    write_manifest(out / 'review_audit.jsonl', decisions)
    report = {'package_id': info['package_id'], 'applied_at': now(), 'decisions': dict(applied),
              'records_preserved': len(rows), 'groups_and_original_splits_preserved': True,
              'training_ready': False, 'next': 'Run quality and grouped split, then training gates',
              'manifest_sha256': sha(out / 'manifest.jsonl')}
    save_json(out / 'review_report.json', report)
    return report


def create_server(package, port=8765):
    """Serve one local package and persist explicit edits/decisions."""
    package = Path(package).resolve(); info = package_data(package); schema = schema_for(info)
    items = {item['id']: item for item in info['items']}
    state_lock = RLock()

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            super().end_headers()

        def do_GET(self):
            if urlsplit(self.path).path in {'/', '/index.html'}:
                try:
                    with state_lock:
                        body = render(package, regenerate_previews=False, write_page=False).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (ValueError, OSError, KeyError) as exc:
                    self.send_json(500, {'error': str(exc)})
                return
            super().do_GET()

        def send_json(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            with state_lock:
                self.persist_POST()

        def persist_POST(self):
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length < 2 or length > 1_000_000:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(length))
                if body.get('package_id') != info['package_id'] or body.get('id') not in items:
                    raise ValueError('Unknown package or record')
                if self.path == '/api/labels':
                    label = inside(package, items[body['id']]['label'])
                    schema.write_labels(label, body.get('boxes', []))
                    self.send_json(200, {'saved': True, 'sha256': sha(label)}); return
                if self.path == '/api/decision':
                    action = body.get('action')
                    mapped = 'approve' if action == 'negative' else action
                    decision = decide(package, body['id'], mapped, str(body.get('reviewer', '')),
                                      str(body.get('note', '')), complete=mapped == 'approve',
                                      provenance_reviewed=mapped == 'approve',
                                      verified_negative=action == 'negative')
                    self.send_json(200, decision); return
                self.send_json(404, {'error': 'Unknown endpoint'})
            except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
                self.send_json(400, {'error': str(exc)})

    handler = partial(Handler, directory=str(package))
    return ThreadingHTTPServer(('127.0.0.1', port), handler)


def serve(package, port=8765):
    server = create_server(package, port)
    print(f'Editor local: http://127.0.0.1:{port}/index.html', flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('export'); p.add_argument('--manifest', nargs='+', required=True)
    p.add_argument('--output', required=True); p.add_argument('--limit', type=int, default=120)
    p.add_argument('--classes', nargs='+', choices=sorted(set(campus_schema.CLASSES + campus22_schema.CLASSES)))
    p.add_argument('--taxonomy', choices=['legacy11', 'campus-v4', 'campus22'], default='legacy11')
    p.add_argument('--selection', choices=['rare-first', 'balanced', 'manifest-order'], default='rare-first')
    p.add_argument('--image-mode', choices=['copy', 'hardlink'], default='copy')
    p = sub.add_parser('refresh'); p.add_argument('--package', required=True)
    p = sub.add_parser('decide'); p.add_argument('--package', required=True); p.add_argument('--id', required=True)
    p.add_argument('--action', choices=['approve', 'reject', 'defer'], required=True)
    p.add_argument('--reviewer', required=True); p.add_argument('--note', required=True)
    p.add_argument('--complete', action='store_true'); p.add_argument('--provenance-reviewed', action='store_true')
    p.add_argument('--verified-negative', action='store_true')
    p = sub.add_parser('apply'); p.add_argument('--package', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('serve'); p.add_argument('--package', required=True); p.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if args.command == 'export':
        info = export(args.manifest, args.output, args.limit, args.classes, args.selection, args.taxonomy,
                      args.image_mode)
        print(json.dumps({'exported': info['rows_exported'], 'gallery': str(Path(args.output)/'index.html')}))
    elif args.command == 'refresh': render(args.package)
    elif args.command == 'decide':
        print(json.dumps(decide(args.package, args.id, args.action, args.reviewer, args.note,
                                complete=args.complete, provenance_reviewed=args.provenance_reviewed,
                                verified_negative=args.verified_negative), ensure_ascii=False))
    elif args.command == 'apply': print(json.dumps(apply(args.package, args.output)))
    else: serve(args.package, args.port)


if __name__ == '__main__':
    main()
