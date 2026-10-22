"""Full on-disk candidate audit. Heuristics flag review; never approve/delete boxes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
from PIL import Image, ImageDraw
import imagehash
import campus22_schema as schema
from campus_inventory import audit


def inspect_pixels(row):
    try:
        path=Path(row['image'])
        if schema.sha(path)!=row['sha256']:raise ValueError('SHA256 mismatch')
        with Image.open(path) as image:
            image.load()
            if image.size!=(row['width'],row['height']):raise ValueError('Dimensions mismatch')
            phash=str(imagehash.phash(image))
        return row['id'],phash,None
    except Exception as exc:return row['id'],None,str(exc)


def main(workers=4):
    active=json.loads((schema.ROOT/'configs/active_dataset.json').read_text(encoding='utf-8'))
    manifest=schema.ROOT.parent/active['manifest'];rows=schema.read_manifest(manifest)
    report=audit(rows,{r['sha256'] for r in rows},False,schema=schema)
    errors=[];by_id={r['id']:r for r in rows}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n,(rid,phash,error) in enumerate(pool.map(inspect_pixels,rows),1):
            if error:errors.append({'id':rid,'error':error})
            else:by_id[rid]['phash']=phash
            if n%2000==0:print(f'Pixels verified: {n}/{len(rows)}',flush=True)
    groups=schema.duplicate_groups(rows)
    conflicts=[]
    for group in groups:
        splits={rows[i].get('original_split') for i in group}&{'train','val','test'}
        if len(splits)>1:conflicts.append({'ids':[rows[i]['id'] for i in group],'source_splits':sorted(splits)})
    suspicious=[];door_rows=[]
    for row in rows:
        try:boxes=schema.read_labels(row['label'])
        except (OSError,ValueError):continue
        flags=[];doors=[b for b in boxes if b[0] in (13,14)]
        if doors:
            if len(doors)>4:flags.append('more_than_four_door_components')
            if any(b[3]*b[4]>.8 for b in doors):flags.append('door_box_over_80_percent_image')
            if any(b[3]*b[4]<.0004 for b in doors):flags.append('very_small_door_box')
            if {b[0] for b in doors}=={13,14}:flags.append('door_and_glass_door_same_image_check_instances')
            door_rows.append((row,doors,flags))
        if len({tuple(b) for b in boxes})!=len(boxes):flags.append('duplicate_box_geometry')
        if flags:suspicious.append({'id':row['id'],'image':row['image'],'label':row['label'],'flags':flags})
    root=schema.ROOT/'reports';root.mkdir(exist_ok=True)
    source_counts=Counter(r['source'] for r,_,_ in door_rows)
    source_schema=json.loads((schema.ROOT.parent/active['class_mapping']).read_text())
    if source_schema['names']!=schema.CLASSES:errors.append({'error':'Active mapping/schema mismatch'})
    sys.path.insert(0,str(schema.ROOT/'training'))
    from preflight import inspect_release
    release=inspect_release(schema.ROOT/'datasets/campus22_release_v1/data.yaml',schema=schema)
    report.update(READY_FOR_TRAINING=False,pixel_errors=errors,decoded_and_hashed_records=len(rows)-len(errors),
        recomputed_phash_records=len(rows)-len(errors),source_partition_conflicts=conflicts,
        suspicious_annotations=suspicious,door_sources=dict(source_counts),release_check=release,
        source_manifest_sha256=schema.sha(manifest),classes=schema.CLASSES,
        blockers=['Incomplete all-class annotation/provenance review','Source partition conflicts unresolved','No approved release with all 22 classes in train/val/test'],
        annotation_semantics='Heuristics and sample inspection only; human bounding-box review remains required',
        safe_automatic_corrections=['Renamed ID 10 trash_bin to trash_can in active contract; no pixel/box changes'])
    schema.save_json(root/'campus22_preflight_final.json',report)
    schema.save_json(root/'door_glass_door_suspicious.json',suspicious)
    text=f'''# Auditoría final campus22\n\nREADY_FOR_TRAINING = **false**\n\n- Registros: {len(rows)}; imágenes únicas: {report['unique_images']}.\n- Cajas/propuestas: {report['boxes_deduplicated_by_pixels_and_geometry']}.\n- Errores de etiquetas/archivos: {len(report['errors'])}; errores de píxeles/hash: {len(errors)}.\n- Imágenes sin cajas: {report['empty_label_records_not_necessarily_negatives']} (no son negativos aprobados).\n- Aprobaciones completas: {report['approved_complete_22_class_records']}.\n- Grupos con particiones de origen diferentes: {len(conflicts)}. No se resolvió leakage mediante barajado.\n- Cero imágenes liberadas en train/val/test; las 22 clases están ausentes del release aprobado.\n\nSe comprobaron existencia, SHA256, decodificación, dimensiones, etiquetas, IDs, límites, duplicados exactos y pHash recalculado. Se agruparon secuencias/origen cuando esos metadatos existen; no se infirieron sesiones independientes de fuentes sin metadatos. Las anotaciones sospechosas son una cola de revisión, no errores semánticos confirmados.\n\nLa corrección automática fue el nombre de ID 10 a trash_can, conservando ID y cajas. No se borraron datos ni se inventaron anotaciones. Falta completar revisión de las 22 clases y resolver grupos antes de liberar un split.\n\nEvidencia detallada: [JSON](campus22_preflight_final.json), [revisión de puertas](door_glass_door_quality.md).\n'''
    (root/'campus22_preflight_final.md').write_text(text,encoding='utf-8')
    selected=[]
    for cls in (13,14):
        relevant=[x for x in door_rows if any(b[0]==cls for b in x[1])]
        relevant.sort(key=lambda x:(not bool(x[2]),x[0]['id']))
        selected.extend(relevant[:8])
    sheet=Image.new('RGB',(1200,320*((len(selected)+2)//3)),'white')
    for index,(row,boxes,flags) in enumerate(selected):
        with Image.open(row['image']) as im:im=im.convert('RGB');im.thumbnail((390,275))
        draw=ImageDraw.Draw(im)
        for cls,x,y,w,h in boxes:draw.rectangle(((x-w/2)*im.width,(y-h/2)*im.height,(x+w/2)*im.width,(y+h/2)*im.height),outline='red',width=2)
        left=(index%3)*400;top=(index//3)*320;sheet.paste(im,(left,top+35))
        ImageDraw.Draw(sheet).text((left+2,top+2),row['id'][:48],fill='black')
    sheet.save(root/'door_glass_door_sample.jpg',quality=90)
    counts=[]
    for cls in (13,14):
        items=[(r,[b for b in boxes if b[0]==cls]) for r,boxes,_ in door_rows]
        counts.append((schema.CLASSES[cls],len({r['sha256'] for r,b in items if b}),sum(len(b) for _,b in items)))
    text='# Revisión específica: door y glass_door\n\nAmbas clases conservan prioridad crítica.\n\n| Clase | Imágenes con cajas | Cajas |\n|---|---:|---:|\n'+''.join(f'| {c} | {n} | {b} |\n' for c,n,b in counts)
    text+='\nLa comprobación geométrica no determina transparencia, apertura, reflejos ni identidad de una puerta. La fuente Open Images Door puede incluir puertas acristaladas; Trans10K deriva cajas por componentes y requiere reconciliar instancias. No se borró ninguna muestra por heurística.\n\n'
    text+='[Muestra con cajas](door_glass_door_sample.jpg). [Candidatos sospechosos](door_glass_door_suspicious.json).\n\n'
    text+='Revisar manualmente: puerta frente a ventana, vidrio frente a material opaco, puertas abiertas/cerradas/dobles, fragmentos del mismo objeto, perspectiva lateral/lejana, interior/exterior y reflejos. Estos atributos no tienen cobertura certificada en el manifiesto; su ausencia no debe inventarse.\n\n'
    text+='Acciones: conteo directo, detección de cajas grandes/pequeñas y múltiples componentes, preservación de originales y generación de muestra/cola. La aprobación semántica completa queda pendiente.\n'
    (root/'door_glass_door_quality.md').write_text(text,encoding='utf-8')
    print(json.dumps({'records':len(rows),'pixel_errors':len(errors),'READY_FOR_TRAINING':False}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);main(parser.parse_args().workers)
