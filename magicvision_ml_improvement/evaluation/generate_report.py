"""Rebuild measured experiment tables; never select a new checkpoint implicitly."""
from pathlib import Path
import json
import csv
ROOT=Path(__file__).resolve().parents[1]


def main():
    rows=[]
    for folder in sorted((ROOT/'reports').glob('E0_*')):
        if not (folder/'run.json').exists():continue
        run=json.loads((folder/'run.json').read_text())
        for score in (.15,.20,.25,.30):
            result=json.loads((folder/f'metrics_{score:.2f}.json').read_text())
            for name,metrics in result['classes'].items():
                rows.append({'experiment':folder.name,'class':name,'threshold':score,'images':result['images'],
                    **metrics,'latency_mean_ms':run['latency_mean_ms'],'latency_p95_ms':run['latency_p95_ms']})
    with (ROOT/'reports/comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=['# Comparación medida del checkpoint operativo','',
        'Mismo checkpoint e imágenes de validación. No se ha usado el conjunto test para seleccionar parámetros.',
        'Las métricas usan correspondencia por clase, IoU ≥ 0.50 y asignación uno a uno en orden de confianza. AP se integra en 101 puntos; no es una reproducción exacta del evaluador histórico.',
        'La compresión app640 es una aproximación PIL, no una captura del S25. Las latencias locales incluyen preprocesamiento del predictor; hubo trabajo concurrente de CPU/disco, por lo que no demuestran rendimiento aislado ni latencia AWS.',
        '', '| Experimento | Umbral | Clase | Precision | Recall | FN | FP/imagen |', '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        if row['class'] in ('person','stairs','obstacle'):
            summary.append(f"| {row['experiment']} | {row['threshold']:.2f} | {row['class']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['fn']} | {row['fp_per_image']:.4f} |")
    summary += ['', 'Decisión: conservar checkpoint y valores operativos 640 / 0.30. Las metas de recall no están alcanzadas en esta validación; falta el lote independiente del campus y límites aceptados de falsas alarmas/latencia. E1/E2/E3 no se presentan como ejecutados.',
        'La clase trash_bin no existe en este checkpoint: no confundir sus ceros/sin soporte con una evaluación de un detector nuevo. Las 10 anotaciones de escaleras en val son insuficientes para generalizar al campus.']
    (ROOT/'reports/comparison.md').write_text('\n'.join(summary),encoding='utf-8')
    print(f'Comparison: {len(rows)} class/threshold rows')


if __name__=='__main__':main()
