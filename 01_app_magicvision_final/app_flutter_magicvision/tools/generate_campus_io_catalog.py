"""Documenta entradas/salidas actuales y ejemplos de campus; no ejecuta modelos."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import xlsxwriter

import generate_cerm_workbook as cerm


APP = Path(__file__).resolve().parents[1]
ROOT = APP.parents[1]
BASE = ROOT / "MAGICVISION_INPUTS_OUTPUTS_CAMPUS"
SOURCE = (APP / "lib/main.dart").read_text(encoding="utf-8")

# Rectángulos sintéticos normalizados: left, top, right, bottom.
BOXES = {
    "lejos_izquierda": [0.10, 0.20, 0.20, 0.40],
    "lejos_frente": [0.45, 0.20, 0.55, 0.40],
    "lejos_derecha": [0.80, 0.20, 0.90, 0.40],
    "cerca_izquierda": [0.05, 0.45, 0.30, 0.85],
    "cerca_frente": [0.35, 0.45, 0.65, 0.85],
    "cerca_derecha": [0.70, 0.45, 0.95, 0.85],
    "grande_derecha": [0.60, 0.20, 1.00, 0.85],
    "inferior_izquierda": [0.05, 0.65, 0.25, 0.85],
    "inferior_frente": [0.40, 0.65, 0.60, 0.85],
    "inferior_derecha": [0.75, 0.65, 0.95, 0.85],
}


def d(label: str, box: str, confidence: float = 0.90) -> dict:
    return {"label": label, "confidence": confidence, "bbox": BOXES[box]}


# El mensaje esperado sirve para detectar errores al editar la matriz documental.
# Las frases VLM son ejemplos redactados, NO resultados de inferencia.
CASES = [
    ("Pasillo", "Sin detecciones", [], "LOW_SAFE", "Se observa un pasillo sin obstáculos evidentes en la imagen."),
    ("Patio", "Persona pequeña a la izquierda", [d("person", "lejos_izquierda")], "LOW_SAFE", "Veo una persona a la izquierda."),
    ("Pasillo", "Persona pequeña al frente", [d("person", "lejos_frente")], "LOW_SAFE", "Veo una persona al frente."),
    ("Biblioteca", "Persona pequeña a la derecha", [d("person", "lejos_derecha")], "LOW_SAFE", "Veo una persona a la derecha."),
    ("Pasillo", "Persona aparentemente cercana a la izquierda", [d("person", "cerca_izquierda")], "M_PERSON", "Veo una persona a la izquierda que parece cercana."),
    ("Acceso al aula", "Persona aparentemente cercana al frente", [d("person", "cerca_frente")], "M_PERSON", "Veo una persona al frente que parece cercana."),
    ("Cafetería", "Persona aparentemente cercana a la derecha", [d("person", "cerca_derecha")], "M_PERSON", "Veo una persona a la derecha que parece cercana."),
    ("Pasillo", "Dos personas, izquierda y derecha", [d("person", "cerca_izquierda"), d("person", "cerca_derecha")], "M_PERSON", "Veo dos personas, una a la izquierda y otra a la derecha."),
    ("Patio", "Tres personas", [d("person", "cerca_izquierda"), d("person", "cerca_frente"), d("person", "cerca_derecha")], "M_PERSON", "Veo tres personas repartidas a la izquierda, al frente y a la derecha."),
    ("Bicicletero", "Bicicleta pequeña a un lado", [d("bicycle", "lejos_derecha")], "LOW_SAFE", "Veo una bicicleta a la derecha."),
    ("Sendero", "Bicicleta aparentemente cercana a la izquierda", [d("bicycle", "cerca_izquierda")], "M_BICYCLE", "Veo una bicicleta a la izquierda que parece cercana."),
    ("Sendero", "Bicicleta aparentemente cercana al frente", [d("bicycle", "cerca_frente")], "M_BICYCLE", "Veo una bicicleta al frente que parece cercana."),
    ("Sendero", "Bicicleta aparentemente cercana a la derecha", [d("bicycle", "cerca_derecha")], "M_BICYCLE", "Veo una bicicleta a la derecha que parece cercana."),
    ("Pabellón", "Escalera pequeña a la izquierda", [d("stairs", "lejos_izquierda")], "H_STAIRS", "Veo una posible escalera a la izquierda."),
    ("Pabellón", "Escalera al frente", [d("stairs", "cerca_frente")], "H_STAIRS", "Veo una posible escalera al frente."),
    ("Pabellón", "Escalera pequeña a la derecha", [d("stairs", "lejos_derecha")], "H_STAIRS", "Veo una posible escalera a la derecha."),
    ("Patio", "Hueco fuera de la zona inferior", [d("pothole", "lejos_frente")], "M_GENERIC", "Veo un posible hueco al frente."),
    ("Sendero", "Hueco abajo a la izquierda", [d("pothole", "inferior_izquierda")], "M_GENERIC", "Veo un posible hueco a la izquierda."),
    ("Sendero", "Hueco abajo al frente", [d("pothole", "inferior_frente")], "H_POTHOLE", "Veo un posible hueco al frente, en la parte inferior de la imagen."),
    ("Sendero", "Hueco abajo a la derecha", [d("pothole", "inferior_derecha")], "M_GENERIC", "Veo un posible hueco a la derecha."),
    ("Patio", "Poste a la izquierda", [d("pole", "cerca_izquierda")], "LOW_SAFE", "Veo un poste a la izquierda."),
    ("Patio", "Poste centrado", [d("pole", "cerca_frente")], "M_POLE", "Veo un poste al frente."),
    ("Patio", "Poste a la derecha", [d("pole", "cerca_derecha")], "LOW_SAFE", "Veo un poste a la derecha."),
    ("Vía interna", "Señal de tránsito", [d("traffic_sign", "lejos_derecha")], "M_SIGN", "Veo una señal de tránsito a la derecha."),
    ("Pasillo", "Obstáculo fuera de la zona inferior", [d("obstacle", "lejos_frente")], "M_GENERIC", "Veo un obstáculo al frente."),
    ("Acceso al aula", "Obstáculo abajo y centrado", [d("obstacle", "inferior_frente")], "H_OBSTACLE", "Veo un obstáculo al frente, en la zona de paso."),
    ("Pasillo", "Obstáculo abajo a la izquierda", [d("obstacle", "inferior_izquierda")], "M_GENERIC", "Veo un obstáculo a la izquierda."),
    ("Estacionamiento", "Auto pequeño a un lado", [d("car", "lejos_izquierda")], "M_VEHICLE", "Veo un auto a la izquierda."),
    ("Estacionamiento", "Auto centrado, área 0.12", [d("car", "cerca_frente")], "H_VEHICLE", "Veo un auto al frente que parece cercano."),
    ("Estacionamiento", "Auto lateral grande, área 0.26", [d("car", "grande_derecha")], "H_VEHICLE", "Veo un auto a la derecha que ocupa gran parte de la imagen."),
    ("Vía interna", "Motocicleta pequeña a un lado", [d("motorcycle", "lejos_derecha")], "M_VEHICLE", "Veo una motocicleta a la derecha."),
    ("Vía interna", "Motocicleta centrada, área 0.12", [d("motorcycle", "cerca_frente")], "H_VEHICLE", "Veo una motocicleta al frente que parece cercana."),
    ("Paradero interno", "Bus pequeño a un lado", [d("bus", "lejos_izquierda")], "M_VEHICLE", "Veo un autobús a la izquierda."),
    ("Paradero interno", "Bus centrado, área 0.12", [d("bus", "cerca_frente")], "H_VEHICLE", "Veo un autobús al frente que parece cercano."),
    ("Sendero", "Dos huecos inferiores, uno a cada lado", [d("pothole", "inferior_izquierda"), d("pothole", "inferior_derecha")], "H_MULTI", "Veo dos posibles huecos, uno a la izquierda y otro a la derecha."),
    ("Pabellón", "Escalera y persona", [d("stairs", "cerca_frente"), d("person", "cerca_derecha")], "H_STAIRS", "Veo una posible escalera al frente y una persona a la derecha."),
    ("Patio en mantenimiento", "Hueco y obstáculo inferiores", [d("pothole", "inferior_izquierda"), d("obstacle", "inferior_frente")], "H_MULTI", "Veo un posible hueco a la izquierda y un obstáculo al frente."),
    ("Estacionamiento", "Dos vehículos laterales aparentemente cercanos", [d("car", "cerca_izquierda"), d("motorcycle", "cerca_derecha")], "H_MULTI", "Veo un auto a la izquierda y una motocicleta a la derecha."),
    ("Patio", "Cuatro tipos: dos personas, bicicleta, señal y poste", [d("person", "cerca_izquierda"), d("person", "cerca_derecha"), d("bicycle", "lejos_izquierda"), d("traffic_sign", "lejos_derecha"), d("pole", "lejos_frente", 0.95)], "M_POLE", "Veo dos personas, una a la izquierda y otra a la derecha. También se observa un poste al frente."),
    ("Pasillo", "Persona con confianza 0.60", [d("person", "cerca_frente", 0.60)], "M_PERSON", "Podría haber una persona al frente, si la imagen permite distinguirla."),
    ("Sendero", "Hueco con confianza 0.40", [d("pothole", "inferior_frente", 0.40)], "H_POTHOLE", "Veo un posible hueco al frente."),
    ("Pasillo", "Obstáculo con confianza 0.40", [d("obstacle", "inferior_frente", 0.40)], "H_OBSTACLE", "Podría haber un obstáculo al frente, si resulta evidente en la imagen."),
]


def assess_boxes(detections: list[dict]) -> dict:
    inputs = []
    for item in detections:
        left, top, right, bottom = item["bbox"]
        assert 0 <= left < right <= 1 and 0 <= top < bottom <= 1
        cx, cy = (left + right) / 2, (top + bottom) / 2
        inputs.append({
            "label": item["label"], "confidence": item["confidence"],
            "center": 0.35 <= cx <= 0.65,
            "bottom_area": cy >= 0.55 or bottom >= 0.68,
            "bottom_ge_070": bottom >= 0.70,
            "area": (right - left) * (bottom - top),
        })
    return cerm.assess(inputs)


def hint(detections: list[dict]) -> str:
    labels = {"person": "persona", "car": "auto", "motorcycle": "motocicleta",
              "bus": "autobus", "pothole": "bache"}
    reliable = [x for x in detections if x["confidence"] >= (
        0.65 if x["label"] == "person" else 0.30 if x["label"] == "pothole" else 0.55)]
    counts = Counter(labels.get(x["label"], x["label"].replace("_", " ")) for x in reliable)
    parts = [label if n == 1 else f"{n} {label}" for label, n in counts.most_common(3)]
    return ("El detector rapido ve: " + ", ".join(parts) + ".") if parts else (
        "El detector rapido no encontro objetos confiables todavia.")


def table(headers: list[str], rows: list[list]) -> str:
    def cell(value):
        return str(value).replace("|", " / ").replace("\n", "<br>")
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(cell(x) for x in row) + " |" for row in rows),
    ])


def main():
    cerm.validate_source_alignment(cerm.build_single_scenarios(), cerm.build_pair_scenarios())
    prompt_function = SOURCE.split("String mobileVlmPromptForDetections(", 1)[1].split("String extractMobileVlmText", 1)[0]
    prompt = "".join(re.findall(r"'([^']*)'", prompt_function.split("return ", 1)[1]))
    assert prompt.endswith("$hints")
    prompt = prompt[:-6] + "{RESUMEN_YOLO}"
    for marker in ["sceneSpeechPersonMinConfidence = 0.65", "sceneSpeechMinConfidence = 0.55", "confidenceThreshold = 0.30", "Duration(seconds: 2)"]:
        assert marker in SOURCE

    inputs = [
        ["Modo rápido", "Captura de cámara → YOLO → CERM", "Clases, confianza y cajas normalizadas", "Riesgo, descripción, recomendación y acción; voz con plantilla"],
        ["Generador visual", "Nueva imagen + prompt + resumen textual de YOLO reciente", "No recibe bbox, riesgo, distancia ni acción de CERM como campos", "Texto libre de descripción; máximo dos frases solicitado por prompt"],
        ["Mensaje completo", "Resultado CERM disponible + texto del generador", "Se combinan después de generar", "Confirmado por YOLO: … Objetos: … Recomendacion: … Contexto visual: …"],
        ["Lectura OCR (separada)", "Imagen capturada en modo lectura", "Texto visible, si el OCR lo reconoce", "Texto reconocido leído en voz alta; no usa el generador visual"],
    ]
    constraints = [
        ["Imagen", "Nueva captura, enviada como ruta local o como imagen base64 al endpoint. No es necesariamente el mismo fotograma usado por YOLO."],
        ["Resumen de YOLO", "Detecciones recientes (ventana de 2 s), filtradas por confianza; máximo tres tipos de objetos, ordenados por cantidad. No se envían coordenadas."],
        ["Umbrales del resumen", "Persona ≥ 0.65; pothole/bache ≥ 0.30; las otras ocho clases configuradas ≥ 0.55. Son filtros del resumen VLM, no umbrales de las reglas CERM."],
        ["Modelo remoto", "LiquidAI/LFM2.5-VL-450M; temperature 0.1, top_p 0.9, max_tokens 110. Requiere endpoint configurado."],
        ["Modelo local", "LiquidAI/LFM2-VL-1.6B; temperature 0.2, topP 0.9, maxTokens 110. Requiere archivos y carga disponibles."],
        ["Prompt de sistema remoto", "Eres una asistente visual para una persona con baja vision. Prioriza seguridad peatonal, obstaculos, personas, vehiculos, escaleras, huecos, postes y senales. Responde en espanol latino, natural, concreto y sin inventar."],
        ["Prompt de sistema local", "Eres una asistente visual para una persona con baja vision. Describe solo lo que puedas inferir de la imagen. Responde en espanol latino con una frase corta y util."],
        ["Salida del modelo", "Texto libre, limpiado de espacios y algunos prefijos. El máximo de dos frases es una instrucción, no una validación estricta; el límite de tokens puede truncar."],
        ["Qué sabe de la universidad", "El prompt actual no identifica la universidad, pabellón, aula ni plano. Esos lugares son contexto del caso de prueba, no campos enviados al modelo."],
    ]
    classes = [
        ["person", "Persona", "Cerca: medio; resto: bajo"],
        ["car", "Auto", "Área ≥ 0.18, o central y área ≥ 0.08: alto; resto: medio"],
        ["motorcycle", "Motocicleta", "Misma regla de vehículos"],
        ["bus", "Bus", "Misma regla de vehículos"],
        ["bicycle", "Bicicleta", "Cerca: medio; resto: bajo"],
        ["stairs", "Escalera", "Alto, cualquiera sea su posición o área"],
        ["pothole", "Hueco", "Central e inferior: alto; resto: medio"],
        ["pole", "Poste", "Central: medio; resto: bajo"],
        ["traffic_sign", "Señal de tránsito", "Medio"],
        ["obstacle", "Obstáculo genérico", "Central e inferior: alto; resto: medio"],
    ]
    rules = [
        ["Área", "(right-left) × (bottom-top), con coordenadas entre 0 y 1"],
        ["Central CERM", "0.35 ≤ centroX ≤ 0.65"],
        ["Inferior CERM", "centroY ≥ 0.55 o bottom ≥ 0.68"],
        ["Zona de peligro", "Central e inferior a la vez"],
        ["Cercanía CERM", "Área ≥ 0.06 o bottom ≥ 0.70; no es distancia física medida"],
        ["Varios peligros", "Dos o más elementos potencialmente peligrosos → alto: escaleras, huecos/obstáculos inferiores o vehículos cercanos. Personas y bicicletas no se cuentan en esta condición."],
        ["Selección de mensaje", "En riesgo alto/medio se usa el objeto con mayor puntaje: peso de clase×100 + peso de zona×10 + área + confianza. Puede ser distinto del objeto que activó la regla."],
        ["Pesos", "Clase: hueco/obstáculo/escalera=5; vehículos=4; bicicleta/poste=3; persona/señal=2. Zona: peligro=3; solo central=1.5; otra=0."],
        ["Dirección en funciones auxiliares", "centroX < 0.38: izquierda; > 0.62: derecha; resto: frente. No confundir estos límites con la zona central de CERM. No se envían en el resumen VLM."],
    ]
    cases = []
    raw_inputs = []
    reached = set()
    for n, (place, scene, detections, expected, sample) in enumerate(CASES, 1):
        case_id = f"U{n:02d}"
        result = assess_boxes(detections)
        assert result["message_id"] == expected, (case_id, result, expected)
        reached.add(expected)
        cases.append([case_id, place, scene, hint(detections), result["risk"], expected, result["speech"], sample])
        raw_inputs.append([case_id, json.dumps(detections, ensure_ascii=False), result["objects"], result["dangerous_count"]])
    assert reached == {m["id"] for m in cerm.MESSAGES if m["id"] != "H_GENERIC"}
    assert {d["label"] for case in CASES for d in case[2]} == set(cerm.MODEL_LABELS)
    messages = [[m["id"], m["risk"], cerm.speech_text(m), m["reachable"]] for m in cerm.MESSAGES]
    limitations = [
        ["Silla, mesa, mochila, tacho, banca", "No son clases específicas del YOLO configurado. Pueden entrar como obstacle si el detector así los reconoce; no está garantizado. El VLM puede sugerirlos como posibles si son evidentes en la imagen."],
        ["Puerta, vidrio, rampa, ascensor", "Sin clase específica ni regla dedicada. No afirmar detección fiable ni estado abierto/cerrado como capacidad validada."],
        ["Piso mojado, cable, derrame, humo", "Sin clase específica ni regla dedicada; una descripción eventual del VLM no equivale a un detector validado para ese peligro."],
        ["Aula 302, biblioteca, laboratorio, baños", "El generador no recibe mapa ni identificación del lugar. OCR separado puede leer un rótulo visible; eso no proporciona navegación interior."],
        ["Letrero SALIDA o EVACUACIÓN", "traffic_sign no implica reconocimiento de cualquier rótulo universitario ni lectura de su contenido. Para leer texto existe el modo OCR."],
        ["Profesor, alumno o personal", "YOLO detecta person, sin determinar identidad, función o pertenencia a la universidad."],
        ["Auto acercándose o persona cruzando", "No se confirma movimiento con seguimiento temporal en este flujo. Cercanía aparente no demuestra aproximación."],
        ["Objeto a dos metros", "Las funciones auxiliares estiman distancia con tamaño aparente sin calibración física; no es una medida validada ni un campo enviado al generador."],
        ["Ruta segura a izquierda o derecha", "Sin mapa de transitabilidad que confirme una ruta alternativa libre. No atribuir esta garantía al generador."],
        ["Camino libre", "La plantilla de riesgo bajo dice El camino parece libre incluso con determinadas detecciones presentes. Es la salida actual; no prueba ausencia real de peligros."],
        ["Imagen cambiante", "YOLO y VLM pueden usar capturas distintas; los indicios YOLO pueden vencer a los 2 s. El modelo puede recibir imagen con resumen vacío o no coincidente temporalmente."],
        ["Fidelidad generativa", "El prompt pide no inventar; el código filtra algunas respuestas poco útiles, pero no verifica semánticamente todas las afirmaciones. Ningún ejemplo VLM de este documento es una salida garantizada."],
    ]
    states = [
        ["YOLO sin detecciones", "CERM entrega LOW_SAFE. Si se solicita VLM y está listo, usa imagen más El detector rapido no encontro objetos confiables todavia."],
        ["Detecciones bajo filtro VLM o vencidas", "El resumen VLM queda vacío; esto no significa necesariamente que CERM no haya detectado objetos."],
        ["Solo consulta rápida", "Se reproduce speechText de CERM; no se ejecuta generación visual."],
        ["Modelo visual no listo", "Se ofrece/reproduce el resumen rápido y se prepara o instala el modelo según el estado. No hay descripción generada disponible en ese momento."],
        ["Generación vacía, error o desconexión durante VLM", "La función devuelve null y no añade mensaje generativo; normalmente ya se emitió la respuesta rápida. La UI puede indicar Descripcion completa no disponible."],
        ["Respuesta VLM considerada poco útil", "Se usa el texto de respaldo construido con el análisis/detecciones disponibles en groundedMobileVlmSpeech."],
        ["Sin cámara disponible", "Se registra/indica Camara desconectada; no se obtiene un nuevo análisis válido. El anuncio de voz depende del flujo y del evento de conexión."],
        ["Error de red, respuesta inválida o captura caducada", "No hay AnalysisResult válido nuevo. Se comunica el error correspondiente; no se convierte el fallo en lista vacía ni en riesgo bajo."],
        ["Imagen oscura, borrosa u objeto tapado", "No hay una salida fija dedicada ni control explícito de calidad en este flujo. Puede producir omisiones, detecciones erróneas o texto incorrecto: debe comprobarse con capturas reales."],
        ["Solicitud concurrente o cancelada", "El flujo puede devolver null o invalidar el resultado; no se genera una nueva descripción válida para esa solicitud."],
        ["OCR sin texto", "En modo lectura: No encontre texto en la imagen."],
    ]
    source_paths = ["lib/main.dart", "lib/core/services/risk_assessment_service.dart", "lib/core/models/detection_result.dart", "lib/core/models/analysis_result.dart", "lib/core/services/vision_analysis_service.dart", "assets/models/labels.txt"]
    sources = [[p, hashlib.sha256((APP / p).read_bytes()).hexdigest()] for p in source_paths]
    case_headers = ["ID", "Lugar del ensayo", "Escena supuesta", "Resumen YOLO del INPUT VLM", "Riesgo CERM", "Plantilla", "OUTPUT rápido actual", "Ejemplo de OUTPUT VLM; no ejecutado"]
    sheets = [
        ("Flujos", ["Flujo", "INPUT", "Detalle", "OUTPUT"], inputs),
        ("Contrato VLM", ["Elemento", "Comportamiento actual"], constraints + [["Prompt de usuario literal", prompt]]),
        ("Clases", ["Clase", "Nombre", "Riesgo si aparece sola"], classes),
        ("Reglas", ["Concepto", "Regla actual"], rules),
        ("Casos campus", case_headers, cases),
        ("Inputs numericos", ["ID", "Detecciones sintéticas; bbox=[left,top,right,bottom]", "Objetos CERM", "Conteo interno de peligros"], raw_inputs),
        ("Mensajes fijos", ["ID", "Riesgo", "speechText literal", "Alcance"], messages),
        ("Limites", ["Situación", "Cobertura actual"], limitations),
        ("Estados", ["Entrada o estado", "Salida o comportamiento"], states),
        ("Fuentes", ["Archivo relativo a la app", "SHA256"], sources),
    ]
    introduction = f"""# MagicVision: entradas y salidas en un entorno universitario

Documento para revisión del profesor. Basado en el código disponible el 20/09/2026.

El INPUT actual del generador visual es una nueva imagen de cámara, instrucciones y un resumen textual de las detecciones recientes de YOLO. Su OUTPUT es texto de descripción. La aplicación combina después ese texto con el análisis de riesgo CERM y lo convierte a voz.

Se documentan las diez clases configuradas, las trece plantillas definidas en CERM (doce alcanzables con las clases y rangos normales), {len(cases)} casos de campus, estados de fallo y límites de cobertura. Las imágenes y las frases generativas posibles no forman una lista finita: esta es una cobertura por clases y ramas del programa con escenarios representativos, no un catálogo de todas las escenas o combinaciones físicas.

**Evidencia:** los casos son sintéticos. No se ha capturado ninguna imagen dentro de la universidad ni ejecutado YOLO, VLM o la app para producir estas filas. Los mensajes CERM se deducen de detecciones explícitas mediante el generador documental existente, cotejando sus plantillas con el código. Los textos VLM son ejemplos redactados para explicar qué podría producir; deben sustituirse por salidas observadas en una evaluación real.

El lugar (aula, biblioteca, patio) es contexto del ensayo. No es una variable que la aplicación envíe al generador, ni evidencia de que reconozca ese lugar.

## 1. Flujos actuales

{table(sheets[0][1], inputs)}

## 2. Todos los componentes de la entrada al generador

{table(sheets[1][1], constraints)}

Prompt de usuario actual, con el resumen insertado al final:

```text
{prompt}
```

Los ejemplos de resumen conservan el texto real del código: puede aparecer `2 persona`, `stairs`, `bicycle`, `pole`, `obstacle` o `traffic sign`. El resumen no traduce todas las clases ni pluraliza todas las etiquetas. El generador recibe ese texto y la imagen para redactar la descripción.

Entre clases con la misma cantidad no hay un criterio explícito de desempate. Puede variar el orden y, si hay más de tres tipos, qué clases empatadas entran en el resumen. Las filas muestran un orden posible; esta variación no cambia las reglas CERM.

## 3. Clases y reglas que determinan el resultado rápido

Las reglas de esta tabla suponen una única detección. En una escena con varios objetos se aplica la prioridad y la regla de múltiples peligros.

{table(sheets[2][1], classes)}

{table(sheets[3][1], rules)}

## 4. Matriz de escenarios dentro de la universidad

Cada fila presupone que YOLO entregó las detecciones numéricas del apartado 5, que siguen recientes cuando se arma el prompt, y que el modelo visual está listo si se solicita descripción completa. La escena visual tendría que concordar con esas detecciones para interpretar el ejemplo generativo. «Lejos» y «cerca» nombran tamaño/posición aparentes, no distancias reales.

La columna INPUT muestra únicamente la parte variable del prompt; el generador también recibe la imagen y las instrucciones del apartado 2. La columna OUTPUT rápido contiene la plantilla completa del código. La última columna es exclusivamente ilustrativa y NO la transcripción de una ejecución ni el mensaje completo final.

{table(case_headers, cases)}

## 5. Entradas numéricas para reproducir el análisis de cada caso

Estos son datos de detección para CERM, no el JSON enviado al VLM. Cada bbox usa el orden `[left, top, right, bottom]` con valores normalizados 0..1. La API real incorpora además `class_id` y metadatos; CERM usa etiqueta, confianza y rectángulo. Los conteos de peligros son internos: no equivalen a la cantidad de personas u objetos ni necesariamente se pronuncian.

{table(sheets[5][1], raw_inputs)}

## 6. Todas las plantillas de voz definidas en CERM

`speechText = descripción + recomendación + "Accion sugerida: " + acción`. El conteo de objetos y el nivel de riesgo son campos del resultado, pero no se insertan explícitamente en todas las frases habladas. El fallback alto H_GENERIC está definido como rama defensiva; no lo alcanza ninguna de las entradas normales con estas diez clases.

{table(sheets[6][1], messages)}

## 7. Cómo se forma el mensaje completo

Ejemplo U19, suponiendo que el VLM devolviera el texto ilustrativo de esa fila:

```text
Confirmado por YOLO: Se detecta un posible hueco en el camino. Objetos: pothole. Recomendacion: Detente y verifica antes de avanzar. Contexto visual: Veo un posible hueco al frente, en la parte inferior de la imagen.
```

La expresión «Confirmado por YOLO» es el prefijo literal de la aplicación; no significa certeza absoluta. El generador no recibe el riesgo ni la recomendación de CERM en su prompt: se incorporan después al mensaje hablado. Si no hay análisis CERM, el ensamblado puede usar el resumen de detecciones; si tampoco existe, puede devolver solo la descripción generada. El mensaje combinado puede superar las dos frases pedidas al VLM.

## 8. Situaciones universitarias fuera de la cobertura específica

{table(sheets[7][1], limitations)}

## 9. Entradas vacías, fallos y disponibilidad

{table(sheets[8][1], states)}

## 10. Qué falta para mostrar resultados reales al profesor

Para cada caso seleccionado: guardar la imagen usada por YOLO y la usada por el VLM, detecciones con sus tiempos, prompt efectivamente enviado, respuesta cruda del generador, mensaje final de voz y latencia. Registrar también lo observado en la escena para comparar conteo, ubicación, omisiones y afirmaciones sin respaldo. Actualmente esta matriz demuestra el contrato y las ramas del código; no acredita precisión, seguridad o latencia en el campus.

Ya existe un inventario más amplio de las combinaciones de reglas CERM en [CERM_entradas_salidas_y_nivel_de_detalle_FINAL.xlsx](CERM_entradas_salidas_y_nivel_de_detalle_FINAL.xlsx). Este documento lo complementa con la entrada del generador y escenas universitarias.

## 11. Trazabilidad

Fuentes relativas a `01_app_magicvision_final/app_flutter_magicvision`:

{table(sheets[9][1], sources)}

Reproducción: `python 01_app_magicvision_final/app_flutter_magicvision/tools/generate_campus_io_catalog.py`. El script genera este documento y el Excel compañero; no modifica el comportamiento de la aplicación.
"""
    BASE.with_suffix(".md").write_text(introduction, encoding="utf-8")
    with xlsxwriter.Workbook(str(BASE.with_suffix(".xlsx"))) as workbook:
        title = workbook.add_format({"bold": True, "font_size": 16, "font_color": "#17365D"})
        note = workbook.add_format({"text_wrap": True, "valign": "top", "bg_color": "#FFF2CC"})
        cell = workbook.add_format({"text_wrap": True, "valign": "top"})
        for name, headers, rows in sheets:
            ws = workbook.add_worksheet(name)
            last = len(headers) - 1
            ws.merge_range(0, 0, 0, last, f"MagicVision · {name}", title)
            ws.merge_range(1, 0, 2, last, "Código al 20/09/2026. Casos sintéticos: CERM deducido del código; ejemplos VLM redactados, NO inferencias observadas. Consulte el documento Markdown para alcance y limitaciones.", note)
            ws.add_table(4, 0, 4 + len(rows), last, {"data": rows, "columns": [{"header": h, "format": cell} for h in headers], "style": "Table Style Medium 2"})
            ws.set_column(0, last, 32, cell)
            widths = [32] * len(headers)
            if name == "Casos campus":
                ws.set_column(0, 0, 8)
                ws.set_column(4, 5, 16)
                ws.set_column(6, 7, 60)
                widths = [8, 32, 32, 32, 16, 16, 60, 60]
            elif len(headers) == 2:
                ws.set_column(1, 1, 110)
                widths[1] = 110
            elif name == "Mensajes fijos":
                ws.set_column(2, 2, 90)
                widths[2] = 90
            elif name == "Inputs numericos":
                ws.set_column(1, 1, 110)
                widths[1] = 110
            for row, values in enumerate(rows, 5):
                lines = max(math.ceil(len(str(value)) / (width * 0.8)) for value, width in zip(values, widths))
                ws.set_row(row, min(409, max(70, lines * 16 + 12)))
            ws.freeze_panes(5, 1)
            ws.set_landscape()
            ws.fit_to_pages(1, 0)
            ws.repeat_rows(4)
            ws.set_paper(9)
    print(json.dumps({"cases": len(cases), "classes": len(cerm.MODEL_LABELS), "defined_templates": len(messages), "normal_templates_covered": len(reached), "files": [str(BASE.with_suffix(ext)) for ext in (".md", ".xlsx")]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
