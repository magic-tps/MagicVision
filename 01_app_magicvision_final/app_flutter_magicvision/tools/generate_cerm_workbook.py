"""Genera el inventario Excel exhaustivo del CERM implementado en MagicVision.

El libro documenta el comportamiento actual de RiskAssessmentService y, en un
apendice separado, el motor de riesgo en tiempo real de main.dart. No propone
ni simula una version futura del algoritmo.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import xlsxwriter


APP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = WORKSPACE_ROOT / "CERM_entradas_salidas_y_nivel_de_detalle_FINAL.xlsx"

MODEL_LABELS = [
    "person",
    "car",
    "motorcycle",
    "bus",
    "bicycle",
    "stairs",
    "pothole",
    "pole",
    "traffic_sign",
    "obstacle",
]
SCENARIO_LABELS = MODEL_LABELS + ["other_label"]
VEHICLES = {"car", "bus", "motorcycle"}
GROUND_HAZARDS = {"pothole", "obstacle"}


MESSAGES = [
    {
        "id": "LOW_SAFE",
        "risk": "bajo",
        "branch": "Lista vacia o fallback bajo",
        "leading": "Cualquiera / ninguno",
        "reachable": "Si",
        "description": "El camino parece libre.",
        "recommendation": "Puedes avanzar lentamente con cuidado.",
        "action": "Avanza con cuidado",
        "literal_condition": "detections.isEmpty, o ninguna regla alto/medio",
    },
    {
        "id": "H_MULTI",
        "risk": "alto",
        "branch": "Alto: varios peligros",
        "leading": "No importa",
        "reachable": "Si",
        "description": "Se detectan varios objetos peligrosos cerca.",
        "recommendation": "Detente y verifica el entorno antes de avanzar.",
        "action": "Detente",
        "literal_condition": "dangerous.length >= 2",
    },
    {
        "id": "H_POTHOLE",
        "risk": "alto",
        "branch": "Alto: leading pothole",
        "leading": "pothole",
        "reachable": "Si",
        "description": "Se detecta un posible hueco en el camino.",
        "recommendation": "Detente y verifica antes de avanzar.",
        "action": "Detente",
        "literal_condition": "highRisk y normalize(leading.label) == pothole",
    },
    {
        "id": "H_STAIRS",
        "risk": "alto",
        "branch": "Alto: leading stairs",
        "leading": "stairs",
        "reachable": "Si",
        "description": "Se detecta una posible escalera o desnivel.",
        "recommendation": "Avanza lentamente y usa apoyo.",
        "action": "Avanza con precaucion",
        "literal_condition": "highRisk y normalize(leading.label) == stairs",
    },
    {
        "id": "H_OBSTACLE",
        "risk": "alto",
        "branch": "Alto: leading obstacle",
        "leading": "obstacle",
        "reachable": "Si",
        "description": "Se detecta un obstaculo en la zona de paso.",
        "recommendation": "Detente y busca una ruta libre antes de avanzar.",
        "action": "Detente",
        "literal_condition": "highRisk y normalize(leading.label) == obstacle",
    },
    {
        "id": "H_VEHICLE",
        "risk": "alto",
        "branch": "Alto: leading vehiculo",
        "leading": "car | bus | motorcycle",
        "reachable": "Si",
        "description": "Se detecta un vehiculo cerca.",
        "recommendation": "Evita avanzar hacia esa direccion.",
        "action": "Mantente en zona segura",
        "literal_condition": "highRisk y leading es vehiculo",
    },
    {
        "id": "H_GENERIC",
        "risk": "alto",
        "branch": "Alto: fallback defensivo",
        "leading": "Otra etiqueta",
        "reachable": "No con YOLO normal; si por llamada directa fuera de rango",
        "description": "Se detecta un riesgo cerca.",
        "recommendation": "Detente y verifica antes de avanzar.",
        "action": "Detente",
        "literal_condition": "highRisk y leading no coincide con ramas anteriores",
    },
    {
        "id": "M_PERSON",
        "risk": "medio",
        "branch": "Medio: leading person",
        "leading": "person",
        "reachable": "Si",
        "description": "Se detecta una persona cerca.",
        "recommendation": "Avanza lentamente y manten distancia.",
        "action": "Avanza con precaucion",
        "literal_condition": "mediumRisk y normalize(leading.label) == person",
    },
    {
        "id": "M_BICYCLE",
        "risk": "medio",
        "branch": "Medio: leading bicycle",
        "leading": "bicycle",
        "reachable": "Si",
        "description": "Se detecta una bicicleta cerca.",
        "recommendation": "Mantente atento al movimiento lateral.",
        "action": "Avanza con precaucion",
        "literal_condition": "mediumRisk y normalize(leading.label) == bicycle",
    },
    {
        "id": "M_POLE",
        "risk": "medio",
        "branch": "Medio: leading pole",
        "leading": "pole",
        "reachable": "Si",
        "description": "Se detecta un poste cerca del centro.",
        "recommendation": "Corrige la trayectoria antes de avanzar.",
        "action": "Desviate con cuidado",
        "literal_condition": "mediumRisk y normalize(leading.label) == pole",
    },
    {
        "id": "M_SIGN",
        "risk": "medio",
        "branch": "Medio: leading traffic_sign",
        "leading": "traffic_sign",
        "reachable": "Si",
        "description": "Se detecta una senal de transito.",
        "recommendation": "Usala como referencia y avanza con cuidado.",
        "action": "Avanza con cuidado",
        "literal_condition": "mediumRisk y normalize(leading.label) == traffic_sign",
    },
    {
        "id": "M_VEHICLE",
        "risk": "medio",
        "branch": "Medio: leading vehiculo",
        "leading": "car | bus | motorcycle",
        "reachable": "Si",
        "description": "Se detecta un vehiculo en el entorno.",
        "recommendation": "Mantente en zona segura y evita acercarte.",
        "action": "Mantente en zona segura",
        "literal_condition": "mediumRisk y leading es vehiculo",
    },
    {
        "id": "M_GENERIC",
        "risk": "medio",
        "branch": "Medio: fallback",
        "leading": "pothole | obstacle | otra",
        "reachable": "Si",
        "description": "Se detectan objetos en el entorno.",
        "recommendation": "Avanza lentamente con atencion.",
        "action": "Avanza con cuidado",
        "literal_condition": "mediumRisk y leading no coincide con ramas anteriores",
    },
]
MESSAGE_BY_ID = {message["id"]: message for message in MESSAGES}


def normalize_label(label: str) -> str:
    return label.lower().strip().replace("-", "_").replace(" ", "_")


def label_weight(label: str) -> float:
    if label in {"pothole", "obstacle", "stairs"}:
        return 5.0
    if label in VEHICLES:
        return 4.0
    if label in {"bicycle", "pole"}:
        return 3.0
    if label in {"person", "traffic_sign"}:
        return 2.0
    return 1.0


def enrich_detection(detection: dict[str, Any]) -> dict[str, Any]:
    item = dict(detection)
    item["label"] = normalize_label(str(item["label"]))
    center = bool(item["center"])
    bottom_area = bool(item["bottom_area"])
    area = float(item["area"])
    bottom_ge_070 = bool(item.get("bottom_ge_070", False))
    item["danger_zone"] = center and bottom_area
    item["near"] = area >= 0.06 or bottom_ge_070
    item["vehicle"] = item["label"] in VEHICLES
    item["potentially_dangerous"] = (
        item["label"] == "stairs"
        or (item["label"] in GROUND_HAZARDS and bottom_area)
        or (item["vehicle"] and item["near"])
    )
    zone_weight = 3.0 if item["danger_zone"] else 1.5 if center else 0.0
    item["danger_score"] = (
        label_weight(item["label"]) * 100
        + zone_weight * 10
        + area
        + float(item["confidence"])
    )
    return item


def high_triggers(detections: list[dict[str, Any]]) -> list[str]:
    dangerous_count = sum(bool(item["potentially_dangerous"]) for item in detections)
    triggers: list[str] = []
    if dangerous_count >= 2:
        triggers.append("dangerous.length >= 2")
    if any(item["label"] == "stairs" for item in detections):
        triggers.append("stairs presente")
    if any(
        item["label"] in GROUND_HAZARDS and item["danger_zone"]
        for item in detections
    ):
        triggers.append("pothole/obstacle en dangerZone")
    if any(
        item["vehicle"]
        and (item["area"] >= 0.18 or (item["center"] and item["area"] >= 0.08))
        for item in detections
    ):
        triggers.append("vehiculo grande o centrado")
    return triggers


def medium_triggers(detections: list[dict[str, Any]]) -> list[str]:
    triggers: list[str] = []
    if any(item["label"] in {"person", "bicycle"} and item["near"] for item in detections):
        triggers.append("person/bicycle cerca")
    if any(item["label"] == "pole" and item["center"] for item in detections):
        triggers.append("pole centrado")
    if any(item["label"] == "traffic_sign" for item in detections):
        triggers.append("traffic_sign presente")
    if any(item["vehicle"] for item in detections):
        triggers.append("vehiculo presente")
    if any(
        item["label"] in GROUND_HAZARDS and not item["danger_zone"]
        for item in detections
    ):
        triggers.append("pothole/obstacle fuera de dangerZone")
    return triggers


def message_for(risk: str, leading: dict[str, Any] | None, dangerous_count: int) -> str:
    if risk == "bajo":
        return "LOW_SAFE"
    label = leading["label"] if leading is not None else ""
    if risk == "alto":
        if dangerous_count >= 2:
            return "H_MULTI"
        if label == "pothole":
            return "H_POTHOLE"
        if label == "stairs":
            return "H_STAIRS"
        if label == "obstacle":
            return "H_OBSTACLE"
        if label in VEHICLES:
            return "H_VEHICLE"
        return "H_GENERIC"
    if label == "person":
        return "M_PERSON"
    if label == "bicycle":
        return "M_BICYCLE"
    if label == "pole":
        return "M_POLE"
    if label == "traffic_sign":
        return "M_SIGN"
    if label in VEHICLES:
        return "M_VEHICLE"
    return "M_GENERIC"


def assess(detections: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = [enrich_detection(item) for item in detections]
    if not items:
        message = MESSAGE_BY_ID["LOW_SAFE"]
        return {
            "risk": "bajo",
            "message_id": "LOW_SAFE",
            "leading": "ninguno",
            "high_triggers": [],
            "medium_triggers": [],
            "dangerous_count": 0,
            "aggregate_confidence": 0.65,
            "objects": "sin objetos peligrosos",
            "speech": speech_text(message),
        }
    sorted_items = sorted(items, key=lambda item: item["danger_score"], reverse=True)
    dangerous_count = sum(bool(item["potentially_dangerous"]) for item in sorted_items)
    high = high_triggers(sorted_items)
    medium = [] if high else medium_triggers(sorted_items)
    risk = "alto" if high else "medio" if medium else "bajo"
    leading = sorted_items[0] if risk in {"alto", "medio"} else items[0]
    message_id = message_for(risk, leading, dangerous_count)
    counts = Counter(item["label"] for item in sorted_items)
    object_parts = [label if count == 1 else f"{label} x{count}" for label, count in counts.most_common()]
    top = sorted_items[:5]
    confidence = sum(float(item["confidence"]) for item in top) / len(top)
    message = MESSAGE_BY_ID[message_id]
    return {
        "risk": risk,
        "message_id": message_id,
        "leading": leading["label"],
        "high_triggers": high,
        "medium_triggers": medium,
        "dangerous_count": dangerous_count,
        "aggregate_confidence": confidence,
        "objects": ", ".join(object_parts),
        "speech": speech_text(message),
        "sorted": sorted_items,
    }


def speech_text(message: dict[str, Any]) -> str:
    return (
        f'{message["description"]} {message["recommendation"]} '
        f'Accion sugerida: {message["action"]}.'
    )


def humanized_speech_text(message: dict[str, Any]) -> str:
    """Mismo contenido de speechText, con tildes para lectura documental."""
    text = speech_text(message)
    replacements = {
        "Accion": "Acción",
        "accion": "acción",
        "atencion": "atención",
        "obstaculo": "obstáculo",
        "vehiculo": "vehículo",
        "direccion": "dirección",
        "precaucion": "precaución",
        "senal": "señal",
        "transito": "tránsito",
        "manten ": "mantén ",
        "Usala": "Úsala",
        "Desviate": "Desvíate",
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    return text


def detail_flags(message: dict[str, Any]) -> dict[str, int]:
    message_id = message["id"]
    specific = message_id in {
        "H_POTHOLE", "H_STAIRS", "H_OBSTACLE", "H_VEHICLE",
        "M_PERSON", "M_BICYCLE", "M_POLE", "M_SIGN", "M_VEHICLE",
    }
    quantity = message_id not in {"LOW_SAFE"}
    horizontal_position = message_id == "M_POLE"  # Dice centro; nunca izquierda/derecha.
    vertical_or_zone = message_id == "H_OBSTACLE"
    proximity = message_id in {
        "H_MULTI", "H_VEHICLE", "H_GENERIC", "M_PERSON", "M_BICYCLE", "M_POLE",
    }
    uncertainty = message_id in {"LOW_SAFE", "H_POTHOLE", "H_STAIRS"}
    return {
        "severidad_explicita_voz": 0,
        "tipo_peligro": int(specific),
        "cantidad": int(quantity),
        "posicion_horizontal": int(horizontal_position),
        "zona_vertical_o_paso": int(vertical_or_zone),
        "proximidad_o_distancia": int(proximity),
        "confianza": 0,
        "incertidumbre_verbal": int(uncertainty),
        "recomendacion": 1,
        "accion_inmediata": 1,
        "fuente": 0,
        "timestamp": 0,
    }


def detail_level(points: int) -> str:
    percentage = points * 100 / 12
    if percentage <= 20:
        return "Minimo"
    if percentage <= 40:
        return "Basico"
    if percentage <= 60:
        return "Intermedio"
    if percentage <= 80:
        return "Detallado"
    return "Exhaustivo"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_source_alignment(
    single_scenarios: list[dict[str, Any]],
    pair_scenarios: list[dict[str, Any]],
) -> None:
    """Detiene la generacion si el inventario deja de coincidir con el codigo."""
    risk_source = (
        APP_ROOT / "lib/core/services/risk_assessment_service.dart"
    ).read_text(encoding="utf-8")
    analysis_source = (
        APP_ROOT / "lib/core/models/analysis_result.dart"
    ).read_text(encoding="utf-8")
    model_labels = (
        APP_ROOT / "assets/models/labels.txt"
    ).read_text(encoding="utf-8").splitlines()
    model_labels = [label.strip() for label in model_labels if label.strip()]

    errors: list[str] = []
    if model_labels != MODEL_LABELS:
        errors.append("MODEL_LABELS no coincide con assets/models/labels.txt")
    if len(MESSAGES) != 13 or len({message["id"] for message in MESSAGES}) != 13:
        errors.append("El catalogo debe contener 13 IDs de mensaje unicos")
    for message in MESSAGES:
        for field in ("description", "recommendation", "action"):
            if message[field] not in risk_source:
                errors.append(f'{message["id"]}.{field} no aparece literalmente en RiskAssessmentService')
    if "Accion sugerida: $suggestedAction." not in analysis_source:
        errors.append("La plantilla speechText cambio en AnalysisResult")

    valid_scenarios = [row for row in single_scenarios if row["valid"] == "Si"]
    risk_counts = Counter(row["risk"] for row in valid_scenarios)
    if len(single_scenarios) != 330 or len(valid_scenarios) != 297:
        errors.append("La matriz de una deteccion debe tener 330 filas, 297 validas")
    if dict(risk_counts) != {"bajo": 57, "medio": 179, "alto": 61}:
        errors.append(f"Distribucion inesperada de escenarios: {dict(risk_counts)}")
    if len(pair_scenarios) != 55:
        errors.append("La matriz de pares de las 10 clases debe tener 55 filas")

    if errors:
        raise RuntimeError("Validacion CERM fallida:\n- " + "\n- ".join(errors))


def make_formats(workbook: xlsxwriter.Workbook) -> dict[str, Any]:
    return {
        "title": workbook.add_format({"bold": True, "font_size": 20, "font_color": "#FFFFFF", "bg_color": "#123047", "align": "left", "valign": "vcenter"}),
        "subtitle": workbook.add_format({"font_size": 10, "font_color": "#334155", "bg_color": "#E8F0F5", "text_wrap": True, "valign": "top"}),
        "section": workbook.add_format({"bold": True, "font_size": 13, "font_color": "#FFFFFF", "bg_color": "#1D607A", "align": "left", "valign": "vcenter"}),
        "body": workbook.add_format({"font_size": 10, "font_color": "#1F2937", "text_wrap": True, "valign": "top"}),
        "body_bold": workbook.add_format({"bold": True, "font_size": 10, "font_color": "#1F2937", "text_wrap": True, "valign": "top"}),
        "note": workbook.add_format({"italic": True, "font_size": 9, "font_color": "#475569", "bg_color": "#F8FAFC", "text_wrap": True, "valign": "top"}),
        "metric": workbook.add_format({"bold": True, "font_size": 16, "font_color": "#123047", "bg_color": "#DDECF3", "align": "center", "valign": "vcenter", "border": 1, "border_color": "#B8CDD8"}),
        "metric_label": workbook.add_format({"bold": True, "font_size": 9, "font_color": "#334155", "bg_color": "#EFF6F9", "align": "center", "valign": "vcenter", "text_wrap": True, "border": 1, "border_color": "#B8CDD8"}),
        "risk_high": workbook.add_format({"bg_color": "#FECACA", "font_color": "#991B1B"}),
        "risk_medium": workbook.add_format({"bg_color": "#FEF3C7", "font_color": "#92400E"}),
        "risk_low": workbook.add_format({"bg_color": "#D1FAE5", "font_color": "#065F46"}),
        "warn": workbook.add_format({"bg_color": "#FDE68A", "font_color": "#78350F", "text_wrap": True}),
        "fail": workbook.add_format({"bg_color": "#FECACA", "font_color": "#991B1B", "text_wrap": True}),
        "pass": workbook.add_format({"bg_color": "#D1FAE5", "font_color": "#065F46", "text_wrap": True}),
        "mono": workbook.add_format({"font_name": "Consolas", "font_size": 9, "font_color": "#0F172A", "bg_color": "#F8FAFC", "text_wrap": True, "valign": "top"}),
    }


def write_table_sheet(
    workbook: xlsxwriter.Workbook,
    formats: dict[str, Any],
    sheet_name: str,
    title: str,
    note: str,
    headers: list[str],
    rows: list[list[Any]],
    widths: list[int] | None = None,
    table_name: str | None = None,
) -> xlsxwriter.worksheet.Worksheet:
    worksheet = workbook.add_worksheet(sheet_name)
    worksheet.hide_gridlines(2)
    worksheet.set_tab_color("#1D607A")
    last_col = max(0, len(headers) - 1)
    worksheet.merge_range(0, 0, 0, last_col, title, formats["title"])
    worksheet.set_row(0, 30)
    worksheet.merge_range(1, 0, 1, last_col, note, formats["subtitle"])
    worksheet.set_row(1, 46)
    start_row = 3
    safe_table_name = table_name or "T_" + "".join(char for char in sheet_name if char.isalnum())
    worksheet.add_table(
        start_row,
        0,
        start_row + len(rows),
        last_col,
        {
            "name": safe_table_name[:250],
            "style": "Table Style Medium 2",
            "columns": [{"header": header} for header in headers],
            "data": rows,
        },
    )
    worksheet.freeze_panes(start_row + 1, 0)
    worksheet.set_default_row(30)
    for index, width in enumerate(widths or [16] * len(headers)):
        worksheet.set_column(index, index, width, formats["body"])
    for index, header in enumerate(headers):
        lower = header.lower()
        if "riesgo" in lower or lower == "nivel":
            first = start_row + 1
            last = start_row + len(rows)
            worksheet.conditional_format(first, index, last, index, {"type": "text", "criteria": "containing", "value": "alto", "format": formats["risk_high"]})
            worksheet.conditional_format(first, index, last, index, {"type": "text", "criteria": "containing", "value": "medio", "format": formats["risk_medium"]})
            worksheet.conditional_format(first, index, last, index, {"type": "text", "criteria": "containing", "value": "bajo", "format": formats["risk_low"]})
        if lower in {"estado", "status", "resultado_qa"}:
            first = start_row + 1
            last = start_row + len(rows)
            worksheet.conditional_format(first, index, last, index, {"type": "text", "criteria": "containing", "value": "PASS", "format": formats["pass"]})
            worksheet.conditional_format(first, index, last, index, {"type": "text", "criteria": "containing", "value": "WARN", "format": formats["warn"]})
            worksheet.conditional_format(first, index, last, index, {"type": "text", "criteria": "containing", "value": "FAIL", "format": formats["fail"]})
    worksheet.autofit()
    if widths:
        for index, width in enumerate(widths):
            worksheet.set_column(index, index, width, formats["body"])
    return worksheet


def build_single_scenarios() -> list[dict[str, Any]]:
    proximity = {
        "P0": ("A < 0.06 y bottom < 0.70", 0.03, False),
        "P1": ("A < 0.06 y bottom >= 0.70", 0.03, True),
        "P2": ("0.06 <= A < 0.08", 0.07, False),
        "P3": ("0.08 <= A < 0.18", 0.12, False),
        "P4": ("A >= 0.18", 0.22, False),
    }
    horizontal = {
        "izquierda": False,
        "centro": True,
        "derecha": False,
    }
    scenarios: list[dict[str, Any]] = []
    scenario_id = 1
    for label, (horizontal_name, center), bottom_area, (prox_code, prox_values) in itertools.product(
        SCENARIO_LABELS,
        horizontal.items(),
        [False, True],
        proximity.items(),
    ):
        prox_description, area, bottom_ge_070 = prox_values
        valid = not (prox_code == "P1" and not bottom_area)
        zone_code = "ZD" if center and bottom_area else "ZC" if center else "ZB" if bottom_area else "ZO"
        if valid:
            detection = {
                "label": label,
                "confidence": 0.80,
                "center": center,
                "bottom_area": bottom_area,
                "bottom_ge_070": bottom_ge_070,
                "area": area,
            }
            enriched = enrich_detection(detection)
            result = assess([detection])
            message = MESSAGE_BY_ID[result["message_id"]]
            points = sum(detail_flags(message).values())
            scenario = {
                "id": f"S{scenario_id:03d}",
                "valid": "Si",
                "label": label,
                "horizontal": horizontal_name,
                "center": center,
                "bottom_area": bottom_area,
                "zone": zone_code,
                "proximity_code": prox_code,
                "proximity_partition": prox_description,
                "area_example": area,
                "bottom_ge_070": bottom_ge_070,
                "near": enriched["near"],
                "danger_zone": enriched["danger_zone"],
                "potentially_dangerous": enriched["potentially_dangerous"],
                "risk": result["risk"],
                "high_trigger": "; ".join(result["high_triggers"]) or "—",
                "medium_trigger": "; ".join(result["medium_triggers"]) or "—",
                "message_id": result["message_id"],
                "description": message["description"],
                "recommendation": message["recommendation"],
                "action": message["action"],
                "speech": result["speech"],
                "detail_points": points,
                "note": "Izquierda/derecha y bbox no se incluyen en el mensaje CERM.",
            }
            scenario_id += 1
        else:
            scenario = {
                "id": "—",
                "valid": "No: P1 implica bottomArea=true",
                "label": label,
                "horizontal": horizontal_name,
                "center": center,
                "bottom_area": bottom_area,
                "zone": zone_code,
                "proximity_code": prox_code,
                "proximity_partition": prox_description,
                "area_example": area,
                "bottom_ge_070": bottom_ge_070,
                "near": "—",
                "danger_zone": "—",
                "potentially_dangerous": "—",
                "risk": "—",
                "high_trigger": "—",
                "medium_trigger": "—",
                "message_id": "—",
                "description": "Combinacion geometrica imposible",
                "recommendation": "—",
                "action": "—",
                "speech": "—",
                "detail_points": "—",
                "note": "bottom >= 0.70 activa isBottomArea por definicion.",
            }
        scenarios.append(scenario)
    return scenarios


def build_pair_scenarios() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (label_a, label_b) in enumerate(itertools.combinations_with_replacement(MODEL_LABELS, 2), start=1):
        detections = [
            {"label": label_a, "confidence": 0.80, "center": True, "bottom_area": True, "bottom_ge_070": False, "area": 0.12},
            {"label": label_b, "confidence": 0.75, "center": True, "bottom_area": True, "bottom_ge_070": False, "area": 0.12},
        ]
        result = assess(detections)
        message = MESSAGE_BY_ID[result["message_id"]]
        rows.append({
            "id": f"M{index:02d}",
            "label_a": label_a,
            "label_b": label_b,
            "profile": "Ambos centro+inferior; A=0.12; conf=0.80/0.75",
            "dangerous_count": result["dangerous_count"],
            "high_trigger": "; ".join(result["high_triggers"]) or "—",
            "medium_trigger": "; ".join(result["medium_triggers"]) or "—",
            "leading": result["leading"],
            "risk": result["risk"],
            "message_id": result["message_id"],
            "description": message["description"],
            "recommendation": message["recommendation"],
            "action": message["action"],
            "objects": result["objects"],
            "confidence": result["aggregate_confidence"],
            "speech": result["speech"],
            "note": "Matriz de interaccion de etiquetas con perfil fijo; otras geometrias se rigen por 04_REGLAS_PRECEDENCIA.",
        })
    return rows


def build_guided_cases() -> list[list[Any]]:
    """Casos pequenos que explican las reglas sin exigir leer la matriz completa."""
    far_person = {"label": "person", "confidence": 0.80, "center": False, "bottom_area": False, "bottom_ge_070": False, "area": 0.01}
    far_stairs = {"label": "stairs", "confidence": 0.80, "center": False, "bottom_area": False, "bottom_ge_070": False, "area": 0.01}
    danger_pothole = {"label": "pothole", "confidence": 0.80, "center": True, "bottom_area": True, "bottom_ge_070": True, "area": 0.06}
    danger_obstacle = {"label": "obstacle", "confidence": 0.80, "center": True, "bottom_area": True, "bottom_ge_070": True, "area": 0.06}
    high_car = {"label": "car", "confidence": 0.80, "center": True, "bottom_area": False, "bottom_ge_070": False, "area": 0.08}
    near_person = {"label": "person", "confidence": 0.80, "center": True, "bottom_area": True, "bottom_ge_070": False, "area": 0.09}
    near_bicycle = {"label": "bicycle", "confidence": 0.80, "center": True, "bottom_area": True, "bottom_ge_070": False, "area": 0.09}
    center_pole = {"label": "pole", "confidence": 0.80, "center": True, "bottom_area": False, "bottom_ge_070": False, "area": 0.01}
    traffic_sign = {"label": "traffic_sign", "confidence": 0.80, "center": False, "bottom_area": False, "bottom_ge_070": False, "area": 0.01}
    medium_car = {"label": "car", "confidence": 0.80, "center": False, "bottom_area": False, "bottom_ge_070": False, "area": 0.04}
    lateral_pothole = {"label": "pothole", "confidence": 0.80, "center": False, "bottom_area": False, "bottom_ge_070": False, "area": 0.015}
    far_bicycle = {"label": "bicycle", "confidence": 0.80, "center": False, "bottom_area": False, "bottom_ge_070": False, "area": 0.01}
    abnormal_other = {"label": "other_label", "confidence": 1000.0, "center": False, "bottom_area": False, "bottom_ge_070": False, "area": 0.01}

    definitions = [
        ("G01", "No se detecta nada", "[]", [], "La lista vacia devuelve bajo con confianza fija 0.65."),
        ("G02", "Persona pequena y lejana", "person; area=.01; lateral/superior", [far_person], "Puede haber un objeto y aun asi decir 'camino libre'."),
        ("G03", "Dos escaleras pequenas", "2 x stairs; area=.01; lateral/superior", [far_stairs, {**far_stairs, "confidence": 0.75}], "Dos stairs cuentan como dos peligros aunque no exista cercania geometrica."),
        ("G04", "Hueco en centro inferior", "pothole; area=.06; centro+inferior", [danger_pothole], "Un peligro de piso en la zona central inferior es alto."),
        ("G05", "Escalera pequena", "stairs; area=.01; lateral/superior", [far_stairs], "stairs siempre activa alto, sin importar tamano o posicion."),
        ("G06", "Obstaculo en centro inferior", "obstacle; area=.06; centro+inferior", [danger_obstacle], "Un obstaculo en la zona de paso es alto."),
        ("G07", "Auto centrado justo en el limite", "car; area=.08; centro", [high_car], "El limite .08 es inclusivo y activa alto si el vehiculo esta centrado."),
        ("G08", "Persona cercana", "person; area=.09", [near_person], "Una persona solo llega a medio; nunca a alto por si sola."),
        ("G09", "Bicicleta cercana", "bicycle; area=.09", [near_bicycle], "Una bicicleta cercana produce medio."),
        ("G10", "Poste centrado", "pole; centro; area=.01", [center_pole], "El poste produce medio por estar en el centro, aunque sea pequeno."),
        ("G11", "Senal de transito", "traffic_sign; cualquier zona valida", [traffic_sign], "Toda traffic_sign produce medio; no se evalua relevancia."),
        ("G12", "Auto pequeno y lateral", "car; area=.04; lateral", [medium_car], "Todo vehiculo es al menos medio aunque no cumpla el umbral alto."),
        ("G13", "Hueco pequeno y lateral", "pothole; area=.015; lateral/superior", [lateral_pothole], "El nivel es medio, pero usa el mensaje generico de objetos."),
        ("G14", "El objeto que habla no activo el alto", "car centro area=.08 + pothole lateral area=.015", [high_car, lateral_pothole], "El auto activa alto, pero pothole gana el orden y el texto habla del hueco."),
        ("G15", "El objeto que habla no activo el medio", "traffic_sign + bicycle lejana", [traffic_sign, far_bicycle], "La senal activa medio, pero bicycle lidera y el texto afirma que esta cerca."),
        ("G16", "Fallback alto con entrada anormal", "stairs valida + other_label confidence=1000", [far_stairs, abnormal_other], "H_GENERIC es alcanzable por la API directa sin validar, no por el productor YOLO normal."),
    ]

    rows: list[list[Any]] = []
    for case_id, situation, input_summary, detections, lesson in definitions:
        result = assess(detections)
        message = MESSAGE_BY_ID[result["message_id"]]
        rows.append([
            case_id,
            situation,
            input_summary,
            len(detections),
            "; ".join(result["high_triggers"]) or "—",
            "; ".join(result["medium_triggers"]) or "—",
            result["leading"],
            result["risk"],
            result["message_id"],
            humanized_speech_text(message),
            result["objects"],
            result["aggregate_confidence"],
            lesson,
            "Ejecutado por la replica documental del codigo",
        ])

    low_message = MESSAGE_BY_ID["LOW_SAFE"]
    rows.append([
        "G17",
        "Modelo ausente o error: salida mock",
        "No llama assess(); AnalysisResult.mockSafe()",
        "—",
        "—",
        "—",
        "—",
        "bajo",
        "LOW_SAFE (texto reutilizado)",
        humanized_speech_text(low_message),
        "sin objetos peligrosos",
        0.55,
        "La voz no revela que es un fallback: suena igual que el camino libre.",
        "Ruta externa al CERM core",
    ])
    return rows


def build_workbook(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(output)
    workbook.set_properties({
        "title": "CERM: entradas, reglas, salidas y nivel de detalle",
        "subject": "Especificacion exhaustiva del CERM implementado en MagicVision",
        "author": "MagicVision / Codex",
        "comments": "Generado desde el codigo fuente vigente; no es una especificacion futura.",
        "company": "MagicVision",
    })
    formats = make_formats(workbook)
    single_scenarios = build_single_scenarios()
    valid_scenarios = [row for row in single_scenarios if row["valid"] == "Si"]
    pair_scenarios = build_pair_scenarios()
    validate_source_alignment(single_scenarios, pair_scenarios)

    # 00 - Portada y resumen.
    ws = workbook.add_worksheet("00_LEEME_RESUMEN")
    ws.hide_gridlines(2)
    ws.set_tab_color("#123047")
    ws.set_column("A:A", 3)
    ws.set_column("B:B", 24)
    ws.set_column("C:C", 24)
    ws.set_column("D:D", 24)
    ws.set_column("E:E", 24)
    ws.set_column("F:F", 24)
    ws.merge_range("B2:F3", "CERM · Entradas, reglas, salidas y nivel de detalle", formats["title"])
    ws.merge_range(
        "B5:F7",
        "Alcance: comportamiento actual de RiskAssessmentService (analisis por captura). "
        "El motor de tiempo real de main.dart se documenta aparte porque usa otras clases, "
        "umbrales y mensajes. CERM es una denominacion de la tesis; el acronimo no aparece "
        "como clase en el repositorio.",
        formats["subtitle"],
    )
    metrics = [
        ("10", "Clases oficiales YOLO"),
        ("3", "Niveles CERM"),
        ("13", "Plantillas textuales unicas"),
        (str(len(valid_scenarios)), "Escenarios validos de 1 objeto"),
        (str(len(pair_scenarios)), "Pares de clases estandarizados"),
    ]
    for col, (value, label) in enumerate(metrics, start=1):
        ws.write(9, col, value, formats["metric"])
        ws.write(10, col, label, formats["metric_label"])
    ws.merge_range("B13:F13", "Como leer la exhaustividad", formats["section"])
    ws.merge_range(
        "B14:F17",
        "Las coordenadas, la confianza y la cantidad de detecciones tienen dominios continuos o abiertos, "
        "por lo que existen infinitas entradas numericas. El libro es exhaustivo respecto de las decisiones: "
        "enumera todos los intervalos que cambian una regla, todas las fronteras y las 13 plantillas textuales, "
        "297 escenarios validos de una deteccion y los 55 pares de clases bajo un perfil comun.",
        formats["body"],
    )
    ws.merge_range("B19:F19", "Flujo implementado", formats["section"])
    ws.merge_range(
        "B20:F22",
        "List<DetectionResult> + source  →  variables derivadas  →  reglas alto/medio/bajo  →  leading  →  "
        "plantilla fija  →  AnalysisResult  →  voz / pantalla / historial",
        formats["mono"],
    )
    ws.merge_range("B24:F24", "Conclusiones rapidas sobre el detalle", formats["section"])
    ws.merge_range(
        "B25:F30",
        "• La voz siempre combina descripcion + recomendacion + accion.\n"
        "• La voz no pronuncia el string de nivel, confianza, fuente, timestamp, bbox, izquierda/derecha ni distancia.\n"
        "• El getter displayText forma 7 lineas para el registro; la tarjeta visual lee los campos directamente.\n"
        "• El motor realtime si construye izquierda/frente/derecha y 1–8 m aproximados, pero NO es el mismo CERM.\n"
        "• La confianza de CERM es la media de hasta 5 scores YOLO ordenados por dangerScore; no es probabilidad del mensaje.\n"
        "• H_GENERIC no aparece con la salida normal de YOLO, pero es alcanzable por llamada directa con valores fuera del rango esperado.",
        formats["body"],
    )
    ws.merge_range("B32:F32", "Fecha y version documental", formats["section"])
    ws.merge_range("B33:F34", f"Generado: {date.today().isoformat()} · Fuente: codigo local vigente · Archivo reproducible con tools/generate_cerm_workbook.py", formats["note"])
    risk_counts = Counter(row["risk"] for row in valid_scenarios)
    ws.write("H2", "Riesgo", formats["body_bold"])
    ws.write("I2", "Escenarios", formats["body_bold"])
    for row_index, risk in enumerate(["alto", "medio", "bajo"], start=2):
        ws.write(row_index, 7, risk)
        ws.write(row_index, 8, risk_counts[risk])
    chart = workbook.add_chart({"type": "column"})
    chart.add_series({
        "name": "Escenarios de una deteccion",
        "categories": "='00_LEEME_RESUMEN'!$H$3:$H$5",
        "values": "='00_LEEME_RESUMEN'!$I$3:$I$5",
        "fill": {"color": "#1D607A"},
    })
    chart.set_title({"name": "Distribucion de los 297 escenarios validos"})
    chart.set_legend({"none": True})
    chart.set_style(10)
    ws.insert_chart("H7", chart, {"x_scale": 1.15, "y_scale": 1.05})

    # 00B - Guia para lectores no tecnicos.
    guide = workbook.add_worksheet("00B_GUIA_FACIL")
    guide.hide_gridlines(2)
    guide.set_tab_color("#2A7F62")
    guide.merge_range("A1:G2", "Guía fácil: qué recibe y qué responde el CERM", formats["title"])
    guide.set_row(0, 28)
    guide.set_row(1, 28)
    guide.merge_range(
        "A3:G3",
        "Esta hoja permite entender el resultado sin conocer Dart ni visión artificial. Los textos con tildes son ayudas de lectura; la hoja 05 conserva exactamente lo que ejecuta la app.",
        formats["subtitle"],
    )
    concept_headers = ["Pregunta", "Respuesta sencilla", "Valor o regla exacta", "Ampliar en"]
    concept_rows = [
        ["¿Qué entra?", "Una lista de objetos detectados en una foto.", "0..N detecciones + source", "01_ENTRADAS_CERM"],
        ["¿Es una API o recibe JSON?", "No. Es una llamada local dentro de la app Flutter.", "assess(List<DetectionResult>, {source})", "01_ENTRADAS_CERM"],
        ["¿Qué sabe de cada objeto?", "Nombre, confianza de YOLO y rectángulo en la imagen.", "label + confidence + boundingBox", "01_ENTRADAS_CERM"],
        ["¿Recibe la foto directamente?", "No. Otro servicio convierte la foto en detecciones primero.", "File/CameraImage → detector → CERM", "01_ENTRADAS_CERM"],
        ["¿Cuántos niveles hay?", "Tres: bajo, medio y alto. Alto tiene prioridad sobre medio.", "{bajo, medio, alto}", "04_REGLAS_PRECEDENCIA"],
        ["¿Cómo entiende 'cerca'?", "Por el tamaño o por lo abajo que llega el objeto en la imagen; no usa metros.", "area≥.06 o bottom≥.70", "03_DERIVADAS_UMBRALES"],
        ["¿Qué es la zona peligrosa?", "La parte central e inferior de la imagen.", "centro X [.35,.65] y zona inferior", "03_DERIVADAS_UMBRALES"],
        ["¿Cuántos mensajes existen?", "13 plantillas textuales únicas; 14 rutas internas porque dos rutas bajas reutilizan el mismo texto.", "1 baja + 6 altas + 6 medias", "05_CATALOGO_MENSAJES"],
        ["¿El texto explica todo?", "No. Habla descripción, recomendación y acción; omite varios datos técnicos.", "Ver matriz inferior", "10_DETALLE_MENSAJE"],
        ["¿Menciona izquierda/derecha?", "No en el CERM estructurado. Un motor en vivo separado sí puede hacerlo.", "CERM: no; realtime: sí", "11_COMPARATIVA_RT"],
        ["¿Da distancia en metros?", "No en el CERM. El motor en vivo hace una estimación no calibrada de 1 a 8 m.", "CERM: no; realtime: 1..8 m", "11_COMPARATIVA_RT"],
        ["¿Confianza significa probabilidad de riesgo?", "No. Es un promedio de hasta cinco scores de detección.", "media top 5 por dangerScore", "03_DERIVADAS_UMBRALES"],
        ["¿'Sin detecciones' significa seguro?", "No necesariamente. La app lo clasifica bajo, pero pudo no detectar o haber fallado.", "LOW_SAFE, confidence=.65", "13_QA_TRAZABILIDAD"],
        ["¿Dónde veo ejemplos?", "Hay casos cortos explicados y matrices amplias para comprobar combinaciones.", "17 guiados + 297 válidos + 55 pares", "07_CASOS_GUIADOS"],
    ]
    guide.add_table(4, 0, 4 + len(concept_rows), len(concept_headers) - 1, {
        "name": "TGuiaConceptos",
        "style": "Table Style Medium 4",
        "columns": [{"header": header} for header in concept_headers],
        "data": concept_rows,
    })

    class_start = 20
    guide.merge_range(class_start, 0, class_start, 6, "Decisión rápida para una sola detección", formats["section"])
    class_headers = ["Objeto", "Puede ser bajo cuando", "Es medio cuando", "Es alto cuando", "Mensaje individual"]
    simple_class_rows = [
        ["Persona (person)", "No parece cerca", "Parece cerca", "Nunca por sí sola", "LOW_SAFE / M_PERSON"],
        ["Auto, bus o motocicleta", "Nunca", "No alcanza el umbral alto", "área≥.18 o centro+área≥.08", "M_VEHICLE / H_VEHICLE"],
        ["Bicicleta (bicycle)", "No parece cerca", "Parece cerca", "Nunca por sí sola", "LOW_SAFE / M_BICYCLE"],
        ["Escalera (stairs)", "Nunca", "Nunca", "Siempre", "H_STAIRS"],
        ["Hueco (pothole)", "Nunca", "Fuera de zona peligrosa", "En zona peligrosa", "M_GENERIC / H_POTHOLE"],
        ["Poste (pole)", "Fuera del centro", "En el centro", "Nunca por sí solo", "LOW_SAFE / M_POLE"],
        ["Señal (traffic_sign)", "Nunca", "Siempre", "Nunca por sí sola", "M_SIGN"],
        ["Obstáculo (obstacle)", "Nunca", "Fuera de zona peligrosa", "En zona peligrosa", "M_GENERIC / H_OBSTACLE"],
        ["Etiqueta desconocida", "Siempre por sí sola", "Nunca por sí sola", "Nunca con datos normales", "LOW_SAFE"],
    ]
    guide.add_table(class_start + 1, 0, class_start + 1 + len(simple_class_rows), len(class_headers) - 1, {
        "name": "TGuiaClases",
        "style": "Table Style Medium 4",
        "columns": [{"header": header} for header in class_headers],
        "data": simple_class_rows,
    })

    detail_start = 33
    guide.merge_range(detail_start, 0, detail_start, 6, "Qué contiene el mensaje hablado del CERM", formats["section"])
    detail_headers = ["Dato", "¿Lo dice la voz?", "¿Existe en la salida estructurada?", "Explicación"]
    simple_detail_rows = [
        ["Descripción del entorno", "Sí", "Sí", "Primera oración del mensaje."],
        ["Recomendación", "Sí", "Sí", "Indica cómo avanzar o verificar."],
        ["Acción sugerida", "Sí", "Sí", "Se añade como 'Accion sugerida: ...'."],
        ["Nivel bajo/medio/alto", "No", "Sí", "La pantalla/estado puede mostrarlo; la voz no lo pronuncia."],
        ["Objetos detectados", "No", "Sí", "La voz CERM no enumera la lista."],
        ["Confianza", "No", "Sí", "La tarjeta visual la muestra como porcentaje."],
        ["Fuente del análisis", "No", "Sí", "Por ejemplo, tflite_float32 o mock."],
        ["Fecha y hora", "No", "Sí", "Se conserva para historial, no se habla."],
        ["Izquierda / derecha", "No", "No", "El CERM descarta el rectángulo al crear AnalysisResult."],
        ["Distancia en metros", "No", "No", "El CERM no calcula distancia física."],
    ]
    guide.add_table(detail_start + 1, 0, detail_start + 1 + len(simple_detail_rows), len(detail_headers) - 1, {
        "name": "TGuiaDetalleVoz",
        "style": "Table Style Medium 4",
        "columns": [{"header": header} for header in detail_headers],
        "data": simple_detail_rows,
    })
    guide.freeze_panes(5, 0)
    guide.set_default_row(31)
    guide.set_column("A:A", 30, formats["body"])
    guide.set_column("B:B", 48, formats["body"])
    guide.set_column("C:C", 42, formats["body"])
    guide.set_column("D:D", 30, formats["body"])
    guide.set_column("E:G", 24, formats["body"])

    # 01 - Entradas.
    input_rows = [
        ["E01", "Directa", "detections", "Lista de los objetos que YOLO cree haber visto", "List<DetectionResult>", "Coleccion discreta", "0..N; N no acotado por el metodo", "0..100 despues del detector y NMS", "cantidad de objetos", "Si", "No valida nulos internos ni duplicados", "Entrada principal", "[] o [DetectionResult(...)]", "risk_assessment_service.dart:5-8; object_detection_service.dart:330-331", "IMPLEMENTADO"],
        ["E01A", "Propiedad de lista", "cantidad N", "Cuantos objetos se entregan al evaluador", "int", "Discreta", "0, 1, 2, 3, ...", "0..100", "objetos", "Derivada", "N=0 corta a bajo; no hay maximo interno", "Vacio y conteo de peligros", 2, "risk_assessment_service.dart:9,24,75", "IMPLEMENTADO"],
        ["E01B", "Propiedad de lista", "orden", "En que orden llegan los objetos", "Secuencia", "Ordinal estructurada", "Cualquier orden", "Confianza descendente antes de NMS", "posicion en lista", "Implicito", "CERM crea una copia y la reordena por dangerScore", "Leading, top 5 y empates", "[car, person]", "risk_assessment_service.dart:22-29; object_detection_service.dart:330-331", "IMPLEMENTADO"],
        ["E01C", "Propiedad de lista", "duplicados", "Si un mismo objeto/clase puede repetirse", "bool conceptual", "Binaria discreta", "Si / No; el metodo los acepta", "NMS reduce solapes de igual clase, pero pueden quedar repeticiones", "—", "Implicito", "CERM no deduplica", "Resumen y dangerousCount", "[stairs, stairs]", "risk_assessment_service.dart:24,238-251; object_detection_service.dart:420-430", "IMPLEMENTADO"],
        ["E02", "Directa por deteccion", "label", "Nombre de la clase detectada", "String", "Nominal abierta", "Cualquier texto, incluso vacio; 10 clases esperadas", "Una de las 10 etiquetas de labels.txt", "sin unidad", "Si", "minusculas + trim + guion/espacio a '_'", "Clase, peso, reglas y mensaje", "Traffic Sign", "detection_result.dart:10; risk_assessment_service.dart:260-262", "IMPLEMENTADO"],
        ["E03", "Directa por deteccion", "confidence", "Que tan segura esta YOLO de la deteccion", "double", "Continua", "Cualquier double; CERM no rechaza <0, >1, NaN o infinito", "[0.30,1.00] con la configuracion actual", "proporcion; 0=0%, 1=100%", "Si", "El productor normaliza y filtra; CERM no lo hace", "dangerScore y confianza agregada", 0.8, "detection_result.dart:11; object_detection_service.dart:21,289-343; risk_assessment_service.dart:141-144", "IMPLEMENTADO"],
        ["E04", "Directa por deteccion", "boundingBox", "Rectangulo que encierra el objeto dentro de la imagen", "Rect", "Estructurada continua", "left, top, right, bottom sin validacion interna", "Rectangulo valido, finito y recortado a la imagen", "coordenadas normalizadas 0..1", "Si", "Se espera normalizado; CERM no lo comprueba", "Posicion, area y proximidad", "Rect.fromLTRB(0.2,0.3,0.5,0.8)", "detection_result.dart:13-25; object_detection_service.dart:389-417", "IMPLEMENTADO"],
        ["E05", "Subcampo bbox", "left", "Borde izquierdo del objeto", "double", "Continua", "Sin rango validado", "[0,1]", "fraccion del ancho; 0=izquierda", "Si", "Ninguna dentro del CERM", "width y centerX", 0.2, "dart:ui Rect; detection_result.dart:17,19", "ESPERADO_NO_VALIDADO"],
        ["E06", "Subcampo bbox", "top", "Borde superior del objeto", "double", "Continua", "Sin rango validado", "[0,1]", "fraccion del alto; 0=arriba", "Si", "Ninguna dentro del CERM", "height y centerY", 0.3, "dart:ui Rect; detection_result.dart:18,20", "ESPERADO_NO_VALIDADO"],
        ["E07", "Subcampo bbox", "right", "Borde derecho del objeto", "double", "Continua", "Sin rango validado", "[0,1]", "fraccion del ancho; 1=derecha", "Si", "Ninguna dentro del CERM", "width y centerX", 0.5, "dart:ui Rect; detection_result.dart:17,19", "ESPERADO_NO_VALIDADO"],
        ["E08", "Subcampo bbox", "bottom", "Borde inferior del objeto", "double", "Continua", "Sin rango validado", "[0,1]", "fraccion del alto; 1=abajo", "Si", "Ninguna dentro del CERM", "height, centerY, isBottomArea, isNear", 0.8, "detection_result.dart:18,20,24; risk_assessment_service.dart:120", "IMPLEMENTADO"],
        ["E09", "Directa opcional", "source", "Texto que identifica de donde salio el analisis", "String", "Nominal abierta", "Cualquier texto, incluso vacio", "tflite_float32 en los llamadores actuales", "sin unidad", "No; default='tflite'", "Se copia sin cambiarlo", "Metadato de salida; no cambia riesgo", "tflite_float32", "risk_assessment_service.dart:7,18,41,55,67; vision_analysis_service.dart:44-66", "IMPLEMENTADO"],
        ["E10", "Upstream; no directa", "imagen/camera frame", "Foto o cuadro de camara que se analiza antes", "File o CameraImage", "Estructurada", "Imagen capturada", "JPEG o YUV420", "pixeles", "No", "ObjectDetectionService la convierte en detecciones", "No entra directamente al CERM", "foto del entorno", "vision_analysis_service.dart:33-70", "CONTEXTO"],
    ]
    write_table_sheet(
        workbook,
        formats,
        "01_ENTRADAS_CERM",
        "Entradas directas y atributos recibidos por CERM",
        "Contrato real: assess(List<DetectionResult> detections, {String source='tflite'}). Se separa lo que la API acepta de lo que normalmente produce YOLO, porque el CERM no valida bbox ni confidence.",
        ["ID", "Nivel_entrada", "Campo", "Significado_en_palabras_simples", "Tipo_Dart", "Naturaleza", "Dominio_aceptado_por_CERM", "Valores_normales_desde_YOLO", "Unidad_escala", "Obligatorio", "Validacion_transformacion", "Incide_en", "Ejemplo", "Fuente_linea", "Estado_evidencia"],
        input_rows,
        [9, 19, 18, 42, 22, 20, 44, 38, 30, 18, 40, 32, 26, 58, 24],
        "TEntradasCerm",
    )

    # 02 - Etiquetas.
    class_rows = [
        [0, "person", "persona", "Persona", 2, "No", "isNear", "Nunca", "bajo | medio", "LOW_SAFE / M_PERSON", "10 clases oficiales"],
        [1, "car", "vehiculo", "Vehiculo", 4, "isNear", "Siempre si no es alto", "area>=0.18 OR (center AND area>=0.08)", "medio | alto", "M_VEHICLE / H_VEHICLE", "Vehiculo"],
        [2, "motorcycle", "vehiculo", "Vehiculo", 4, "isNear", "Siempre si no es alto", "area>=0.18 OR (center AND area>=0.08)", "medio | alto", "M_VEHICLE / H_VEHICLE", "Vehiculo"],
        [3, "bus", "vehiculo", "Vehiculo", 4, "isNear", "Siempre si no es alto", "area>=0.18 OR (center AND area>=0.08)", "medio | alto", "M_VEHICLE / H_VEHICLE", "Vehiculo"],
        [4, "bicycle", "bicicleta", "Movilidad", 3, "No", "isNear", "Nunca", "bajo | medio", "LOW_SAFE / M_BICYCLE", "No cuenta dangerous"],
        [5, "stairs", "escalera/desnivel", "Peligro de paso", 5, "Siempre", "Nunca", "Siempre", "alto", "H_STAIRS", "Riesgo alto sin importar posicion/tamano"],
        [6, "pothole", "hueco/bache", "Peligro de piso", 5, "isBottomArea", "NOT isDangerZone", "isDangerZone", "medio | alto", "M_GENERIC / H_POTHOLE", "Medio usa texto generico"],
        [7, "pole", "poste", "Obstaculo vertical", 3, "No", "isCenterArea", "Nunca", "bajo | medio", "LOW_SAFE / M_POLE", "Fuera del centro queda bajo"],
        [8, "traffic_sign", "senal de transito", "Referencia", 2, "No", "Siempre", "Nunca", "medio", "M_SIGN", "Siempre medio"],
        [9, "obstacle", "obstaculo", "Peligro de paso", 5, "isBottomArea", "NOT isDangerZone", "isDangerZone", "medio | alto", "M_GENERIC / H_OBSTACLE", "Medio usa texto generico"],
        ["—", "OTHER", "etiqueta no reconocida", "Abierta", 1, "No", "Nunca", "Nunca", "bajo", "LOW_SAFE", "Puede anunciar camino libre aunque haya objeto"],
    ]
    write_table_sheet(workbook, formats, "02_ETIQUETAS", "Diccionario de clases y comportamiento individual", "Pesos y condiciones exactos del codigo para UNA deteccion. La normalizacion no traduce sinonimos espanoles; solo cambia mayusculas, guiones y espacios. En escenas con varios objetos, el mensaje puede corresponder a otra clase o a H_MULTI.", ["class_id", "label_normalizada", "texto_es", "grupo", "labelWeight", "potentiallyDangerous_cuando", "medio_cuando", "alto_cuando", "riesgos_posibles_1_objeto", "mensajes_posibles_1_objeto", "observacion"], class_rows, [10, 20, 22, 22, 12, 30, 30, 42, 26, 34, 36], "TEtiquetasCerm")

    # 03 - Derivadas y umbrales.
    derived_rows = [
        ["D01", "normalizedLabel", "lower().trim().replace('-', '_').replace(' ', '_')", "String", "—", "label", "Reglas, peso, resumen", "Traffic Sign -> traffic_sign", "risk_assessment_service.dart:260-262"],
        ["D02", "centerX", "boundingBox.center.dx", "double continuo", "—", "left,right", "isCenterArea", "(left+right)/2", "detection_result.dart:17"],
        ["D03", "centerY", "boundingBox.center.dy", "double continuo", "—", "top,bottom", "isBottomArea", "(top+bottom)/2", "detection_result.dart:18"],
        ["D04", "width", "boundingBox.width", "double continuo", "Esperado >=0", "left,right", "area", "right-left", "detection_result.dart:19"],
        ["D05", "height", "boundingBox.height", "double continuo", "Esperado >=0", "top,bottom", "area", "bottom-top", "detection_result.dart:20"],
        ["D06", "area", "width * height", "double continuo", "0.06 / 0.08 / 0.18 inclusivos", "width,height", "near, vehiculo alto, dangerScore", 0.12, "detection_result.dart:21"],
        ["D07", "isCenterArea", "0.35 <= centerX <= 0.65", "bool", "0.35 y 0.65 incluidos", "centerX", "dangerZone, pole, vehiculo, zoneWeight", "x=.35 -> true", "detection_result.dart:23"],
        ["D08", "isBottomArea", "centerY >= 0.55 OR bottom >= 0.68", "bool", "0.55 y 0.68 incluidos", "centerY,bottom", "dangerZone, dangerous piso", "bottom=.68 -> true", "detection_result.dart:24"],
        ["D09", "isDangerZone", "isCenterArea AND isBottomArea", "bool", "—", "D07,D08", "alto pothole/obstacle; zoneWeight=3", "centro+inferior", "detection_result.dart:25"],
        ["D10", "isNear", "area >= 0.06 OR bottom >= 0.70", "bool", "0.06 y 0.70 incluidos", "area,bottom", "person/bicycle medio; dangerous vehiculo", "A=.06 -> true", "risk_assessment_service.dart:119-121"],
        ["D11", "isVehicle", "label in {car,bus,motorcycle}", "bool", "3 etiquetas", "normalizedLabel", "alto/medio vehiculo", "car -> true", "risk_assessment_service.dart:123-125"],
        ["D12", "potentiallyDangerous", "stairs OR (pothole/obstacle AND bottom) OR (vehicle AND near)", "bool", "—", "label,D08,D10", "dangerous.length", "pothole lateral inferior -> true", "risk_assessment_service.dart:109-116"],
        ["D13", "dangerousCount", "count(potentiallyDangerous)", "int discreto", ">=2 activa alto", "detections", "H_MULTI", "2", "risk_assessment_service.dart:24,75"],
        ["D14", "labelWeight", "5/4/3/2/1 segun clase", "double discreto", "{1,2,3,4,5}", "label", "dangerScore", "pothole=5", "risk_assessment_service.dart:127-135"],
        ["D15", "zoneWeight", "3 dangerZone; 1.5 centro no inferior; 0 resto", "double discreto", "{0,1.5,3}", "D07,D09", "dangerScore", "ZD=3", "risk_assessment_service.dart:136-140"],
        ["D16", "dangerScore", "labelWeight*100 + zoneWeight*10 + area + confidence", "double continuo", "Sin umbral de riesgo", "D14,D15,area,confidence", "Ordena leading y top5", "530.92", "risk_assessment_service.dart:141-145"],
        ["D17", "leading", "sorted.first por dangerScore", "DetectionResult", "Una deteccion", "detections", "Elige plantilla alto/medio", "Puede no ser el disparador", "risk_assessment_service.dart:22-32"],
        ["D18", "aggregateConfidence", "media confidence de primeros 5 por dangerScore", "double continuo", "N<=5 usa todas; N>5 descarta resto", "sorted", "AnalysisResult.confidence", "No es probabilidad de riesgo", "risk_assessment_service.dart:254-258"],
        ["D19", "objectSummary", "conteo por label normalizada; label o label xN", "List<String>", "0..numero etiquetas", "sorted", "detectedObjects", "person x2", "risk_assessment_service.dart:238-251"],
    ]
    write_table_sheet(workbook, formats, "03_DERIVADAS_UMBRALES", "Variables derivadas y fronteras de decision", "Todos los comparadores >= y <= indicados son inclusivos. No existe validacion defensiva para NaN, infinito, rectangulos invertidos o coordenadas fuera de 0..1.", ["ID", "Variable", "Formula_exacta", "Tipo", "Umbral_dominio", "Dependencias", "Uso", "Ejemplo_nota", "Fuente_linea"], derived_rows, [8, 24, 54, 22, 28, 28, 34, 30, 44], "TDerivadasCerm")

    # 04 - Reglas y precedencia.
    rule_rows = [
        [0, "R_EMPTY", "Cortocircuito", "bajo", "detections.isEmpty", "Finaliza inmediatamente", "Todas", "LOW_SAFE", "confidence=0.65 fija", "risk_assessment_service.dart:9-19"],
        [1, "R_SORT", "Preparacion", "—", "Ordenar dangerScore descendente", "Antes de reglas", "Todas", "—", "Define leading y top5", "risk_assessment_service.dart:22-29"],
        [2, "H_MULTI", "Alto", "alto", "dangerous.length >= 2", "Primera condicion alta", "Solo items potentiallyDangerous", "H_MULTI", "Duplicados cuentan; dos objetos cualquiera NO bastan; stairs cuenta aun lejos", "risk_assessment_service.dart:75,109-116"],
        [3, "H_STAIRS", "Alto", "alto", "Existe stairs", "Despues de H_MULTI", "stairs", "Segun leading", "Escalera siempre alta", "risk_assessment_service.dart:76-79"],
        [4, "H_GROUND", "Alto", "alto", "pothole/obstacle AND dangerZone", "Despues de stairs", "pothole, obstacle", "Segun leading", "Centro+inferior", "risk_assessment_service.dart:79-82"],
        [5, "H_VEHICLE", "Alto", "alto", "vehiculo AND (area>=.18 OR (center AND area>=.08))", "Ultima alta", "car,bus,motorcycle", "Segun leading", "Umbrales inclusivos", "risk_assessment_service.dart:83-87"],
        [6, "M_PERSON_BIKE", "Medio", "medio", "person/bicycle AND near", "Solo si no hubo alto", "person,bicycle", "Segun leading", "near: A>=.06 o bottom>=.70", "risk_assessment_service.dart:92-97"],
        [7, "M_POLE", "Medio", "medio", "pole AND center", "Solo si no hubo alto", "pole", "Segun leading", "Centro inclusivo .35-.65", "risk_assessment_service.dart:98"],
        [8, "M_SIGN", "Medio", "medio", "traffic_sign presente", "Solo si no hubo alto", "traffic_sign", "Segun leading", "No depende de zona/tamano", "risk_assessment_service.dart:99"],
        [9, "M_VEHICLE", "Medio", "medio", "cualquier vehiculo", "Solo si no hubo alto", "car,bus,motorcycle", "Segun leading", "Vehiculo nunca queda bajo", "risk_assessment_service.dart:100"],
        [10, "M_GROUND", "Medio", "medio", "pothole/obstacle AND NOT dangerZone", "Solo si no hubo alto", "pothole,obstacle", "M_GENERIC si leading coincide", "Peligro de piso nunca queda bajo", "risk_assessment_service.dart:101-104"],
        [11, "R_LOW", "Fallback", "bajo", "Ninguna regla anterior", "Ultima salida", "person/bicycle lejos; pole lateral; other", "LOW_SAFE", "Puede haber objetos detectados", "risk_assessment_service.dart:59-68"],
        [12, "R_MESSAGE", "Mensaje", "—", "Seleccionar por label de leading", "Despues de decidir nivel", "Alto/medio", "13 plantillas", "Leading puede no ser el disparador; hay 14 rutas porque LOW_SAFE se reutiliza", "risk_assessment_service.dart:9-68,147-235"],
    ]
    write_table_sheet(workbook, formats, "04_REGLAS_PRECEDENCIA", "Reglas de riesgo y orden de evaluacion", "El nivel se decide por existencia de alguna condicion; el texto se decide despues usando leading. Esa separacion explica mensajes que pueden describir un objeto distinto del que activo el nivel.", ["Orden", "Rule_ID", "Fase", "Nivel_resultante", "Condicion_exacta", "Precedencia", "Clases", "Message_ID", "Efecto_observacion", "Fuente_linea"], rule_rows, [8, 20, 16, 18, 52, 24, 26, 22, 38, 42], "TReglasCerm")

    # 05 - Catalogo de mensajes.
    message_rows: list[list[Any]] = []
    for message in MESSAGES:
        speech = speech_text(message)
        words = len(speech.replace(".", "").split())
        flags = detail_flags(message)
        points = sum(flags.values())
        message_rows.append([
            message["id"], message["risk"], message["branch"], message["leading"], message["reachable"],
            message["literal_condition"], message["description"], message["recommendation"], message["action"], speech,
            humanized_speech_text(message), words, len(speech), round(words / 2.5, 1), points, round(points * 100 / 12, 1), detail_level(points),
        ])
    write_table_sheet(workbook, formats, "05_CATALOGO_MENSAJES", "Las 13 plantillas textuales unicas del CERM", "Hay 14 rutas internas de salida: vacio y fallback bajo comparten LOW_SAFE. speechText_literal conserva exactamente la ortografia del codigo; la version legible solo agrega tildes para el lector. TTS estimado usa 150 palabras/minuto.", ["Message_ID", "Riesgo", "Ruta_plantilla", "Leading_aplicable", "Alcanzable", "Condicion", "description_literal", "recommendation_literal", "suggestedAction_literal", "speechText_literal", "speechText_legible_con_tildes", "Palabras", "Caracteres", "TTS_estimado_s", "Puntos_detalle_12", "Indice_detalle_pct", "Nivel_detalle_QA"], message_rows, [18, 12, 24, 26, 38, 48, 45, 50, 28, 92, 92, 10, 12, 16, 18, 18, 20], "TCatalogoMensajes")

    # 06 - Atributos exactos de salida.
    output_attribute_rows = [
        ["S01", "riskLevel", "String", "Nominal ordinal discreta", "{bajo, medio, alto}", "Reglas de 04_REGLAS_PRECEDENCIA", "medio", "Si", "AnalysisResult y todos los consumidores", "No existe unknown/error"],
        ["S02", "description", "String", "Nominal discreta", "13 textos unicos de 05_CATALOGO_MENSAJES", "Plantilla elegida por riesgo + leading", "Se detecta una persona cerca.", "Si", "Voz, tarjeta visual, historial, VLM", "No incluye direccion ni distancia"],
        ["S03", "recommendation", "String", "Nominal discreta", "12 textos unicos dentro de las 13 plantillas", "Plantilla elegida", "Avanza lentamente y manten distancia.", "Si", "Voz, tarjeta visual, historial, VLM", "H_POTHOLE y H_GENERIC comparten recomendacion"],
        ["S04", "suggestedAction", "String", "Nominal discreta", "{Detente; Avanza con precaucion; Mantente en zona segura; Desviate con cuidado; Avanza con cuidado}", "Plantilla elegida", "Detente", "Si", "Voz, tarjeta visual y status; metodo toHistoryJson", "El historial real no lo guarda como campo separado"],
        ["S05", "detectedObjects", "List<String>", "Coleccion nominal discreta", "[] o [label, label xN, ...]; longitud abierta", "Conteo de todas las etiquetas normalizadas", "[person x2, car]", "Si", "Pantalla, historial, grounding VLM", "Orden por frecuencia; desempate no especificado"],
        ["S06", "confidence", "double", "Continua", "assess([])=0.65; lista no vacia=media top5 por dangerScore; mockSafe=0.55; sin clamp interno", "aggregateConfidence", 0.775, "Si", "Tarjeta visual y metodo toHistoryJson", "El historial real no lo guarda como campo separado; no es probabilidad de riesgo"],
        ["S07", "timestamp", "DateTime", "Temporal continua", "Cualquier instante DateTime.now()", "Generado al crear AnalysisResult", "2026-09-02T16:30:00", "Si", "Historial JSON", "No aparece en speechText/displayText"],
        ["S08", "source", "String", "Nominal abierta", "Default tflite; usados: tflite, tflite_float32, mock; acepta cualquier texto", "Copia del argumento source", "tflite_float32", "Si", "Pantalla e historial", "No cambia nivel"],
        ["SD01", "isMock", "bool derivado", "Binaria discreta", "{true,false}", "source == 'mock'", "false", "Derivada", "Logica de app", "No modifica el mensaje por si sola"],
        ["SD02", "sourceText", "String derivado", "Nominal", "mock->Demo mock; tflite_float32->YOLO26m float32; tflite->TFLite; otro->literal", "switch(source)", "YOLO26m float32", "Derivada", "displayText", "Mapeo exacto"],
        ["SD03", "detectedObjectsText", "String derivado", "Texto", "Vacio->sin objetos peligrosos; no vacio->join(', ')", "detectedObjects", "person x2, car", "Derivada", "displayText", "La frase vacia puede interpretarse como confirmacion"],
        ["SD04", "speechText", "String derivado", "Texto de plantilla", "description + recommendation + ' Accion sugerida: ' + suggestedAction + '.'", "3 campos", "Ver 05_CATALOGO_MENSAJES", "Derivada", "TTS", "Omite nivel, objetos, confidence, timestamp y source"],
        ["SD05", "displayText", "String derivado", "Texto de 7 lineas", "description; riesgo; objetos; recomendacion; accion; confianza%; fuente", "7 campos/derivadas", "Ver 06B_CANALES_SALIDA", "Derivada", "Se guarda dentro de CommandHistoryEntry.spokenText", "La tarjeta visual real NO usa este getter; omite timestamp, bbox, direccion y distancia"],
        ["SD06", "historyJson", "Map<String,Object?>", "Estructurada", "type=environment + los 8 atributos de AnalysisResult", "toHistoryJson()", "Mapa JSON", "Derivada", "Sin consumidor en la app actual", "Metodo disponible, pero la persistencia real usa CommandHistoryEntry.toJson"],
        ["SD07", "statusText", "String derivado externo", "Nominal discreta", "8 combinaciones: bajo/Avanza con cuidado; alto/{Detente, Avanza con precaucion, Mantente en zona segura}; medio/{Avanza con precaucion, Desviate con cuidado, Avanza con cuidado, Mantente en zona segura}", "Riesgo + suggestedAction", "Riesgo alto: Detente", "Derivada", "Barra de estado", "Cualquier nivel desconocido se presenta como bajo"],
    ]
    write_table_sheet(workbook, formats, "06_ATRIBUTOS_SALIDA", "Atributos de salida y todos sus dominios", "S01-S08 son los ocho atributos almacenados en AnalysisResult. SD01-SD07 son presentaciones o propiedades derivadas; no agregan informacion espacial perdida.", ["ID", "Campo", "Tipo", "Naturaleza", "Valores_dominio", "Construccion", "Ejemplo", "Obligatorio_derivado", "Consumidores", "Observacion"], output_attribute_rows, [10, 24, 24, 24, 86, 48, 32, 22, 44, 58], "TAtributosSalida")

    # 06B - Canales de salida.
    channel_rows = [
        ["AnalysisResult", "Objeto estructurado", "Siempre", 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, "8 atributos exactos; no guarda bbox"],
        ["speechText", "description + recommendation + accion", "Cuando la app lo pronuncia", 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, "No pronuncia nivel, inventario, confianza, fuente ni tiempo"],
        ["displayText", "Getter de 7 lineas", "Al construir historial", 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, "Se guarda en spokenText; la tarjeta visual no usa este getter"],
        ["Tarjeta visual real", "_EnvironmentAnalysisSummary", "Resultado en pantalla", 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, "Objetos: maximo 5 grupos + indicador +N"],
        ["toHistoryJson", "Metodo disponible", "No llamado actualmente", 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, "No es la persistencia usada por la app"],
        ["Historial real", "CommandHistoryEntry", "Al registrar analisis", 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, "Accion/confianza no son campos estructurados; displayText queda en spokenText pero la vista/exportacion environment no lo lee"],
        ["_statusForAnalysisResult", "Riesgo X: accion", "Barra de estado", 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, "Cualquier string no alto/medio cae como bajo"],
        ["yoloGroundingForSpeech", "Confirmado por YOLO + descripcion + objetos + recomendacion", "VLM", 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, "No comprueba analysis.isMock"],
        ["Motor realtime", "Plantilla dinamica de main.dart", "HUD/TTS en vivo", "danger/caution/info", 1, 0, 0, 0, 0, 0, 0, 1, 1, "Es otro motor; no es AnalysisResult"],
    ]
    write_table_sheet(workbook, formats, "06B_CANALES_SALIDA", "Campos disponibles por canal de salida", "1=presente como dato util del canal; 0=ausente. Direccion y distancia existen solo en el motor realtime paralelo. La persistencia real no usa AnalysisResult.toHistoryJson().", ["Canal", "Construccion", "Disparador", "Nivel_riesgo", "Descripcion", "Recomendacion", "Accion", "Objetos", "Confianza", "Timestamp", "Fuente", "Direccion", "Distancia", "Observacion"], channel_rows, [26, 42, 28, 16, 14, 16, 12, 12, 12, 12, 12, 12, 12, 66], "TSalidasCanales")

    # 07 - Casos guiados.
    guided_rows = build_guided_cases()
    write_table_sheet(
        workbook,
        formats,
        "07_CASOS_GUIADOS",
        "17 casos explicados paso a paso",
        "Incluye cada plantilla normal, limites, multiobjeto, dos casos donde leading no activo el nivel, el fallback alto fuera de rango y la salida mock. 'Ejecutado' significa evaluado por una replica fiel del codigo dentro de este generador.",
        ["Case_ID", "Situacion_en_palabras_simples", "Entrada_resumida", "N", "Condiciones_alto_activas", "Condiciones_medio_activas", "Leading", "Riesgo", "Message_ID", "Mensaje_hablado_legible", "Objetos_salida", "Confianza_salida", "Que_demuestra", "Origen_verificacion"],
        guided_rows,
        [12, 38, 46, 8, 48, 48, 18, 12, 26, 96, 30, 18, 64, 46],
        "TCasosGuiados",
    )

    # 07B - Escenarios de una deteccion.
    scenario_headers = ["Scenario_ID", "Valido", "label", "sector_horizontal", "isCenterArea", "isBottomArea", "zona_codigo", "proximidad_codigo", "particion_area_bottom", "area_ejemplo", "bottom_ge_0_70", "isNear", "isDangerZone", "potentiallyDangerous", "Riesgo", "Disparador_alto", "Disparador_medio", "Message_ID", "description", "recommendation", "suggestedAction", "speechText", "Puntos_detalle", "Nota"]
    scenario_rows = [[row[key] for key in ["id", "valid", "label", "horizontal", "center", "bottom_area", "zone", "proximity_code", "proximity_partition", "area_example", "bottom_ge_070", "near", "danger_zone", "potentially_dangerous", "risk", "high_trigger", "medium_trigger", "message_id", "description", "recommendation", "action", "speech", "detail_points", "note"]] for row in single_scenarios]
    write_table_sheet(workbook, formats, "07_CASOS_1_OBJETO", "Cobertura exhaustiva por equivalencia · una deteccion", "330 filas logicas: 297 combinaciones geometricamente validas y 33 imposibles marcadas. Izquierda y derecha se expanden aunque el CERM las colapsa como no-centro; asi se hace visible que no aparecen en el mensaje.", scenario_headers, scenario_rows, [14, 26, 18, 20, 15, 15, 12, 18, 34, 14, 16, 12, 16, 22, 12, 38, 38, 18, 44, 48, 28, 92, 16, 42], "TCasosUnObjeto")

    # 08 - Pares de clases.
    pair_headers = ["Case_ID", "label_A", "label_B", "Perfil_comun", "dangerousCount", "Disparador_alto", "Disparador_medio", "leading", "Riesgo", "Message_ID", "description", "recommendation", "suggestedAction", "detectedObjects", "aggregateConfidence", "speechText", "Nota"]
    pair_rows = [[row[key] for key in ["id", "label_a", "label_b", "profile", "dangerous_count", "high_trigger", "medium_trigger", "leading", "risk", "message_id", "description", "recommendation", "action", "objects", "confidence", "speech", "note"]] for row in pair_scenarios]
    write_table_sheet(workbook, formats, "08_CASOS_MULTI", "Interaccion multiobjeto · 55 pares de las 10 clases", "Cada par usa el mismo perfil para aislar la interaccion de etiquetas: centro+inferior, area 0.12 y confianza 0.80/0.75. Para otras geometrias, combinar esta hoja con 03 y 04.", pair_headers, pair_rows, [12, 18, 18, 48, 18, 42, 42, 18, 12, 18, 44, 48, 28, 30, 22, 92, 52], "TCasosMulti")

    # 09 - Fronteras.
    frontier_rows = [
        ["F01", "confidence upstream", 0.299999, 0.30, 0.300001, ">=0.30 en detector", "CERM no aplica este umbral", "object_detection_service.dart:21,289-318; CERM sin filtro"],
        ["F02", "IoU NMS", 0.649999, 0.65, 0.650001, ">0.65 suprime", "Exactamente .65 se conserva", "object_detection_service.dart:420-430"],
        ["F03", "centerX izquierda", 0.349999, 0.35, 0.350001, ">=0.35", "En .35 entra al centro", "detection_result.dart:23"],
        ["F04", "centerX derecha", 0.649999, 0.65, 0.650001, "<=0.65", "En .65 sigue en centro", "detection_result.dart:23"],
        ["F05", "centerY inferior", 0.549999, 0.55, 0.550001, ">=0.55", "En .55 activa isBottomArea", "detection_result.dart:24"],
        ["F06", "bottom inferior", 0.679999, 0.68, 0.680001, ">=0.68", "En .68 activa isBottomArea", "detection_result.dart:24"],
        ["F07", "area near", 0.059999, 0.06, 0.060001, ">=0.06", "En .06 activa isNear", "risk_assessment_service.dart:120"],
        ["F08", "bottom near", 0.699999, 0.70, 0.700001, ">=0.70", "En .70 activa isNear y ya es bottom", "risk_assessment_service.dart:120"],
        ["F09", "vehiculo alto centro", 0.079999, 0.08, 0.080001, ">=0.08 si center", "En centro .08 cambia medio->alto", "risk_assessment_service.dart:84-85"],
        ["F10", "vehiculo alto global", 0.179999, 0.18, 0.180001, ">=0.18", "En lateral .18 cambia medio->alto", "risk_assessment_service.dart:84"],
        ["F11", "dangerousCount", 1, 2, 3, ">=2", "En 2 fuerza H_MULTI", "risk_assessment_service.dart:75"],
        ["F12", "aggregateConfidence top N", 4, 5, 6, "take(5)", "La sexta deteccion no participa", "risk_assessment_service.dart:254-258"],
        ["F13", "detector cap", 99, 100, 101, "take(100)", "Maximo 100 detecciones post-NMS", "object_detection_service.dart:330-331"],
        ["F14", "confidence display", -0.01, "0..1", 1.01, "clamp 0..100%", "Solo presentacion; valor estructurado no cambia", "analysis_result.dart:42-53; main.dart:5968-5970"],
    ]
    write_table_sheet(workbook, formats, "09_PRUEBAS_FRONTERA", "Valores justo debajo, en y encima de cada umbral", "Estas fronteras son suficientes para pruebas unitarias de inclusividad. NaN/infinito requieren casos adicionales porque no estan rechazados por el CERM.", ["ID", "Variable_regla", "Debajo", "En_umbral", "Encima", "Comparador", "Resultado_en_umbral", "Fuente_linea"], frontier_rows, [10, 28, 16, 18, 16, 22, 42, 44], "TPruebasFrontera")

    # 10 - Nivel de detalle.
    detail_rows = []
    for message in MESSAGES:
        flags = detail_flags(message)
        points = sum(flags.values())
        speech = speech_text(message)
        words = len(speech.replace(".", "").split())
        detail_rows.append([
            message["id"], message["risk"], *flags.values(), points,
            round(points * 100 / 12, 1), detail_level(points), words,
            len(speech), round(words / 2.5, 1),
            "PROPUESTO_QA: mide contenido presente, no exactitud ni seguridad.",
        ])
    detail_headers = ["Message_ID", "Riesgo", *detail_flags(MESSAGES[0]).keys(), "Puntos_12", "Indice_pct", "Nivel_QA", "Palabras", "Caracteres", "TTS_estimado_s", "Interpretacion"]
    write_table_sheet(workbook, formats, "10_DETALLE_MENSAJE", "Indice descriptivo de detalle por plantilla", "Rubrica PROPUESTA_QA, no salida del CERM. Cada flag vale 1 si el contenido aparece en speechText. Escala: 0-20 Minimo; >20-40 Basico; >40-60 Intermedio; >60-80 Detallado; >80 Exhaustivo.", detail_headers, detail_rows, [18, 12] + [18] * 12 + [12, 14, 14, 12, 14, 16, 58], "TDetalleMensaje")

    # 11 - Comparativa con tiempo real.
    rt_map = {
        "person": ("persona", "caution"),
        "car": ("auto", "caution"),
        "motorcycle": ("motocicleta", "caution"),
        "bus": ("autobus", "caution"),
        "bicycle": ("bicycle", "info"),
        "stairs": ("stairs", "info"),
        "pothole": ("bache", "danger"),
        "pole": ("pole", "info"),
        "traffic_sign": ("traffic_sign", "info"),
        "obstacle": ("obstacle", "info"),
    }
    core_risks = {
        "person": "bajo | medio", "car": "medio | alto", "motorcycle": "medio | alto", "bus": "medio | alto",
        "bicycle": "bajo | medio", "stairs": "alto", "pothole": "medio | alto", "pole": "bajo | medio",
        "traffic_sign": "medio", "obstacle": "medio | alto",
    }
    comparison_rows = []
    for label in MODEL_LABELS:
        canonical, realtime_risk = rt_map[label]
        comparison_rows.append([
            "CLASE", label, core_risks[label], canonical, realtime_risk,
            "Si" if realtime_risk == "danger" else "No",
            "Si" if label in {"bicycle", "stairs", "pole", "traffic_sign", "obstacle"} else "No",
            "Core: .35/.65; RT: .38/.62",
            "RT estima 1..8 m; core no conserva distancia",
            "main.dart:4968-5326",
        ])
    comparison_rows.extend([
        ["REGLA", "Direccion RT", "No sale en CERM", "left/front/right", "x<.38 / .38-.62 / x>.62", "—", "—", "Tres sectores", "Se verbaliza", "main.dart:5275-5288"],
        ["REGLA", "Distancia RT", "No sale en CERM", "meters", "round(clamp(.55/apparentSize,1,8))", "—", "—", "—", "No calibrada por clase/camara", "main.dart:5317-5326"],
        ["REGLA", "Nearby RT", "No equivalente", "nearby", "distance<=4", "—", "—", "—", "Activa alerta automatica", "main.dart:5198-5206"],
        ["REGLA", "Cooldown RT", "No temporal", "label|zone", "4 segundos", "—", "—", "—", "No usa tracking/ID/distancia", "main.dart:2537-2554"],
        ["REGLA", "HUD vacio", "Core: LOW_SAFE", "sin detecciones", "Ruta despejada / Sin peligros activos", "—", "—", "—", "Tambien ocurre si detector falla", "main.dart:2973-3006"],
    ])
    write_table_sheet(workbook, formats, "11_COMPARATIVA_RT", "CERM estructurado vs motor paralelo en tiempo real", "No mezclar estos resultados en la tesis: RiskAssessmentService gobierna AnalysisResult por captura, mientras main.dart clasifica HUD/TTS en vivo con otro vocabulario.", ["Tipo_fila", "Label_regla", "Riesgo_CERM_core", "Canonico_RT", "Riesgo_RT", "Alerta_auto_RT", "Clase_omitida_RT", "Direccion", "Distancia", "Fuente_linea"], comparison_rows, [14, 22, 26, 22, 24, 18, 20, 34, 46, 40], "TComparativaRT")

    # 12 - Variaciones de mensajes realtime.
    rt_spoken_labels = [
        ("bache", "bache | pothole | hole | huecos | holes", "Si: pothole"),
        ("grieta", "grieta | crack | cracks", "No"),
        ("hueco", "hueco | hoyo", "No"),
        ("alcantarilla abierta", "alcantarilla | open_manhole | manhole | alcantarilla_abierta", "No"),
    ]
    rt_directions = [("left", "a la izquierda"), ("front", "al frente"), ("right", "a la derecha")]
    rt_rows: list[list[Any]] = [
        ["RT00001", "SIN_DETECCIONES", "—", "—", "Si", "—", "—", "—", 0, "No detecto objetos urbanos confiables en este momento.", "Fija"],
        ["RT00002", "SIN_DANGER", "objeto no danger", "—", "Si", "—", "—", "—", 0, "No detecto un riesgo alto inmediato. Para una lectura completa, dime: que ves.", "Fija"],
    ]
    rt_id = 3
    for (label, accepted_labels, current_model), (direction_code, direction_text), meters, extra_count in itertools.product(rt_spoken_labels, rt_directions, range(1, 9), range(0, 100)):
        extra_text = "" if extra_count == 0 else f" Tambien detecto {extra_count} peligro{'s' if extra_count != 1 else ''} mas."
        output_text = f"Peligro {direction_text}, aproximadamente {meters} metros: {label}.{extra_text}"
        rt_rows.append([f"RT{rt_id:05d}", "DANGER", label, accepted_labels, current_model, direction_code, direction_text, meters, extra_count, output_text, "Exhaustivo bajo el limite operativo de 100 detecciones"])
        rt_id += 1
    if len(rt_rows) != 9602:
        raise RuntimeError(f"Se esperaban 9602 variaciones realtime y se generaron {len(rt_rows)}")
    write_table_sheet(workbook, formats, "12_VARIACIONES_RT", "9,602 variaciones del resumen de peligro realtime", "Apendice separado: NO pertenece al CERM core. Expansion exhaustiva bajo el productor actual: 4 nombres hablados unicos, 3 direcciones, 8 distancias enteras y 100 cantidades de peligros extra (0..99), mas 2 textos fijos. El modelo oficial actual solo produce pothole→bache; los demas son aliases aceptados por las funciones.", ["ID", "Rama", "Label_hablada", "Labels_entrada_aceptados", "Posible_con_modelo_oficial", "Direccion_codigo", "Direccion_hablada", "Distancia_m", "Peligros_extra", "Mensaje_resultante", "Cobertura"], rt_rows, [14, 20, 24, 54, 26, 20, 24, 14, 16, 104, 46], "TVariacionesRT")

    # 13 - QA y trazabilidad.
    source_files = [
        APP_ROOT / "lib/core/services/risk_assessment_service.dart",
        APP_ROOT / "lib/core/models/detection_result.dart",
        APP_ROOT / "lib/core/models/analysis_result.dart",
        APP_ROOT / "lib/core/services/object_detection_service.dart",
        APP_ROOT / "lib/core/services/vision_analysis_service.dart",
        APP_ROOT / "lib/main.dart",
        APP_ROOT / "assets/models/labels.txt",
        APP_ROOT / "test/widget_test.dart",
        APP_ROOT / "README.md",
        APP_ROOT / "ARQUITECTURA_APLICACION.md",
    ]
    qa_rows: list[list[Any]] = [
        ["COBERTURA", "10 etiquetas oficiales documentadas", "PASS", 10, 10, "02_ETIQUETAS", "—"],
        ["COBERTURA", "3 niveles documentados", "PASS", 3, 3, "04_REGLAS_PRECEDENCIA", "alto/medio/bajo"],
        ["COBERTURA", "13 plantillas textuales unicas", "PASS", len(MESSAGES), 13, "05_CATALOGO_MENSAJES", "14 rutas internas: vacio y fallback bajo reutilizan LOW_SAFE"],
        ["COBERTURA", "8 atributos AnalysisResult documentados", "PASS", 8, 8, "06_ATRIBUTOS_SALIDA", "Mas 7 propiedades/presentaciones derivadas"],
        ["COBERTURA", "Casos guiados", "PASS", len(guided_rows), 17, "07_CASOS_GUIADOS", "Incluye precedencia, fuera de rango y mock"],
        ["COBERTURA", "Escenarios validos de una deteccion", "PASS", len(valid_scenarios), 297, "07_CASOS_1_OBJETO", "33 combinaciones imposibles tambien visibles"],
        ["COBERTURA", "Pares de etiquetas", "PASS", len(pair_scenarios), 55, "08_CASOS_MULTI", "Perfil comun"],
        ["COBERTURA", "Fronteras de decision", "PASS", len(frontier_rows), len(frontier_rows), "09_PRUEBAS_FRONTERA", "Incluye upstream y display"],
        ["LIMITACION", "Combinaciones numericas literales", "WARN", "Infinitas", "Particiones equivalentes", "00_LEEME_RESUMEN", "confidence/bbox/N son continuos o abiertos"],
        ["PRUEBAS", "Tests unitarios directos del CERM", "WARN", 0, ">0 y cobertura de reglas/fronteras", "test/widget_test.dart:140-213", "Los tests actuales cubren el motor realtime, no RiskAssessmentService"],
        ["SEGURIDAD", "Fallo/modelo ausente produce mock bajo", "FAIL", "mock conf=.55", "Estado unknown/unavailable", "vision_analysis_service.dart:33-75; analysis_result.dart:70-80", "La voz dice camino libre y no revela el fallback"],
        ["SEGURIDAD", "Error de inferencia puede convertirse en lista vacia", "FAIL", "bajo conf=.65 source=tflite_float32", "Estado de error distinguible", "object_detection_service.dart:99-125; risk_assessment_service.dart:9-19", "Puede ser indistinguible de una captura genuinamente vacia"],
        ["SEGURIDAD", "Cero detecciones produce bajo", "WARN", "conf=.65", "Distinguir sin detecciones de libre", "risk_assessment_service.dart:9-19", "No prueba ausencia de peligro"],
        ["CONSISTENCIA", "Core y realtime usan mismas clases/reglas", "FAIL", "No", "Si", "11_COMPARATIVA_RT", "5 clases oficiales quedan info en realtime"],
        ["DETALLE", "CERM conserva bbox/direccion/distancia", "FAIL", "No", "Segun requisito", "06B_CANALES_SALIDA", "Se descartan al construir AnalysisResult"],
        ["CALIBRACION", "confidence es probabilidad de riesgo", "WARN", "No", "Calibracion requerida", "03_DERIVADAS_UMBRALES", "Media de score YOLO por dangerScore"],
        ["CONSISTENCIA", "leading siempre es quien activo el nivel", "FAIL", "No", "Mensaje alineado con disparador", "07_CASOS_GUIADOS G14-G15", "Puede describir otro objeto o afirmar cercania no cumplida"],
        ["DETERMINISMO", "Desempate secundario en dangerScore/conteos", "WARN", "No existe", "Criterio explicito", "risk_assessment_service.dart:22-23,244-251", "Empates pueden cambiar plantilla u orden de objetos"],
    ]
    for path in source_files:
        qa_rows.append(["HASH_SHA256", str(path.relative_to(APP_ROOT)), "PASS" if path.exists() else "FAIL", sha256(path) if path.exists() else "AUSENTE", "Snapshot reproducible", "Archivo local", "Hash calculado al generar"])
    write_table_sheet(workbook, formats, "13_QA_TRAZABILIDAD", "Controles de cobertura, hallazgos y huellas del codigo", "PASS confirma cobertura documental, no calidad del algoritmo. WARN/FAIL reflejan el comportamiento actual auditado y permiten rastrear futuras modificaciones.", ["Categoria", "Control_elemento", "Resultado_QA", "Valor_observado", "Esperado_objetivo", "Evidencia", "Nota_impacto"], qa_rows, [18, 44, 18, 70, 34, 44, 52], "TQATrazabilidad")

    # 14 - Glosario.
    glossary_rows = [
        ["UCSE", "Unidad de Captura y Sensado del Entorno", "Denominacion de tesis para captura, preprocesamiento y deteccion", "No es solo el sensor fisico", "Propuesta documental; no aparece como simbolo en codigo"],
        ["CERM", "Expansion no definida en el repositorio", "Nombre usado por la tesis/solicitud para el comportamiento de RiskAssessmentService", "No atribuir al codigo la expansion 'Componente de Evaluacion de Riesgo para la Movilidad' sin una fuente de tesis", "Convencion documental; RiskAssessmentService es el simbolo real"],
        ["DetectionResult", "Deteccion normalizada", "label + confidence + boundingBox", "No incluye distancia real", "detection_result.dart"],
        ["AnalysisResult", "Salida estructurada CERM", "8 atributos de resultado", "No conserva bbox/direccion/distancia", "analysis_result.dart"],
        ["isCenterArea", "Zona horizontal central", "centerX en [0.35,0.65]", "Difiere del realtime [.38,.62]", "detection_result.dart:23"],
        ["isBottomArea", "Zona inferior", "centerY>=.55 o bottom>=.68", "No equivale a distancia", "detection_result.dart:23"],
        ["isDangerZone", "Centro e inferior", "isCenterArea AND isBottomArea", "No es corredor libre calculado", "detection_result.dart:24"],
        ["isNear", "Heuristica de cercania", "area>=.06 o bottom>=.70", "No son metros", "risk_assessment_service.dart:119-121"],
        ["dangerous", "Lista interna de potencialmente peligrosos", "stairs; peligro de piso inferior; vehiculo near", "No incluye person/bicycle/pole/sign", "risk_assessment_service.dart:109-116"],
        ["leading", "Primera deteccion por dangerScore", "Elige el texto alto/medio", "Puede no haber activado la regla", "risk_assessment_service.dart:22-32"],
        ["dangerScore", "Puntuacion de orden", "peso clase*100 + peso zona*10 + area + confidence", "No es nivel ni probabilidad", "risk_assessment_service.dart:127-145"],
        ["speechText", "Mensaje hablado CERM", "Descripcion + recomendacion + accion", "No pronuncia nivel/objetos/confianza/fuente/tiempo/direccion/distancia", "analysis_result.dart:38-40"],
        ["Motor realtime", "Clasificador paralelo en main.dart", "HUD, alertas, direccion y distancia aproximada", "No debe llamarse CERM core sin aclaracion", "main.dart:4968-5350"],
        ["Particion de equivalencia", "Conjunto de entradas con la misma decision", "Permite cubrir dominios continuos finitamente", "No es muestreo estadistico", "Metodologia de este libro"],
    ]
    write_table_sheet(workbook, formats, "14_GLOSARIO", "Glosario operacional UCSE/CERM", "Definiciones vinculadas al codigo actual y aclaraciones para evitar afirmaciones mas fuertes que la implementacion.", ["Termino", "Expansion_definicion", "Significado_operacional", "No_confundir_con", "Fuente"], glossary_rows, [26, 48, 52, 44, 48], "TGlosario")

    workbook.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Ruta del .xlsx de salida")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_workbook(args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
