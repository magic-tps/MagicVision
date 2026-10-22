// Generated from the active dataset; do not edit independently.
const campusClassNames = <String>[
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
  "trash_can",
  "ramp",
  "curb_step",
  "door",
  "glass_door",
  "chair",
  "table_desk",
  "backpack",
  "construction_barrier",
  "handrail",
  "cone",
  "fire_extinguisher",
];
const campusTaxonomy = "magicvision-campus22-v1";
const campusPolicyJson = r'''{"version": "detection-depth-v1", "priorities": {"critical": ["stairs", "pothole", "curb_step", "glass_door", "door", "obstacle", "ramp", "construction_barrier"], "high": ["trash_can", "cone", "person", "bicycle", "motorcycle", "car", "bus"], "medium": ["chair", "table_desk", "backpack", "pole"], "contextual": ["traffic_sign", "handrail", "fire_extinguisher"]}, "names_es": {"person": "Persona", "car": "Auto", "motorcycle": "Motocicleta", "bus": "Autobús", "bicycle": "Bicicleta", "stairs": "Escalera", "pothole": "Hueco", "pole": "Poste", "traffic_sign": "Señal de tránsito", "obstacle": "Obstáculo", "trash_can": "Tacho de basura", "ramp": "Rampa", "curb_step": "Desnivel", "door": "Puerta", "glass_door": "Puerta de vidrio", "chair": "Silla", "table_desk": "Mesa", "backpack": "Mochila", "construction_barrier": "Barrera de obra", "handrail": "Pasamanos", "cone": "Cono", "fire_extinguisher": "Extintor"}, "direction": {"left_max": 0.38, "right_min": 0.62}, "distance_thresholds_m": {"critical": 1.0, "high": 2.0, "medium": 4.0}, "geometry": {"bottom_min": 0.68, "large_area": 0.18, "near_area": 0.06}, "roi": {"default": [0.25, 0.25, 0.75, 0.75], "ground": [0.25, 0.65, 0.75, 0.95], "stairs": [0.2, 0.8, 0.8, 0.98], "door": [0.3, 0.35, 0.7, 0.85]}, "ground_classes": ["stairs", "pothole", "curb_step", "ramp", "trash_can", "cone", "construction_barrier", "chair", "table_desk", "backpack", "person", "car", "bus", "motorcycle", "bicycle", "pole"], "depth": {"model": "yolo26m-depth.pt", "imgsz": 640, "min_value": 0.02, "max_value": 150.0, "min_valid_fraction": 0.5, "min_pixels": 8, "trim_percentiles": [5, 95], "quantile": 25, "require_calibration_for_meters": true}, "tracking": {"iou_threshold": 0.3, "max_gap_ms": 3000, "history_size": 5, "smoothing": "rolling_median", "ema_alpha": 0.35, "approaching_min_samples": 3, "approaching_drop_m": 0.4, "approaching_min_span_ms": 800}, "alerts": {"cooldown_ms": 7000, "distance_change_m": 1.0, "max_per_cycle": 2}, "walking": {"enabled_by_default": false, "analysis_interval_ms": 1000, "max_result_age_ms": 3000, "evaluate_intervals_ms": [1000, 500, 333, 200], "stop_after_errors": 3}}''';
