# Entrenamiento externo y modelo para SageMaker

El entrenamiento y reentrenamiento de MagicVision ocurren fuera de AWS. SageMaker se utiliza exclusivamente para inferencia del checkpoint YOLO26m completo.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install ultralytics opencv-python roboflow kaggle
python ml/scripts/train_yolo.py --model yolo26m.pt --data ml/configs/data.yaml --epochs 80 --imgsz 640 --batch 8 --workers 4 --device 0 --name modelo_final_yolo26m_80e_640_b8
```

Dataset final: `ml/datasets/dataset_final_magicvision`, con splits train/val/test. Los scripts de descarga, preparación, validación y entrenamiento se conservan. Ver README_DATASET.md.

Checkpoint conservado: `runs/detect/modelo_final_yolo26m_80e_640_b8/weights/best.pt`. Mejor época 50, early stopping en época 75; mAP50=0.58750, mAP50-95=0.35306, precisión=0.68426, recall=0.55655, según CSV histórico. No son métricas de latencia/precisión medidas tras migrar a UVC/cloud.

```powershell
python backend/package_model.py --weights runs/detect/modelo_final_yolo26m_80e_640_b8/weights/best.pt
```

Subir el artefacto generado a S3 privado y servirlo con el contenedor SageMaker descrito en `docs/AWS_DEPLOYMENT.md`. No se convierte a TFLite ni se incluye YOLO en Flutter. El notebook `ml/notebooks/magicvision_training_colab.ipynb` termina empaquetando y descargando `model.tar.gz`.

El export anterior permanece únicamente como evidencia en `ml/legacy_exports/`. El exportador antiguo está en el respaldo previo a la migración, no en el pipeline activo.
