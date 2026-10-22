@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
echo Entrenamiento YOLO26m - 22 clases - dataset anterior mas datos nuevos
echo 24.226 imagenes: 5.104 anteriores y 19.122 nuevas utilizables.
echo Separacion: 16.570 entrenamiento, 5.238 validacion y 2.418 prueba.
echo Se conservan las 998 revisiones; el resto usa sus etiquetas originales.
echo 100 epocas maximas, paciencia 20, imagen 640, batch maximo 4.
echo En cada epoca: duracion, tiempo transcurrido y tiempo restante estimado.
echo Al finalizar: tiempo total, epocas completadas y ruta del mejor modelo.
echo Se comprueban los archivos y la GPU antes de entrenar.
echo Evaluacion experimental: las etiquetas originales pueden estar incompletas.
echo.
set "CAMPUS22_PYTHON=C:\Users\magic\AppData\Local\Programs\Python\Python311\python.exe"
if not exist "%CAMPUS22_PYTHON%" set "CAMPUS22_PYTHON=python"
if "%~1"=="--check" (
  "%CAMPUS22_PYTHON%" magicvision_ml_improvement\training\campus22.py check --existing-labels --data magicvision_ml_improvement\datasets\campus22_full_existing_v1\data.yaml --report magicvision_ml_improvement\reports\campus22_full_existing_preflight.json
  exit /b
)
"%CAMPUS22_PYTHON%" -u magicvision_ml_improvement\training\campus22.py train --existing-labels --data magicvision_ml_improvement\datasets\campus22_full_existing_v1\data.yaml --weights magicvision_ml_improvement\models\yolo26m.pt --report magicvision_ml_improvement\reports\campus22_full_existing_training.json
set "CAMPUS22_EXIT=%ERRORLEVEL%"
echo.
if not "%CAMPUS22_EXIT%"=="0" (
  echo El entrenamiento se detuvo. Revisa el mensaje anterior y el reporte.
) else (
  echo Entrenamiento terminado. Pesos en magicvision_ml_improvement\training\runs.
)
pause
exit /b %CAMPUS22_EXIT%
