@echo off
setlocal

set "PACKAGE_ROOT=%~dp0"
set "CODE_ROOT=%PACKAGE_ROOT%code"
set "OUTPUT_ROOT=%PACKAGE_ROOT%..\experiment_outputs\size_weighted"
set "CONDA_EXE=D:\Users\nieyuhang\miniconda3\Scripts\conda.exe"

if not exist "%CONDA_EXE%" (
  echo Conda executable not found: %CONDA_EXE%
  exit /b 1
)

cd /d "%CODE_ROOT%"
if errorlevel 1 exit /b 1

echo Running package preflight...
"%CONDA_EXE%" run --no-capture-output -n polyolefin_ml python -u -m scripts.server_batch.run_job --bundle "%PACKAGE_ROOT%" --output-dir "%OUTPUT_ROOT%\preflight" --variant region_size_weighted --seed 0 --action prepare
if errorlevel 1 exit /b 1

echo Starting six-job GPU queue...
"%CONDA_EXE%" run --no-capture-output -n polyolefin_ml python -u -m scripts.server_batch.launch --bundle "%PACKAGE_ROOT%" --output-dir "%OUTPUT_ROOT%\jobs" --gpus 0 1
if errorlevel 1 exit /b 1

echo Collecting validation-only results...
"%CONDA_EXE%" run --no-capture-output -n polyolefin_ml python -u -m scripts.server_batch.collect --bundle "%PACKAGE_ROOT%" --output-dir "%OUTPUT_ROOT%\jobs"
if errorlevel 1 exit /b 1

echo Experiment completed: %OUTPUT_ROOT%
endlocal
