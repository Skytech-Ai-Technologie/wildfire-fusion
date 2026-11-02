@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [run_Benchmark] .venv\Scripts\python.exe not found - run this from the wildfire-fusion repo root.
    pause
    exit /b 1
)

rem newest trained model first: Ultralytics runs keep weights\best.pt,
rem the custom dual-stream loop keeps best.pt directly in the run folder
set "WEIGHTS="
for /f "delims=" %%d in ('dir /b /ad /o-d runs\* 2^>nul') do (
    if not defined WEIGHTS if exist "runs\%%d\weights\best.pt" set "WEIGHTS=runs\%%d\weights\best.pt"
    if not defined WEIGHTS if exist "runs\%%d\best.pt" set "WEIGHTS=runs\%%d\best.pt"
)

if not defined WEIGHTS (
    echo [run_Benchmark] no trained model found under runs\ - train one first ^(or resume training^).
    pause
    exit /b 1
)

echo [run_Benchmark] evaluating %WEIGHTS% on the dfire test split ...
echo [run_Benchmark] (mAP@.5, image-level P/R/F1, bootstrap CI, confusion matrix, latency)
.venv\Scripts\python.exe benchmark\run_benchmark.py --table iv --dataset dfire --weights "%WEIGHTS%"

echo.
echo [run_Benchmark] done - results written to results\table_iv.csv
pause
