@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [resume_Training] .venv\Scripts\python.exe not found - run this from the wildfire-fusion repo root.
    pause
    exit /b 1
)

rem pick the newest run folder that still has a resumable checkpoint:
rem   - Ultralytics runs (single/baseline): weights\last.pt
rem   - custom dual-stream loop:           resume.pt
set "RUNDIR="
set "KIND="
for /f "delims=" %%d in ('dir /b /ad /o-d runs\* 2^>nul') do (
    if not defined RUNDIR if exist "runs\%%d\weights\last.pt" (
        set "RUNDIR=runs\%%d"
        set "KIND=ultralytics"
    )
    if not defined RUNDIR if exist "runs\%%d\resume.pt" (
        set "RUNDIR=runs\%%d"
        set "KIND=dual"
    )
)

if not defined RUNDIR (
    echo [resume_Training] no run with a resumable checkpoint found under runs\ - nothing to resume.
    pause
    exit /b 1
)

if "%KIND%"=="dual" (
    echo [resume_Training] resuming dual-stream run %RUNDIR% ...
    .venv\Scripts\python.exe src\train_perceptual.py --arch dual --resume "%RUNDIR%"
) else (
    echo [resume_Training] resuming from %RUNDIR%\weights\last.pt ...
    .venv\Scripts\python.exe -c "import sys; from ultralytics import YOLO; YOLO(sys.argv[1]).train(resume=True)" "%RUNDIR%\weights\last.pt"
)

echo.
echo [resume_Training] training process ended.
pause
