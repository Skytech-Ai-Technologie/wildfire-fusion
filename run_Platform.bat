@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [run_Platform] .venv\Scripts\python.exe not found - run this from the wildfire-fusion repo root.
    pause
    exit /b 1
)

echo [run_Platform] live wildfire detection platform
echo [run_Platform]   perceptual : dual-stream YOLOv8n + CBAM  (runs\dual_*\best.pt)
echo [run_Platform]   contextual : entropy-gated transformer    (runs\contextual_*\best.pt)
echo [run_Platform]   decisional : Dempster-Shafer fusion + alarm
echo.
echo [run_Platform] input sources (also switchable live from the UI):
echo [run_Platform]   PC camera    default, device 0   --camera N
echo [run_Platform]   video file   --source file --path clip.mp4
echo [run_Platform]   YouTube      --source youtube --youtube "https://youtu.be/XXXX"
echo [run_Platform]   test pattern --source synthetic
echo.
echo [run_Platform] opening http://127.0.0.1:8080 - press Ctrl+C here to stop.
echo.

.venv\Scripts\python.exe scripts\run_platform.py %*

echo.
echo [run_Platform] platform stopped.
pause
