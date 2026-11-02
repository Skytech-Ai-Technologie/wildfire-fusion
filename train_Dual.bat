@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [train_Dual] .venv\Scripts\python.exe not found - run this from the wildfire-fusion repo root.
    pause
    exit /b 1
)

echo [train_Dual] starting dual-stream training on D-Fire
echo [train_Dual]   arch: RGB + LWIR YOLOv8n streams with CBAM, 150 epochs, batch 16
echo [train_Dual]   expect roughly 6-12 h on a laptop GPU; checkpoints save every epoch
echo [train_Dual]   interrupted? just run resume_Training.bat afterwards
echo.
.venv\Scripts\python.exe src\train_perceptual.py --dataset dfire --arch dual

echo.
echo [train_Dual] training ended - grade it with run_Benchmark.bat
pause
