@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  LabKnowMat-lite Environment Setup
echo ============================================================
echo.

where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] conda not found in PATH. Please install Miniconda first.
    pause
    exit /b 1
)

echo [Step 1/3] Creating conda environment (includes pip packages)...
conda env create -f environment.yml --name labknowmat-lite --yes
if %errorlevel% neq 0 (
    echo [WARN] Environment may already exist. Trying to update...
    conda env update -f environment.yml --name labknowmat-lite --prune
)

echo.
call conda activate labknowmat-lite
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate conda environment.
    pause
    exit /b 1
)

echo.
echo [Step 2/3] Installing sam3 from local source...
if exist "sam3\pyproject.toml" (
    pip install -e ./sam3
    if %errorlevel% neq 0 (
        echo [ERROR] sam3 installation failed.
        pause
        exit /b 1
    )
) else (
    echo [WARN] sam3/pyproject.toml not found. Skipping sam3 install.
)

echo.
echo [Step 3/3] Verifying installation...
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import cv2; print('opencv', cv2.__version__)"
python -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import paddleocr; print('paddleocr OK')"
python -c "import sam3; print('sam3 OK')"

echo.
echo ============================================================
echo  Setup complete! Activate with: conda activate labknowmat-lite
echo ============================================================
pause
