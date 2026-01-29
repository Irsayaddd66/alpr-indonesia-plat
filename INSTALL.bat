@echo off
echo ========================================
echo   INSTALLING DEPENDENCIES
echo ========================================
echo.

echo [1/4] Uninstalling old PyTorch...
pip uninstall torch torchvision torchaudio -y

echo.
echo [2/4] Installing PyTorch (CPU version)...
pip install torch torchvision torchaudio

echo.
echo [3/4] Installing other dependencies...
pip install ultralytics opencv-python easyocr numpy

echo.
echo [4/4] Verifying installation...
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import easyocr; print('EasyOCR: OK')"
python -c "from ultralytics import YOLO; print('Ultralytics: OK')"

echo.
echo ========================================
echo   INSTALLATION COMPLETE!
echo ========================================
echo.
pause
