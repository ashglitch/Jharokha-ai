@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo Fine-tuning your saved model so it also sees thin cracks in wide shots.
echo A backup of your current weights is saved as weights\crack_unet_before_resume.pt
python train.py --resume --encoder resnet18 --crop 256 --crops-per-image 2 --epochs 6 --lr 2e-4 --thin-prob 0.5
echo.
echo Done. Now double-click run_app.bat
pause
