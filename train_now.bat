@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Creating virtual environment, please wait...
  py -m venv .venv
  call .venv\Scripts\activate.bat
  pip install -r requirements.txt
  pip install -r requirements-train.txt
) else (
  call .venv\Scripts\activate.bat
)
python prepare_data.py all
python train.py --encoder resnet18 --crop 256 --crops-per-image 2 --epochs 15
echo.
echo Training finished. Now double-click run_app.bat
pause
