@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo Open http://127.0.0.1:5000 in your browser
python app.py
pause
