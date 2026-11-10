@echo off
cd /d "%~dp0"
echo Installing dependencies...
pip install -q -r requirements.txt
echo.
echo Starting LOI Audio Scoring...
echo.
echo ==========================================
echo   Open http://localhost:8501 in browser
echo ==========================================
echo.
start "" "http://localhost:8501"
python app\app.py
pause
