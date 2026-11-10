@echo off
REM Double-click to launch the LOI frontend, then open the browser.
cd /d "%~dp0"
start "" http://127.0.0.1:5000
python app.py
pause
