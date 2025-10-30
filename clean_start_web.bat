@echo off
echo ============================================================
echo Stopping ALL Python processes...
echo ============================================================

taskkill /F /IM python.exe /T 2>nul
taskkill /F /IM pythonw.exe /T 2>nul

echo.
echo Waiting 3 seconds...
timeout /t 3 /nobreak >nul

echo.
echo ============================================================
echo Starting fresh PhotoMind web server...
echo ============================================================

cd /d "d:\Downloads\Project1_Photomind"
set PYTHONPATH=src
call conda activate clipsex
python demos\web_app.py

pause
