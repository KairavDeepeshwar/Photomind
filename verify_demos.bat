@echo off
cd /d d:\Downloads\Project1_Photomind
set PYTHONPATH=src
call conda activate clipsex
echo.
echo ==========================================
echo Running Demo Verification
echo ==========================================
echo.
python demos/verify_demos.py
pause
