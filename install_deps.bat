@echo off
cd /d d:\Downloads\Project1_Photomind
call conda activate clipsex
echo Installing Phase 2 dependencies...
pip install faiss-cpu matplotlib pyyaml
echo.
echo Dependencies installed successfully!
