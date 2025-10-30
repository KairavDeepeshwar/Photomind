@echo off
cd /d d:\Downloads\Project1_Photomind
set PYTHONPATH=src
call conda activate clipsex
echo.
echo ==========================================
echo Running All FAISS Tests (Phase 2)
echo ==========================================
echo.
pytest tests/test_faiss_index.py -v
