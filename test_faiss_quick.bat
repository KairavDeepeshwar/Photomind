@echo off
cd /d d:\Downloads\Project1_Photomind
set PYTHONPATH=src
call conda activate clipsex
echo.
echo ==========================================
echo Testing FAISS Integration (Quick Check)
echo ==========================================
echo.
pytest tests/test_faiss_index.py::test_auto_select_index_type -v
pytest tests/test_faiss_index.py::test_create_flat_index -v
pytest tests/test_faiss_index.py::test_search_faiss_index_flat -v
echo.
echo ==========================================
echo FAISS Quick Tests Complete
echo ==========================================
