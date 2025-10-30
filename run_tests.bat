@echo off
cd /d d:\Downloads\Project1_Photomind
set PYTHONPATH=src
call conda activate clipsex
pytest tests/test_search_index.py tests/test_faiss_index.py -v
