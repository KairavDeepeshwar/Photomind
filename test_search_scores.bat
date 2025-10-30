@echo off
cd /d "d:\Downloads\Project1_Photomind"
set PYTHONPATH=src
call conda activate clipsex

echo.
echo Testing query: "a person wearing a hat"
echo ============================================================
echo.

python -c "import sys; from pathlib import Path; sys.path.insert(0, 'src'); from clipsex.model import CLIPModel, CLIPSpec; from clipsex.embedder import encode_text, topk_similar; from clipsex.index_store import load_index; index_dir = Path('index_out'); embeddings, paths, meta = load_index(index_dir); clip_spec = CLIPSpec(model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'); clip = CLIPModel(clip_spec).load(); query = 'a person wearing a hat'; query_emb = encode_text(query, clip); indices, scores = topk_similar(query_emb, embeddings, k=8); print('\nRank  Score      Image'); print('='*50); [print(f'{rank:<6}{score:<11.6f}{Path(paths[idx]).name}') for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1)]; print('='*50)"

echo.
pause
