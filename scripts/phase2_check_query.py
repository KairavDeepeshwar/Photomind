from pathlib import Path
from clipsex.model import CLIPModel, CLIPSpec
from clipsex.embedder import encode_text, topk_similar
from clipsex.index_store import load_index

# Load index built in the previous step
emb, paths, meta = load_index(Path("index_out"))
print("Index:", emb.shape, "items")

# Load CLIP (CPU/GPU auto)
clip = CLIPModel(CLIPSpec()).load()

# Try a few queries
queries = [
    "picture of a leaf",
    "picture of a buddha sitting",
    "close-up portrait of a person",
]
for q in queries:
    tv = encode_text(q, clip)
    idx, scores = topk_similar(tv, emb, k=3)
    print(f"\nQuery: {q}")
    for i, s in zip(idx, scores):
        print(f"  {paths[i]}  score={float(s):.4f}")
