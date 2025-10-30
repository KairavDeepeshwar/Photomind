"""
Quick test to show similarity scores for "a person wearing a hat"
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clipsex.model import CLIPModel, CLIPSpec
from clipsex.embedder import encode_text, topk_similar
from clipsex.index_store import load_index

# Load index and model
index_dir = Path(__file__).parent.parent / "index_out"
embeddings, paths, meta = load_index(index_dir)

clip_spec = CLIPSpec(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")
clip = CLIPModel(clip_spec).load()

# Search
query = "a person wearing a hat"
print(f"\n🔍 Query: '{query}'\n")
print("=" * 70)

query_emb = encode_text(query, clip)
indices, scores = topk_similar(query_emb, embeddings, k=8)  # All 8 images

print(f"{'Rank':<6} {'Score':<12} {'Image':<30}")
print("=" * 70)

for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
    img_path = Path(paths[idx])
    print(f"{rank:<6} {score:<12.6f} {img_path.name:<30}")

print("=" * 70)
print("\n✨ Higher scores = more similar to query")
print("🎯 image3.png (man with hat) should be near the top!")
