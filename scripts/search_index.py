"""
Search PhotoMind index with natural language queries.

Usage:
    python scripts/search_index.py <index_dir> <query> [k] [--faiss]

Arguments:
    index_dir: Directory containing the index
    query: Natural language query
    k: Number of results to return (default: 5)
    --faiss: Use FAISS index (auto-detected if not specified)

Examples:
    python scripts/search_index.py index_out "person wearing sunglasses" 5
    python scripts/search_index.py index_faiss "sunset beach" 10 --faiss
"""
from pathlib import Path
import sys
import argparse
from clipsex.model import CLIPModel, CLIPSpec
from clipsex.embedder import encode_text, topk_similar
from clipsex.index_store import load_index
from clipsex.faiss_index import load_faiss_index, search_faiss_index

def detect_index_type(index_dir: Path) -> str:
    """Detect if index is FAISS or NumPy based on files present."""
    if (index_dir / "index.faiss").exists():
        return "faiss"
    elif (index_dir / "embeddings.npz").exists():
        return "numpy"
    else:
        raise ValueError(f"No valid index found in {index_dir}")

def search(index_dir: Path, query: str, k: int = 5, use_faiss: bool = None):
    """Search index with natural language query."""
    # Auto-detect index type if not specified
    if use_faiss is None:
        index_type = detect_index_type(index_dir)
        use_faiss = (index_type == "faiss")
    
    print(f"Loading index from {index_dir}...")
    
    if use_faiss:
        # Load FAISS index
        index, paths, meta = load_faiss_index(index_dir)
        print(f"Loaded {meta.num_images} images ({meta.model_name}, {meta.pretrained})")
        print(f"Index type: {meta.index_type}")
        
        # Initialize CLIP model
        print(f"\nInitializing CLIP model...")
        spec = CLIPSpec(model_name=meta.model_name, pretrained=meta.pretrained)
        clip = CLIPModel(spec).load()
        
        # Encode query
        print(f"\nSearching for: '{query}'")
        query_emb = encode_text(query, clip)
        
        # Search FAISS index
        scores, indices = search_faiss_index(index, query_emb, k=k)
    else:
        # Load NumPy index
        embeddings, paths, meta = load_index(index_dir)
        print(f"Loaded {meta.num_images} images ({meta.model_name}, {meta.pretrained})")
        
        # Initialize CLIP model
        print(f"\nInitializing CLIP model...")
        spec = CLIPSpec(model_name=meta.model_name, pretrained=meta.pretrained)
        clip = CLIPModel(spec).load()
        
        # Encode query and search
        print(f"\nSearching for: '{query}'")
        query_emb = encode_text(query, clip)
        indices, scores = topk_similar(query_emb, embeddings, k=k)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"Top {k} Results:")
    print(f"{'='*80}")
    
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        print(f"\n{rank}. Score: {score:.4f}")
        print(f"   Path: {paths[idx]}")
    
    print(f"\n{'='*80}")
    return indices, scores, paths

def main():
    parser = argparse.ArgumentParser(
        description="Search PhotoMind index with natural language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("index_dir", help="Directory containing the index")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("k", nargs="?", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--faiss", action="store_true", help="Force FAISS index (auto-detected if not set)")
    
    args = parser.parse_args()
    
    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        print(f"Error: Index directory '{index_dir}' does not exist.")
        sys.exit(1)
    
    use_faiss = args.faiss if args.faiss else None
    search(index_dir, args.query, args.k, use_faiss)

if __name__ == "__main__":
    main()
