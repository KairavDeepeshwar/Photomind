"""
Build PhotoMind index from a folder of images.

Usage:
    python scripts/build_index.py <images_folder> <out_dir> [--faiss] [--index-type TYPE]

Arguments:
    images_folder: Directory containing images to index
    out_dir: Output directory for index files
    --faiss: Use FAISS index instead of numpy (default: numpy)
    --index-type: FAISS index type (Flat, IVF, HNSW). Auto-selected if not specified.

Examples:
    python scripts/build_index.py my_photos index_out
    python scripts/build_index.py my_photos index_faiss --faiss
    python scripts/build_index.py my_photos index_faiss --faiss --index-type HNSW
"""
from pathlib import Path
import argparse
from clipsex.io_utils import build_manifest
from clipsex.model import CLIPModel, CLIPSpec
from clipsex.embedder import encode_images
from clipsex.index_store import save_index, IndexMeta
from clipsex.faiss_index import (
    create_faiss_index,
    save_faiss_index,
    FAISSIndexMeta,
    auto_select_index_type
)

def main(folder: str, out_dir: str, use_faiss: bool = False, index_type: str = None):
    """Build index from images folder."""
    print(f"Building index from: {folder}")
    print(f"Output directory: {out_dir}")
    print(f"Using FAISS: {use_faiss}")
    
    # Build manifest
    valid, rejected = build_manifest(Path(folder))
    print(f"Valid: {len(valid)} | Rejected: {len(rejected)}")
    
    if len(valid) == 0:
        print("Error: No valid images found!")
        return
    
    # Load CLIP model and encode images
    print("\nLoading CLIP model...")
    clip = CLIPModel(CLIPSpec()).load()
    
    print("\nEncoding images...")
    emb = encode_images(valid, clip, batch_size=32)
    
    print(f"\nGenerated {emb.shape[0]} embeddings of dimension {emb.shape[1]}")
    
    # Save index
    if use_faiss:
        # FAISS index
        if index_type:
            selected_type = index_type
        else:
            selected_type = auto_select_index_type(len(valid))
        
        print(f"\nCreating FAISS index (type: {selected_type})...")
        index, params = create_faiss_index(emb, index_type=selected_type)
        
        meta = FAISSIndexMeta(
            model_name=clip.spec.model_name,
            pretrained=clip.spec.pretrained,
            dim=emb.shape[1],
            num_images=emb.shape[0],
            index_type=params["index_type"],
            nlist=params.get("nlist"),
            nprobe=params.get("nprobe"),
            M=params.get("M"),
            efConstruction=params.get("efConstruction"),
            efSearch=params.get("efSearch")
        )
        
        save_faiss_index(index, valid, meta, Path(out_dir))
        print(f"\n✅ FAISS index saved to {out_dir}")
        print(f"   Index type: {meta.index_type}")
        if meta.nlist:
            print(f"   IVF clusters: {meta.nlist}, nprobe: {meta.nprobe}")
        if meta.M:
            print(f"   HNSW M: {meta.M}, efSearch: {meta.efSearch}")
    else:
        # NumPy index
        meta = IndexMeta(
            model_name=clip.spec.model_name,
            pretrained=clip.spec.pretrained,
            dim=emb.shape[1],
            num_images=emb.shape[0]
        )
        save_index(emb, valid, meta, Path(out_dir))
        print(f"\n✅ NumPy index saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build PhotoMind index from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("folder", help="Directory containing images to index")
    parser.add_argument("out_dir", help="Output directory for index files")
    parser.add_argument("--faiss", action="store_true", help="Use FAISS index (default: numpy)")
    parser.add_argument(
        "--index-type",
        choices=["Flat", "IVF", "HNSW"],
        help="FAISS index type (auto-selected if not specified)"
    )
    
    args = parser.parse_args()
    main(args.folder, args.out_dir, args.faiss, args.index_type)
