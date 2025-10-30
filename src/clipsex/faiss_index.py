"""
FAISS-based approximate nearest neighbor search for scalable image retrieval.

Supports multiple index types:
- Flat: Exact search, best for < 10K images
- IVF: Inverted file index, good for 10K-500K images
- HNSW: Hierarchical Navigable Small World, best for 100K+ images
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Literal

import numpy as np
import faiss

from .io_utils import ImageRecord


IndexType = Literal["Flat", "IVF", "HNSW"]


@dataclass
class FAISSIndexMeta:
    """Metadata for FAISS index."""
    model_name: str
    pretrained: str
    dim: int
    num_images: int
    index_type: IndexType
    # IVF-specific parameters
    nlist: Optional[int] = None  # Number of clusters for IVF
    nprobe: Optional[int] = None  # Number of clusters to search
    # HNSW-specific parameters
    M: Optional[int] = None  # Number of connections per layer
    efConstruction: Optional[int] = None  # Construction time search depth
    efSearch: Optional[int] = None  # Search time depth


def auto_select_index_type(num_images: int) -> IndexType:
    """
    Automatically select the best index type based on collection size.
    
    Rules:
    - < 10,000: Flat (exact search is fast enough)
    - 10,000 - 100,000: IVF (good balance)
    - > 100,000: HNSW (best for very large collections)
    """
    if num_images < 10_000:
        return "Flat"
    elif num_images < 100_000:
        return "IVF"
    else:
        return "HNSW"


def create_faiss_index(
    embeddings: np.ndarray,
    index_type: Optional[IndexType] = None,
    nlist: Optional[int] = None,
    M: int = 32,
    efConstruction: int = 40
) -> Tuple[faiss.Index, dict]:
    """
    Create a FAISS index from embeddings.
    
    Args:
        embeddings: L2-normalized embeddings (N, D)
        index_type: Type of index to create. If None, auto-selected.
        nlist: Number of clusters for IVF (default: sqrt(N))
        M: HNSW connections per layer (default: 32)
        efConstruction: HNSW construction depth (default: 40)
    
    Returns:
        Tuple of (faiss.Index, params_dict)
    """
    num_images, dim = embeddings.shape
    
    # Auto-select index type if not specified
    if index_type is None:
        index_type = auto_select_index_type(num_images)
    
    # Ensure embeddings are float32 and contiguous
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
    
    params = {"index_type": index_type}
    
    if index_type == "Flat":
        # Exact search using inner product (cosine similarity for normalized vectors)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    
    elif index_type == "IVF":
        # Inverted file index with product quantization
        if nlist is None:
            nlist = min(int(np.sqrt(num_images)), num_images // 39)
            nlist = max(nlist, 1)  # Ensure at least 1 cluster
        
        # Use IndexIVFFlat for simplicity (can upgrade to IndexIVFPQ for compression)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train the index
        index.train(embeddings)
        index.add(embeddings)
        
        # Set search parameters (nprobe = number of clusters to search)
        nprobe = min(max(nlist // 10, 1), nlist)
        index.nprobe = nprobe
        
        params["nlist"] = nlist
        params["nprobe"] = nprobe
    
    elif index_type == "HNSW":
        # Hierarchical Navigable Small World graph
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = efConstruction
        index.add(embeddings)
        
        # Set search-time parameters
        efSearch = max(16, M * 2)
        index.hnsw.efSearch = efSearch
        
        params["M"] = M
        params["efConstruction"] = efConstruction
        params["efSearch"] = efSearch
    
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    return index, params


def search_faiss_index(
    index: faiss.Index,
    query_embedding: np.ndarray,
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index for top-k similar items.
    
    Args:
        index: FAISS index
        query_embedding: Query vector (1, D) or (D,)
        k: Number of results to return
    
    Returns:
        Tuple of (scores, indices) - both shape (k,) or (n,) if k > n
    """
    # Ensure query is 2D and float32
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))
    
    # Limit k to index size
    actual_k = min(k, index.ntotal)
    
    # Search
    scores, indices = index.search(query_embedding, actual_k)
    
    # Return as 1D arrays
    return scores[0], indices[0]


def save_faiss_index(
    index: faiss.Index,
    records: List[ImageRecord],
    meta: FAISSIndexMeta,
    out_dir: Path
) -> None:
    """
    Save FAISS index and metadata to disk.
    
    Args:
        index: FAISS index to save
        records: List of image records
        meta: Index metadata
        out_dir: Output directory
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, str(out_dir / "index.faiss"))
    
    # Save metadata and paths
    paths = [str(r.path) for r in records]
    payload = {
        "meta": asdict(meta),
        "paths": paths
    }
    (out_dir / "index.json").write_text(json.dumps(payload, indent=2))
    
    print(f"FAISS index saved: {meta.index_type} with {meta.num_images} images")


def load_faiss_index(in_dir: Path) -> Tuple[faiss.Index, List[str], FAISSIndexMeta]:
    """
    Load FAISS index and metadata from disk.
    
    Args:
        in_dir: Directory containing index files
    
    Returns:
        Tuple of (index, paths, meta)
    """
    in_dir = Path(in_dir)
    
    # Load FAISS index
    index = faiss.read_index(str(in_dir / "index.faiss"))
    
    # Load metadata
    payload = json.loads((in_dir / "index.json").read_text())
    paths = payload["paths"]
    
    # Extract only the fields that FAISSIndexMeta expects
    meta_dict = payload["meta"]
    faiss_meta = FAISSIndexMeta(
        model_name=meta_dict.get("model_name"),
        pretrained=meta_dict.get("pretrained"),
        dim=meta_dict.get("dim"),
        num_images=meta_dict.get("num_images"),
        index_type=meta_dict.get("index_type", "Flat"),
        nlist=meta_dict.get("nlist"),
        nprobe=meta_dict.get("nprobe"),
        M=meta_dict.get("M"),
        efConstruction=meta_dict.get("efConstruction"),
        efSearch=meta_dict.get("efSearch")
    )
    
    # Verify consistency
    assert index.ntotal == len(paths), "Index size / paths length mismatch"
    assert index.ntotal == faiss_meta.num_images, "Index size / metadata mismatch"
    
    return index, paths, faiss_meta


def add_to_faiss_index(
    index: faiss.Index,
    new_embeddings: np.ndarray,
    existing_paths: List[str],
    new_records: List[ImageRecord]
) -> Tuple[faiss.Index, List[str]]:
    """
    Add new embeddings to an existing FAISS index (incremental update).
    
    Args:
        index: Existing FAISS index
        new_embeddings: New embeddings to add (M, D)
        existing_paths: Current list of paths
        new_records: New image records
    
    Returns:
        Tuple of (updated_index, updated_paths)
    """
    # Ensure embeddings are float32 and contiguous
    new_embeddings = np.ascontiguousarray(new_embeddings.astype(np.float32))
    
    # Add to index
    index.add(new_embeddings)
    
    # Update paths
    new_paths = [str(r.path) for r in new_records]
    updated_paths = existing_paths + new_paths
    
    return index, updated_paths


def benchmark_index(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    k: int = 10
) -> dict:
    """
    Benchmark search performance of a FAISS index.
    
    Args:
        index: FAISS index to benchmark
        query_embeddings: Query vectors (N, D)
        k: Number of results per query
    
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    query_embeddings = np.ascontiguousarray(query_embeddings.astype(np.float32))
    num_queries = query_embeddings.shape[0]
    
    # Warmup
    index.search(query_embeddings[:1], k)
    
    # Actual benchmark
    start = time.perf_counter()
    scores, indices = index.search(query_embeddings, k)
    end = time.perf_counter()
    
    total_time = end - start
    avg_time = total_time / num_queries
    qps = num_queries / total_time
    
    return {
        "total_time_sec": total_time,
        "avg_time_ms": avg_time * 1000,
        "queries_per_sec": qps,
        "num_queries": num_queries,
        "k": k,
        "index_size": index.ntotal
    }
