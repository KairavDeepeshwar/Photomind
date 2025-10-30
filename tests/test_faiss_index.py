"""
Tests for FAISS index functionality.
"""
from pathlib import Path
import numpy as np
import pytest
import faiss

from clipsex.faiss_index import (
    auto_select_index_type,
    create_faiss_index,
    search_faiss_index,
    save_faiss_index,
    load_faiss_index,
    add_to_faiss_index,
    benchmark_index,
    FAISSIndexMeta
)
from clipsex.io_utils import ImageRecord


def test_auto_select_index_type():
    """Test automatic index type selection based on dataset size."""
    assert auto_select_index_type(1000) == "Flat"
    assert auto_select_index_type(9999) == "Flat"
    assert auto_select_index_type(10_000) == "IVF"
    assert auto_select_index_type(50_000) == "IVF"
    assert auto_select_index_type(100_000) == "HNSW"
    assert auto_select_index_type(500_000) == "HNSW"


def test_create_flat_index():
    """Test creation of Flat FAISS index."""
    embeddings = np.random.randn(100, 512).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    index, params = create_faiss_index(embeddings, index_type="Flat")
    
    assert isinstance(index, faiss.IndexFlatIP)
    assert index.ntotal == 100
    assert index.d == 512
    assert params["index_type"] == "Flat"


def test_create_ivf_index():
    """Test creation of IVF FAISS index."""
    embeddings = np.random.randn(1000, 128).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    index, params = create_faiss_index(embeddings, index_type="IVF", nlist=10)
    
    assert isinstance(index, faiss.IndexIVFFlat)
    assert index.ntotal == 1000
    assert index.d == 128
    assert params["index_type"] == "IVF"
    assert params["nlist"] == 10
    assert "nprobe" in params


def test_create_hnsw_index():
    """Test creation of HNSW FAISS index."""
    embeddings = np.random.randn(500, 256).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    index, params = create_faiss_index(embeddings, index_type="HNSW", M=16)
    
    assert isinstance(index, faiss.IndexHNSWFlat)
    assert index.ntotal == 500
    assert index.d == 256
    assert params["index_type"] == "HNSW"
    assert params["M"] == 16


def test_search_faiss_index_flat():
    """Test searching Flat FAISS index."""
    # Create index with known embeddings
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
    ], dtype=np.float32)
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    index, _ = create_faiss_index(embeddings, index_type="Flat")
    
    # Query similar to first embedding
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    query = query / np.linalg.norm(query)
    
    scores, indices = search_faiss_index(index, query, k=2)
    
    assert len(scores) == 2
    assert len(indices) == 2
    assert indices[0] == 0  # First embedding should be most similar
    assert scores[0] > scores[1]  # Scores should be descending


def test_search_faiss_index_2d_query():
    """Test search with 2D query array."""
    embeddings = np.random.randn(50, 128).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    index, _ = create_faiss_index(embeddings, index_type="Flat")
    
    # 2D query
    query = np.random.randn(1, 128).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    scores, indices = search_faiss_index(index, query, k=5)
    
    assert len(scores) == 5
    assert len(indices) == 5
    assert all(0 <= idx < 50 for idx in indices)


def test_save_and_load_faiss_index(tmp_path):
    """Test saving and loading FAISS index."""
    # Create index
    embeddings = np.random.randn(100, 512).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    index, params = create_faiss_index(embeddings, index_type="Flat")
    
    # Create metadata and records
    records = [
        ImageRecord(
            path=tmp_path / f"image_{i}.jpg",
            width=800,
            height=600,
            format="JPEG"
        )
        for i in range(100)
    ]
    
    meta = FAISSIndexMeta(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        dim=512,
        num_images=100,
        index_type="Flat"
    )
    
    # Save
    index_dir = tmp_path / "faiss_index"
    save_faiss_index(index, records, meta, index_dir)
    
    # Verify files exist
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "index.json").exists()
    
    # Load
    loaded_index, loaded_paths, loaded_meta = load_faiss_index(index_dir)
    
    # Verify
    assert loaded_index.ntotal == 100
    assert loaded_index.d == 512
    assert len(loaded_paths) == 100
    assert loaded_meta.num_images == 100
    assert loaded_meta.index_type == "Flat"
    assert loaded_meta.model_name == "ViT-B-32"


def test_save_and_load_ivf_index(tmp_path):
    """Test saving and loading IVF index with parameters."""
    embeddings = np.random.randn(1000, 256).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    index, params = create_faiss_index(embeddings, index_type="IVF")
    
    records = [
        ImageRecord(
            path=tmp_path / f"img_{i}.jpg",
            width=1024,
            height=768,
            format="JPEG"
        )
        for i in range(1000)
    ]
    
    meta = FAISSIndexMeta(
        model_name="ViT-L-14",
        pretrained="openai",
        dim=256,
        num_images=1000,
        index_type="IVF",
        nlist=params["nlist"],
        nprobe=params["nprobe"]
    )
    
    index_dir = tmp_path / "ivf_index"
    save_faiss_index(index, records, meta, index_dir)
    
    loaded_index, loaded_paths, loaded_meta = load_faiss_index(index_dir)
    
    assert isinstance(loaded_index, faiss.IndexIVFFlat)
    assert loaded_meta.nlist is not None
    assert loaded_meta.nprobe is not None


def test_add_to_faiss_index():
    """Test incremental addition to FAISS index."""
    # Initial embeddings
    initial_emb = np.random.randn(50, 128).astype(np.float32)
    norms = np.linalg.norm(initial_emb, axis=1, keepdims=True)
    initial_emb = initial_emb / norms
    
    index, _ = create_faiss_index(initial_emb, index_type="Flat")
    initial_paths = [f"image_{i}.jpg" for i in range(50)]
    
    # New embeddings
    new_emb = np.random.randn(20, 128).astype(np.float32)
    norms = np.linalg.norm(new_emb, axis=1, keepdims=True)
    new_emb = new_emb / norms
    
    new_records = [
        ImageRecord(
            path=Path(f"new_{i}.jpg"),
            width=800,
            height=600,
            format="JPEG"
        )
        for i in range(20)
    ]
    
    # Add to index
    updated_index, updated_paths = add_to_faiss_index(
        index, new_emb, initial_paths, new_records
    )
    
    assert updated_index.ntotal == 70
    assert len(updated_paths) == 70
    assert updated_paths[:50] == initial_paths
    assert all("new_" in path for path in updated_paths[50:])


def test_benchmark_index():
    """Test index benchmarking."""
    embeddings = np.random.randn(1000, 128).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    index, _ = create_faiss_index(embeddings, index_type="Flat")
    
    # Create query embeddings
    query_emb = np.random.randn(10, 128).astype(np.float32)
    norms = np.linalg.norm(query_emb, axis=1, keepdims=True)
    query_emb = query_emb / norms
    
    stats = benchmark_index(index, query_emb, k=10)
    
    assert "total_time_sec" in stats
    assert "avg_time_ms" in stats
    assert "queries_per_sec" in stats
    assert stats["num_queries"] == 10
    assert stats["k"] == 10
    assert stats["index_size"] == 1000
    assert stats["avg_time_ms"] > 0


def test_cosine_similarity_preserved():
    """Test that FAISS inner product equals cosine similarity for normalized vectors."""
    # Create normalized vectors
    embeddings = np.random.randn(100, 64).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    query = np.random.randn(1, 64).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    # FAISS search
    index, _ = create_faiss_index(embeddings, index_type="Flat")
    faiss_scores, faiss_indices = search_faiss_index(index, query, k=5)
    
    # Manual cosine similarity
    manual_sims = (embeddings @ query.T).squeeze()
    manual_indices = np.argsort(-manual_sims)[:5]
    manual_scores = manual_sims[manual_indices]
    
    # Compare
    np.testing.assert_array_equal(faiss_indices, manual_indices)
    np.testing.assert_allclose(faiss_scores, manual_scores, rtol=1e-5)


def test_index_with_small_dataset():
    """Test FAISS works with very small datasets."""
    embeddings = np.random.randn(5, 32).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    # Flat should work
    index, _ = create_faiss_index(embeddings, index_type="Flat")
    assert index.ntotal == 5
    
    # IVF should handle gracefully
    index_ivf, params = create_faiss_index(embeddings, index_type="IVF")
    assert index_ivf.ntotal == 5
    assert params["nlist"] >= 1


def test_search_k_larger_than_index():
    """Test search when k > index size."""
    embeddings = np.random.randn(10, 64).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    index, _ = create_faiss_index(embeddings, index_type="Flat")
    
    query = np.random.randn(1, 64).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    # Request more results than available
    scores, indices = search_faiss_index(index, query, k=20)
    
    # Should return all 10 results
    assert len(scores) == 10
    assert len(indices) == 10
