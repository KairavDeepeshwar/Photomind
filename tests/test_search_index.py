"""
Tests for search_index.py script and search functionality.
"""
from pathlib import Path
import numpy as np
import pytest

from clipsex.model import CLIPModel, CLIPSpec
from clipsex.embedder import encode_text, topk_similar
from clipsex.index_store import save_index, load_index, IndexMeta
from clipsex.io_utils import ImageRecord


@pytest.fixture
def mock_index_dir(tmp_path):
    """Create a mock index directory with sample embeddings."""
    # Create 10 random embeddings (dimension 512, typical for ViT-B-32)
    dim = 512
    num_images = 10
    embeddings = np.random.randn(num_images, dim).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    embeddings = embeddings / norms
    
    # Create mock image records
    records = [
        ImageRecord(
            path=tmp_path / f"image_{i}.jpg",
            width=800,
            height=600,
            format="JPEG"
        )
        for i in range(num_images)
    ]
    
    # Create mock image files (empty is fine for this test)
    for rec in records:
        rec.path.touch()
    
    # Save index
    meta = IndexMeta(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        dim=dim,
        num_images=num_images
    )
    index_dir = tmp_path / "index"
    save_index(embeddings, records, meta, index_dir)
    
    return index_dir, embeddings, records


def test_topk_similar_basic():
    """Test topk_similar returns correct top-k results."""
    # Create a simple test case
    image_mat = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
    ], dtype=np.float32)
    
    # Normalize
    norms = np.linalg.norm(image_mat, axis=1, keepdims=True)
    image_mat = image_mat / norms
    
    # Query vector similar to first embedding
    query = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    
    indices, scores = topk_similar(query, image_mat, k=2)
    
    assert len(indices) == 2
    assert len(scores) == 2
    assert indices[0] == 0  # First embedding should be most similar
    assert scores[0] > scores[1]  # Scores should be descending


def test_topk_similar_with_k_larger_than_n():
    """Test topk_similar when k > number of images."""
    image_mat = np.random.randn(3, 512).astype(np.float32)
    norms = np.linalg.norm(image_mat, axis=1, keepdims=True)
    image_mat = image_mat / norms
    
    query = np.random.randn(1, 512).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    indices, scores = topk_similar(query, image_mat, k=10)
    
    # Should return all 3 results
    assert len(indices) == 3
    assert len(scores) == 3


def test_load_index(mock_index_dir):
    """Test loading index returns correct data."""
    index_dir, original_embeddings, original_records = mock_index_dir
    
    embeddings, paths, meta = load_index(index_dir)
    
    assert embeddings.shape == original_embeddings.shape
    assert len(paths) == len(original_records)
    assert meta.num_images == len(original_records)
    assert meta.dim == 512
    assert meta.model_name == "ViT-B-32"
    
    # Check embeddings are still normalized
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_encode_text_normalization():
    """Test that encode_text returns L2-normalized embeddings."""
    spec = CLIPSpec(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")
    clip = CLIPModel(spec).load()
    
    query_emb = encode_text("a photo of a cat", clip)
    
    # Check shape
    assert query_emb.shape[0] == 1
    assert query_emb.shape[1] > 0  # Should have some dimensionality
    
    # Check normalization
    norm = np.linalg.norm(query_emb)
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_search_integration(mock_index_dir):
    """Integration test: load index and perform search."""
    index_dir, embeddings, records = mock_index_dir
    
    # Load index
    loaded_emb, paths, meta = load_index(index_dir)
    
    # Load CLIP model
    spec = CLIPSpec(model_name=meta.model_name, pretrained=meta.pretrained)
    clip = CLIPModel(spec).load()
    
    # Encode query
    query_emb = encode_text("test query", clip)
    
    # Search
    indices, scores = topk_similar(query_emb, loaded_emb, k=5)
    
    # Verify results
    assert len(indices) == 5
    assert len(scores) == 5
    assert all(0 <= idx < len(paths) for idx in indices)
    assert all(0.0 <= score <= 1.0 for score in scores)
    
    # Scores should be in descending order
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


def test_search_script_import():
    """Test that search_index.py can be imported without errors."""
    import sys
    from pathlib import Path
    
    # Add scripts to path
    scripts_path = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(scripts_path))
    
    try:
        import search_index
        assert hasattr(search_index, 'search')
        assert hasattr(search_index, 'main')
    finally:
        sys.path.pop(0)


def test_cosine_similarity_properties():
    """Test mathematical properties of cosine similarity computation."""
    # Create two identical vectors
    vec1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    image_mat = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    
    indices, scores = topk_similar(vec1, image_mat, k=1)
    
    # Cosine similarity of identical normalized vectors should be 1.0
    assert np.isclose(scores[0], 1.0, atol=1e-5)
    
    # Orthogonal vectors
    vec2 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    image_mat2 = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    
    indices, scores = topk_similar(vec2, image_mat2, k=1)
    
    # Cosine similarity of orthogonal vectors should be 0.0
    assert np.isclose(scores[0], 0.0, atol=1e-5)
