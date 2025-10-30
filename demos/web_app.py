"""
PhotoMind Web Demo - Interactive web interface for Phase 1 & 2 features.

Features:
- Image gallery view of indexed photos
- Natural language search with visual results
- FAISS vs NumPy comparison
- Performance metrics
- Live search as you type
"""

import os
import sys
import json
import time
import base64
from pathlib import Path
from io import BytesIO
from typing import List, Tuple, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
from PIL import Image

from clipsex.model import CLIPModel, CLIPSpec
from clipsex.embedder import encode_text, topk_similar
from clipsex.index_store import load_index
from clipsex.faiss_index import load_faiss_index, search_faiss_index, create_faiss_index

# Default CLIP model specification
DEFAULT_CLIP_SPEC = CLIPSpec(
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'photomind-demo-2025'

# Global state
state = {
    'numpy_index': None,
    'faiss_index': None,
    'clip_model': None,
    'image_paths': [],
    'embeddings': None,
    'index_meta': None,
    'photos_dir': None,
    'index_dir': None
}

def init_app():
    """Initialize application state."""
    base_dir = Path(__file__).parent.parent
    state['photos_dir'] = base_dir / "my_photos"
    state['index_dir'] = base_dir / "index_out"
    
    # Load CLIP model
    print("🔄 Loading CLIP model...")
    state['clip_model'] = CLIPModel(DEFAULT_CLIP_SPEC).load()
    print("✅ CLIP model loaded")
    
    # Load NumPy index
    if (state['index_dir'] / "embeddings.npz").exists():
        print("🔄 Loading NumPy index...")
        embeddings, paths, meta = load_index(state['index_dir'])
        state['numpy_index'] = {
            'embeddings': embeddings,
            'paths': paths,
            'meta': meta
        }
        state['embeddings'] = embeddings
        state['image_paths'] = paths
        state['index_meta'] = meta
        print(f"✅ Loaded {len(paths)} images from NumPy index")
    
    # Try to load FAISS index
    if (state['index_dir'] / "index.faiss").exists():
        print("🔄 Loading FAISS index...")
        try:
            index_obj, paths, meta = load_faiss_index(state['index_dir'])
            state['faiss_index'] = {
                'index': index_obj,
                'paths': paths,
                'meta': meta
            }
            print(f"✅ Loaded FAISS index ({meta.index_type})")
        except Exception as e:
            print(f"⚠️ Could not load FAISS index: {e}")

def get_image_base64(image_path: Path) -> str:
    """Convert image to base64 for embedding in HTML."""
    try:
        with Image.open(image_path) as img:
            # Resize for web display
            img.thumbnail((400, 400))
            buffered = BytesIO()
            img.save(buffered, format=img.format or "PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/{img.format.lower() if img.format else 'png'};base64,{img_str}"
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return ""

def search_numpy(query: str, k: int = 5) -> List[Dict]:
    """Search using NumPy index."""
    if not state['numpy_index']:
        return [], 0
    
    start = time.perf_counter()
    query_emb = encode_text(query, state['clip_model'])
    indices, scores = topk_similar(
        query_emb,
        state['numpy_index']['embeddings'],
        k=k
    )
    elapsed = (time.perf_counter() - start) * 1000
    
    # topk_similar returns 1D arrays, ensure they're iterable
    indices_arr = np.atleast_1d(indices)
    scores_arr = np.atleast_1d(scores)
    
    results = []
    for idx, score in zip(indices_arr, scores_arr):
        img_path = Path(state['numpy_index']['paths'][idx])
        results.append({
            'path': str(img_path.name),
            'full_path': str(img_path),
            'score': float(score),
            'image_b64': get_image_base64(img_path)
        })
    
    return results, elapsed

def search_faiss_wrapper(query: str, k: int = 5) -> List[Dict]:
    """Search using FAISS index."""
    if not state['faiss_index']:
        return [], 0
    
    start = time.perf_counter()
    query_emb = encode_text(query, state['clip_model'])
    scores, indices = search_faiss_index(  # Correct order: scores, indices
        state['faiss_index']['index'],
        query_emb,
        k=k
    )
    elapsed = (time.perf_counter() - start) * 1000
    
    # Ensure indices and scores are arrays, and convert indices to int
    indices_arr = np.atleast_1d(indices).astype(int)
    scores_arr = np.atleast_1d(scores)
    
    results = []
    for idx, score in zip(indices_arr, scores_arr):
        img_path = Path(state['faiss_index']['paths'][idx])
        results.append({
            'path': str(img_path.name),
            'full_path': str(img_path),
            'score': float(score),
            'image_b64': get_image_base64(img_path)
        })
    
    return results, elapsed

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/gallery')
def gallery():
    """Get all indexed images."""
    if not state['image_paths']:
        return jsonify({'error': 'No index loaded'}), 400
    
    images = []
    for path in state['image_paths'][:20]:  # Limit to 20 for performance
        img_path = Path(path)
        images.append({
            'name': img_path.name,
            'path': str(img_path),
            'image_b64': get_image_base64(img_path)
        })
    
    return jsonify({
        'total': len(state['image_paths']),
        'displayed': len(images),
        'images': images
    })

@app.route('/api/search', methods=['POST'])
def search():
    """Search endpoint."""
    data = request.json
    query = data.get('query', '')
    k = data.get('k', 5)
    method = data.get('method', 'numpy')  # 'numpy', 'faiss', or 'both'
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    response = {}
    
    if method in ['numpy', 'both']:
        if state['numpy_index']:
            results, elapsed = search_numpy(query, k)
            print(f"DEBUG: NumPy search returned {len(results)} results for k={k}")
            for i, r in enumerate(results[:3]):
                print(f"  Result {i+1}: {r['path']} - score {r['score']:.4f}")
            response['numpy'] = {
                'results': results,
                'elapsed_ms': round(elapsed, 2),
                'count': len(results)
            }
        else:
            response['numpy'] = {'error': 'NumPy index not loaded'}
    
    if method in ['faiss', 'both']:
        if state['faiss_index']:
            results, elapsed = search_faiss_wrapper(query, k)
            response['faiss'] = {
                'results': results,
                'elapsed_ms': round(elapsed, 2),
                'count': len(results),
                'index_type': state['faiss_index']['meta'].index_type
            }
        else:
            response['faiss'] = {
                'error': 'FAISS index not loaded. Build it with: python scripts/build_index.py my_photos index_out --faiss'
            }
    
    return jsonify(response)

@app.route('/api/stats')
def stats():
    """Get system statistics."""
    stats_data = {
        'numpy_loaded': state['numpy_index'] is not None,
        'faiss_loaded': state['faiss_index'] is not None,
        'clip_loaded': state['clip_model'] is not None,
        'num_images': len(state['image_paths']),
    }
    
    if state['numpy_index']:
        stats_data['numpy_dim'] = state['numpy_index']['embeddings'].shape[1]
    
    if state['faiss_index']:
        stats_data['faiss_index_type'] = state['faiss_index']['meta'].index_type
        stats_data['faiss_ntotal'] = state['faiss_index']['index'].ntotal
    
    if state['index_meta']:
        stats_data['clip_model'] = f"{state['index_meta'].model_name}::{state['index_meta'].pretrained}"
    
    return jsonify(stats_data)

@app.route('/api/compare', methods=['POST'])
def compare():
    """Compare NumPy vs FAISS performance."""
    data = request.json
    query = data.get('query', '')
    k = data.get('k', 5)
    trials = data.get('trials', 10)
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    results = {}
    
    # NumPy benchmark
    if state['numpy_index']:
        times = []
        for _ in range(trials):
            start = time.perf_counter()
            query_emb = encode_text(query, state['clip_model'])
            topk_similar(query_emb, state['numpy_index']['embeddings'], k=k)
            times.append((time.perf_counter() - start) * 1000)
        
        results['numpy'] = {
            'mean_ms': round(np.mean(times), 3),
            'std_ms': round(np.std(times), 3),
            'min_ms': round(np.min(times), 3),
            'max_ms': round(np.max(times), 3)
        }
    
    # FAISS benchmark
    if state['faiss_index']:
        times = []
        for _ in range(trials):
            start = time.perf_counter()
            query_emb = encode_text(query, state['clip_model'])
            search_faiss_index(state['faiss_index']['index'], query_emb, k=k)
            times.append((time.perf_counter() - start) * 1000)
        
        results['faiss'] = {
            'mean_ms': round(np.mean(times), 3),
            'std_ms': round(np.std(times), 3),
            'min_ms': round(np.min(times), 3),
            'max_ms': round(np.max(times), 3),
            'index_type': state['faiss_index']['meta'].index_type
        }
    
    # Calculate speedup
    if 'numpy' in results and 'faiss' in results:
        results['speedup'] = round(results['numpy']['mean_ms'] / results['faiss']['mean_ms'], 2)
    
    return jsonify(results)

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 PhotoMind Web Demo")
    print("=" * 60)
    init_app()
    print("\n" + "=" * 60)
    print("🌐 Starting web server...")
    print("📍 Open http://localhost:5000 in your browser")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server.\n")
    
    # Disable debug mode to prevent memory issues with reloader
    app.run(debug=False, host='0.0.0.0', port=5000)
