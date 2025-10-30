from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from .io_utils import ImageRecord

@dataclass
class IndexMeta:
    model_name: str
    pretrained: str
    dim: int
    num_images: int
    index_type: Optional[str] = None  # For FAISS indices

def save_index(embeddings: np.ndarray, records: List[ImageRecord], meta: IndexMeta, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "embeddings.npz", embeddings=embeddings.astype(np.float32))
    paths = [str(r.path) for r in records]
    payload = {"meta": asdict(meta), "paths": paths}
    (out_dir / "index.json").write_text(json.dumps(payload, indent=2))

def load_index(in_dir: Path) -> tuple[np.ndarray, List[str], IndexMeta]:
    in_dir = Path(in_dir)
    data = np.load(in_dir / "embeddings.npz")
    emb = data["embeddings"].astype(np.float32)
    payload = json.loads((in_dir / "index.json").read_text())
    paths = payload["paths"]
    m = payload["meta"]
    
    # Only extract fields that IndexMeta expects, ignore extra FAISS parameters
    meta_fields = {
        'model_name': m.get('model_name'),
        'pretrained': m.get('pretrained'),
        'dim': m.get('dim'),
        'num_images': m.get('num_images'),
        'index_type': m.get('index_type')
    }
    meta = IndexMeta(**meta_fields)
    
    assert emb.shape[0] == len(paths), "Embeddings/paths length mismatch"
    return emb, paths, meta
