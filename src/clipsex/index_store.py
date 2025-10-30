from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from .io_utils import ImageRecord

@dataclass
class IndexMeta:
    model_name: str
    pretrained: str
    dim: int
    num_images: int

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
    meta = IndexMeta(**m)
    assert emb.shape[0] == len(paths), "Embeddings/paths length mismatch"
    return emb, paths, meta
