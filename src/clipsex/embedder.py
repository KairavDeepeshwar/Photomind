from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .io_utils import ImageRecord
from .model import CLIPModel

def encode_images(records: List[ImageRecord], clip: CLIPModel, batch_size: int = 32) -> np.ndarray:
    """Encode verified images into L2-normalized embeddings (float32)."""
    feats = []
    for i in tqdm(range(0, len(records), batch_size), desc="Encoding images"):
        batch = records[i:i+batch_size]
        imgs = []
        for r in batch:
            im = Image.open(r.path).convert("RGB")
            imgs.append(clip.preprocess(im))
        batch_tensor = torch.stack(imgs)
        emb = clip.encode_image_batch(batch_tensor)
        feats.append(emb.cpu())
    out = torch.cat(feats, dim=0).contiguous().cpu().numpy().astype(np.float32)
    # out should already be unit-norm; enforce to be safe
    norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    out = out / norms
    return out

def encode_text(query: str, clip: CLIPModel) -> np.ndarray:
    """Encode a single query into L2-normalized embedding (1, D)."""
    t = clip.encode_text([query]).detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    return t / norms

def topk_similar(text_vec: np.ndarray, image_mat: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of top-k cosine similarities."""
    # text_vec shape (1, D), image_mat shape (N, D)
    sims = (image_mat @ text_vec.T).squeeze(1)  # (N,)
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]
