import numpy as np
import pytest
from pathlib import Path
from clipsex.io_utils import build_manifest
from clipsex.model import CLIPModel, CLIPSpec
from clipsex.embedder import encode_images, encode_text, topk_similar
from clipsex.index_store import save_index, load_index, IndexMeta

@pytest.mark.slow
def test_encode_small_cpu(tmp_path):
    # Create 2 small images via Phase 1 fixtures
    from PIL import Image
    p1 = tmp_path / "a.jpg"; Image.new("RGB",(64,64),(120,30,10)).save(p1)
    p2 = tmp_path / "b.png"; Image.new("RGB",(70,50),(20,130,210)).save(p2)
    valid, rejected = build_manifest(tmp_path)
    assert len(valid) == 2

    clip = CLIPModel(CLIPSpec(), device="cpu").load()
    img_emb = encode_images(valid, clip, batch_size=2)
    assert img_emb.shape[0] == 2 and img_emb.ndim == 2
    # unit-norm check
    norms = np.linalg.norm(img_emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)

    txt = encode_text("a colorful square", clip)
    assert txt.shape[0] == 1

    idx, scores = topk_similar(txt, img_emb, k=2)
    assert len(idx) == 2 and len(scores) == 2

@pytest.mark.slow
def test_save_and_load_index(tmp_path):
    # minimal fake embeddings
    emb = np.eye(3, dtype=np.float32)
    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.jpg"
        p.write_bytes(b"x"); paths.append(p)
    from clipsex.index_store import save_index, load_index, IndexMeta
    meta = IndexMeta(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", dim=3, num_images=3)
    # build fake ImageRecord list for saving
    from clipsex.io_utils import ImageRecord
    records = [ImageRecord(path=p, width=1, height=1, format="JPEG") for p in paths]
    out = tmp_path / "index"
    save_index(emb, records, meta, out)
    emb2, paths2, meta2 = load_index(out)
    assert emb2.shape == emb.shape
    assert len(paths2) == 3
    assert meta2.model_name == "ViT-B-32"
