from pathlib import Path
from clipsex.io_utils import (
    SUPPORTED_EXTS,
    is_supported_image,
    verify_image,
    list_images,
    build_manifest,
    extract_zip,
)

def test_supported_exts_has_common():
    for ext in (".jpg", ".jpeg", ".png"):
        assert ext in SUPPORTED_EXTS

def test_is_supported_image(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.txt").write_text("hi")
    assert is_supported_image(tmp_path / "a.jpg") is True
    assert is_supported_image(tmp_path / "b.txt") is False

def test_verify_image_metadata(tmp_path):
    from PIL import Image
    p = tmp_path / "ok.png"
    Image.new("RGB", (80, 60), (0, 0, 0)).save(p, format="PNG")
    rec = verify_image(p)
    assert rec is not None
    assert rec.width == 80 and rec.height == 60
    assert rec.path == p

def test_verify_image_rejects_corrupt(tmp_path):
    p = tmp_path / "bad.jpeg"
    p.write_bytes(b"not_image")
    assert verify_image(p) is None

def test_manifest_filters_corrupt(tmp_images):
    root, good_paths, bad = tmp_images
    files = list_images(root)
    # extension discovery includes broken.png even if corrupt
    assert (root / "broken.png") in files
    valid, rejected = build_manifest(root)
    assert set([r.path for r in valid]) == set(good_paths)
    assert (root / "broken.png") in rejected

def test_extract_zip(tmp_zip, tmp_path):
    dest = tmp_path / "out"
    out = extract_zip(tmp_zip, dest)
    assert len(out) >= 3  # at least good + corrupt extracted
    for p in out:
        assert p.exists()
