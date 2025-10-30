from pathlib import Path
import zipfile
import pytest
from PIL import Image

@pytest.fixture
def tmp_images(tmp_path: Path):
    # Create several valid images
    good = []
    for i in range(3):
        p = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (64 + i, 64 + i), (10 * i, 20, 30)).save(p, format="JPEG")
        good.append(p)
    # Corrupted image-like file
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not_an_image")
    # Non-image file
    (tmp_path / "note.txt").write_text("hello")
    return tmp_path, good, bad

@pytest.fixture
def tmp_zip(tmp_images):
    root, good, bad = tmp_images
    z = root / "photos.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in good + [bad]:
            zf.write(p, arcname=p.name)
    return z

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
