from __future__ import annotations
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

from PIL import Image, UnidentifiedImageError

# Supported file extensions for discovery
SUPPORTED_EXTS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"
}

@dataclass(frozen=True)
class ImageRecord:
    """Minimal metadata for a verified image."""
    path: Path
    width: int
    height: int
    format: str

def is_supported_image(path: Path) -> bool:
    """Fast extension-based filter without IO."""
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTS

def verify_image(path: Path) -> Optional[ImageRecord]:
    """
    Open with Pillow in a corruption-safe manner and return metadata.
    Returns None if the file is not a valid image or is unsupported/corrupted.
    """
    if not is_supported_image(path):
        return None
    try:
        # verify() checks integrity; must reopen to access metadata
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im2:
            w, h = im2.size
            fmt = im2.format or path.suffix.upper().lstrip(".")
        return ImageRecord(path=path, width=w, height=h, format=str(fmt))
    except (UnidentifiedImageError, OSError, ValueError):
        return None

def list_images(root: Path) -> List[Path]:
    """Recursively list candidate image files by extension only."""
    root = Path(root)
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if is_supported_image(p)]

def build_manifest(root: Path) -> Tuple[List[ImageRecord], List[Path]]:
    """
    Verify all candidate images and split into (valid_records, rejected_paths).
    Sorting is applied for reproducible order.
    """
    candidates = list_images(root)
    valid: List[ImageRecord] = []
    rejected: List[Path] = []
    for p in candidates:
        rec = verify_image(p)
        if rec is None:
            rejected.append(p)
        else:
            valid.append(rec)
    valid.sort(key=lambda r: str(r.path).lower())
    rejected.sort(key=lambda p: str(p).lower())
    return valid, rejected

def extract_zip(zip_path: Path, dest_dir: Path, overwrite: bool = False) -> List[Path]:
    """
    Extract images from a .zip into dest_dir and return extracted file paths.
    - Flattens internal paths to filenames (prevents zip-slip).
    - Skips existing files unless overwrite=True.
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    if not zip_path.exists():
        raise FileNotFoundError(f"zip not found: {zip_path}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    extracted: List[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            # Flatten path and protect against traversal
            target = dest_dir / Path(info.filename).name
            if target.exists() and not overwrite:
                extracted.append(target)
                continue
            with zf.open(info, "r") as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted.append(target)
    return extracted
