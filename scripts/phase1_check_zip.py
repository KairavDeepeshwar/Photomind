from pathlib import Path
from clipsex.io_utils import build_manifest

valid, rejected = build_manifest(Path("my_photos"))
print("Valid:")
for r in valid:
    print(f" - {r.path.name} [{r.width}x{r.height}] fmt={r.format}")
print("\nRejected:")
for p in rejected:
    print(f" - {p.name}")
