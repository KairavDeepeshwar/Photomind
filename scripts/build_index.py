from pathlib import Path
from clipsex.io_utils import build_manifest
from clipsex.model import CLIPModel, CLIPSpec
from clipsex.embedder import encode_images
from clipsex.index_store import save_index, IndexMeta

def main(folder: str, out_dir: str):
    valid, rejected = build_manifest(Path(folder))
    print(f"Valid: {len(valid)} | Rejected: {len(rejected)}")
    clip = CLIPModel(CLIPSpec()).load()
    emb = encode_images(valid, clip, batch_size=32)
    meta = IndexMeta(
      model_name=clip.spec.model_name,
      pretrained=clip.spec.pretrained,
      dim=emb.shape[1],
      num_images=emb.shape[0]
    )
    save_index(emb, valid, meta, Path(out_dir))
    print(f"Index saved to {out_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python scripts/build_index.py <images_folder> <out_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
