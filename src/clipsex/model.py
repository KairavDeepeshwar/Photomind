from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import open_clip

class ModelNotInitialized(Exception):
    pass

@dataclass
class CLIPSpec:
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"  # robust default from open_clip

class CLIPModel:
    def __init__(self, spec: Optional[CLIPSpec] = None, device: Optional[str] = None):
        self.spec = spec or CLIPSpec()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self.tokenizer = None

    def load(self) -> "CLIPModel":
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.spec.model_name,
            pretrained=self.spec.pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.spec.model_name)
        self.model.eval()
        return self

    def encode_image_batch(self, batch_tensor) -> torch.Tensor:
        if self.model is None:
            raise ModelNotInitialized("Call load() first.")
        with torch.no_grad():
            feats = self.model.encode_image(batch_tensor.to(self.device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        if self.model is None or self.tokenizer is None:
            raise ModelNotInitialized("Call load() first.")
        with torch.no_grad():
            toks = self.tokenizer(prompts).to(self.device)
            feats = self.model.encode_text(toks)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats
