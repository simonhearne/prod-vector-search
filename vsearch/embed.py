"""Text (Qwen3-Embedding-0.6B) and image/text (SigLIP ViT-L-16-256) embedders.

All outputs are float32 and L2-normalised. Models load lazily on first use.
"""
from pathlib import Path

import numpy as np

from vsearch.config import QUERY_PROMPT

TEXT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SIGLIP_ARCH, SIGLIP_PRETRAINED = "ViT-L-16-SigLIP-256", "webli"


def _device(device: str | None) -> str:
    if device:
        return device
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def truncate_mrl(vecs: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka truncation: keep the first `dim` dims and renormalise."""
    return _unit(np.array(vecs[:, :dim], dtype=np.float32, copy=True))


class TextEmbedder:
    def __init__(self, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.device = _device(device)
        self.model = SentenceTransformer(TEXT_MODEL, device=self.device)
        self.model.max_seq_length = 512

    def encode_docs(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return _unit(self.model.encode(list(texts), batch_size=batch_size,
                                       normalize_embeddings=True, convert_to_numpy=True))

    def encode_queries(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        return _unit(self.model.encode(list(texts), batch_size=batch_size, prompt=QUERY_PROMPT,
                                       normalize_embeddings=True, convert_to_numpy=True))


class ImageEmbedder:
    def __init__(self, device: str | None = None):
        import open_clip
        self.device = _device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            SIGLIP_ARCH, pretrained=SIGLIP_PRETRAINED, device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(SIGLIP_ARCH)

    def encode_images(self, paths: list[Path], batch_size: int = 64) -> np.ndarray:
        import torch
        from PIL import Image
        out = []
        for i in range(0, len(paths), batch_size):
            batch = torch.stack([self.preprocess(Image.open(p).convert("RGB"))
                                 for p in paths[i:i + batch_size]]).to(self.device)
            with torch.no_grad():
                out.append(self.model.encode_image(batch).float().cpu().numpy())
        return _unit(np.concatenate(out))

    def encode_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        import torch
        out = []
        for i in range(0, len(texts), batch_size):
            toks = self.tokenizer(list(texts[i:i + batch_size])).to(self.device)
            with torch.no_grad():
                out.append(self.model.encode_text(toks).float().cpu().numpy())
        return _unit(np.concatenate(out))
