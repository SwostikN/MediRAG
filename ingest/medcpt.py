"""MedCPT encoder wrappers.

Article encoder = embed chunks at ingest time. Query encoder = embed the
user query at retrieval time. Both output 768-d vectors, [CLS]-pooled from
the last hidden state, per the MedCPT HF model card.

Paper: Jin et al., 2023. *MedCPT.* Bioinformatics 39(11).
Models: `ncbi/MedCPT-Article-Encoder`, `ncbi/MedCPT-Query-Encoder`.

CPU is fine at the Week 3 scale (hundreds of docs, ~40 chunks/sec on a
modern laptop). Lazy-load: the model only downloads on first use, so
importing this module is cheap.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from transformers import AutoModel, AutoTokenizer


_ARTICLE_MODEL_ID = "ncbi/MedCPT-Article-Encoder"
_QUERY_MODEL_ID = "ncbi/MedCPT-Query-Encoder"
_ARTICLE_MAX_LEN = 512
_QUERY_MAX_LEN = 64


class _Encoder:
    def __init__(self, model_id: str, max_length: int) -> None:
        self.model_id = model_id
        self.max_length = max_length
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id)
        self._model.eval()

    @torch.inference_mode()
    def _encode(self, inputs) -> list[list[float]]:
        self._lazy_load()
        enc = self._tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        out = self._model(**enc)
        cls = out.last_hidden_state[:, 0, :]
        return cls.cpu().tolist()


class ArticleEncoder(_Encoder):
    def __init__(self) -> None:
        super().__init__(_ARTICLE_MODEL_ID, _ARTICLE_MAX_LEN)

    def encode(self, pairs: Iterable[tuple[str, str]], batch_size: int = 8) -> list[list[float]]:
        """Encode an iterable of (title, content) pairs.

        The article encoder expects two segments per input. For patient-ed
        chunks without a distinct abstract, passing (doc_title, chunk_text)
        is the right shape.
        """
        pairs = list(pairs)
        if not pairs:
            return []
        embeddings: list[list[float]] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            embeddings.extend(self._encode([list(p) for p in batch]))
        return embeddings


class QueryEncoder(_Encoder):
    def __init__(self) -> None:
        super().__init__(_QUERY_MODEL_ID, _QUERY_MAX_LEN)

    def encode_one(self, text: str) -> list[float]:
        return self._encode([text])[0]


def to_pgvector_literal(vec: list[float]) -> str:
    """Format a Python list as a pgvector string literal, e.g. '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
