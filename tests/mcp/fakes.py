# -*- coding: utf-8 -*-
"""测试替身：确定性哈希 embedding，无网络无 key 可跑。"""
import hashlib
from typing import List

DIM = 256


class FakeEmbedding:
    """256 维确定性哈希向量：同文本同向量，无需 API key。"""

    def __init__(self, dim: int = DIM):
        self.dim = dim

    def _vector(self, text: str) -> List[float]:
        out = []
        counter = 0
        while len(out) < self.dim:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            for b in digest:
                out.append(b / 255.0)
                if len(out) >= self.dim:
                    break
            counter += 1
        return out

    def embed_text(self, content: str) -> List[float]:
        return self._vector(content)

    def embed_documents(self, contents: List[str]) -> List[List[float]]:
        return [self._vector(c) for c in contents]
