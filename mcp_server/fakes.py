# -*- coding: utf-8 -*-
"""测试模式的 fake 组件（与 tests/mcp/fakes.py 同逻辑）。"""
from typing import List

from .config import Config
from .registry import SourceRegistry


class FakeEmbedding:
    """256 维确定性哈希向量：同文本同向量，无需 API key。"""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vector(self, text: str) -> List[float]:
        import hashlib
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


class FakeAgent:
    """无 key 协议验证用：固定脚本（先检索一次再回答）。"""

    def __init__(self, vector_db, registry: SourceRegistry, config: Config):
        self.vector_db = vector_db
        self.registry = registry

    def run(self, query: str) -> dict:
        results = self.vector_db.search(query, limit=3)
        sources_used = []
        for r in results:
            rec = self.registry.find_by_chunk_id(r["id"])
            if rec and rec["source_id"] not in [s["source_id"] for s in sources_used]:
                sources_used.append({
                    "source_id": rec["source_id"],
                    "source_name": rec["source_name"],
                })
        if results:
            top = results[0]["text"]
            answer = f"[test_mode] 检索到 {len(results)} 条片段，最相关：{top[:200]}"
        else:
            answer = "[test_mode] 库中没有相关内容"
        return {
            "answer": answer,
            "sources_used": sources_used,
            "tool_trace": [
                {"tool": "search_documents", "args": {"query": query, "top_k": 3},
                 "ok": True, "truncated": False},
            ],
        }
