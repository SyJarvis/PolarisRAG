# -*- coding: utf-8 -*-
"""Source Registry：JSON 边车文件，记录入库来源的元数据与原文。

写入采用临时文件 + os.replace 原子替换；进程内内存缓存，写穿。
v1 为单进程使用（stdio），无文件锁（设计文档 R4）。
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .ids import new_source_id

LOGGER = logging.getLogger(__name__)

REGISTRY_VERSION = 1
VALID_MODES = ("fast", "smart")


class SourceRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: Dict = self._load_or_init()

    # ---------- 加载与持久化 ----------

    def _load_or_init(self) -> Dict:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("sources"), list):
                    if data.get("version") == REGISTRY_VERSION:
                        return data
                    LOGGER.warning(
                        "registry 版本不匹配(%s)，按 v1 结构继续加载", data.get("version")
                    )
                    return data
            except (json.JSONDecodeError, OSError) as e:
                LOGGER.error("registry 加载失败，初始化为空: %s", e)
        return {"version": REGISTRY_VERSION, "sources": []}

    def _flush(self) -> None:
        """原子写：临时文件 + os.replace。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    # ---------- CRUD ----------

    def create_source(
        self, source_id: str, source_name: str, mode: str, text: str
    ) -> Dict:
        """登记新来源，status=pending。text 原文先落盘，入库失败可重试。"""
        if mode not in VALID_MODES:
            raise ValueError(f"mode 必须是 {VALID_MODES}，收到: {mode!r}")
        if not source_id:
            source_id = new_source_id()
        if not source_name:
            source_name = f"source_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        record = {
            "source_id": source_id,
            "source_name": source_name,
            "mode": mode,
            "status": "pending",
            "text": text,
            "chunk_ids": [],
            "chunks": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "error": None,
        }
        self._data["sources"].append(record)
        self._flush()
        return self._summary(record)

    def mark_indexed(self, source_id: str, chunk_ids: List[int]) -> None:
        rec = self._find(source_id)
        if rec is None:
            raise KeyError(f"source 不存在: {source_id}")
        rec["status"] = "indexed"
        rec["chunk_ids"] = list(chunk_ids)
        rec["chunks"] = len(chunk_ids)
        rec["error"] = None
        self._flush()

    def mark_failed(self, source_id: str, error: str) -> None:
        rec = self._find(source_id)
        if rec is None:
            raise KeyError(f"source 不存在: {source_id}")
        rec["status"] = "failed"
        rec["error"] = error
        self._flush()

    def get_source(self, source_id: str) -> Optional[Dict]:
        rec = self._find(source_id)
        return dict(rec) if rec else None

    def list_sources(self) -> List[Dict]:
        """来源摘要列表（不含 text 全文）。"""
        return [self._summary(r) for r in self._data["sources"]]

    def stats(self) -> Dict:
        sources = self._data["sources"]
        indexed = [r for r in sources if r.get("status") == "indexed"]
        return {
            "sources": len(sources),
            "indexed": len(indexed),
            "chunks": sum(len(r.get("chunk_ids", [])) for r in indexed),
        }

    # ---------- 内部 ----------

    def _find(self, source_id: str) -> Optional[Dict]:
        for r in self._data["sources"]:
            if r["source_id"] == source_id:
                return r
        return None

    def find_by_chunk_id(self, chunk_id: int) -> Optional[Dict]:
        """按 Milvus chunk 主键反查所属 source（检索结果标注来源用）。"""
        for r in self._data["sources"]:
            if chunk_id in r.get("chunk_ids", []):
                return r
        return None

    @staticmethod
    def _summary(rec: Dict) -> Dict:
        return {
            k: rec.get(k)
            for k in (
                "source_id", "source_name", "mode", "status",
                "chunks", "created_at", "error",
            )
        }
