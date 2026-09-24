# -*- coding: utf-8 -*-
"""fast/smart 整理入库（设计文档 §3.2）。"""
import logging
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Config
from .ids import new_source_id, next_chunk_id
from .registry import SourceRegistry

LOGGER = logging.getLogger(__name__)

VALID_MODES = ("fast", "smart")


class IngestError(Exception):
    """入库失败（参数非法或底层错误）。"""


class Ingestor:
    def __init__(self, config: Config, registry: SourceRegistry, vector_db):
        self.config = config
        self.registry = registry
        self.vector_db = vector_db
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

    def add_text(
        self, text: str, source_name: str = "", mode: str = "fast"
    ) -> dict:
        """文本整理入库。先落盘 registry（保原文），再切分入库。"""
        self._validate(text, source_name, mode)

        source_id = new_source_id()
        self.registry.create_source(source_id, source_name, mode, text)

        if mode == "smart":
            # P2 实现；schema 与校验先行（设计文档 §3.2）
            raise NotImplementedError("smart 模式将在 P2 提供，当前请使用 fast")

        chunks: List[str] = self.splitter.split_text(text)
        if not chunks:
            self.registry.mark_failed(source_id, "切分结果为空")
            raise IngestError("文本切分结果为空，未入库")

        # 预生成 id：registry 记录真实 chunk id，检索结果可反查来源
        chunk_ids = [next_chunk_id() for _ in chunks]

        try:
            inserted = self.vector_db.insert(docs=chunks, ids=chunk_ids)
        except Exception as e:
            self.registry.mark_failed(source_id, str(e))
            LOGGER.exception("入库失败 source=%s", source_id)
            raise IngestError(f"向量入库失败: {e}") from e

        if inserted != len(chunks):
            err = f"入库数量不一致：期望 {len(chunks)}，实际 {inserted}"
            self.registry.mark_failed(source_id, err)
            raise IngestError(err)

        self.registry.mark_indexed(source_id, chunk_ids)
        rec = self.registry.get_source(source_id)
        LOGGER.info("入库成功 source=%s chunks=%d mode=%s", source_id, inserted, mode)
        return {
            "source_id": source_id,
            "source_name": rec["source_name"],
            "chunks": rec["chunks"],
            "mode": mode,
        }

    def _validate(self, text: str, source_name: Optional[str], mode: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise IngestError("text 不能为空")
        if len(text) > self.config.max_text_len:
            raise IngestError(
                f"text 长度 {len(text)} 超过上限 {self.config.max_text_len}"
            )
        if mode not in VALID_MODES:
            raise IngestError(f"mode 必须是 {VALID_MODES}，收到: {mode!r}")
        if mode == "smart" and len(text) > self.config.smart_max_len:
            raise IngestError(
                f"smart 模式长度上限为 {self.config.smart_max_len}，"
                f"当前 {len(text)}，请使用 fast 模式"
            )
        if source_name and len(source_name) > 200:
            raise IngestError("source_name 长度不能超过 200 字符")
