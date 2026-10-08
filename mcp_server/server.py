# -*- coding: utf-8 -*-
"""PolarisRAG MCP Server（stdio 入口，设计文档 §4、§9）。"""
import json
import logging
import sys

from mcp.server import MCPServer

from polarisrag.vector_database import MilvusDB

from .agent import RAGAgent
from .config import Config, load_config
from .fakes import FakeAgent, FakeEmbedding
from .ingest import Ingestor
from .registry import SourceRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)

config: Config = load_config()

_registry = SourceRegistry(config.registry_path)

_resources = None  # (vector_db, ingestor, agent)，懒加载


def _get_resources():
    """懒加载重量级组件（embedding 维度探测 / Milvus / Agent）。

    import 本模块不再触发任何网络请求；首次工具调用才初始化。
    上游 Embedding 服务不可用时仅该次调用报错，而非整个 server 无法启动
    （stdio 场景 Host 拉起进程后工具列表仍可枚举，HTTP 场景服务可正常起监听）。
    """
    global _resources
    if _resources is None:
        if config.test_mode:
            embedding = FakeEmbedding()
        else:
            from polarisrag.embedding import OpenAIEmbedding
            embedding = OpenAIEmbedding(
                api_key=config.embedding_api_key,
                model=config.embedding_model,
                base_url=config.embedding_base_url,
            )
        vector_db = MilvusDB(
            db_file=str(config.db_file),
            embedding_model=embedding,
            collection_name=config.collection,
        )
        ingestor = Ingestor(config, _registry, vector_db)
        if config.test_mode:
            agent = FakeAgent(vector_db, _registry, config)
        else:
            agent = RAGAgent(config, _registry, vector_db)
        _resources = (vector_db, ingestor, agent)
    return _resources

mcp = MCPServer("PolarisRAG")


def _status_payload() -> dict:
    stats = _registry.stats()
    return {
        "collection": config.collection,
        "sources": stats["sources"],
        "indexed": stats["indexed"],
        "chunks": stats["chunks"],
        "embedding_model": "fake-256d" if config.test_mode else config.embedding_model,
        "decision_model": "fake-agent" if config.test_mode else config.llm_model,
        "test_mode": config.test_mode,
    }


@mcp.tool()
def rag_query(query: str) -> dict:
    """基于已入库文档进行检索问答。服务端决策模型自动编排检索（先检索后回答），
    返回带来源引用的回答。前提：库中已通过 rag_add_text 入库文档。

    Args:
        query: 用户问题（1..2000 字符）
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 不能为空")
    if len(query) > 2000:
        raise ValueError("query 长度不能超过 2000 字符")
    _, _, agent = _get_resources()
    return agent.run(query)


@mcp.tool()
def rag_add_text(text: str, source_name: str = "", mode: str = "fast") -> dict:
    """上传一段文本并整理入库（切分、向量化，写入本地向量库），
    可通过 source_name 命名来源。mode=fast 机械切分；smart 模式暂未提供。

    Args:
        text: 文本内容（1..100000 字符）
        source_name: 可选来源名（≤200 字符）
        mode: "fast" 或 "smart"
    """
    _, ingestor, _ = _get_resources()
    return ingestor.add_text(text, source_name, mode)


@mcp.tool()
def rag_status() -> dict:
    """查询向量库状态：集合名、来源数、片段总数与模型配置摘要（脱敏）。"""
    return _status_payload()


@mcp.resource("polaris://status")
def status_resource() -> str:
    """服务状态快照（JSON 文本）。"""
    return json.dumps(_status_payload(), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
