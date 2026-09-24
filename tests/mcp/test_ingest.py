# -*- coding: utf-8 -*-
import pytest

from mcp_server.config import Config
from mcp_server.fakes import FakeEmbedding
from mcp_server.ingest import IngestError, Ingestor
from mcp_server.registry import SourceRegistry
from polarisrag.vector_database import MilvusDB


def _config(tmp_path, max_text_len=100_000):
    return Config(
        llm_api_key="sk", llm_base_url=None, llm_model="m",
        embedding_api_key="sk", embedding_base_url=None,
        embedding_model="e", home=tmp_path / "home",
        collection="t", max_text_len=max_text_len,
        smart_max_len=20_000, max_iterations=5,
    )


@pytest.fixture()
def deps(tmp_path):
    cfg = _config(tmp_path)
    cfg.home.mkdir(parents=True, exist_ok=True)  # milvus-lite 要求 db 父目录存在
    registry = SourceRegistry(cfg.registry_path)
    db = MilvusDB(db_file=str(cfg.db_file), embedding_model=FakeEmbedding(),
                  collection_name=cfg.collection)
    return cfg, registry, Ingestor(cfg, registry, db), db, tmp_path


def test_fast_ingest_success(deps):
    cfg, registry, ingestor, db, tmp_path = deps
    text = "PolarisRAG 是一个检索增强生成框架。" * 30  # 约 600 字，1-2 chunks
    result = ingestor.add_text(text, source_name="intro")

    assert result["mode"] == "fast"
    assert result["chunks"] >= 1
    assert result["source_name"] == "intro"
    assert result["source_id"].startswith("src_")

    # registry 状态与原文保全
    rec = registry.get_source(result["source_id"])
    assert rec["status"] == "indexed"
    assert rec["text"] == text

    # 可检索到且能反查来源
    hits = db.search("PolarisRAG", limit=3)
    assert hits
    owner = registry.find_by_chunk_id(hits[0]["id"])
    assert owner["source_id"] == result["source_id"]


def test_auto_source_name(deps):
    _, registry, ingestor, _, _ = deps
    result = ingestor.add_text("some text")
    assert result["source_name"].startswith("source_")


def test_empty_text_rejected(deps):
    _, _, ingestor, _, _ = deps
    with pytest.raises(IngestError):
        ingestor.add_text("   ")
    with pytest.raises(IngestError):
        ingestor.add_text("")


def test_too_long_rejected(deps):
    cfg, _, ingestor, _, _ = deps
    ingestor.config = _config(deps[4], max_text_len=100)
    with pytest.raises(IngestError, match="上限"):
        ingestor.add_text("x" * 200)


def test_invalid_mode_rejected(deps):
    _, _, ingestor, _, _ = deps
    with pytest.raises(IngestError):
        ingestor.add_text("text", mode="turbo")


def test_smart_not_implemented(deps):
    _, registry, ingestor, _, _ = deps
    with pytest.raises(NotImplementedError):
        ingestor.add_text("text", mode="smart")


def test_insert_failure_marks_failed(deps):
    _, registry, ingestor, _, _ = deps

    class BrokenDB:
        def insert(self, docs, ids=None):
            raise RuntimeError("milvus down")

    broken = Ingestor(ingestor.config, registry, BrokenDB())
    # create_source 先行，insert 失败后状态应为 failed 且保留原文
    try:
        broken.add_text("keep this text", mode="fast")
    except IngestError:
        pass
    recs = registry.list_sources()
    assert recs and recs[0]["status"] == "failed"
    full = registry.get_source(recs[0]["source_id"])
    assert full["text"] == "keep this text"
