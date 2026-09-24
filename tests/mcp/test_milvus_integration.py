# -*- coding: utf-8 -*-
"""MilvusDB 集成测试：fake embedding + milvus-lite 本地文件（无网络无 key）。"""
import pytest

from polarisrag.vector_database import MilvusDB
from tests.mcp.fakes import FakeEmbedding


@pytest.fixture()
def db(tmp_path):
    db = MilvusDB(
        db_file=str(tmp_path / "test.db"),
        embedding_model=FakeEmbedding(),
        collection_name="test_mcp",
    )
    yield db
    # milvus-lite 句柄释放后再由 tmp_path 统一清理


def test_double_insert_no_conflict(db):
    """回归：旧实现 id=enumerate 从 0 起，第二次 insert 主键冲突。"""
    n1 = db.insert(["alpha content about cats", "beta content about dogs"])
    n2 = db.insert(["gamma content about birds", "delta about fish", "epsilon about cows"])
    assert n1 == 2
    assert n2 == 3

    results = db.search("cats", limit=10)
    assert len(results) == 5  # 两批数据都在


def test_search_returns_structured_results(db):
    db.insert(["polaris rag documentation intro", "unrelated weather report"])
    results = db.search("polaris", limit=2)

    assert isinstance(results, list)
    assert len(results) >= 1
    item = results[0]
    assert set(item.keys()) == {"id", "text", "distance"}
    assert isinstance(item["id"], int)
    assert isinstance(item["distance"], float)
    assert "polaris" in item["text"]


def test_search_missing_collection_returns_empty(tmp_path):
    db = MilvusDB(
        db_file=str(tmp_path / "empty.db"),
        embedding_model=FakeEmbedding(),
        collection_name="never_created",
    )
    assert db.search("anything") == []


def test_search_without_embedding_model_raises(tmp_path):
    db = MilvusDB(
        db_file=str(tmp_path / "x.db"),
        embedding_model=None,
        embedding_dim=8,
        collection_name="c",
    )
    import pytest as _pytest
    with _pytest.raises(ValueError):
        db.search("q")


def test_query_still_works_after_id_change(db):
    """既有 query() 行为不受影响：insert 后可检索到上下文文本。"""
    db.insert(["stable api check content"])
    ctx = db.query("stable api check", limit=1, similarity=-1)
    assert "stable api check" in ctx
