# -*- coding: utf-8 -*-
import json

import pytest

from mcp_server.registry import SourceRegistry


@pytest.fixture()
def registry(tmp_path):
    return SourceRegistry(tmp_path / "registry.json")


def test_create_source_pending_and_persisted(registry, tmp_path):
    rec = registry.create_source("src_abc123", "doc1", "fast", "hello world")
    assert rec["status"] == "pending"
    assert rec["source_id"] == "src_abc123"

    # 落盘后可解析、字段齐全
    data = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["sources"][0]["text"] == "hello world"


def test_auto_name_and_auto_id(registry):
    rec = registry.create_source("", "", "fast", "body")
    assert rec["source_id"].startswith("src_")
    assert rec["source_name"].startswith("source_")


def test_invalid_mode_rejected(registry):
    with pytest.raises(ValueError):
        registry.create_source("src_a", "n", "turbo", "t")


def test_lifecycle_pending_to_indexed(registry):
    registry.create_source("src_a", "n", "fast", "t")
    registry.mark_indexed("src_a", [11, 22, 33])
    rec = registry.get_source("src_a")
    assert rec["status"] == "indexed"
    assert rec["chunks"] == 3
    assert rec["chunk_ids"] == [11, 22, 33]


def test_mark_failed_keeps_text(registry):
    registry.create_source("src_a", "n", "fast", "keep me")
    registry.mark_failed("src_a", "boom")
    rec = registry.get_source("src_a")
    assert rec["status"] == "failed"
    assert rec["error"] == "boom"
    assert rec["text"] == "keep me"


def test_get_missing_returns_none(registry):
    assert registry.get_source("src_nothing") is None


def test_list_sources_hides_text(registry):
    registry.create_source("src_a", "n", "fast", "secret body")
    listing = registry.list_sources()
    assert listing[0]["source_id"] == "src_a"
    assert "text" not in listing[0]


def test_stats_counts_indexed_only(registry):
    registry.create_source("src_a", "a", "fast", "t1")
    registry.create_source("src_b", "b", "fast", "t2")
    registry.mark_indexed("src_a", [1, 2])
    registry.mark_failed("src_b", "x")

    s = registry.stats()
    assert s == {"sources": 2, "indexed": 1, "chunks": 2}


def test_reload_from_disk(tmp_path):
    path = tmp_path / "registry.json"
    r1 = SourceRegistry(path)
    r1.create_source("src_a", "n", "fast", "persisted")

    r2 = SourceRegistry(path)
    assert r2.get_source("src_a")["text"] == "persisted"


def test_corrupt_file_falls_back_to_empty(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text("{not json", encoding="utf-8")
    r = SourceRegistry(path)
    assert r.stats() == {"sources": 0, "indexed": 0, "chunks": 0}


def test_unknown_source_raises(registry):
    with pytest.raises(KeyError):
        registry.mark_indexed("src_missing", [1])
