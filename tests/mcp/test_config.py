# -*- coding: utf-8 -*-
import pytest

from mcp_server.config import Config, ConfigError, load_config


def _base_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-emb")
    monkeypatch.setenv("POLARIS_MCP_HOME", str(tmp_path / "home"))


def test_missing_keys_exit_2(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("POLARIS_FAKE_EMBEDDINGS", raising=False)
    monkeypatch.setenv("POLARIS_MCP_HOME", str(tmp_path / "h"))
    with pytest.raises(SystemExit) as ei:
        load_config()
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "LLM_API_KEY" in err and "EMBEDDING_API_KEY" in err


def test_fake_mode_skips_key_check(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("POLARIS_FAKE_EMBEDDINGS", "1")
    monkeypatch.setenv("POLARIS_MCP_HOME", str(tmp_path / "h"))
    cfg = load_config()
    assert cfg.test_mode is True


def test_home_dir_created(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    cfg = load_config()
    assert cfg.home.exists()


def test_invalid_int_env(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("POLARIS_MAX_TEXT_LEN", "not-a-number")
    with pytest.raises(ConfigError):
        load_config()


def test_int_below_minimum(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("POLARIS_MAX_ITERATIONS", "0")
    with pytest.raises(ConfigError):
        load_config()


def test_smart_gt_max_rejected(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("POLARIS_MAX_TEXT_LEN", "5000")
    monkeypatch.setenv("POLARIS_SMART_MAX_LEN", "8000")
    with pytest.raises(ConfigError):
        load_config()


def test_defaults(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    cfg = load_config()
    assert cfg.llm_model == "gpt-4o-mini"
    assert cfg.embedding_model == "text-embedding-3-small"
    assert cfg.collection == "polaris_mcp"
    assert cfg.max_text_len == 100_000
    assert cfg.max_iterations == 5
    assert cfg.test_mode is False
    assert cfg.registry_path == cfg.home / "registry.json"
    assert cfg.db_file == cfg.home / "milvus_data.db"
