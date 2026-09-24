# -*- coding: utf-8 -*-
import pytest

from mcp_server.config import Config, ConfigError, load_config


def _base_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-emb")
    monkeypatch.setenv("POLARIS_MCP_HOME", str(tmp_path / "home"))
    # 隔离本地 config/mcp.toml（开发者可能有真实配置文件）
    monkeypatch.setenv("POLARIS_MCP_CONFIG", str(tmp_path / "no-such.toml"))


def test_missing_keys_exit_2(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("POLARIS_FAKE_EMBEDDINGS", raising=False)
    monkeypatch.setenv("POLARIS_MCP_HOME", str(tmp_path / "h"))
    monkeypatch.setenv("POLARIS_MCP_CONFIG", str(tmp_path / "no-such.toml"))
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
    monkeypatch.setenv("POLARIS_MCP_CONFIG", str(tmp_path / "no-such.toml"))
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


# ---------- TOML 配置 ----------


def _write_toml(monkeypatch, tmp_path, content: str):
    cfg_path = tmp_path / "mcp.toml"
    cfg_path.write_text(content, encoding="utf-8")
    monkeypatch.setenv("POLARIS_MCP_CONFIG", str(cfg_path))
    return cfg_path


def test_toml_provides_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("POLARIS_FAKE_EMBEDDINGS", raising=False)
    _write_toml(monkeypatch, tmp_path, (
        '[llm]\napi_key = "toml-llm-key"\nmodel = "toml-model"\n'
        '[embedding]\napi_key = "toml-emb-key"\n'
        '[server]\nhome = "%s"\ncollection = "toml_col"\n'
        % (tmp_path / "toml_home")
    ))
    cfg = load_config()
    assert cfg.llm_api_key == "toml-llm-key"
    assert cfg.embedding_api_key == "toml-emb-key"
    assert cfg.llm_model == "toml-model"
    assert cfg.collection == "toml_col"
    assert cfg.test_mode is False
    assert (tmp_path / "toml_home").exists()


def test_env_overrides_toml(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    _write_toml(monkeypatch, tmp_path, (
        '[llm]\napi_key = "toml-key"\n'
        '[embedding]\napi_key = "toml-emb"\n'
    ))
    cfg = load_config()
    assert cfg.llm_api_key == "sk-test"       # 环境变量优先
    assert cfg.embedding_api_key == "sk-emb"


def test_toml_empty_string_treated_as_unset(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("POLARIS_FAKE_EMBEDDINGS", raising=False)
    _write_toml(monkeypatch, tmp_path, (
        '[llm]\napi_key = ""\n'
        '[embedding]\napi_key = ""\n'
    ))
    with pytest.raises(SystemExit) as ei:
        load_config()
    assert ei.value.code == 2


def test_toml_invalid_syntax(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    _write_toml(monkeypatch, tmp_path, "[llm\nbroken =")
    with pytest.raises(ConfigError):
        load_config()


def test_toml_int_type_check(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    _write_toml(monkeypatch, tmp_path, (
        '[llm]\napi_key = "k"\n'
        '[embedding]\napi_key = "e"\n'
        '[server]\nmax_iterations = "five"\n'
    ))
    with pytest.raises(ConfigError):
        load_config()


def test_toml_int_below_minimum(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    _write_toml(monkeypatch, tmp_path, '[server]\nmax_iterations = 0\n')
    with pytest.raises(ConfigError):
        load_config()


def test_toml_defaults_when_absent(monkeypatch, tmp_path):
    """TOML 只给 key，其余项走代码默认值。"""
    _write_toml(monkeypatch, tmp_path, (
        '[llm]\napi_key = "k"\n'
        '[embedding]\napi_key = "e"\n'
    ))
    monkeypatch.setenv("POLARIS_MCP_HOME", str(tmp_path / "env_home"))
    cfg = load_config()
    assert cfg.llm_model == "gpt-4o-mini"
    assert cfg.max_iterations == 5
    assert cfg.collection == "polaris_mcp"


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
