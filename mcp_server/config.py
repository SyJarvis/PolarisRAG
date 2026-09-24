# -*- coding: utf-8 -*-
"""配置读取（TOML 文件 + 环境变量覆盖）与启动校验（设计文档 §8）。

优先级：环境变量 > config/mcp.toml > 代码内默认值。
TOML 使用标准库 tomllib（Python 3.11+），不引入第三方依赖。
真实 key 只应写入 config/mcp.toml（已 gitignore）或环境变量，不得提交仓库。
"""
import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG_PATH = Path("config/mcp.toml")


@dataclass
class Config:
    llm_api_key: Optional[str]
    llm_base_url: Optional[str]
    llm_model: str
    embedding_api_key: Optional[str]
    embedding_base_url: Optional[str]
    embedding_model: str
    home: Path
    collection: str
    max_text_len: int
    smart_max_len: int
    max_iterations: int
    test_mode: bool = field(default=False)

    @property
    def registry_path(self) -> Path:
        return self.home / "registry.json"

    @property
    def db_file(self) -> Path:
        return self.home / "milvus_data.db"


class ConfigError(Exception):
    """配置缺失或非法。"""


def _fail(errors: List[str]) -> None:
    msg = "配置错误:\n" + "\n".join(f"  - {e}" for e in errors)
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def _load_toml() -> Dict[str, Any]:
    """读取 TOML 配置；文件不存在返回 {}，解析失败抛 ConfigError。"""
    path = Path(os.getenv("POLARIS_MCP_CONFIG", str(DEFAULT_CONFIG_PATH)))
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"配置文件解析失败 {path}: {e}")
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件根节点必须是表: {path}")
    return data


def _section(toml_cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    sec = toml_cfg.get(section, {})
    return sec if isinstance(sec, dict) else {}


def _opt(
    toml_cfg: Dict[str, Any],
    section: str,
    key: str,
    env_name: str,
    default: Optional[str],
) -> Optional[str]:
    """字符串配置项：环境变量 > TOML > 默认值；空串视为未设置。"""
    env = os.getenv(env_name)
    if env is not None and env != "":
        return env
    val = _section(toml_cfg, section).get(key)
    if val is not None and val != "":
        if not isinstance(val, str):
            raise ConfigError(
                f"配置 {section}.{key} 必须是字符串，收到: {val!r}"
            )
        return val
    return default


def _int_opt(
    toml_cfg: Dict[str, Any],
    section: str,
    key: str,
    env_name: str,
    default: int,
    minimum: int,
) -> int:
    """整数配置项：环境变量 > TOML > 默认值；校验类型与下限。"""
    env = os.getenv(env_name)
    if env is not None and env != "":
        try:
            val = int(env)
        except ValueError:
            raise ConfigError(f"{env_name} 必须是整数，收到: {env!r}")
        if val < minimum:
            raise ConfigError(f"{env_name} 必须 ≥ {minimum}，收到: {val}")
        return val
    val = _section(toml_cfg, section).get(key, default)
    if isinstance(val, bool) or not isinstance(val, int):
        raise ConfigError(f"配置 {section}.{key} 必须是整数，收到: {val!r}")
    if val < minimum:
        raise ConfigError(f"配置 {section}.{key} 必须 ≥ {minimum}，收到: {val}")
    return val


def load_config() -> Config:
    """读取并校验配置；错误时打印到 stderr 并 exit(2)。"""
    errors: List[str] = []
    test_mode = os.getenv("POLARIS_FAKE_EMBEDDINGS", "0") in ("1", "true", "True")
    toml_cfg = _load_toml()

    llm_api_key = _opt(toml_cfg, "llm", "api_key", "LLM_API_KEY", None)
    embedding_api_key = _opt(
        toml_cfg, "embedding", "api_key", "EMBEDDING_API_KEY", None
    )

    if not test_mode:
        if not llm_api_key:
            errors.append("LLM_API_KEY 未设置（环境变量或配置 llm.api_key）")
        if not embedding_api_key:
            errors.append("EMBEDDING_API_KEY 未设置（环境变量或配置 embedding.api_key）")
    if errors:
        _fail(errors)

    home = Path(
        _opt(toml_cfg, "server", "home", "POLARIS_MCP_HOME", "./polaris_mcp")
    ).resolve()
    home.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        llm_api_key=llm_api_key,
        llm_base_url=_opt(toml_cfg, "llm", "base_url", "LLM_BASE_URL", None),
        llm_model=_opt(toml_cfg, "llm", "model", "LLM_MODEL", "gpt-4o-mini"),
        embedding_api_key=embedding_api_key,
        embedding_base_url=_opt(
            toml_cfg, "embedding", "base_url", "EMBEDDING_BASE_URL", None
        ),
        embedding_model=_opt(
            toml_cfg, "embedding", "model", "EMBEDDING_MODEL",
            "text-embedding-3-small",
        ),
        home=home,
        collection=_opt(
            toml_cfg, "server", "collection", "POLARIS_MCP_COLLECTION",
            "polaris_mcp",
        ),
        max_text_len=_int_opt(
            toml_cfg, "server", "max_text_len", "POLARIS_MAX_TEXT_LEN",
            100_000, 1000,
        ),
        smart_max_len=_int_opt(
            toml_cfg, "server", "smart_max_len", "POLARIS_SMART_MAX_LEN",
            20_000, 1000,
        ),
        max_iterations=_int_opt(
            toml_cfg, "server", "max_iterations", "POLARIS_MAX_ITERATIONS",
            5, 1,
        ),
        test_mode=test_mode,
    )
    if cfg.smart_max_len > cfg.max_text_len:
        raise ConfigError(
            f"smart_max_len({cfg.smart_max_len}) 不能大于 "
            f"max_text_len({cfg.max_text_len})"
        )
    return cfg
