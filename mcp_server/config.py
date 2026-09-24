# -*- coding: utf-8 -*-
"""环境变量读取与启动校验（设计文档 §8）。"""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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


def _int_env(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ConfigError(f"{name} 必须是整数，收到: {raw!r}")
    if v < minimum:
        raise ConfigError(f"{name} 必须 ≥ {minimum}，收到: {v}")
    return v


def load_config() -> Config:
    """读取并校验配置；错误时打印到 stderr 并 exit(2)。"""
    errors: List[str] = []
    test_mode = os.getenv("POLARIS_FAKE_EMBEDDINGS", "0") in ("1", "true", "True")

    llm_api_key = os.getenv("LLM_API_KEY")
    embedding_api_key = os.getenv("EMBEDDING_API_KEY")

    if not test_mode:
        if not llm_api_key:
            errors.append("LLM_API_KEY 未设置（决策/整理 LLM 必需）")
        if not embedding_api_key:
            errors.append("EMBEDDING_API_KEY 未设置（向量化必需）")
    if errors:
        _fail(errors)

    home = Path(os.getenv("POLARIS_MCP_HOME", "./polaris_mcp")).resolve()
    home.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        llm_api_key=llm_api_key,
        llm_base_url=os.getenv("LLM_BASE_URL"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        embedding_api_key=embedding_api_key,
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        home=home,
        collection=os.getenv("POLARIS_MCP_COLLECTION", "polaris_mcp"),
        max_text_len=_int_env("POLARIS_MAX_TEXT_LEN", 100_000, 1000),
        smart_max_len=_int_env("POLARIS_SMART_MAX_LEN", 20_000, 1000),
        max_iterations=_int_env("POLARIS_MAX_ITERATIONS", 5, 1),
        test_mode=test_mode,
    )
    if cfg.smart_max_len > cfg.max_text_len:
        raise ConfigError(
            f"POLARIS_SMART_MAX_LEN({cfg.smart_max_len}) 不能大于 "
            f"POLARIS_MAX_TEXT_LEN({cfg.max_text_len})"
        )
    return cfg
