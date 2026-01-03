# -*- coding: utf-8 -*-
"""
LLM 模型实现

基于 LangChain 1.0，专注于 OpenAI 生态
"""
import os
from typing import List, Dict, Optional, Iterator

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.language_models import BaseChatModel
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseChatModel = object


class OpenAILLM:
    """
    OpenAI LLM 封装

    基于 LangChain 1.0 的 ChatOpenAI
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        is_memory: bool = False,
        system_prompt: Optional[str] = None
    ):
        """
        初始化 OpenAI LLM

        Args:
            api_key: OpenAI API 密钥，如果为 None 则从环境变量读取
            base_url: API 基础 URL，如果为 None 则从环境变量读取
            model: 模型名称
            temperature: 温度参数
            is_memory: 是否启用记忆（未使用，保留兼容性）
            system_prompt: 系统提示词（未使用，保留兼容性）
        """
        # 获取 API 密钥
        if api_key:
            os.environ["LLM_API_KEY"] = api_key
        else:
            api_key = os.getenv("LLM_API_KEY")
            if api_key is None:
                raise ValueError("LLM_API_KEY 未设置。请设置环境变量或传入 api_key 参数")

        # 获取 base_url
        if base_url:
            os.environ["LLM_BASE_URL"] = base_url
        print(f"LLM_BASE_URL: {os.environ['LLM_BASE_URL']}")
        print(f"LLM_API_KEY: {os.environ['LLM_API_KEY']}")
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.is_memory = is_memory
        self.system_prompt = system_prompt

        # 创建 LangChain ChatOpenAI 实例
        self.client = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            base_url=self.base_url
        )

    def set_system_prompt(self, prompt: str):
        """设置系统提示词（保留兼容性）"""
        self.system_prompt = prompt

    def chat(self, content: str, *, history: List[Dict] = None, **kwargs) -> str:
        """
        聊天

        Args:
            content: 用户消息
            history: 历史消息（未使用，保留兼容性）
            **kwargs: 其他传递给 invoke 的参数

        Returns:
            模型回复（字符串）

        Raises:
            RuntimeError: 如果 langchain 未安装
        """
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("langchain 未安装，请运行: pip install langchain langchain-openai")

        # 使用 LangChain 的 invoke 方法
        response = self.client.invoke(content, **kwargs)

        # 提取响应内容
        return response.content

    def stream(self, content: str, *, history: List[Dict] = None, **kwargs) -> Iterator[str]:
        """
        流式输出

        Args:
            content: 用户消息
            history: 历史消息（未使用，保留兼容性）
            **kwargs: 其他传递给 stream 的参数

        Yields:
            输出文本片段

        Raises:
            RuntimeError: 如果 langchain 未安装
        """
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("langchain 未安装，请运行: pip install langchain langchain-openai")

        for chunk in self.client.stream(content, **kwargs):
            yield chunk


class ZhipuLLM:
    """
    Zhipu LLM（已移除）

    v2.0 已移除 Zhipu 支持
    如需使用其他 LLM，请直接使用 LangChain 的对应组件
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ZhipuLLM 已在 v2.0 中移除。"
            "请使用 OpenAILLM 或 LangChain 的 ChatOpenAI。"
            "如需使用其他 LLM 提供商（如 Anthropic），请直接使用 LangChain 的对应组件。"
        )


class Qwen2LLM:
    """
    Qwen2 LLM（已移除）

    v2.0 已移除 Qwen2 支持
    如需使用，请直接使用 LangChain 的对应组件
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Qwen2LLM 已在 v2.0 中移除。"
            "请使用 OpenAILLM 或 LangChain 的 ChatOpenAI。"
        )


class Qwen2VLLLM:
    """
    Qwen2VL LLM（已移除）

    v2.0 已移除 Qwen2VL 支持
    如需使用，请直接使用 LangChain 的对应组件
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Qwen2VLLM 已在 v2.0 中移除。"
            "请使用 OpenAILLM 或 LangChain 的 ChatOpenAI。"
        )


__all__ = [
    "OpenAILLM",
    # 以下类已移除
    # "ZhipuLLM",
    # "Qwen2LLM",
    # "Qwen2VLLLM",
]
