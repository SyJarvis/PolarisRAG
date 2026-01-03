# -*- coding: utf-8 -*-
"""
Embedding 模型实现

基于 LangChain 1.0，专注于 OpenAI 生态
"""
import os
from typing import List, Optional, Any

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.embeddings import Embeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Embeddings = object

from transformers import AutoTokenizer, AutoModel
import torch


class OpenAIEmbedding:
    """
    OpenAI 嵌入模型

    基于 LangChain 1.0 的 OpenAIEmbeddings
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None
    ) -> None:
        """
        初始化 OpenAI 嵌入模型

        Args:
            api_key: OpenAI API 密钥，如果为 None 则从环境变量读取
            model_name: 模型名称，默认 "text-embedding-3-small"
            base_url: API 基础 URL，如果为 None 则从环境变量读取
        """
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError(
                "langchain 未安装，请运行: pip install langchain langchain-openai"
            )

        # 获取 API 密钥
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("EMBEDDING_API_KEY")
            if self.api_key is None:
                raise ValueError("EMBEDDING_API_KEY 未设置。请设置环境变量或传入 api_key 参数")

        # 获取 base_url
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = os.getenv("EMBEDDING_BASE_URL")

        # 设置模型名称
        self.model_name = model

        # 创建 LangChain OpenAIEmbeddings 实例
        params = {
            "model": self.model_name,
            "api_key": self.api_key
        }

        if self.base_url:
            params["base_url"] = self.base_url

        self.client = OpenAIEmbeddings(**params)

    def embed_text(self, content: str, **kwargs) -> List[float]:
        """
        将单个文本转换为向量

        Args:
            content: 输入文本
            **kwargs: 其他传递给 LangChain 的参数

        Returns:
            文本的向量表示
        """
        if not isinstance(content, str):
            raise TypeError("content 必须是字符串类型")

        # 使用 LangChain 的 embed_query 方法
        result = self.client.embed_query(content, **kwargs)
        return result

    def embed_documents(self, contents: List[str], **kwargs) -> List[List[float]]:
        """
        将多个文本批量转换为向量

        Args:
            contents: 文本列表
            **kwargs: 其他传递给 LangChain 的参数

        Returns:
            文本向量的列表
        """
        if not isinstance(contents, list):
            raise TypeError("contents 必须是列表类型")

        if len(contents) == 0:
            raise ValueError("contents 列表不能为空")

        # 使用 LangChain 的 embed_documents 方法
        results = self.client.embed_documents(contents, **kwargs)
        return results

    def embed_query(self, text: str) -> List[float]:
        """
        兼容性方法：嵌入查询文本

        Args:
            text: 查询文本

        Returns:
            文本的向量表示
        """
        return self.embed_text(text)


class HFEmbedding:
    """
    HuggingFace 嵌入模型

    支持本地模型，用于需要本地部署的场景
    """

    def __init__(self, pretrain_dir: str = None, *inputs, **kwargs) -> None:
        """
        初始化 HuggingFace 嵌入模型

        Args:
            pretrain_dir: 预训练模型路径
            *inputs: 额外位置参数
            **kwargs: 额外关键字参数
        """
        self.pretrained_model_path = pretrain_dir
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_dir, *inputs, **kwargs)
        self.model = AutoModel.from_pretrained(pretrain_dir, *inputs, **kwargs)

    def embed_text(self, content: str, **kwargs) -> List[float]:
        """
        编码文本

        Args:
            content: 文本内容
            **kwargs: 额外参数

        Returns:
            向量表示
        """
        if isinstance(content, str):
            contents = [content.strip()]
        else:
            raise Exception("content must be str")

        return self.__embedding(contents, **kwargs).tolist()[0]

    def embed_documents(self, contents: List[str], **kwargs) -> List[List[float]]:
        """
        编码文档

        Args:
            contents: 文本列表
            **kwargs: 额外参数

        Returns:
            向量列表
        """
        return self.__embedding(contents, **kwargs).tolist()

    def __embedding(self, contents: List[str], **kwargs):
        """
        内部嵌入方法

        Args:
            contents: 文本列表
            **kwargs: 额外参数

        Returns:
            向量张量
        """
        encoded_input = self.tokenizer(contents, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


class ZhipuEmbedding:
    """
    Zhipu 嵌入模型（已移除）

    v2.0 已移除 Zhipu 支持
    如需使用，请使用 OpenAIEmbedding 或 LangChain 的对应组件
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ZhipuEmbedding 已在 v2.0 中移除。"
            "请使用 OpenAIEmbedding 或 LangChain 的 OpenAIEmbeddings。"
            "迁移指南请参考 docs/MIGRATION_GUIDE.md"
        )


class BGEEmbedding:
    """
    BGE 嵌入模型（已移除）

    v2.0 已移除 BGE 专用实现
    如需使用，请使用 HFEmbedding 或 LangChain 的 HuggingFaceEmbeddings
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "BGEEmbedding 已在 v2.0 中移除。"
            "请使用 HFEmbedding 或 LangChain 的 HuggingFaceEmbeddings。"
            "迁移指南请参考 docs/MIGRATION_GUIDE.md"
        )


__all__ = [
    "OpenAIEmbedding",
    "HFEmbedding",
    # 以下类已移除
    # "ZhipuEmbedding",
    # "BGEEmbedding",
]
