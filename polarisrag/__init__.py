# -*- coding: utf-8 -*-
"""
PolarisRAG - 基于 OpenAI 的 RAG 系统

特性：
- 自动检测模式：有向量库用 RAG，否则用纯 LLM
- 专注于 OpenAI 生态（使用 OpenAILLM 和 OpenAIEmbedding）
- 支持三种配置方式：字典、配置文件、直接实例化
"""
from .polarisrag import PolarisRAG, QueryParam
from .llm import OpenAILLM
from .embedding import OpenAIEmbedding, HFEmbedding
from .vector_database import (
    BaseVectorDB,
    MilvusDB,
    VectorDB
)
from .const import (
    DEFAULT_RAG_TEMPLATE,
    DEFAULT_LLM_TEMPLATE
)

__version__ = "0.1.0"
__author__ = "Runke Zhong"
__url__ = "https://github.com/SyJarvis/PolarisRAG"

__all__ = [
    # 核心类
    "PolarisRAG",
    "QueryParam",
    
    # LLM 模型
    "OpenAILLM",
    
    # Embedding 模型
    "OpenAIEmbedding",
    "HFEmbedding",
    
    # 向量数据库
    "BaseVectorDB",
    "MilvusDB",
    "VectorDB",
    
    # 提示词模板
    "DEFAULT_RAG_TEMPLATE",
    "DEFAULT_LLM_TEMPLATE",
]
