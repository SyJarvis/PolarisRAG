# -*- coding: utf-8 -*-
"""
PolarisRAG v2.0 常量配置
基于 LangChain 1.0 重构，专注于 OpenAI 生态
"""

# 模式常量
MODE_LLM_ONLY = "llm_only"  # 纯 LLM 模式
MODE_RAG = "rag"  # RAG 模式

# LLM 默认配置
DEFAULT_LLM_MODEL = {
    "type": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.7
}

# Embedding 默认配置
DEFAULT_EMBEDDING_MODEL = {
    "type": "openai",
    "model": "text-embedding-3-small"
}

# 向量存储默认配置
DEFAULT_VECTOR_STORAGE = {
    "type": "milvus",
    "connection_args": {},
    "collection_name": "polarisrag"
}

# 默认配置（完整）
DEFAULT_CONFIG = {
    "mode": MODE_RAG,
    "llm": DEFAULT_LLM_MODEL,
    "embedding": DEFAULT_EMBEDDING_MODEL,
    "vector_store": DEFAULT_VECTOR_STORAGE
}

# Milvus 配置
MilvusDB_CONF = {
    "db_file": "milvus_data.db",
    "collection_name": "polarisrag",
    "embedding_dim": 1536,  # text-embedding-3-small 的维度
    "search_params": {
        "metric_type": "IP",
        "params": {}
    },
    "output_fields": ["text"],
    "limit": 3
}

# 检索相似度阈值
similarity = 0.5

# LangChain 默认配置
LANGCHAIN_CONFIG = {
    "retriever_k": 3,  # 检索返回的文档数量
    "chunk_size": 1000,  # 文本分块大小
    "chunk_overlap": 200  # 分块重叠
}

# 提示词模板
DEFAULT_RAG_TEMPLATE = """使用以下上下文回答问题。如果不知道答案，就说不知道。总是使用中文回答。

上下文:
{context}

问题: {question}

回答:"""

DEFAULT_LLM_TEMPLATE = """回答用户的问题。总是使用中文回答。

问题: {question}

回答:"""
