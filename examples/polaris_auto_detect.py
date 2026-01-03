# -*- coding: utf-8 -*-
"""
PolarisRAG 自动检测模式示例

展示 PolarisRAG 如何自动检测模式：
- 有向量存储 → RAG 模式
- 无向量存储 → 纯 LLM 模式
"""
import os
from polarisrag import PolarisRAG
from polarisrag.llm import OpenAILLM
from polarisrag.embedding import OpenAIEmbedding
from polarisrag.vector_database import MilvusDB

# 示例 1：纯 LLM 模式（不配置向量存储）
print("=" * 60)
print("示例 1：纯 LLM 模式（不配置向量存储）")
print("=" * 60)

llm_only = PolarisRAG(
    llm_model={"model": "gpt-4o-mini", "temperature": 0.7},
    # 不配置 vector_storage，自动使用纯 LLM 模式
)

response = llm_only.chat("你好，介绍一下你自己")
print(f"回答: {response}\n")

# 示例 2：RAG 模式（配置向量存储）
print("=" * 60)
print("示例 2：RAG 模式（配置向量存储）")
print("=" * 60)

rag_mode = PolarisRAG(
    llm_model={"model": "gpt-4o-mini", "temperature": 0.7},
    embedding_model={"model": "text-embedding-3-small"},
    vector_storage={
        "db_file": "milvus_data.db",  # 本地文件存储（默认）
        "collection_name": "demo_collection"
    }
)

# 添加一些文档
documents = [
    "BERT 是一种预训练的语言模型，由 Google 在 2018 年提出。",
    "GPT-4 是 OpenAI 开发的大型语言模型，具有强大的语言理解和生成能力。",
    "向量数据库用于存储和检索高维向量数据，常用于 RAG 系统。"
]

print("添加文档到向量数据库...")
rag_mode.vector_storage.insert(documents)
print(f"成功添加 {len(documents)} 个文档\n")

# 提问
question = "BERT 是什么？"
print(f"问题: {question}")
response = rag_mode.chat(question)
print(f"回答: {response}\n")

# 示例 3：使用默认配置的 RAG 模式
print("=" * 60)
print("示例 3：使用默认配置的 RAG 模式")
print("=" * 60)

rag_default = PolarisRAG(
    llm_model={"model": "gpt-4o-mini"},
    embedding_model={"model": "text-embedding-3-small"},
    # vector_storage 使用默认配置（本地文件）
)

# 添加文档
rag_default.vector_storage.insert(documents)

question = "什么是向量数据库？"
print(f"问题: {question}")
response = rag_default.chat(question)
print(f"回答: {response}\n")

# 示例 4：使用字典配置的简洁方式
print("=" * 60)
print("示例 4：字典配置方式")
print("=" * 60)

# 纯 LLM 模式
llm_simple = PolarisRAG(llm_model={"model": "gpt-4o-mini"})
print("模式：纯 LLM（无向量存储）")

# RAG 模式
rag_simple = PolarisRAG(
    llm_model={"model": "gpt-4o-mini"},
    embedding_model={"model": "text-embedding-3-small"},
    vector_storage={"db_file": "my_vector.db", "collection_name": "simple"}
)
print("模式：RAG（有向量存储）")

# 添加文档
rag_simple.vector_storage.insert(["Python 是一种流行的编程语言。"])

# 测试
print(f"\n纯 LLM 模式回答:")
print(llm_simple.chat("什么是 Python？"))

print(f"\nRAG 模式回答:")
rag_simple.vector_storage.insert(["Python 是一种高级编程语言，语法简洁易懂。"])
print(rag_simple.chat("什么是 Python？"))

print("=" * 60)
print("所有示例执行完成！")
print("=" * 60)

