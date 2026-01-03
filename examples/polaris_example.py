# -*- coding: utf-8 -*-
"""
PolarisRAG v2.0 基础使用示例

使用基于 LangChain 1.0 的新 API
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from polarisrag import PolarisRAG

# 设置 API 密钥（如果未设置）
import os
if not os.getenv("LLM_API_KEY"):
    print("警告: 未设置 LLM_API_KEY 环境变量")
    print("请设置: export LLM_API_KEY='your-api-key'")

print("="*80)
print("PolarisRAG 基础示例")
print("="*80 + "\n")

# 方式 1: 使用字典配置
print("方式 1: 使用字典配置")
print("-" * 80)

rag = PolarisRAG(
    llm_model={
        "model": "glm-4.7",
        "temperature": 0.7
    },
    embedding_model={
        "model": "Qwen/Qwen3-Embedding-8B"
    },
    vector_storage={
        "type": "milvus",
        "collection_name": "demo_collection",
        "drop_old": False
    }
)

# 加载文档
print("加载文档...")
rag.load_document("documents/test.txt")
print("文档加载完成\n")

# 初始化
print("初始化 RAG 系统...")
rag.init()
print("初始化完成\n")

# 查询
query = "什么是BERT?"
print(f"查询: {query}\n")

result = rag.chat(query)

print(f"回答:\n{result}\n")

print("="*80)
print("示例运行完成")
print("="*80)
