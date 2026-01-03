# -*- coding: utf-8 -*-
"""
PolarisRAG v2.0 文件夹加载示例

使用基于 LangChain 1.0 的新 API
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from polarisrag import PolarisRAG, MODE_RAG

# 设置 API 密钥（如果未设置）
import os
if not os.getenv("OPENAI_API_KEY"):
    print("警告: 未设置 OPENAI_API_KEY 环境变量")
    print("请设置: export OPENAI_API_KEY='your-api-key'")

print("="*80)
print("PolarisRAG v2.0 文件夹加载示例")
print("="*80 + "\n")

# 创建 RAG 系统
rag = PolarisRAG(
    mode=MODE_RAG,
    llm={
        "model": "gpt-4o-mini",
        "temperature": 0.7
    },
    embeddings={
        "model": "text-embedding-3-small"
    },
    vector_store={
        "type": "milvus",
        "connection_args": {
            "host": "localhost",
            "port": 19530
        },
        "collection_name": "demo_folder"
    }
)

print("初始化 RAG 系统...")
rag.init()
print("初始化完成\n")

print("从工作目录加载文档...")
print("工作目录: documents\n")

# 从文件夹加载文档
rag.load_documents("documents")
print("文档加载完成\n")

# 查询
query = "BERT 的主要特点是什么?"
print(f"查询: {query}\n")

result = rag.chat(query)
print(f"回答:\n{result}\n")

print("="*80)
print("示例运行完成")
print("="*80)
