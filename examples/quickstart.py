# -*- coding: utf-8 -*-
"""
PolarisRAG 快速上手示例

环境变量（必须）：
- LLM_API_KEY: LLM 服务密钥
- EMBEDDING_API_KEY: Embedding 服务密钥（使用 RAG 检索时必须）

可选（OpenAI 兼容服务）：
- LLM_BASE_URL / EMBEDDING_BASE_URL
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

import os

from polarisrag import PolarisRAG

WORKING_DIR = "documents"

rag = PolarisRAG(working_dir=WORKING_DIR)
rag.init_rag()

# 插入文档（传入文件路径，支持 txt / md / pdf）
rag.insert(os.path.join(WORKING_DIR, "test.txt"))

print(rag.chat("什么是BERT"))
