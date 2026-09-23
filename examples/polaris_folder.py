# -*- coding: utf-8 -*-
"""
文件夹加载示例：将工作目录下全部文档（txt/md/pdf）加载进向量库后查询

环境变量同 quickstart.py（LLM_API_KEY / EMBEDDING_API_KEY）
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from polarisrag import PolarisRAG

WORKING_DIR = "documents"

rag = PolarisRAG(working_dir=WORKING_DIR)
rag.init_rag()

# 加载 WORKING_DIR 下所有文档
rag.load_document()

print(rag.chat("什么是BERT"))
