# -*- coding: utf-8 -*-
"""
core 工作流示例：使用 Graph / Node 构建可编排的 RAG 管道

环境变量同 quickstart.py（LLM_API_KEY / EMBEDDING_API_KEY）
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

import os
import sys

if not (os.getenv("LLM_API_KEY") and os.getenv("EMBEDDING_API_KEY")):
    print("请先设置 LLM_API_KEY / EMBEDDING_API_KEY 环境变量")
    sys.exit(1)

from polarisrag.core import (
    Graph,
    RAGWorkflowBuilder,
    QueryNode,
    RetrievalNode,
    GenerationNode,
)
from polarisrag.embedding import OpenAIEmbedding
from polarisrag.llm import OpenAILLM
from polarisrag.vector_database import MilvusDB

WORKING_DIR = "documents"
os.makedirs(WORKING_DIR, exist_ok=True)

# 组件初始化
embedding_model = OpenAIEmbedding()
llm_model = OpenAILLM()
vector_db = MilvusDB(
    db_file=os.path.join(WORKING_DIR, "workflow.db"),
    embedding_model=embedding_model,
    collection_name="workflow",
    drop_old=True,
)

# 准备语料
vector_db.insert([
    "PolarisRAG 是一个检索增强生成（RAG）框架。",
    "BERT 是 Google 在 2018 年提出的预训练语言模型。",
    "Milvus 是一个开源向量数据库。",
])

# ============================================================================
# 示例 1: RAGWorkflowBuilder 链式构建（推荐）
# ============================================================================
print("=" * 60)
print("示例 1: RAGWorkflowBuilder 链式构建")
print("=" * 60)

builder = RAGWorkflowBuilder()
builder.add_query() \
       .add_retrieval(vector_db, top_k=3) \
       .add_generation(llm_model)

result = builder.execute("BERT 是什么？")
print("回答:", result.get("answer"))

# ============================================================================
# 示例 2: 手动创建节点并用 Graph 编排（灵活，可插入自定义 Node）
# ============================================================================
print("=" * 60)
print("示例 2: 手动 Graph 编排")
print("=" * 60)

graph = Graph()
graph.add_node(QueryNode("q"))
graph.add_node(RetrievalNode(vector_db=vector_db, top_k=3, name="r"))
graph.add_node(GenerationNode(llm_model=llm_model, name="g"))
graph.add_edge("q", "r")
graph.add_edge("r", "g")

result = graph.execute_workflow({"q": {"text": "PolarisRAG 是什么？"}})
print("回答:", result.get("answer"))
