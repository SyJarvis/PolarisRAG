# -*- coding: utf-8 -*-
"""
使用OpenAIEmbedding的示例演示
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

import os
from polarisrag import PolarisRAG

# 示例1: 使用字典配置OpenAIEmbedding
print("=== 示例1: 使用字典配置 ===")
WORKING_DIR = "documents"

embedding_conf = {
    "class_name": "OpenAIEmbedding",
    "class_param": {
        "model_name": "Qwen/Qwen3-Embedding-8B",
        "api_key": os.getenv("LLM_API_KEY"),
        "base_url": os.getenv("LLM_BASE_URL")
    }
}

llm_model_conf = {
    "class_name": "OpenAILLM",
    "class_param": {
        "model": "glm-4.7",
        "api_key": os.getenv("LLM_API_KEY"),
        "base_url": os.getenv("LLM_BASE_URL")
    }
}

vector_conf = {
    "class_name": "MilvusDB",
    "class_param": {}
}

rag = PolarisRAG(
    working_dir=WORKING_DIR,
    embedding_model=embedding_conf,
    vector_storage=vector_conf,
    llm_model=llm_model_conf
)

# 初始化RAG系统
rag.init_rag()

# 加载文档
rag.load_document()

# 查询示例
result = rag.chat("什么是BERT?")
print(result)


# 示例2: 直接使用OpenAIEmbedding组件
print("\n=== 示例2: 直接使用组件 ===")
from polarisrag.embedding import OpenAIEmbedding
from polarisrag.llm import OpenAILLM
from polarisrag.vector_database import MilvusDB
from polarisrag.utils import FolderLoader

# 初始化组件
embedding_model = OpenAIEmbedding(
    api_key=os.getenv("EMBEDDING_API_KEY"),
    model_name="Qwen/Qwen3-Embedding-8B",
    base_url=os.getenv("EMBEDDING_BASE_URL")
)

llm_model = OpenAILLM(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    model="glm-4.7"
)

# 加载文档
loader = FolderLoader(folder_path=WORKING_DIR)
docs = loader.get_all_chunk_content()

# 创建向量数据库
vector_db = MilvusDB({"embedding_model": embedding_model})
vector_db.create_collection("openai_collection")
vector_db.insert(docs=docs, collection_name="openai_collection")

# 创建RAG实例
rag = PolarisRAG(
    working_dir=WORKING_DIR,
    embedding_model=embedding_model,
    vector_storage=vector_db,
    llm_model=llm_model
)

# 查询
result = rag.chat("如何下载BERT模型?")
print(result)


# 示例3: 测试OpenAIEmbedding的基本功能
print("\n=== 示例3: 测试OpenAIEmbedding基本功能 ===")
embedding = OpenAIEmbedding(
    api_key=os.getenv("EMBEDDING_API_KEY"),
    base_url=os.getenv("EMBEDDING_BASE_URL"),
    model_name="Qwen/Qwen3-Embedding-8B"
)

# 测试单个文本嵌入
text = "这是一个测试文本"
vector = embedding.embed_text(text)
print(f"文本: {text}")
print(f"向量维度: {len(vector)}")
print(f"向量前10维: {vector[:10]}")

# 测试批量文本嵌入
texts = ["文本1", "文本2", "文本3"]
vectors = embedding.embed_documents(texts)
print(f"\n批量嵌入:")
for i, (txt, vec) in enumerate(zip(texts, vectors)):
    print(f"{i+1}. {txt} -> 向量维度: {len(vec)}")

# 测试文本相似度
text1 = "机器学习是人工智能的一个分支"
text2 = "深度学习属于机器学习"
text3 = "今天天气很好"

similarity_12 = embedding.compare(text1, text2)
similarity_13 = embedding.compare(text1, text3)

print(f"\n文本相似度:")
print(f"'{text1}' 和 '{text2}' 的相似度: {similarity_12:.4f}")
print(f"'{text1}' 和 '{text3}' 的相似度: {similarity_13:.4f}")

# 测试向量相似度
vec1 = embedding.embed_text(text1)
vec2 = embedding.embed_text(text2)
similarity_v = embedding.compare_v(vec1, vec2)
print(f"向量相似度: {similarity_v:.4f}")

print("\n=== 所有测试完成 ===")

