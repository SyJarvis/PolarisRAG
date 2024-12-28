# -*- coding: utf-8 -*-
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

import os
from polarisrag import PolarisRAG
from polarisrag.embedding import ZhipuEmbedding
from polarisrag.vector_database import MilvusDB
from polarisrag.llm import ZhipuLLM
from polarisrag.utils import FolderLoader
# api_key
ZHIPUAI_API_KEY = ""
# 工作空间
WORKING_DIR = "documents"
embedding_model = ZhipuEmbedding(api_key=ZHIPUAI_API_KEY)
llm_model = ZhipuLLM(api_key=os.getenv("ZHIPUAI_API_KEY"))
loader = FolderLoader(folder_path=WORKING_DIR)
docs = loader.get_all_chunk_content()
vector_db = MilvusDB({"embedding_model": embedding_model})
# 创建集合
vector_db.create_collection("default")
vector_db.insert(docs=docs)
rag = PolarisRAG(
    working_dir=WORKING_DIR,
    embedding_model=embedding_model,
    vector_storage=vector_db,
    llm_model=llm_model,
    is_memory=True
)
res = rag.chat("什么是BERT")
print(res)