# -*- coding: utf-8 -*-
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
import os

from polarisrag import PolarisRAG
from polarisrag.embedding import ZhipuEmbedding
from polarisrag.vector_database import VectorDB
from polarisrag.llm import ZhipuLLM
WORKING_DIR = "documents"
embedding_model = ZhipuEmbedding(api_key=os.getenv("ZHIPUAI_API_KEY"))
llm_model = ZhipuLLM(api_key=os.getenv("ZHIPUAI_API_KEY"))
from polarisrag.utils import FolderLoader
loader = FolderLoader(folder_path="documents")
docs = loader.get_all_chunk_content()
vector_db = VectorDB(docs, embedding_model)
vector_db.load_vector()
rag = PolarisRAG(
    working_dir=WORKING_DIR,
    embedding_model=embedding_model,
    vector_storage=vector_db,
    llm_model=llm_model,
    is_memory=True
)
res = rag.chat("如何补办学生证")
print(res)