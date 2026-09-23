# -*- coding: utf-8 -*-
"""
字典配置示例：通过 class_name / class_param 配置组件，并复用已有本地向量库文件

环境变量同 quickstart.py（LLM_API_KEY / EMBEDDING_API_KEY）
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

import os

from polarisrag import PolarisRAG

WORKING_DIR = "documents"

embedding_conf = {
    "class_name": "OpenAIEmbedding",
    "class_param": {}
}
vector_conf = {
    "class_name": "MilvusDB",
    "class_param": {
        # 本地向量库文件：首次运行会创建，之后可复用
        "db_file": os.path.join(WORKING_DIR, "milvus_data.db"),
    }
}
llm_conf = {
    "class_name": "OpenAILLM",
    "class_param": {}
}

rag = PolarisRAG(
    working_dir=WORKING_DIR,
    use_config_manager=False,
    embedding_model=embedding_conf,
    vector_storage=vector_conf,
    llm_model=llm_conf,
)
rag.init_rag()

print(rag.chat("什么是BERT"))
