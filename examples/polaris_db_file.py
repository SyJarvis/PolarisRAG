# -*- coding: utf-8 -*-
import os
# 加载api_key
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
os.environ["ZHIPUAI_API_KEY"] = ""

from polarisrag import PolarisRAG
# 定义工作空间
WORKING_DIR = "documents"
vector_conf = {
    "class_name": "MilvusDB",
    "class_param": {
        "db_file": "milvus_data.db"
    }
}
rag = PolarisRAG(working_dir=WORKING_DIR,
                 vector_storage=vector_conf)
# 初始化rag,加载embedding、vector、llm
rag.init_rag()
print(
    rag.chat("什么是BERT")
)
print(
    rag.chat("如何下载BERT-base-chinese预训练模型")
)