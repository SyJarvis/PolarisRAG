# -*- coding: utf-8 -*-
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
from polarisrag import PolarisRAG
WORKING_DIR = "documents"
embedding_conf = {
    "class_name": "HFEmbedding",
    "class_param": {
        "pretrain_dir": '/root/.cache/modelscope/hub/BAAI/bge-large-zh-v1___5'
    }
}
vector_conf = {
    "class_name": "MilvusDB",
    "class_param": {}
}
llm_model_conf = {
    "class_name": "ZhipuLLM",
    "class_param": {
        "model": "glm-4-flash",
        "is_memory": "True"
    }
}
rag = PolarisRAG(working_dir=WORKING_DIR,
                 embedding_model=embedding_conf,
                 vector_storage=vector_conf,
                 llm_model=llm_model_conf)
rag.init_rag()
with open("documents/test.txt", 'r') as f:
    rag.insert(f.read())
result = rag.chat("什么是BERT")
print(result)
result = rag.chat("如何下载BERT-base-chinese预训练模型")
print(result)