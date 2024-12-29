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
        "model": "glm-4v-plus",
        "is_memory": "True"
    }
}
rag = PolarisRAG(working_dir=WORKING_DIR,
                 embedding_model=embedding_conf,
                 vector_storage=vector_conf,
                 llm_model=llm_model_conf)
rag.init_rag()
content = {
    "text": "图片中3月20日有哪些课程",
    "image": "documents/test.png"
}
result = rag.chat(content)
print(result)
result = rag.chat("那3月27日有几个课程，分别是哪些？")
print(result)