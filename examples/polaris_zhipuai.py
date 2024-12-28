# -*- coding: utf-8 -*-
import os
# 加载api_key
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
os.environ["ZHIPUAI_API_KEY"] = "fb8a7db01584a5264bd4e11e242832f5.R9NuV9SxI3hxl6Rd"

from polarisrag import PolarisRAG
# 定义工作空间
WORKING_DIR = "documents"
rag = PolarisRAG(working_dir=WORKING_DIR)
# 初始化rag,加载embedding、vector、llm
rag.init_rag()
# 插入数据
with open("documents/test.txt", 'r') as f:
    rag.insert(f.read())
print(
    rag.chat("什么是BERT")
)
print(
    rag.chat("如何下载BERT-base-chinese预训练模型")
)