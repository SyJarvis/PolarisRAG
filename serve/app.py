# -*- coding: utf-8 -*-
"""
PolarisRAG Gradio 服务

启动：
    pip install "gradio<6"          # 可选组件，不随库安装
    python serve/app.py

环境变量：
    LLM_API_KEY          必填，LLM 服务密钥
    LLM_BASE_URL         可选，OpenAI 兼容服务地址
    EMBEDDING_API_KEY    使用 RAG 检索时必填
    EMBEDDING_BASE_URL   可选
    LLM_MODEL            LLM 模型名，默认 gpt-4o-mini
    EMBEDDING_MODEL      Embedding 模型名，默认 text-embedding-3-small
    GRADIO_SERVER_NAME   监听地址，默认 0.0.0.0
    GRADIO_PORT          端口，默认 7860
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

import os

from polarisrag import PolarisRAG
from polarisrag.llm import OpenAILLM
from polarisrag.embedding import OpenAIEmbedding
from polarisrag.vector_database import MilvusDB

WORKING_DIR = "documents"

llm = OpenAILLM(model=os.getenv("LLM_MODEL", "gpt-4o-mini"))
emb = OpenAIEmbedding(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
vec = MilvusDB(
    db_file=os.path.join(WORKING_DIR, "milvus_data.db"),
    embedding_model=emb,
    collection_name="polaris_serve",
)
rag = PolarisRAG(
    llm_model=llm,
    embedding_model=emb,
    vector_storage=vec,
    working_dir=WORKING_DIR,
    use_config_manager=False,
)
rag.init_rag()

# RAG 模式：把工作目录下的语料（txt/md/pdf）载入向量库
if rag.embedding_model is not None:
    rag.load_document()


def mode_banner() -> str:
    if rag.llm_model is None:
        return "未初始化：请设置 LLM_API_KEY 后重启服务"
    if rag.embedding_model is not None and rag.vector_storage is not None:
        return "RAG 检索模式（语料目录 documents/）"
    return "纯 LLM 对话模式（未配置 EMBEDDING_API_KEY）"


def chat(message: str, history: list) -> str:
    # 说明：OpenAILLM 当前不维护对话历史（chat 的 history 参数未实现），
    # 因此这里显式拼接最近 6 轮以保证连续对话体验。
    parts = [f"{m['role']}: {m['content']}" for m in history[-6:]]
    parts.append(f"user: {message}")
    return rag.chat("\n".join(parts))


if __name__ == "__main__":
    import gradio as gr

    demo = gr.ChatInterface(
        fn=chat,
        type="messages",
        title="PolarisRAG",
        description=mode_banner(),
    )
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
    )
