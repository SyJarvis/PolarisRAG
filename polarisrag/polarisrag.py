# -*- coding: utf-8 -*-
import asyncio
import os
from typing import (
    List,
    Dict,
    Optional,
    Any,
    Union
)
from dataclasses import dataclass,asdict,field
from datetime import datetime
from langchain_core.messages import BaseMessage
from .base import (
    BaseLLM,
    BaseEmbedding,
    BaseVectorDB
)

from .llm import (
    ZhipuLLM
)

from .embedding import (
    ZhipuEmbedding
)

from .vector_database import (
    VectorDB
)

from .utils import (
    FolderLoader
)

from polarisrag.prompt import DGCSXY_TEMPLATE


@dataclass
class PolarisRAG:

    working_dir: str = field(
        default_factory=lambda: f"./polarisrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    # vector_storage: VectorDB = field(
    #     default_factory=lambda: VectorDB
    # )

    vector_storage: Union[VectorDB, str] = field(
        default_factory=lambda: "VectorDB"
    )

    # embedding_model: BaseEmbedding = field(
    #     default_factory=lambda: BaseEmbedding
    # )
    embedding_model: Union[BaseEmbedding, str] = field(
        default_factory=lambda: "zhipu_embedding"
    )

    llm_model: BaseLLM = field(
        default_factory=lambda: BaseLLM
    )

    is_memory: bool = False

    def __post_init__(self):
        # 测试一段文本获取embedding_dim
        self.embedding_dim = len(self.embedding_model.embed_text("This is a test"))

        # 判断是否有工作目录
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        # 判断是否有向量数据库
        if not isinstance(self.embedding_model, BaseEmbedding):
            raise Exception("embedding_model must be a BaseEmbedding instance")

        # 判断是否有向量数据库存储
        if not isinstance(self.vector_storage, BaseVectorDB):
            raise Exception("vector_storage must be a BaseVectorDB instance")

        # 判断是否有LLM
        if not isinstance(self.llm_model, BaseLLM):
            raise Exception("llm_model must be a BaseLLM instance")

    def chat(self, prompt: str, system_prompt=None, history_messages: List=[], **kwargs) -> str:
        """
        info = self.vector_storage.query(prompt)
        context = "".join(info)
        from polarisrag.prompt import DGCSXY_TEMPLATE
        prompt = DGCSXY_TEMPLATE.format(context=context, question=question)
        """
        return self.llm_model.chat(prompt)

    def insert(self, f):
        pass

    def load_conf(self):
        pass


@dataclass
class QueryParam:
    pass





