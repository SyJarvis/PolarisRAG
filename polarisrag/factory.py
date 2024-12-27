# -*- coding: utf-8 -*-
"""
工厂方法，根据字符串名称来创建对应的类实例
"""
from typing import (
    Dict,
    Type, Any,
)
from .base import BaseFactory, BaseEmbedding, BaseLLM


class EmbeddingFactory(BaseFactory):
    """工厂方法"""
    _register_embeddings: Dict[str, type[BaseEmbedding]] = {}

    def run(self):
        self._register(BaseEmbedding)
        return self

    # def _register(self):
    #     # 自动注册所有继承自BaseEmbedding的类
    #     for subclass in BaseEmbedding.__subclasses__():
    #         self.register(subclass.__name__.lower(), subclass)

    def register(self, name: str, cls_type: Type[BaseEmbedding]):
        self._register_embeddings[name] = cls_type

    def create(self, name: str, **kwargs) -> BaseEmbedding:
        embedding_class = self._register_embeddings.get(name)
        if embedding_class:
            return embedding_class(name=name, **kwargs)
        else:
            raise ValueError(f"Unknown embedding model:{name}")


class LLMFactory(BaseFactory):
    """LLM工厂方法"""

    _register_llm: Dict[str, type[BaseLLM]]

    def __init__(self):
        pass

    def run(self):
        self._register(BaseLLM)
        return self

    def create(self, name: str, **kwargs):
        """
        创建一个实例对象
        """
        llm_class = self._register_llm.get(name)
        if llm_class:
            return llm_class(name=name, **kwargs)
        else:
            raise ValueError(f"Unknown llm model:{name}")

    def register(self, name: str, cls_type: Type[BaseLLM]):
        self._register_llm[name] = cls_type


class VectorDBFactory(BaseFactory):
    """向量数据库工厂方法"""

    def run(self):
        pass

    def create(self, name: str, **kwargs):
        pass

    def register(self, name: str, cls_type: Type[Any]):
        pass


class AutoFactory:
    """
    自动注册所有的embedding, llm_model, vector_db
    """

    def run(self):
        pass

    def create(self, name: str, **kwargs):
        pass

    def register(self, name: str, cls_type: Type[Any]):
        pass
