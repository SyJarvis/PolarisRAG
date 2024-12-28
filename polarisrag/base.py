# -*- coding: utf-8 -*-
"""

"""
from __future__ import annotations
import warnings
from dataclasses import Field
from typing import (
    List,
    Any,
    Dict,
    Optional,
    Type
)
from abc import ABC, abstractmethod
import logging
from langchain_core.messages import BaseMessage

class BaseFileReader:
    pass


class BaseLLM(ABC):

    # client: Any = None
    #
    # model: str = None
    #
    # base_url: str = None
    #
    # api_key: str = None

    def __init__(self, model: str, api_key: str, base_url: str = None):
        """

        """

    @abstractmethod
    def chat(self, content: str, history: List[Dict[BaseMessage, BaseMessage]] = None) -> str:
        """
        聊天方法
        """

    def stream(self, content: str, history: List[Dict[str, str]] = None):
        pass


class BaseVectorDB(ABC):
    """
    向量数据的增删改查
    """
    @abstractmethod
    def query(self, content: str, limit: int = 3) -> str:
        """

        """

    @property
    def embeddings(self) -> Optional[BaseEmbedding]:
        """Access the query embedding object if available"""
        return None

    @abstractmethod
    def check(self) -> bool:
        """
        检查状态是否可用
        """


class BaseEmbedding(ABC):
    """
    基础的
    """

    @abstractmethod
    def embed_text(self, content: str) -> List[float]:
        """
        content: 输入的是字符串
        List[float]: 输出的是向量
        """

    @abstractmethod
    def embed_documents(self, contents: List[str]) -> List[List[float]]:
        """
        contents: 输入的是字符串列表
        List[List[float]]: 输出的是向量列表
        """

    def compare_v(self, vector: List[float], vector2: List[float]) -> float:
        """
        vector: 输入的是向量
        vector2: 输出的是向量
        """
        pass

    def compare(self, text: str, text2: str):
        """
        text: 输入的是字符串
        text2: 输入的是字符串
        """
        pass

    def check(self) -> bool:
        """
        检查
        """


class BaseChatPromptTemplate(ABC):
    """聊天提示模板类"""

    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        """

        """


class BaseFactory(ABC):
    """
    工厂类
    """
    @abstractmethod
    def run(self):
        self._register()
        return self

    @classmethod
    def create(cls, name: str, **kwargs):
        """
        创建实例对象
        """

    @abstractmethod
    def register(self, name: str, cls_type: Type[Any]):
        """
        注册对象
        """

    def _register(self, base_cls: Type[Any]):
        """
        自动注册全部对象
        """
        for subclass in base_cls.__subclasses__():
            self.register(subclass.__name__.lower(), subclass)
