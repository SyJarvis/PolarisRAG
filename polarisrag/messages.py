# -*- coding: utf-8 -*-
"""
消息类
"""
from typing import Dict, Any, List
from abc import ABC


class BaseMessage(ABC):
    """
    消息基类
    """
    def __init__(self, content: str):
        self.content = content

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(content={self.content!r})"


class HumanMessage(BaseMessage):
    """
    人类消息
    """
    def __init__(self, content: str):
        super().__init__(content)


class AIMessage(BaseMessage):
    """
    AI 消息
    """
    def __init__(self, content: str):
        super().__init__(content)


class SystemMessage(BaseMessage):
    """
    系统消息
    """
    def __init__(self, content: str):
        super().__init__(content)
