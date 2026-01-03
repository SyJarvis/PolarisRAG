# -*- coding: utf-8 -*-
"""
后端适配器
"""
from typing import Any, Dict
from abc import ABC, abstractmethod


class BackendAdapter:
    """
    后端适配器
    """
    def __init__(self, backend_type: str, backend_instance: Any):
        self.backend_type = backend_type
        self.backend_instance = backend_instance

    def chat(self, content: str, **kwargs) -> str:
        """
        聊天接口
        """
        if hasattr(self.backend_instance, 'chat'):
            return self.backend_instance.chat(content, **kwargs)
        elif hasattr(self.backend_instance, 'generate'):
            return str(self.backend_instance.generate([HumanMessage(content=content)]))
        else:
            raise NotImplementedError(f"后端 {self.backend_type} 不支持 chat 接口")

    def generate(self, messages: list, **kwargs):
        """
        生成接口
        """
        if hasattr(self.backend_instance, 'generate'):
            return self.backend_instance.generate(messages, **kwargs)
        else:
            raise NotImplementedError(f"后端 {self.backend_type} 不支持 generate 接口")


# 延迟导入消息类以避免循环导入
def _import_messages():
    from .messages import BaseMessage
    return BaseMessage


# 保持全局引用
_HumanMessage = None


def HumanMessage(content: str):
    """
    获取 HumanMessage
    """
    global _HumanMessage
    if _HumanMessage is None:
        from .messages import HumanMessage as _HM
        _HumanMessage = _HM
    return _HumanMessage(content)
