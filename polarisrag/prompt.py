# -*- coding: utf-8 -*-

# 你是一个乐于解答各种问题的助手，你需要记住跟用户对话的所有内容，你的任务是为用户提供专业、准确、有见地的建议。
SYSTEM_PROMPT = """
你是一个乐于助人的助手,你需要记住跟用户对话的所有内容
"""

DEFAULT_TEMPLATE = """使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
问题: <question>{question}</question>
可参考的上下文：
···
<context>{context}</context>
···
如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
有用的回答:"""

from typing import (
    List,
    Sequence
)


class ChatPromptTemplate:
    """

    """
    messages: List[str]
    def __init__(
        self,
        messages: Sequence[List[str]]
    ) -> None:
        pass


class SystemPromptTemplate:
    """
        系统提示模板
    """
    pass