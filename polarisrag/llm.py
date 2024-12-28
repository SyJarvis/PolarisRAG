# -*- coding: utf-8 -*-
import os
from typing import List, Dict

from langchain_zhipu import ChatZhipuAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
# 上下文记忆
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage
# 流式输出

from .base import BaseLLM
from .prompt import SYSTEM_PROMPT


class ZhipuLLM(BaseLLM):
    """
    智谱AI的封装
    """

    def __init__(self, api_key: str = None,
                 base_url = None,
                 model: str = "glm-4-flash",
                 temperature: float = 0.8,
                 is_memory: bool = False,
                 system_prompt: str = None):
        super().__init__(model, api_key, base_url)
        if api_key:
            os.environ["ZHIPUAI_API_KEY"] = api_key
        else:
            api_key = os.environ.get("ZHIPUAI_API_KEY") if "ZHIPUAI_API_KEY" in os.environ else api_key
            if api_key is None:
                raise ValueError("ZHIPUAI_API_KEY is None")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.is_memory = True if is_memory == "True" or is_memory == True else False
        self.client = ChatZhipuAI(
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key
        )
        self.memory = None

        if system_prompt is None:
            self.system_prompt = SYSTEM_PROMPT
        else:
            self.system_prompt = system_prompt

        if self.is_memory is True:
            self.memory = ConversationBufferMemory(return_messages=True)
            self.memory.load_memory_variables({})
            self.chat_prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("user", "{input}")
            ])
        else:
            self.memory = None
            self.chat_prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("user", "{input}")
            ])
        self.history = []
        self.output_parser = StrOutputParser()

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def chat(self, content: str, history: List[Dict[BaseMessage, BaseMessage]] = None, **kwargs) -> str:
        """

        """
        if history is not None:
            self.history = history
        # else:
        #     self.history = self.memory.load_memory_variables({})['history']
        chain = self.chat_prompt | self.client | self.output_parser
        response = chain.invoke({"input": content, "history": self.history})

        if self.is_memory:
            self.memory.save_context({"input": content}, {"history": response})
            self.history = self.memory.load_memory_variables({})['history']
        return response

    def get_history(self) -> List:
        """
            获取历史记录
        """
        self.history = self.memory.load_memory_variables({})['history']
        return self.history

    def set_history(self, question, answer) -> bool:
        assert len(question) > 0 and len(answer) > 0, "question and answer can't be empty"
        try:
            self.memory.save_context({"input": question}, {"output": answer})
            self.memory.load_memory_variables({})
            return True
        except Exception as e:
            return False

    def stream(self, content: str, history: List[Dict[str, str]] = None):
        if history is not None:
            self.history = history
        chain = self.chat_prompt | self.client | self.output_parser
        for chunk in chain.stream({"input": content, "history": self.history}):
            print(chunk)


class Qwen2LLM(BaseLLM):
    """

    """
    def __init__(self, model: str, api_key: str, base_url: str = None):
        super().__init__(model, api_key, base_url)

    def chat(self, content: str, history: List[Dict[BaseMessage, BaseMessage]] = None) -> str:
        pass

    def stream(self, content: str, history: List[Dict[str, str]] = None):
        super().stream(content, history)


