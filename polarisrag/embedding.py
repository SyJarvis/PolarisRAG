# -*- coding: utf-8 -*-
from torch import Tensor

from .base import BaseEmbedding

import os
from zhipuai import ZhipuAI
from openai import OpenAI, Embedding
from abc import ABC
from typing import (
    List,
    Dict,
    Type,
    Any
)
import numpy as np
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
import torch


class ZhipuEmbedding(BaseEmbedding):

    def __init__(
        self,
        api_key: str = "",
        model_name="embedding-2",
        name: str=None
    ) -> None:
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("ZHIPUAI_API_KEY")
        self.model_supplier = "ZHIPUAI"
        self.model_name = model_name
        self.embedding_model = ZhipuAI(api_key=api_key)
        self.name = name

    def embed_text(self, content: str = "", model_name=None):
        if len(content) <= 0:
            raise Exception("content length must be equal 1")
        response = self.embedding_model.embeddings.create(
            model=self.model_name,  # 填写需要调用的模型名称
            input=content  # 填写需要计算的文本内容,
        )
        return response.data[0].embedding

    def embed_documents(self, content_list: List[str], model_name=None):
        assert len(content_list) > 0, "content length must be equal 1"
        content_vector_list = []
        for content in content_list:
            content_vector_list.append(self.embed_text(content=content))
        return content_vector_list

    def compare_v(cls, vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

    def compare(self, text1: str, text2: str):
        embed1 = self.embedding_model.embeddings.create(
            model=self.model_name,  # 填写需要调用的模型名称
            input=text1  # 填写需要计算的文本内容,
        ).data[0].embedding
        embed2 = self.embedding_model.embeddings.create(
            model=self.model_name,  # 填写需要调用的模型名称
            input=text2  # 填写需要计算的文本内容,
        ).data[0].embedding
        return np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))


class OpenAIEmbedding(BaseEmbedding):
    """

    """
    def __init__(
        self,
        api_key: str = "",
        model_name: str = "",
        name: str = None
    ) -> None:
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.name = name

        self.client = Embedding.create(
            model="text-embedding-ada-002",

        )

    def embed_text(self, content: str) -> List[float]:
        pass

    def embed_documents(self, contents: List[str]) -> List[List[float]]:
        pass


class HFEmbedding(BaseEmbedding, ABC):
    """
    huggingface_embedding
    """

    def __init__(self, pretrained_model_path: str = None) -> None:
        self.pretrained_model_path = pretrained_model_path

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *inputs, **kwargs):
        r"""

        """
        cls.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, *inputs, **kwargs)
        cls.model = AutoModel.from_pretrained(pretrained_model_path, *inputs, **kwargs)
        return cls(pretrained_model_path)

    def embed_text(self, content: str, **kwargs) -> List[float]:
        """
        编码文本
        """
        if isinstance(content, str):
            contents = [content]
        else:
            raise Exception("content must be str")

        return self.__embedding(contents, **kwargs).tolist()[0]

    def embed_documents(self, contents: List[str], **kwargs) -> Tensor:
        """
        编码文档
        """
        return self.__embedding(contents, **kwargs)

    def __embedding(self, contents: List[str], **kwargs) -> Tensor:
        """
        padding=True, truncation=True, return_tensors='pt'
        """
        encoded_input = HFEmbedding.tokenizer(contents, **kwargs)
        with torch.no_grad():
            model_output = HFEmbedding.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def compare(self, text: str, text2: str):
        pass

    def compare_v(self, vector: List[float], vector2: List[float]) -> float:
        pass


@dataclass
class BGEEmbedding(HFEmbedding):
    """

    """
    def __init__(self, model_dir="moka-ai/m3e-base"):
        super().__init__(model_dir=model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained()

    def embed_text(self, content: str) -> List[float]:
        pass

    def embed_documents(self, contents: List[str]) -> List[List[float]]:
        pass


class AutoEmbedding:

    def __init__(self):
        pass



