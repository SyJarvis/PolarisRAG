# -*- coding: utf-8 -*-
from torch import Tensor
from sentence_transformers import SentenceTransformer
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

from .base import BaseEmbedding
class ZhipuEmbedding(BaseEmbedding):

    def __init__(
        self,
        api_key: str = None,
        model_name="embedding-2"
    ) -> None:
        if api_key is None:
            self.api_key = os.getenv("ZHIPUAI_API_KEY")
        else:
            self.api_key = api_key
        self.model_name = model_name
        self.embedding_model = ZhipuAI(api_key=self.api_key)
        # 确保embedding可以使用
        self.check()

    def embed_text(self, content: str = "", model_name=None):
        if len(content) <= 0:
            raise Exception("content length must be equal 1")
        response = self.embedding_model.embeddings.create(
            model=self.model_name,  # 填写需要调用的模型名称
            input=content  # 填写需要计算的文本内容,
        )
        return response.data[0].embedding

    def check(self):
        try:
            self.embed_text("this is a test")
            return True
        except Exception as e:
            raise Exception(e)

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

    def __init__(self, pretrain_dir: str = None, *inputs, **kwargs) -> None:
        self.pretrained_model_path = pretrain_dir
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_dir, *inputs, **kwargs)
        self.model = AutoModel.from_pretrained(pretrain_dir, *inputs, **kwargs)

    def embed_text(self, content: str, **kwargs) -> List[float]:
        """
        编码文本
        """
        if isinstance(content, str):
            contents = [content.strip()]
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
        encoded_input = self.tokenizer(contents, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def compare(self, text: str, text2: str):
        pass

    def compare_v(self, vector: List[float], vector2: List[float]) -> float:
        pass


@dataclass
class BGEEmbedding(BaseEmbedding):
    """

    """
    def __init__(
            self,
            model_name_or_path: str = "BAAI/bge-small-en-v1.5"):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    def init(self):
        self.model = AutoModel.from_pretrained(self.model_name_or_path)
        self.model.eval()

    def embed_text(self, content: str) -> List[float]:
        pass

    def embed_documents(self, contents: List[str]) -> List[List[float]]:
        pass






