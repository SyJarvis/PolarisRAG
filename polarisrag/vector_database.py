# -*- coding: utf-8 -*-
from tqdm import tqdm
from typing import (
    List,
    Dict
)
import os
import json
import numpy as np

from .base import BaseVectorDB, BaseEmbedding


class VectorDB(BaseVectorDB):

    def __init__(self, docs: List, embedding_model: BaseEmbedding) -> None:
        self.docs = docs
        self.embedding_model = embedding_model
        self.vectors = []
        self.document = []

    def get_vector(self):
        for doc in tqdm(self.docs):
            self.vectors.append(self.embedding_model.embed_text(doc))
        return self.vectors

    def export_data(self, data_path="db"):
        try:
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            with open(f"{data_path}/document.json", 'w', encoding='utf-8') as f:
                json.dump(self.docs, f, ensure_ascii=False)
            with open(f"{data_path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)
        except Exception:
            return False
        return True

    # 加载json文件中的向量和字块，得到向量列表、字块列表,默认路径为'database'
    def load_vector(self, path: str = 'db') -> None:
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/document.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)
        # 求向量的余弦相似度，传入两个向量和一个embedding模型，返回一个相似度

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return self.embedding_model.compare_v(vector1, vector2)

    # 求一个字符串和向量列表里的所有向量的相似度，表进行排序，返回相似度前k个的子块列表
    def query(self, query: str, k: int = 3) -> List[str]:
        query_vector = self.embedding_model.embed_text(query)
        result = np.array([self.get_similarity(query_vector, vector)
                           for vector in self.vectors])
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()


from pymilvus import MilvusClient
from tqdm import tqdm
class MilvusDB(BaseVectorDB):
    """

    """
    def __init__(self, db_file=None):
        if db_file is None:
            db_file = "milvus_data.db"
        self.client = MilvusClient(uri=db_file)
        self.embedding_dim = None
        self.embedding_model = None

    def init(self, embedding_model=None):
        """初始化"""
        assert self.embedding_model is None, "embedding_model has been initialized"
        if not isinstance(embedding_model, BaseEmbedding):
            raise Exception("embedding_model must be an instance")
        self.embedding_model = embedding_model

    def get_text_vector(self, text: str) -> Dict:
        """
        获取文本向量
        """
        text_vector_dict = {
            "vector": self.embedding_model.embed_text(text),
            "text": text
        }
        return text_vector_dict

    def insert(self, collection_name: str, docs: List[str], **kwargs):
        """插入数据"""
        desc = kwargs["desc"] if "desc" in kwargs else "Creating embeddings"
        data = []
        for i, line in enumerate(tqdm(docs, desc=desc)):
            data.append({"id": i}.update(self.get_text_vector(line)))
        insert_res = self.client.insert(collection_name=collection_name, data=data)




    def query(self, content: str, limit: int = 3) -> str:
        pass




