# -*- coding: utf-8 -*-
from tqdm import tqdm
from typing import (
    List,
    Dict,
    Union
)
import os
import json
import numpy as np
from .const import MilvusDB_CONF
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
import json


class MilvusDB(BaseVectorDB):
    """
    向量数据库
    """
    def __init__(self, config: Union[Dict, None]=None):
        if config is None:
            config = {}
        self.db_file = config["db_file"] if "db_file" in config else MilvusDB_CONF["db_file"]
        self.client = MilvusClient(uri=self.db_file)
        self.collection_name = config["collection_name"] if "collection_name" in config else MilvusDB_CONF["collection_name"]
        self.embedding_model = config["embedding_model"] if "embedding_model" in config else None
        if self.embedding_model is None:
            raise Exception("embedding_model must be specified")
        self.embedding_dim = config["embedding_dim"] if "embedding_dim" in config else len(self.embedding_model.embed_text("this is a test"))

    def set_embedding_model(self, embedding_model: BaseEmbedding):
        if isinstance(embedding_model, BaseEmbedding):
            try:
                self.embedding_model = embedding_model
                self.embedding_dim = len(self.embedding_model.embed_text("this is a test"))
            except Exception as e:
                raise Exception("embedding_model must be an instance of BaseEmbedding")

    def get_text_vector(self, text: str) -> Dict:
        """
        获取文本向量
        """
        text_vector_dict = {
            "vector": self.embedding_model.embed_text(text),
            "text": text
        }
        return text_vector_dict

    def create_collection(self, collection_name: str=None, embedding_dim:int=None,
                          metric_type="IP",
                          consistency_level="Strong",
                          *args, **kwargs):
        """创建collection"""
        assert isinstance(self.client, MilvusClient), "client must be an instance of MilvusClient"
        embedding_dim = self.embedding_dim if embedding_dim is None else embedding_dim
        collection_name = self.collection_name if collection_name is None else collection_name
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
        try:
            self.client.create_collection(collection_name=collection_name,
                                          dimension=embedding_dim,
                                          metric_type=metric_type,
                                          consistency_level=consistency_level)
            return True
        except Exception as e:
            return False

    def insert(self, docs: List[str], collection_name: Union[str, None] = None, **kwargs) -> int:
        """插入数据"""
        assert len(docs) > 0, "docs must be a list and length must be greater than 0"
        collection_name = self.collection_name if collection_name is None else collection_name
        if not self.is_exists_collection(collection_name):
            self.create_collection(collection_name)
        assert isinstance(self.client, MilvusClient), "client must be an instance of MilvusClient"
        desc = kwargs["desc"] if "desc" in kwargs else "Creating embeddings"
        data = []
        for i, line in enumerate(tqdm(docs, desc=desc)):
            data_dict = self.get_text_vector(line)
            data_dict["id"] = i
            data.append(data_dict)

        insert_res = self.client.insert(collection_name=collection_name, data=data)
        insert_count = insert_res["insert_count"]
        return insert_count

    def query(self, question: str, collection_name:str=None, limit: int=None,
               search_params=None, output_fields=None, *args, **kwargs):
        assert isinstance(self.client, MilvusClient), "client must be an instance of MilvusClient"
        collection_name = self.collection_name if collection_name is None else collection_name
        if not self.is_exists_collection(collection_name):
            raise Exception("not this collection_name")
        limit = MilvusDB_CONF['limit'] if limit is None else limit
        search_params = MilvusDB_CONF['search_params'] if search_params is None else search_params
        output_fields = MilvusDB_CONF['output_fields'] if output_fields is None else output_fields
        data = [self.embedding_model.embed_text(question)]
        search_res = self.client.search(
            collection_name=collection_name,
            data=data,
            limit=limit,
            search_params=search_params,
            output_fields=output_fields
        )
        retrieved_lines_with_distances = [
            (res["entity"]["text"], res["distance"]) for res in search_res[0]
        ]
        context = "\n".join([line_with_distance[0] for line_with_distance in retrieved_lines_with_distances])
        return context

    def get_all_collections(self):
        return self.client.list_collections()

    def set_collection_name(self, collection_name:str):
        self.collection_name = collection_name

    def is_exists_collection(self, collection_name:str):
        if self.client.has_collection(collection_name):
            return True
        else:
            return False

    def check(self) -> bool:
        """检查状态"""
        if self.embedding_model is None:
            raise Exception("embedding_model is None, you must be run init_embedding_model() first")
        if self.is_exists_collection(MilvusDB_CONF["collection_name"]):
            return True
        else:
            return False