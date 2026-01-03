# -*- coding: utf-8 -*-
"""
向量数据库实现

提供三种向量存储实现：
1. BaseVectorDB - 抽象基类
2. MilvusDB - 基于 LangChain Community 的 Milvus（推荐）
3. VectorDB - 简单的内存实现（用于测试）
"""
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Any
from abc import ABC, abstractmethod

from .base import BaseEmbedding
from .const import MilvusDB_CONF, similarity

try:
    from pymilvus import MilvusClient
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

try:
    from langchain_community.vectorstores import Milvus as LangChainMilvus
    from langchain_core.vectorstores import VectorStore as LangChainVectorStore
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LangChainMilvus = object
    LangChainVectorStore = object


class BaseVectorDB(ABC):
    """
    向量数据的基类

    所有向量存储实现都必须继承此类并实现抽象方法
    """

    @abstractmethod
    def insert(self, docs: List, **kwargs) -> int:
        """
        插入文档

        Args:
            docs: 文档列表
            **kwargs: 其他参数

        Returns:
            插入的文档数量
        """
        pass

    @abstractmethod
    def query(self, query: str, **kwargs) -> str:
        """
        查询相关文档

        Args:
            query: 查询文本
            **kwargs: 其他参数（如 limit, similarity）

        Returns:
            相关文档的上下文文本
        """
        pass

    @abstractmethod
    def check(self) -> bool:
        """
        检查向量存储状态

        Returns:
            True 如果可用，False 否则
        """
        pass


class VectorDB(BaseVectorDB):
    """
    简单的内存向量数据库

    用于测试或小规模场景
    不依赖外部向量服务，所有数据存储在内存中
    """

    def __init__(self, docs: List[str] = None, embedding_model: BaseEmbedding = None):
        """
        初始化内存向量数据库

        Args:
            docs: 文档列表
            embedding_model: 嵌入模型
        """
        self.docs = docs if docs is not None else []
        self.embedding_model = embedding_model
        self.vectors = []
        self.document = []

    def insert(self, docs: List[str], **kwargs) -> int:
        """
        插入文档

        Args:
            docs: 文档列表
            **kwargs: 忽略

        Returns:
            插入的文档数量
        """
        # 保存文档
        self.docs.extend(docs)
        self.document.extend(docs)

        return len(docs)

    def query(self, query: str, limit: int = 3, similarity: float = similarity, **kwargs) -> str:
        """
        查询相关文档

        Args:
            query: 查询文本
            limit: 返回的文档数量
            similarity: 相似度阈值

        Returns:
            相关文档的上下文文本
        """
        if self.embedding_model is None:
            raise ValueError("embedding_model must be specified")

        # 如果还没有向量，先生成
        if not self.vectors and self.document:
            for doc in tqdm(self.document, desc="生成向量"):
                vector = self.embedding_model.embed_text(doc)
                self.vectors.append(vector)

        # 如果没有向量，直接返回
        if not self.vectors:
            return ""

        import numpy as np

        # 生成查询向量
        query_vector = self.embedding_model.embed_text(query)

        # 计算相似度
        similarities = []
        for vector in self.vectors:
            dot_product = np.dot(query_vector, vector)
            magnitude = np.linalg.norm(query_vector) * np.linalg.norm(vector)
            if not magnitude:
                sim = 0
            else:
                sim = dot_product / magnitude
            similarities.append(sim)

        # 过滤低于阈值的
        filtered_indices = [i for i, sim in enumerate(similarities) if sim >= similarity]

        # 取前 k 个
        top_indices = sorted(filtered_indices, key=lambda i: similarities[i], reverse=True)[:limit]

        # 返回上下文
        context = "\n".join([self.document[i] for i in top_indices])
        return context

    def check(self) -> bool:
        """
        检查向量存储状态

        Returns:
            True 如果可用
        """
        return self.embedding_model is not None and len(self.document) > 0


class MilvusDB(BaseVectorDB):
    """
    Milvus 向量数据库

    基于 LangChain Community 的 Milvus 实现
    支持：
    1. 连接到 Milvus 服务（推荐）
    2. 本地文件存储（使用本地 Milvus 实例）

    默认配置：
    - 连接到本地 Milvus（使用 db_file）
    - 支持配置为远程服务
    """

    def __init__(
            self,
            db_file: str = None,
            host: str = "localhost",
            port: int = 19530,
            embedding_model: BaseEmbedding = None,
            collection_name: str = "polarisrag",
            embedding_dim: int = None,
            metric_type: str = "IP",
            drop_old: bool = False,
            *args,
            **kwargs
    ):
        """
        初始化 Milvus 向量数据库

        Args:
            db_file: 本地数据库文件路径（推荐方式，使用本地 Milvus）
            host: Milvus 服务主机（远程模式）
            port: Milvus 服务端口（远程模式）
            embedding_model: 嵌入模型
            collection_name: 集合名称
            embedding_dim: 向量维度
            metric_type: 距离类型（IP/COSINE/L2）
            drop_old: 是否删除旧集合
        """
        self.db_file = db_file
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.metric_type = metric_type
        self.embedding_model = embedding_model
        self.client = None
        self.collection = None

        # 确定嵌入维度
        if embedding_dim is None:
            if embedding_model is not None:
                # 尝试嵌入一个测试文本
                try:
                    self.embedding_dim = len(self.embedding_model.embed_text("test"))
                except Exception as e:
                    raise ValueError(f"无法确定嵌入维度: {e}")
            else:
                # 使用默认值
                self.embedding_dim = 1536  # text-embedding-3-small 的默认维度

        # 初始化客户端
        self._init_client()

    def _init_client(self):
        """
        初始化 Milvus 客户端
        """
        # 如果指定了 db_file，使用本地文件模式（推荐）
        if self.db_file:
            self.uri = self.db_file
            self.client = MilvusClient(uri=self.uri)
        else:
            # 否则连接到 Milvus 服务
            self.uri = f"{self.host}:{self.port}"
            self.client = MilvusClient(uri=self.uri)

    def set_embedding_model(self, embedding_model: BaseEmbedding):
        """
        设置嵌入模型
        """
        self.embedding_model = embedding_model

    def create_collection(self, collection_name: str = None, **kwargs) -> bool:
        """
        创建或切换集合

        Args:
            collection_name: 集合名称

        Returns:
            成功返回 True，失败返回 False
        """
        if collection_name:
            self.collection_name = collection_name
        return self._create_collection(**kwargs)

    def _create_collection(self, embedding_dim: int = None, drop_old: bool = False) -> bool:
        """
        创建集合的内部方法
        """
        dim = embedding_dim if embedding_dim is not None else self.embedding_dim

        # 如果集合已存在且不需要删除，直接返回
        if self.client.has_collection(self.collection_name) and not drop_old:
            return True

        # 如果集合存在且需要删除，先删除
        if self.client.has_collection(self.collection_name) and drop_old:
            try:
                self.client.drop_collection(self.collection_name)
            except Exception as e:
                raise RuntimeError(f"删除集合失败: {e}")

        # 创建新集合
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=dim,
                metric_type=self.metric_type,
                consistency_level="Strong"
            )
            self.collection = self.collection_name
            return True
        except Exception as e:
            raise RuntimeError(f"创建集合失败: {e}")

    def insert(self, docs: List[str], collection_name: str = None, **kwargs) -> int:
        """
        插入文档

        Args:
            docs: 文档列表
            collection_name: 集合名称

        Returns:
            插入的文档数量
        """
        if not docs:
            raise ValueError("docs 列表不能为空")

        if self.embedding_model is None:
            raise ValueError("embedding_model 未设置")

        # 使用指定的集合名称或默认集合名称
        actual_collection_name = collection_name if collection_name else self.collection_name

        # 确保集合存在
        if not self.client.has_collection(actual_collection_name):
            self._create_collection(drop_old=False)

        # 准备数据
        data = []
        for i, text in enumerate(tqdm(docs, desc="创建嵌入并插入")):
            # 生成嵌入
            vector = self.embedding_model.embed_text(text)
            
            # 构建数据项
            data.append({
                "id": i,
                "vector": vector,
                "text": text
            })

        # 插入数据
        insert_res = self.client.insert(collection_name=actual_collection_name, data=data)
        insert_count = insert_res["insert_count"]

        # 持久化
        try:
            self.client.flush()
        except Exception:
            pass  # 本地模式可能不需要 flush

        return insert_count

    def query(self, query: str, collection_name: str = None, limit: int = 3, output_fields: List[str] = None, similarity: float = similarity, **kwargs) -> str:
        """
        查询相关文档

        Args:
            query: 查询文本
            collection_name: 集合名称
            limit: 返回的文档数量
            output_fields: 返回字段列表
            similarity: 相似度阈值

        Returns:
            相关文档的上下文文本
        """
        if self.embedding_model is None:
            raise ValueError("embedding_model 未设置")

        # 使用指定的集合名称或默认集合名称
        actual_collection_name = collection_name if collection_name else self.collection_name

        # 检查集合是否存在
        if not self.client.has_collection(actual_collection_name):
            raise ValueError(f"集合 '{actual_collection_name}' 不存在")

        # 设置默认参数
        if limit is None:
            limit = MilvusDB_CONF['limit']
        if output_fields is None:
            output_fields = MilvusDB_CONF['output_fields']

        # 生成查询向量
        query_vector = self.embedding_model.embed_text(query)

        # 执行搜索
        search_res = self.client.search(
            collection_name=actual_collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=output_fields
        )

        # 解析结果
        retrieved_lines_with_distances = [
            (res["entity"]["text"], res["distance"])
            for res in search_res[0]
        ]

        # 根据相似度过滤
        context = ""
        for line_with_distance in retrieved_lines_with_distances:
            if line_with_distance[1] < similarity:
                continue
            context += line_with_distance[0] + "\n"

        return context

    def get_all_collections(self) -> List[str]:
        """
        获取所有集合名称

        Returns:
            集合名称列表
        """
        try:
            return self.client.list_collections()
        except Exception as e:
            raise RuntimeError(f"获取集合列表失败: {e}")

    def set_collection_name(self, collection_name: str):
        """
        设置默认集合名称

        Args:
            collection_name: 集合名称
        """
        self.collection_name = collection_name

    def is_exists_collection(self, collection_name: str = None) -> bool:
        """
        检查集合是否存在

        Args:
            collection_name: 集合名称

        Returns:
            存在返回 True，否则返回 False
        """
        actual_collection_name = collection_name if collection_name else self.collection_name
        return self.client.has_collection(actual_collection_name)

    def check(self) -> bool:
        """
        检查向量存储状态

        Returns:
            True 如果可用
        """
        if self.embedding_model is None:
            return False

        try:
            # 检查客户端连接
            # 如果是本地文件模式，检查文件是否存在
            if self.db_file:
                import os
                return os.path.exists(self.db_file) if self.db_file else True
            else:
                # 远程模式，尝试连接
                return self.client is not None
        except Exception as e:
            raise RuntimeError(f"向量存储检查失败: {e}") from e


__all__ = [
    "BaseVectorDB",
    "MilvusDB",
    "VectorDB"
]