# -*- coding: utf-8 -*-
import asyncio
import os
from typing import (
    Dict,
    Union,
    Type
)
from dataclasses import dataclass,asdict,field
from datetime import datetime
from .base import (
    BaseLLM,
    BaseEmbedding,
    BaseVectorDB
)

from .llm import (
    ZhipuLLM
)

from .embedding import (
    ZhipuEmbedding,
    HFEmbedding
)

from .vector_database import (
    VectorDB,
    MilvusDB
)

from .utils import (
    FolderLoader,
    load_yaml,
    load_json
)

from .prompt import (
    DEFAULT_TEMPLATE,
    SystemPromptTemplate
)

from .const import (
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_STORAGE
)

@dataclass
class PolarisRAG:

    config = None

    working_dir: str = field(
        default_factory=lambda: f"./polarisrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    vector_storage: Union[BaseVectorDB, Dict] = field(
        default_factory=lambda: DEFAULT_VECTOR_STORAGE
    )

    embedding_model: Union[BaseEmbedding, Dict] = field(
        default_factory=lambda: DEFAULT_EMBEDDING_MODEL
    )

    llm_model: Union[BaseLLM, Dict] = field(
        default_factory=lambda: DEFAULT_LLM_MODEL
    )

    role: Union[SystemPromptTemplate, str] = field(
        default_factory=lambda: DEFAULT_TEMPLATE
    )

    is_memory: bool = False

    def __post_init__(self):
        # 加载配置文件
        self.config = {
            "vector_storage": self.vector_storage,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model
        }
        # 判断是否有工作目录
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        # 加载文件处理器
        self.file_loader = FolderLoader(folder_path=self.working_dir)

    def chat(self, question: str, system_prompt=None, history_messages=None, **kwargs) -> str:
        """
        聊天
        """
        if system_prompt is None:
            global DEFAULT_TEMPLATE
            system_prompt = DEFAULT_TEMPLATE
        context = self.vector_storage.query(question)
        prompt = system_prompt.format(context=context, question=question)
        if history_messages is None:
            history_messages = []
        return self.llm_model.chat(prompt)

    def init_rag(self):
        # 加载默认模型
        if isinstance(self.embedding_model, BaseEmbedding):
            if "embedding_model" in self.config:
                del self.config["embedding_model"]

        # 判断是否有向量数据库存储
        if isinstance(self.vector_storage, BaseVectorDB):
            if "vector_storage" in self.config:
                del self.config["vector_storage"]

        # 判断是否有LLM
        if isinstance(self.llm_model, BaseLLM):
            if "llm_model" in self.config:
                del self.config["llm_model"]
        try:
            for key, value in self.config.items():
                if key == "embedding_model":
                    embedding_dict = value
                elif key == "vector_storage":
                    vector_dict = value
                elif key == "llm_model":
                    llm_dict = value
                if "class_name" not in value and "class_param" not in value:
                    raise Exception("class_name and class_param must be in the config file")

            embedding_dict["class_param"] = embedding_dict["class_param"] if "class_param" in embedding_dict else {}
            llm_dict["class_param"] = llm_dict["class_param"] if "class_param" in llm_dict else {}
            vector_dict["class_param"] = vector_dict["class_param"] if "class_param" in vector_dict else {}
            self.embedding_model = self.get_embedding_model_instance(embedding_dict["class_name"], **embedding_dict["class_param"])
            self.llm_model = self.get_llm_model_instance(llm_dict["class_name"], **llm_dict["class_param"])
            vector_dict["class_param"]["embedding_model"] = self.embedding_model
            self.vector_storage = self.get_vector_storage_instance(vector_dict["class_name"], **vector_dict["class_param"])
        except Exception as e:
            raise Exception(f"init_rag error: {e}")

    def load_document(self, *args, **kwargs):
        pass

    def insert(self, f:str):
        assert len(f) > 0, "f length must be greater than 0"
        docs = self.file_loader.split_documents(f)
        self.vector_storage.insert(docs=docs)

    def load_conf(self, conf: Union[str, dict]):
        if isinstance(conf, str):
            conf_dict = self._load_conf_file(conf)
        elif isinstance(conf, dict):
            conf_dict = self._load_conf_dict(conf)
        else:
            raise Exception("conf must be a str or dict")
        self.config = conf_dict
        self._add_dict_to_temp_env(conf_dict)

    def _load_conf_file(self, file_name: str) -> Dict:
        ext = file_name.split(".")[-1]
        if ext == "json":
            conf_dict = load_json(file_name)
        elif ext == "yaml" or ext == "yml":
            conf_dict = load_yaml(file_name)
        else:
            raise Exception("conf file must be a json or yaml file")
        return conf_dict

    def _load_conf_dict(self, conf_dict) -> Dict:
        return conf_dict

    def _add_dict_to_temp_env(self, env_dict: Dict):
        """添加到临时环境变量"""
        for key, value in env_dict.items():
            os.environ[key] = value
        return os.environ

    def _get_vector_storage(self) -> Type[BaseVectorDB]:
        """获取所有向量存储对象"""
        return {
            "MilvusDB": MilvusDB,
        }

    def _get_embedding_model(self) -> Type[BaseEmbedding]:
        """获取所有嵌入模型"""
        return {
            "ZhipuEmbedding": ZhipuEmbedding,
            "HFEmbedding": HFEmbedding
        }

    def _get_llm_model(self) -> Type[BaseLLM]:
        """获取所有LLM模型"""
        return {
            "ZhipuLLM": ZhipuLLM
        }

    def get_vector_storage_instance(self, key: str, **kwargs):
        storage_dict = self._get_vector_storage()
        if key in storage_dict:
            return storage_dict[key](kwargs)
        else:
            raise Exception(f"Vector storage {key} not found")

    def get_embedding_model_instance(self, key: str, **kwargs):
        embedding_dict = self._get_embedding_model()
        if key in embedding_dict:
            return embedding_dict[key](**kwargs)
        else:
            raise Exception(f"Embedding model {key} not found")

    def get_llm_model_instance(self, key: str, **kwargs):
        llm_dict = self._get_llm_model()
        self._add_dict_to_temp_env(kwargs)
        if key in llm_dict:
            return llm_dict[key](**kwargs)
        else:
            raise Exception(f"LLM model {key} not found")



@dataclass
class QueryParam:
    pass





