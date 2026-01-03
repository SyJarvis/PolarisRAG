# -*- coding: utf-8 -*-
import asyncio
import os
from typing import (
    Dict,
    Union,
    List,
    Any,
    Optional
)
from dataclasses import dataclass, asdict, field, InitVar
from datetime import datetime
from .base import (
    BaseLLM,
    BaseEmbedding,
    BaseVectorDB
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

try:
    from polarisrag.config_manager import PARAM_NAME_TO_CLASS
    from polarisrag.config_manager import get_config_manager, ConfigManager
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False

from polarisrag.messages import HumanMessage, SystemMessage, BaseMessage


@dataclass
class PolarisRAG:
    """
    PolarisRAG - 基于 OpenAI 的 RAG 系统

    特性：
    - 自动检测模式：有向量库用 RAG，否则用纯 LLM
    - 专注于 OpenAI 生态（使用 OpenAILLM 和 OpenAIEmbedding）
    - 支持三种配置方式：字典、配置文件、直接实例化
    - 支持配置管理
    - 支持 Graph 工作流
    """

    config: Optional[Dict] = None
    config_path: Optional[str] = field(
        default_factory=lambda: None
    )
    use_config_manager: bool = field(
        default=True
    )

    # 可选组件 - 默认为 None，不强制依赖
    vector_storage: Optional[Union[BaseVectorDB, Dict]] = field(
        default=None
    )

    embedding_model: Optional[Union[BaseEmbedding, Dict]] = field(
        default=None
    )

    # 核心组件 - 必须有 LLM
    llm_model: Union[BaseLLM, Dict] = field(
        default_factory=lambda: DEFAULT_LLM_MODEL
    )

    role: Union[SystemPromptTemplate, str] = field(
        default_factory=lambda: DEFAULT_TEMPLATE
    )

    node_adapter: InitVar[List] = None

    def __post_init__(self, node_adapter=None):
        # 初始化消息历史
        self.messages = []
        
        # 初始化节点适配器
        if node_adapter is None:
            self.node_adapter = []
        else:
            self.node_adapter = node_adapter
        
        # 初始化工作目录
        self.working_dir = f"./polarisrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        self.file_loader = FolderLoader(folder_path=self.working_dir)

        # 使用配置管理器
        if self.use_config_manager and CONFIG_MANAGER_AVAILABLE:
            self.config_manager = get_config_manager()
            self._apply_new_config()
        else:
            # 如果提供了配置路径，加载配置
            if self.config_path:
                try:
                    self.config = self._load_conf_file(self.config_path)
                    self._apply_config()
                except FileNotFoundError:
                    print(f"配置文件 {self.config_path} 不存在，使用默认配置")
                    self.config = {}
            else:
                # 没有配置文件路径
                # 检查用户传入的是配置字典还是实例
                # 如果是实例，不需要放入 config
                self.config = {}
                # 如果传入的是字典配置，放入 config
                if isinstance(self.llm_model, dict):
                    self.config["llm_model"] = self.llm_model
                if isinstance(self.embedding_model, dict):
                    self.config["embedding_model"] = self.embedding_model
                if isinstance(self.vector_storage, dict):
                    self.config["vector_storage"] = self.vector_storage

            # 确保 LLM 模型被初始化
            if not hasattr(self, 'llm_model') or self.llm_model is None:
                # 如果配置中有 LLM 配置，从配置初始化
                if "llm_model" in self.config and isinstance(self.config["llm_model"], dict):
                    llm_dict = self.config["llm_model"]
                    if "class_name" in llm_dict:
                        class_param = llm_dict.get("class_param", {})
                        self.llm_model = self.get_llm_model_instance(llm_dict["class_name"], **class_param)
                else:
                    self.llm_model = DEFAULT_LLM_MODEL
                self.messages.append(SystemMessage(content="你是一个乐于帮助人的助手"))

    def _apply_new_config(self):
        """从配置管理器应用配置"""
        # 获取 LLM 配置
        llm_configs = self.config_manager.get_llm_configs()
        
        # 如果启用了多 LLM 模式
        if self.config_manager.is_multi_llm_enabled():
            self.multi_llm_enabled = True
            self.multi_llm_configs = llm_configs
            self.multi_llm_consensus_strategy = self.config_manager.get_consensus_strategy()
        else:
            # 单 LLM 模式 - 初始化第一个 LLM 作为主 LLM
            if llm_configs:
                main_llm_config = llm_configs[0]
                from polarisrag.llm import OpenAILLM
                self.llm_model = OpenAILLM(**main_llm_config['config_params'])
                self.messages.append(SystemMessage(content="你是一个乐于帮助人的助手"))

        # 初始化嵌入模型
        embedding_config = self.config_manager.get_embedding_config()
        if embedding_config:
            from polarisrag.embedding import OpenAIEmbedding
            self.embedding_model = OpenAIEmbedding(**embedding_config['backend_config'])
        
        # 初始化向量数据库
        vector_db_config = self.config_manager.get_vector_db_config()
        if vector_db_config:
            # 默认使用 MilvusDB 本地文件
            if 'db_file' not in vector_db_config:
                vector_db_config['db_file'] = os.path.join(self.working_dir, 'milvus_data.db')
            self.vector_storage = MilvusDB(**vector_db_config)
            if self.embedding_model:
                self.vector_storage.set_embedding_model(self.embedding_model)

    def _apply_config(self):
        """应用配置到实例"""
        if not self.config:
            return
            
        for key, node_config in self.config.items():
            if key == 'LLM_BACKEND_CONF':
                if CONFIG_MANAGER_AVAILABLE:
                    adapter_config_class = PARAM_NAME_TO_CLASS[node_config['backend_name']]
                    from polarisrag.backends import BackendAdapter
                    adapter = BackendAdapter(node_config['backend_name'], adapter_config_class(**node_config['backend_config']))
                    self.llm_model = adapter
                    self.messages.append(SystemMessage(content="你是一个乐于帮助人的助手"))
                
            elif key == 'VECTOR_STORAGE_CONF':
                # 向量存储是可选的
                try:
                    pass  # 保留接口
                except Exception as e:
                    print(f"无法初始化向量存储: {e}，功能将不可用")
                    self.vector_storage = None
                    
            elif key == 'EMBEDDING_BACKEND_CONF':
                # 嵌入模型是可选的
                try:
                    pass  # 保留接口
                except Exception as e:
                    print(f"无法初始化嵌入模型: {e}，功能将不可用")
                    self.embedding_model = None

    def chat(self, content: Union[Dict, str], system_prompt=None, history_messages=None, **kwargs) -> str:
        """
        核心聊天功能，自动检测模式

        自动检测是否有向量存储：
        - 有向量存储 → RAG 模式
        - 无向量存储 → 纯 LLM 模式
        """
        # 自动检测模式
        has_vector_storage = (
            hasattr(self, 'vector_storage') and 
            self.vector_storage is not None and
            hasattr(self, 'embedding_model') and 
            self.embedding_model is not None
        )
        
        if has_vector_storage:
            # RAG 模式
            return self._chat_with_rag(content, system_prompt, history_messages, **kwargs)
        else:
            # 纯 LLM 模式
            return self._chat_with_llm(content, system_prompt, history_messages, **kwargs)

    def _chat_with_rag(self, content: Union[Dict, str], system_prompt=None, history_messages=None, **kwargs) -> str:
        """
        使用 RAG 模式进行聊天

        步骤：
        1. 提取查询文本
        2. 使用向量存储检索相关文档
        3. 将检索结果和查询组合
        4. 调用 LLM 生成回答
        """
        if self.vector_storage is None:
            raise ValueError("向量存储未初始化")
        
        if self.embedding_model is None:
            raise ValueError("嵌入模型未初始化")
        
        if self.llm_model is None:
            raise ValueError("LLM 模型未初始化")

        # 处理输入
        if isinstance(content, str):
            query = content
        elif isinstance(content, Dict):
            query = content.get("text", "")
            if not query:
                raise ValueError("content 中缺少 'text' 字段")
        else:
            raise ValueError("content 必须是字符串或字典")

        # 使用向量存储检索相关文档
        try:
            context = self.vector_storage.query(query, limit=3)
        except Exception as e:
            print(f"向量检索失败: {e}，退回到纯 LLM 模式")
            return self._chat_with_llm(content, system_prompt, history_messages, **kwargs)

        # 构建提示词
        if context:
            prompt = f"基于以下上下文回答问题：\n\n上下文：\n{context}\n\n问题：{query}"
        else:
            prompt = query

        # 调用 LLM
        try:
            # 尝试调用 chat 方法
            return self.llm_model.chat(prompt, history=history_messages, **kwargs)
        except Exception as e:
            raise ValueError(f"LLM 调用失败: {e}")

    def _chat_with_llm(self, content: Union[Dict, str], system_prompt=None, history_messages=None, **kwargs) -> str:
        """
        使用纯 LLM 模式进行聊天

        直接调用 LLM，不使用向量存储
        """
        if self.llm_model is None:
            raise ValueError("LLM 模型未初始化")

        # 处理输入
        if isinstance(content, str):
            # 使用 chat 方法
            return self.llm_model.chat(content, history=history_messages, **kwargs)

        elif isinstance(content, Dict):
            text = content.get("text", "")
            if not text:
                raise Exception("text is required")

            # 处理图片（如果需要）
            file_data = {}
            if "image" in content:
                from .utils import open_image
                image = content["image"]
                ext = image.split(".")[-1]
                image_data = open_image(image, ext=ext)
                file_data["image_data"] = image_data

            response = self.llm_model.chat(content=text, history=history_messages, **kwargs)
            return response
        else:
            raise ValueError(f"不支持的 content 类型: {type(content)}")

    def execute_workflow(self, graph: Any, initial_inputs: Dict[str, Any] = None) -> Any:
        """
        执行工作流

        Args:
            graph: 工作流图
            initial_inputs: 初始输入

        Returns:
            工作流执行结果

        Note:
            此功能需要 core/graph.py 和 core/node.py，目前不可用
        """
        raise RuntimeError("Graph 工作流功能当前不可用，如需使用请实现 core/graph.py 和 core/node.py")
    
    # 注意：Graph 和 nodes 功能需要实现 core/graph.py 和 core/node.py
    # 以下方法暂时禁用
    
    # def create_parallel_llm_workflow(self, llm_configs: List[Dict[str, Any]], consensus_strategy: str = "majority"):
    #     """创建并行 LLM 工作流"""
    #     pass
    
    # def enable_multi_llm_mode(self, llm_configs: List[Dict[str, Any]] = None, consensus_strategy: str = None):
    #     """启用多 LLM 模式"""
    #     pass
    
    # def disable_multi_llm_mode(self):
    #     """禁用多 LLM 模式"""
    #     pass
    
    # def _chat_with_multi_llm(self, content: Union[Dict, str], system_prompt=None, history_messages=None, **kwargs) -> str:
    #     """使用多 LLM 模式进行聊天"""
    #     pass

    def load_conf(self, conf: Union[str, dict]):
        """
        加载配置
        """
        if isinstance(conf, str):
            conf_dict = self._load_conf_file(conf)
        elif isinstance(conf, dict):
            conf_dict = self._load_conf_dict(conf)
        else:
            raise Exception("conf must be a str or dict")
        self.config = conf_dict
        self._apply_config()
        self._add_dict_to_temp_env(conf_dict)

    def _load_conf_file(self, file_name: str) -> Dict:
        """
        从文件加载配置
        """
        ext = file_name.split(".")[-1]
        if ext == "json":
            conf_dict = load_json(file_name)
        elif ext == "yaml" or ext == "yml":
            conf_dict = load_yaml(file_name)
        else:
            raise Exception("conf file must be a json or yaml file")
        return conf_dict

    def _load_conf_dict(self, conf_dict) -> Dict:
        """
        从字典加载配置
        """
        return conf_dict

    def _add_dict_to_temp_env(self, env_dict: Dict):
        """
        添加到临时环境变量
        """
        for key, value in env_dict.items():
            os.environ[key] = value
        return os.environ

    def _get_vector_storage(self):
        """
        获取所有向量存储对象
        """
        return {
            "MilvusDB": MilvusDB,
            "VectorDB": VectorDB
        }

    def _get_embedding_model(self):
        """
        获取所有嵌入模型
        """
        from .embedding import OpenAIEmbedding, HFEmbedding
        return {
            "OpenAIEmbedding": OpenAIEmbedding,
            "HFEmbedding": HFEmbedding
        }

    def _get_llm_model(self):
        """
        获取所有 LLM 模型
        """
        from .llm import OpenAILLM
        return {
            "OpenAILLM": OpenAILLM
        }

    def get_vector_storage_instance(self, key: str, **kwargs):
        """
        获取向量存储实例
        """
        storage_dict = self._get_vector_storage()
        if key in storage_dict:
            return storage_dict[key](**kwargs)
        else:
            raise Exception(f"Vector storage {key} not found")

    def get_embedding_model_instance(self, key: str, **kwargs):
        """
        获取嵌入模型实例
        """
        embedding_dict = self._get_embedding_model()
        if key in embedding_dict:
            return embedding_dict[key](**kwargs)
        else:
            raise Exception(f"Embedding model {key} not found")

    def get_llm_model_instance(self, key: str, **kwargs):
        """
        获取 LLM 模型实例
        """
        llm_dict = self._get_llm_model()
        self._add_dict_to_temp_env(kwargs)
        if key in llm_dict:
            return llm_dict[key](**kwargs)
        else:
            raise Exception(f"LLM model {key} not found")

    def get_available_components(self):
        """
        获取关于所有可用和已启用组件的信息

        Returns:
            包含可用组件列表和当前默认值的字典
        """
        # 获取可用组件
        llm_list = []
        embedding_list = []
        vector_list = []
        
        if self.use_config_manager and hasattr(self, 'config_manager'):
            # 使用配置管理器获取可用组件
            config = self.config_manager.get_config()
            
            # 获取可用的 LLM 提供商/模型
            backends_config = config.get('backends', {})
            for backend_type, backend_config in backends_config.items():
                if backend_config.get('enabled', False):
                    providers = backend_config.get('providers', {})
                    for provider_name, provider_config in providers.items():
                        if provider_config.get('enabled', False):
                            for model in provider_config.get('models', []):
                                llm_list.append(f"{backend_type}:{provider_name}:{model}")
            
            # 获取可用的嵌入模型
            embedding_config = config.get('embedding', {})
            if embedding_config.get('enabled', True):
                embedding_backend_type = embedding_config.get('backend_type', 'openai')
                embedding_provider = embedding_config.get('provider', embedding_backend_type)
                embedding_model_name = embedding_config.get('model', 'text-embedding-3-small')
                embedding_list.append(f"{embedding_backend_type}:{embedding_provider}:{embedding_model_name}")
            
            # 向量存储列表
            vector_db_config = config.get('vector_db', {})
            vector_storage_type = vector_db_config.get('type', 'milvus')
            vector_list = [vector_storage_type]
            
        else:
            # 回退到原始方法
            llm_list = [k for k in self._get_llm_model().keys()]
            embedding_list = [k for k in self._get_embedding_model().keys()]
            vector_list = [k for k in self._get_vector_storage().keys()]
        
        # 获取当前默认值
        current_model = None
        current_embedding_model = None
        current_vector_db = None
        current_memory = None
        
        # 确定当前模型
        if hasattr(self, 'llm_model') and self.llm_model:
            if self.use_config_manager and hasattr(self, 'config_manager'):
                config = self.config_manager.get_config()
                if hasattr(self, 'multi_llm_enabled') and self.multi_llm_enabled:
                    if hasattr(self, 'multi_llm_configs'):
                        current_model = []
                        for llm_config in self.multi_llm_configs:
                            backend_type = llm_config.get('backend_type', 'unknown')
                            model_name = llm_config['config_params'].get('model', 'unknown')
                            current_model.append(f"{backend_type}:{model_name}")
                    else:
                        current_model = "多 LLM 模式已启用但未加载配置"
                else:
                    # 单 LLM 模式
                    backends_config = config.get('backends', {})
                    for backend_type, backend_config in backends_config.items():
                        if backend_config.get('enabled', False):
                            default_provider_name = backend_config.get('default_provider')
                            if default_provider_name:
                                providers = backend_config.get('providers', {})
                                if default_provider_name in providers:
                                    provider_config = providers[default_provider_name]
                                    if provider_config.get('isdefault', False):
                                        model_list = provider_config.get('models', [])
                                        default_model = model_list[0] if model_list else 'default-model'
                                        current_model = f"{backend_type}:{default_provider_name}:{default_model}"
                                        break
                            for provider_name, provider_config in providers.items():
                                if provider_config.get('enabled', True) and provider_config.get('isdefault', False):
                                    model_list = provider_config.get('models', [])
                                    default_model = model_list[0] if model_list else 'default-model'
                                    current_model = f"{backend_type}:{provider_name}:{default_model}"
                                    break
                        if isinstance(current_model, str) and ':' in current_model:
                            break
                    else:
                        default_provider = config.get('llm_defaults', {}).get('default_backend', 'openai')
                        default_model = config.get('llm_defaults', {}).get('default_model', 'gpt-4o-mini')
                        current_model = f"{default_provider}:{default_model}"
            else:
                current_model = str(self.llm_model) if hasattr(self, 'llm_model') else "未设置"
        
        # 获取当前嵌入模型
        if self.use_config_manager and hasattr(self, 'config_manager'):
            embedding_config = self.config_manager.get_embedding_config()
            provider = embedding_config.get("backend_name", "openai")
            model = embedding_config.get("backend_config", {}).get("model", "text-embedding-3-small")
            current_embedding_model = f"{provider}:{model}"
        else:
            current_embedding_model = str(self.embedding_model) if hasattr(self, 'embedding_model') else "未设置"
        
        # 获取当前向量 DB
        if self.use_config_manager and hasattr(self, 'config_manager'):
            vector_db_config = self.config_manager.get_vector_db_config()
            current_vector_db = vector_db_config.get('storage_name', 'milvus')
        else:
            current_vector_db = str(self.vector_storage) if hasattr(self, 'vector_storage') else "未设置"
        
        current_memory = "basic_memory" if hasattr(self, 'memory') else "未配置内存"
        
        return {
            'model_list': llm_list,
            'embedding_list': embedding_list,
            'vector_list': vector_list,
            'default_model': current_model,
            'default_embedding_model': current_embedding_model,
            'default_vector_db': current_vector_db,
            'memory': current_memory
        }

    def print_components_info(self):
        """
        打印格式化的组件信息概述
        """
        components = self.get_available_components()
        
        print("=" * 60)
        print("POLARIS RAG 组件信息")
        print("=" * 60)
        
        print(f"\n📋 可用的 LLM 模型: {len(components['model_list'])}")
        for model in components['model_list']:
            print(f"  • {model}")
        
        print(f"\n🔍 可用的嵌入模型: {len(components['embedding_list'])}")
        for model in components['embedding_list']:
            print(f"  • {model}")
        
        print(f"\n🗄️  可用的向量存储: {len(components['vector_list'])}")
        for storage in components['vector_list']:
            print(f"  • {storage}")
        
        print(f"\n⚙️  当前配置:")
        print(f"  默认模型: {components['default_model']}")
        print(f"  默认嵌入: {components['default_embedding_model']}")
        print(f"  默认向量数据库: {components['default_vector_db']}")
        print(f"  内存系统: {components['memory']}")
        
        if self.use_config_manager and hasattr(self, 'config_manager'):
            print(f"\n🌐 配置系统: 启用")
            print(f"  多 LLM 模式: {'启用' if self.config_manager.is_multi_llm_enabled() else '禁用'}")
        else:
            print(f"\n🌐 配置系统: 旧版")
        
        print("=" * 60)

    def init_rag(self):
        """
        初始化 RAG 组件

        根据 self.config 字典中的配置初始化：
        - embedding_model: 嵌入模型
        - vector_storage: 向量存储
        - llm_model: LLM 模型（如果需要）

        如果这些组件已经实例化，则跳过初始化
        """
        try:
            # 如果 embedding_model 已经是实例，跳过初始化
            if self.embedding_model is not None:
                if "embedding_model" in self.config:
                    del self.config["embedding_model"]

            # 如果 vector_storage 已经是实例，跳过初始化
            if self.vector_storage is not None:
                if "vector_storage" in self.config:
                    del self.config["vector_storage"]

            # 如果 llm_model 已经是实例，跳过初始化
            if self.llm_model is not None:
                if "llm_model" in self.config:
                    del self.config["llm_model"]

            # 如果没有需要初始化的配置，直接返回
            if not self.config:
                return

            # 提取配置
            embedding_dict = None
            vector_dict = None
            llm_dict = None

            for key, value in self.config.items():
                if key == "embedding_model":
                    embedding_dict = value
                elif key == "vector_storage":
                    vector_dict = value
                elif key == "llm_model":
                    llm_dict = value

            # 检查配置格式
            for dict_name, config_dict in [("embedding", embedding_dict), ("vector", vector_dict), ("llm", llm_dict)]:
                if config_dict is not None:
                    if "class_name" not in config_dict:
                        raise Exception(f"{dict_name}: class_name must be in the config file")
                    if "class_param" not in config_dict:
                        # 如果没有 class_param，使用空字典
                        config_dict["class_param"] = {}

            # 初始化嵌入模型
            if embedding_dict is not None:
                embedding_dict["class_param"] = embedding_dict.get("class_param", {})
                self.embedding_model = self.get_embedding_model_instance(
                    embedding_dict["class_name"],
                    **embedding_dict["class_param"]
                )
                print(f"✓ 嵌入模型初始化成功: {embedding_dict['class_name']}")

            # 初始化 LLM 模型
            if llm_dict is not None:
                llm_dict["class_param"] = llm_dict.get("class_param", {})
                self.llm_model = self.get_llm_model_instance(
                    llm_dict["class_name"],
                    **llm_dict["class_param"]
                )
                print(f"✓ LLM 模型初始化成功: {llm_dict['class_name']}")

            # 初始化向量存储
            if vector_dict is not None:
                vector_dict["class_param"] = vector_dict.get("class_param", {})
                # 将嵌入模型传递给向量存储
                if self.embedding_model is not None:
                    vector_dict["class_param"]["embedding_model"] = self.embedding_model
                self.vector_storage = self.get_vector_storage_instance(
                    vector_dict["class_name"],
                    **vector_dict["class_param"]
                )
                print(f"✓ 向量存储初始化成功: {vector_dict['class_name']}")

        except Exception as e:
            raise Exception(f"init_rag error: {e}")

    # 文档操作方法 - RAG 相关
    def load_document(self, folder_path: str = None, *args, **kwargs):
        """
        加载文档到向量存储

        Args:
            folder_path: 文档文件夹路径。如果为 None，则加载 working_dir 下的所有文档
            *args: 其他参数
            **kwargs: 其他参数

        Returns:
            成功返回 True，失败返回 False
        """
        # 检查是否具备 RAG 能力
        if not hasattr(self, 'vector_storage') or self.vector_storage is None:
            print("警告: 向量存储未初始化，文档加载功能不可用")
            return False
        if not hasattr(self, 'embedding_model') or self.embedding_model is None:
            print("警告: 嵌入模型未初始化，文档加载功能不可用")
            return False

        try:
            # 确定文件夹路径
            if folder_path is None:
                if os.path.exists(self.working_dir) and len(os.listdir(self.working_dir)) > 0:
                    folder_path = self.working_dir
                else:
                    print(f"错误: 工作目录 {self.working_dir} 为空")
                    return False
            else:
                # 检查文件夹是否存在
                if not os.path.exists(folder_path):
                    print(f"错误: 文件夹 {folder_path} 不存在")
                    return False

            # 重新创建 FolderLoader 实例以加载新文件夹
            self.file_loader = FolderLoader(folder_path=folder_path)

            # 获取所有文档的切分内容
            docs = self.file_loader.get_all_chunk_content()

            if not docs:
                print("警告: 没有找到可加载的文档")
                return False

            print(f"找到 {len(docs)} 个文档片段，正在插入向量存储...")

            # 插入到向量存储
            self.vector_storage.insert(docs=docs)

            print(f"✓ 成功加载 {len(docs)} 个文档片段到向量存储")
            return True

        except Exception as e:
            print(f"加载文档失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def insert(self, f: str):
        """
        插入单个文档到向量存储

        Args:
            f: 文件路径

        Returns:
            成功返回 True，失败返回 False
        """
        # 检查是否具备 RAG 能力
        if not hasattr(self, 'vector_storage') or self.vector_storage is None:
            print("警告: 向量存储未初始化，文档插入功能不可用")
            return False
        if not hasattr(self, 'embedding_model') or self.embedding_model is None:
            print("警告: 嵌入模型未初始化，文档插入功能不可用")
            return False

        # 检查文件路径
        assert len(f) > 0, "文件路径长度必须大于 0"

        try:
            # 检查文件是否存在
            if not os.path.exists(f):
                print(f"错误: 文件 {f} 不存在")
                return False

            # 获取文件扩展名
            ext = f.split(".")[-1].lower()

            # 根据扩展名读取文件内容
            if ext in self.file_loader.ext_func_dict:
                read_func = self.file_loader.ext_func_dict[ext]
                content = read_func(f)
            else:
                print(f"错误: 不支持的文件类型 '{ext}'，支持的类型: {list(self.file_loader.ext_func_dict.keys())}")
                return False

            if not content:
                print("警告: 文件内容为空")
                return False

            # 切分文档
            docs = self.file_loader.split_documents(content)

            if not docs:
                print("警告: 没有从文件中提取到文档片段")
                return False

            print(f"找到 {len(docs)} 个文档片段，正在插入向量存储...")

            # 插入到向量存储
            self.vector_storage.insert(docs=docs)

            print(f"✓ 成功插入 {len(docs)} 个文档片段到向量存储")
            return True

        except Exception as e:
            print(f"插入文档失败: {e}")
            import traceback
            traceback.print_exc()
            return False


@dataclass
class QueryParam:
    """
    查询参数（保留用于向后兼容）
    """
    pass

