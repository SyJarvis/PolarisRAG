# -*- coding: utf-8 -*-
"""
配置管理器
"""
from typing import Dict, Any, List
import os


class ConfigManager:
    """
    配置管理器
    """
    def __init__(self):
        self._config = self._load_default_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """
        加载默认配置
        """
        return {
            'llm_defaults': {
                'default_backend': 'openai',
                'default_model': 'glm-4.7'
            },
            'embedding_defaults': {
                'default_backend': 'openai',
                'default_model': 'Qwen/Qwen3-Embedding-8B'
            },
            'vector_db_defaults': {
                'default_storage': 'milvus'
            },
            'backends': {
                'openai': {
                    'enabled': True,
                    'default_provider': 'openai',
                    'providers': {
                        'openai': {
                            'enabled': True,
                            'isdefault': True,
                            'models': ['glm-4.7', 'gpt-4o', 'gpt-3.5-turbo']
                        }
                    }
                }
            },
            'embedding': {
                'enabled': True,
                'backend_type': 'openai',
                'provider': 'openai',
                'model': 'Qwen/Qwen3-Embedding-8B',
                'backend_name': 'openai',
                'backend_config': {
                    'model': 'Qwen/Qwen3-Embedding-8B'
                }
            },
            'vector_db': {
                'type': 'milvus',
                'storage_name': 'milvus',
                'db_file': 'milvus_data.db',
                'collection_name': 'polarisrag'
            },
            'multi_llm': {
                'enabled': False,
                'consensus_strategy': 'majority'
            }
        }

    def get_config(self) -> Dict[str, Any]:
        """
        获取完整配置
        """
        return self._config

    def get_llm_configs(self) -> List[Dict[str, Any]]:
        """
        获取 LLM 配置列表
        """
        return [{
            'backend_type': 'openai',
            'config_params': {
                'model': self._config['llm_defaults']['default_model']
            }
        }]

    def get_embedding_config(self) -> Dict[str, Any]:
        """
        获取嵌入模型配置
        """
        return self._config.get('embedding', {})

    def get_vector_db_config(self) -> Dict[str, Any]:
        """
        获取向量数据库配置
        """
        return self._config.get('vector_db', {})

    def is_multi_llm_enabled(self) -> bool:
        """
        检查是否启用多 LLM 模式
        """
        return self._config.get('multi_llm', {}).get('enabled', False)

    def get_consensus_strategy(self) -> str:
        """
        获取共识策略
        """
        return self._config.get('multi_llm', {}).get('consensus_strategy', 'majority')


# 全局配置管理器实例
_config_manager = None


def get_config_manager() -> ConfigManager:
    """
    获取全局配置管理器实例
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# 参数名称到类的映射
PARAM_NAME_TO_CLASS = {
    'openai': None,  # 将在需要时动态导入
}
