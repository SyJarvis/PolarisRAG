# -*- coding: utf-8 -*-

MilvusDB_CONF = {
    "db_file": "milvus_data.db",
    "collection_name": "default",
    "embedding_dim": 10,
    "search_params": {
        "metric_type": "IP",
        "params": {}
    },
    "output_fields": ["text"],
    "limit": 3
}

DEFAULT_LLM_MODEL = {
    "class_name": "ZhipuLLM",
    "class_param": {
        "model": "glm-4-flash"
    }
}

DEFAULT_EMBEDDING_MODEL = {
    "from": "ZHIPUAI",
    "class_name": "ZhipuEmbedding",
    "class_param": {}
}

DEFAULT_VECTOR_STORAGE = {
    "class_name": "MilvusDB",
    "class_param": {}
}