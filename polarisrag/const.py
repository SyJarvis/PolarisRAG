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