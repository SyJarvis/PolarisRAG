<center><h2>🚀 PolarisRAG: Fast and Easy Retrieval-Augmented Generation</h2></center>
<p align="center">
    <img src="https://i.postimg.cc/qvR7FBb3/polaris-RAG.png" width="600"/>
<p>


## Install

* Install from source (Recommend)

```bash
cd PolarisRAG
pip install -e .
```
* Install from PyPI
```bash
pip install polarisrag
```

## Quick Start

* 示例代码见 `examples`：`quickstart.py`（快速上手）、`polaris_folder.py`（目录加载）、`polaris_db_file.py`（字典配置/复用向量库）、`polaris_graph_rag.py`（工作流编排）
* **v2.0 说明**：已移除 Zhipu / Qwen2 专用封装，统一聚焦 OpenAI 兼容生态。通过 `LLM_BASE_URL` 可接入任意 OpenAI 兼容服务（DeepSeek、GLM、vLLM、Ollama 等）。
* 环境变量：

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `LLM_API_KEY` | 是 | LLM 服务密钥 |
| `LLM_BASE_URL` | 否 | OpenAI 兼容服务地址 |
| `EMBEDDING_API_KEY` | 使用 RAG 检索时必填 | Embedding 服务密钥 |
| `EMBEDDING_BASE_URL` | 否 | Embedding 服务地址 |

* Maybe you can try loading environment variables like this. Create a new `.env` file

```
LLM_API_KEY="sk-..."
LLM_BASE_URL="https://api.openai.com/v1"
EMBEDDING_API_KEY="sk-..."
EMBEDDING_BASE_URL="https://api.openai.com/v1"
```

```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
```

* Download the demo text "A Christmas Carol by Charles Dickens":

```bash
mkdir -p documents && curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > documents/book.txt
```

* 模式自动检测：初始化了向量库和 Embedding 时自动使用 RAG 模式，否则退化为纯 LLM 对话。

Use the below Python snippet (in a script) to initialize PolarisRAG and perform queries:

```python
import os
os.environ["LLM_API_KEY"] = "sk-..."
os.environ["EMBEDDING_API_KEY"] = "sk-..."  # RAG 检索需要
from polarisrag import PolarisRAG

# 定义工作空间
WORKING_DIR = "documents"
rag = PolarisRAG(working_dir=WORKING_DIR)
# 初始化rag,加载embedding、vector、llm
rag.init_rag()
# 插入数据（传入文件路径，支持 txt / md / pdf）
rag.insert("documents/book.txt")
print(rag.chat("Scrooge 的侄子是谁"))
```

Use polarisrag by loading all documents in your working directory

```python
import os
os.environ["LLM_API_KEY"] = "sk-..."
os.environ["EMBEDDING_API_KEY"] = "sk-..."
from polarisrag import PolarisRAG

WORKING_DIR = "documents"
rag = PolarisRAG(working_dir=WORKING_DIR)
rag.init_rag()
# 加载工作目录下的全部文档到向量数据库
rag.load_document()
print(rag.chat("Scrooge 的侄子是谁"))
```

Use PolarisRAG through a dictionary configuration

```python
import os
os.environ["LLM_API_KEY"] = "sk-..."
os.environ["EMBEDDING_API_KEY"] = "sk-..."
from polarisrag import PolarisRAG

WORKING_DIR = "documents"
embedding_conf = {
    "class_name": "OpenAIEmbedding",
    "class_param": {}
}
vector_conf = {
    "class_name": "MilvusDB",
    "class_param": {
        # 本地向量库文件（推荐，无需部署 Milvus 服务）
        "db_file": "documents/milvus_data.db",
    }
}
llm_model_conf = {
    "class_name": "OpenAILLM",
    "class_param": {}
}
rag = PolarisRAG(working_dir=WORKING_DIR,
                 use_config_manager=False,
                 embedding_model=embedding_conf,
                 vector_storage=vector_conf,
                 llm_model=llm_model_conf)
rag.init_rag()
rag.insert("documents/book.txt")
result = rag.chat("Scrooge 的侄子是谁")
print(result)
```

Use polarisrag through the component

```python
import os
from polarisrag import PolarisRAG
from polarisrag.embedding import OpenAIEmbedding
from polarisrag.vector_database import MilvusDB
from polarisrag.llm import OpenAILLM
from polarisrag.utils import FolderLoader

# 工作空间（放入 txt/md/pdf 文档）
WORKING_DIR = "documents"
embedding_model = OpenAIEmbedding()   # 读取 EMBEDDING_API_KEY
llm_model = OpenAILLM()               # 读取 LLM_API_KEY
loader = FolderLoader(folder_path=WORKING_DIR)
docs = loader.get_all_chunk_content()
vector_db = MilvusDB(embedding_model=embedding_model)
# 插入文档（默认集合，自动创建）
vector_db.insert(docs=docs)
rag = PolarisRAG(
    working_dir=WORKING_DIR,
    embedding_model=embedding_model,
    vector_storage=vector_db,
    llm_model=llm_model,
)
print(
    rag.chat("Scrooge 的侄子是谁")
)
```

Use polarisrag through an existing vector database

```python
import os
os.environ["LLM_API_KEY"] = "sk-..."
os.environ["EMBEDDING_API_KEY"] = "sk-..."
from polarisrag import PolarisRAG

WORKING_DIR = "documents"
# 向量数据库配置
vector_conf = {
    "class_name": "MilvusDB",
    "class_param": {
        # 数据库文件名
        "db_file": "documents/milvus_data.db",
    }
}
embedding_conf = {
    "class_name": "OpenAIEmbedding",
    "class_param": {}
}
rag = PolarisRAG(working_dir=WORKING_DIR,
                use_config_manager=False,
                embedding_model=embedding_conf,
                vector_storage=vector_conf)
# 初始化rag,加载embedding、vector、llm
rag.init_rag()
print(
    rag.chat("Scrooge 的侄子是谁")
)
```

## 自定义模型

组件参数名为 `model`（无环境变量开关），不设置时默认 `gpt-4o-mini` / `text-embedding-3-small`：

```python
# 字典配置方式（class_param 中传 model）
rag = PolarisRAG(
    use_config_manager=False,
    llm_model={
        "class_name": "OpenAILLM",
        "class_param": {
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
        },
    },
    embedding_model={
        "class_name": "OpenAIEmbedding",
        "class_param": {"model": "text-embedding-ada-002"},
    },
)

# 组件实例方式（接入 OpenAI 兼容服务 / 本地模型）
from polarisrag.llm import OpenAILLM
from polarisrag.embedding import HFEmbedding

llm = OpenAILLM(model="qwen-plus",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
embedding = HFEmbedding(pretrain_dir="/path/to/bge-m3")  # 本地模型，参数为 pretrain_dir
```

> 注：`config/` 目录下的 yaml 当前不生效（`ConfigManager` 使用代码内默认值，暂不读取文件），模型名请通过上述方式传入。

## 工作流编排（进阶）

`polarisrag.core` 提供基于 Graph/Node 的 RAG 管道编排（Query → Retrieval → Prompt → Generation），完整示例见 `examples/polaris_graph_rag.py`：

```python
from polarisrag.core import RAGWorkflowBuilder

builder = RAGWorkflowBuilder()
builder.add_query() \
       .add_retrieval(vector_db, top_k=3) \
       .add_generation(llm_model)
result = builder.execute("BERT 是什么？")
print(result["answer"])
```

## 🌟Citation

```bibtex
@article{guo2024polarisrag,
title={PolarisRAG: Fast and Easy Retrieval-Augmented Generation},
author={Runke Zhong},
year={2024}
}
```
**保持热爱，奔赴星海！**

*这个世界上唯有两样东西能让我们的心灵感到深深的震撼：一是我们头上灿烂的星空，一是我们内心崇高的道德法则*
