<center><h2>🚀 PolarisRAG: Fast and Easy Retrieval-Augmented Generation</h2></center>
![image](https://github.com/SyJarvis/PolarisRAG/blob/main/assets/polarisRAG.png)

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
* All the code can be found in the `examples`
* Set ZhipuAI API key in environment if using ZhipuAI models: `export ZHIPUAI_API_KEY="537...".`
* Download the demo text "A Christmas Carol by Charles Dickens":
```bash
curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./book.txt
```
*  Maybe you can try loading environment variables like this. Create a new `.env` file

```
ZHIPUAI_API_KEY="537..."
OPENAI_API_KEY="sk-..."
OPENAI_BASE_URL="https://api..."
QWEN2_API_KEY="sk-no-key-required"
QWEN2_BASE_URL="http://127.0.0.1:8080/v1"
```

```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
```
Use the below Python snippet (in a script) to initialize PolarisRAG and perform queries:

```python
import os
os.environ["ZHIPUAI_API_KEY"] = ""
from polarisrag import PolarisRAG
# 定义工作空间
WORKING_DIR = "documents"
rag = PolarisRAG(working_dir=WORKING_DIR)
# 初始化rag,加载embedding、vector、llm
rag.init_rag()
# 插入数据
with open("documents/test.txt", 'r') as f:
    rag.insert(f.read())
print(rag.chat("什么是BERT"))
print(rag.chat("如何下载BERT-base-chinese预训练模型"))
```

Use PolarisRAG through a dictionary configuration

```python
from polarisrag import PolarisRAG
WORKING_DIR = "documents"
conf_dict2 = {
    "embedding_model": {
        "class_name": "HFEmbedding",
        "class_param": {
            "pretrain_dir": '/root/.cache/modelscope/hub/BAAI/bge-large-zh-v1___5'
        }
    },
    "vector_storage": {
        "class_name": "MilvusDB",
        "class_param": {}
    },
    "llm_model": {
        "class_name": "ZhipuLLM",
        "class_param": {
            "model": "glm-4-flash",
            "is_memory": "True"
        }
    }
}
rag = PolarisRAG(working_dir=WORKING_DIR, **conf_dict2)
rag.init_rag()
with open("documents/test.txt", 'r') as f:
    rag.insert(f.read())
result = rag.chat("什么是BERT")
print(result)
result = rag.chat("如何下载BERT-base-chinese预训练模型")
print(result)
```


## 🌟Citation

```python
@article{guo2024polarisrag,
title={PolarisRAG: Fast and Easy Retrieval-Augmented Generation},
author={Runke Zhong},
year={2024}
}
```
**保持热爱，奔赴星海！**

*这个世界上唯有两样东西能让我们的心灵感到深深的震撼：一是我们头上灿烂的星空，一是我们内心崇高的道德法则*

