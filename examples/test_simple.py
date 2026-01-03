# -*- coding: utf-8 -*-
"""
简单的 RAG 测试
"""
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from polarisrag import PolarisRAG
from polarisrag.llm import OpenAILLM
from polarisrag.embedding import OpenAIEmbedding
from polarisrag.vector_database import MilvusDB


def test_direct_instance():
    """测试直接传入实例的方式"""
    print("=" * 60)
    print("测试：直接传入实例")
    print("=" * 60)

    # 创建测试文档
    test_file = "./test_simple.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("""
人工智能（AI）是计算机科学的一个分支。

机器学习是人工智能的一个子集，它使计算机能够从数据中学习。

深度学习是机器学习的一个分支，它使用多层神经网络。

机器学习的主要类型：
1. 监督学习
2. 无监督学习
3. 强化学习

深度学习的应用：
- 图像识别
- 自然语言处理
- 语音识别
        """)

    try:
        # 创建 LLM、Embedding 和 VectorDB 实例
        llm = OpenAILLM(model="glm-4.7", temperature=0.7)
        embedding = OpenAIEmbedding(model="Qwen/Qwen3-Embedding-8B")
        vector_db = MilvusDB(
            db_file="./test_simple_milvus.db",
            embedding_model=embedding,
            collection_name="test_simple_collection"
        )

        # 创建 PolarisRAG 实例
        rag = PolarisRAG(
            llm_model=llm,
            embedding_model=embedding,
            vector_storage=vector_db,
            use_config_manager=False
        )

        # 插入单个文件
        print("\n插入文档...")
        success = rag.insert(test_file)
        if success:
            print("✓ 文档插入成功")
        else:
            print("✗ 文档插入失败")
            return False

        # 测试 RAG 查询
        print("\n测试 RAG 查询...")
        query = "什么是机器学习？"
        response = rag.chat(query)
        print(f"问题: {query}")
        print(f"回答: {response}")

        print("\n✓ 测试通过！")
        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists("./test_simple_milvus.db"):
            os.remove("./test_simple_milvus.db")


if __name__ == "__main__":
    success = test_direct_instance()
    if success:
        print("\n🎉 测试通过！")
    else:
        print("\n❌ 测试失败")
        sys.exit(1)

