# -*- coding: utf-8 -*-
"""
PolarisRAG 功能测试示例
"""
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from polarisrag import PolarisRAG
from polarisrag.llm import OpenAILLM
from polarisrag.embedding import OpenAIEmbedding
from polarisrag.vector_database import MilvusDB
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


def test_rag_basic():
    """
    测试基本的 RAG 功能
    """
    print("=" * 60)
    print("测试 1: 基本配置方式 - 直接传入实例")
    print("=" * 60)

    # 创建测试文档
    test_dir = "./documents"
    os.makedirs(test_dir, exist_ok=True)

    # 创建测试文件
    test_file = os.path.join(test_dir, "test.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("""
PolarisRAG 是一个基于 OpenAI 的检索增强生成（RAG）系统。
它支持自动检测模式：如果有向量存储，使用 RAG 模式；否则使用纯 LLM 模式。
系统包含三个核心组件：
1. LLM 模型 - 负责生成回答
2. 嵌入模型 - 将文本转换为向量
3. 向量数据库 - 存储和检索文档向量

PolarisRAG 支持多种配置方式：
- 直接传入实例
- 使用字典配置
- 使用配置文件（JSON/YAML）
- 使用配置管理器

支持的向量数据库包括：
- MilvusDB（推荐）
- 简单的内存向量数据库

支持的嵌入模型包括：
- OpenAI Embedding
- HuggingFace Embedding
        """)

    try:
        # 方法1: 直接传入实例
        from polarisrag.llm import OpenAILLM
        from polarisrag.embedding import OpenAIEmbedding
        from polarisrag.vector_database import MilvusDB

        llm = OpenAILLM(model="glm-4.7", temperature=0.7)
        embedding = OpenAIEmbedding(model="Qwen/Qwen3-Embedding-8B")
        vector_db = MilvusDB(
            db_file="./test_milvus.db",
            embedding_model=embedding,
            collection_name="test_collection"
        )

        rag = PolarisRAG(
            llm_model=llm,
            embedding_model=embedding,
            vector_storage=vector_db,
            use_config_manager=False
        )

        # 加载文档
        print("\n加载文档...")
        success = rag.load_document(folder_path=test_dir)
        if success:
            print("✓ 文档加载成功")
        else:
            print("✗ 文档加载失败")
            return False

        # 测试 RAG 查询
        print("\n测试 RAG 查询...")
        query = "PolarisRAG 支持哪些配置方式？"
        response = rag.chat(query)
        print(f"问题: {query}")
        print(f"回答: {response}")

        # 测试纯 LLM 模式（删除向量存储）
        print("\n测试纯 LLM 模式...")
        rag.vector_storage = None
        response_llm = rag.chat("什么是 Python？")
        print(f"纯 LLM 回答: {response_llm}")

        print("\n✓ 测试 1 通过！")
        return True

    except Exception as e:
        print(f"\n✗ 测试 1 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理测试文件
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        if os.path.exists("./test_milvus.db"):
            os.remove("./test_milvus.db")


def test_rag_with_config():
    """
    测试使用字典配置的 RAG 功能
    """
    print("\n" + "=" * 60)
    print("测试 2: 字典配置方式")
    print("=" * 60)

    # 创建测试文档
    test_dir = "./test_documents2"
    os.makedirs(test_dir, exist_ok=True)

    test_file = os.path.join(test_dir, "test.md")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("""
# Python 是什么？

Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年首次发布。

## Python 的特点

1. 简单易学 - 语法简洁清晰
2. 面向对象 - 支持多种编程范式
3. 跨平台 - 可以在 Windows、Linux、Mac 等系统上运行
4. 丰富的库 - 拥有大量的第三方库
5. 开源免费 - 完全免费使用

## Python 的应用领域

- Web 开发（Django、Flask）
- 数据科学（NumPy、Pandas）
- 人工智能（TensorFlow、PyTorch）
- 自动化脚本
        """)

    try:
        # 方法2: 使用字典配置
        config = {
            "llm_model": {
                "class_name": "OpenAILLM",
                "class_param": {
                    "model": "glm-4.7",
                    "temperature": 0.7
                }
            },
            "embedding_model": {
                "class_name": "OpenAIEmbedding",
                "class_param": {
                    "model": "Qwen/Qwen3-Embedding-8B"
                }
            },
            "vector_storage": {
                "class_name": "MilvusDB",
                "class_param": {
                    "db_file": "./test_milvus2.db",
                    "collection_name": "test_collection2"
                }
            }
        }

        rag = PolarisRAG(
            config=config,
            use_config_manager=False
        )

        # 初始化 RAG 组件
        print("\n初始化 RAG 组件...")
        rag.init_rag()

        # 加载文档
        print("\n加载文档...")
        success = rag.load_document(folder_path=test_dir)
        if success:
            print("✓ 文档加载成功")
        else:
            print("✗ 文档加载失败")
            return False

        # 测试查询
        print("\n测试查询...")
        query = "Python 有哪些特点？"
        response = rag.chat(query)
        print(f"问题: {query}")
        print(f"回答: {response}")

        print("\n✓ 测试 2 通过！")
        return True

    except Exception as e:
        print(f"\n✗ 测试 2 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理测试文件
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        if os.path.exists("./test_milvus2.db"):
            os.remove("./test_milvus2.db")


def test_insert_single_file():
    """
    测试插入单个文件的功能
    """
    print("\n" + "=" * 60)
    print("测试 3: 插入单个文件")
    print("=" * 60)

    # 创建测试文件
    test_file = "./test_single_file.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("""
机器学习是人工智能的一个分支。

机器学习的主要类型：
1. 监督学习 - 使用标记数据进行训练
2. 无监督学习 - 使用未标记数据
3. 强化学习 - 通过奖励和惩罚学习

常用的机器学习算法：
- 线性回归
- 决策树
- 支持向量机
- 神经网络
        """)

    try:
        # 创建 RAG 实例
        from polarisrag.llm import OpenAILLM
        from polarisrag.embedding import OpenAIEmbedding
        from polarisrag.vector_database import MilvusDB

        llm = OpenAILLM(model="glm-4.7", temperature=0.7)
        embedding = OpenAIEmbedding(model="Qwen/Qwen3-Embedding-8B")
        vector_db = MilvusDB(
            db_file="./test_milvus3.db",
            embedding_model=embedding,
            collection_name="test_collection3"
        )

        rag = PolarisRAG(
            llm_model=llm,
            embedding_model=embedding,
            vector_storage=vector_db,
            use_config_manager=False
        )

        # 插入单个文件
        print("\n插入单个文件...")
        success = rag.insert(test_file)
        if success:
            print("✓ 文件插入成功")
        else:
            print("✗ 文件插入失败")
            return False

        # 测试查询
        print("\n测试查询...")
        query = "机器学习有哪些主要类型？"
        response = rag.chat(query)
        print(f"问题: {query}")
        print(f"回答: {response}")

        print("\n✓ 测试 3 通过！")
        return True

    except Exception as e:
        print(f"\n✗ 测试 3 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists("./test_milvus3.db"):
            os.remove("./test_milvus3.db")


if __name__ == "__main__":
    print("PolarisRAG 功能测试")
    print("=" * 60)

    # 检查是否有 API key
    # if not os.getenv("OPENAI_API_KEY"):
    #     print("\n警告: 未设置 OPENAI_API_KEY 环境变量")
    #     print("请先设置 API key:")
    #     print("export OPENAI_API_KEY='your-api-key-here'")
    #     print("\n或者直接在代码中设置:")
    #     print("os.environ['OPENAI_API_KEY'] = 'your-api-key-here'")
    #     sys.exit(1)

    # 运行测试
    results = []
    results.append(("基本 RAG 功能", test_rag_basic()))
    results.append(("字典配置方式", test_rag_with_config()))
    results.append(("插入单个文件", test_insert_single_file()))

    # 打印测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")

    # 检查是否所有测试都通过
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 部分测试失败")
        sys.exit(1)

