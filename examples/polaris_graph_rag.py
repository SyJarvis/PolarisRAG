# -*- coding: utf-8 -*-
"""
RAG 工作流使用示例
演示如何使用 Graph/Node 架构构建 RAG 系统
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

import os
from polarisrag import PolarisRAG
from polarisrag.core import (
    Graph,
    RAGWorkflowBuilder,
    QueryNode,
    EmbeddingNode,
    RetrievalNode,
    PromptNode,
    GenerationNode
)
from polarisrag.embedding import ZhipuEmbedding, OpenAIEmbedding
from polarisrag.llm import ZhipuLLM, OpenAILLM
from polarisrag.vector_database import MilvusDB
from polarisrag.utils import FolderLoader

# ============================================================================
# 示例 1: 使用 RAGWorkflowBuilder 快速构建（推荐）
# ============================================================================

def example_1_builder():
    """
    使用 RAGWorkflowBuilder 快速构建 RAG 工作流
    最简单、最推荐的方式
    """
    print("\n" + "="*80)
    print("示例 1: 使用 RAGWorkflowBuilder 快速构建")
    print("="*80 + "\n")

    # 1. 初始化组件
    print("1. 初始化组件...")
    embedding_model = ZhipuEmbedding(api_key=os.getenv("ZHIPUAI_API_KEY"))
    vector_db = MilvusDB({"embedding_model": embedding_model})

    # 加载文档
    WORKING_DIR = "documents"
    loader = FolderLoader(folder_path=WORKING_DIR)
    docs = loader.get_all_chunk_content()

    # 创建向量库
    vector_db.create_collection("demo_collection")
    vector_db.insert(docs=docs, collection_name="demo_collection")
    print(f"   已加载 {len(docs)} 个文档片段")

    llm_model = ZhipuLLM(api_key=os.getenv("ZHIPUAI_API_KEY"))

    # 2. 构建工作流
    print("\n2. 构建 RAG 工作流...")
    builder = RAGWorkflowBuilder()
    builder.add_query() \
           .add_embedding(embedding_model) \
           .add_retrieval(vector_db, top_k=3) \
           .add_generation(llm_model)

    graph = builder.build()
    print("   工作流构建完成")

    # 3. 执行查询
    print("\n3. 执行查询...")
    query = "什么是BERT?"
    print(f"   问题: {query}")

    result = builder.execute(query)

    # 4. 显示结果
    print("\n4. 查询结果:")
    print(f"   {result['answer']}\n")


# ============================================================================
# 示例 2: 手动创建和连接节点（灵活）
# ============================================================================

def example_2_manual():
    """
    手动创建和连接节点
    展示更高的灵活性
    """
    print("\n" + "="*80)
    print("示例 2: 手动创建和连接节点")
    print("="*80 + "\n")

    # 1. 初始化组件
    print("1. 初始化组件...")
    embedding_model = OpenAIEmbedding(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    vector_db = MilvusDB({"embedding_model": embedding_model})

    # 加载文档
    WORKING_DIR = "documents"
    loader = FolderLoader(folder_path=WORKING_DIR)
    docs = loader.get_all_chunk_content()

    # 创建向量库
    vector_db.create_collection("manual_collection")
    vector_db.insert(docs=docs, collection_name="manual_collection")
    print(f"   已加载 {len(docs)} 个文档片段")

    llm_model = OpenAILLM(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo"
    )

    # 2. 创建节点
    print("\n2. 创建工作流节点...")
    graph = Graph()

    query_node = QueryNode(name="Query")
    embedding_node = EmbeddingNode(embedding_model, name="Embedding")
    retrieval_node = RetrievalNode(vector_db, top_k=3, name="Retrieval")
    prompt_node = PromptNode(name="Prompt")
    generation_node = GenerationNode(llm_model, name="Generation")

    print("   已创建 5 个节点")

    # 3. 添加节点到图
    print("\n3. 添加节点到图中...")
    graph.add_node(query_node)
    graph.add_node(embedding_node)
    graph.add_node(retrieval_node)
    graph.add_node(prompt_node)
    graph.add_node(generation_node)

    # 4. 连接节点（构建数据流）
    print("\n4. 连接节点...")
    graph.add_edge("Query", "Embedding")
    graph.add_edge("Embedding", "Retrieval")
    graph.add_edge("Retrieval", "Prompt")
    graph.add_edge("Prompt", "Generation")

    # 5. 执行工作流
    print("\n5. 执行工作流...")
    query = "如何下载BERT模型?"
    print(f"   问题: {query}")

    initial_inputs = {
        "Query": {"text": query}
    }

    result = graph.execute_workflow(initial_inputs)

    # 6. 显示结果
    print("\n6. 查询结果:")
    print(f"   {result['answer']}\n")


# ============================================================================
# 示例 3: 可视化工作流
# ============================================================================

def example_3_visualize():
    """
    生成工作流可视化图
    方便调试和展示
    """
    print("\n" + "="*80)
    print("示例 3: 可视化工作流")
    print("="*80 + "\n")

    # 1. 构建工作流
    print("1. 构建工作流...")
    embedding_model = ZhipuEmbedding(api_key=os.getenv("ZHIPUAI_API_KEY"))
    vector_db = MilvusDB({"embedding_model": embedding_model})
    llm_model = ZhipuLLM(api_key=os.getenv("ZHIPUAI_API_KEY"))

    builder = RAGWorkflowBuilder()
    builder.add_query("Query") \
           .add_embedding(embedding_model, "Embedding") \
           .add_retrieval(vector_db, name="Retrieval") \
           .add_prompt(name="Prompt") \
           .add_generation(llm_model, "Generation")

    graph = builder.build()

    # 2. 生成 DOT 文件
    print("\n2. 生成可视化文件...")
    dot_path = "rag_workflow.dot"
    graph.save_dot(dot_path)
    print(f"   DOT 文件已保存到: {dot_path}")

    # 3. 显示 DOT 内容
    print("\n3. DOT 内容:")
    print("-" * 80)
    with open(dot_path, 'r', encoding='utf-8') as f:
        print(f.read())
    print("-" * 80)

    print("\n4. 生成图片（需要安装 Graphviz）:")
    print(f"   dot -Tpng {dot_path} -o rag_workflow.png")
    print(f"   然后打开 rag_workflow.png 查看工作流图\n")


# ============================================================================
# 示例 4: 自定义提示词模板
# ============================================================================

def example_4_custom_prompt():
    """
    使用自定义提示词模板
    """
    print("\n" + "="*80)
    print("示例 4: 使用自定义提示词模板")
    print("="*80 + "\n")

    # 1. 初始化组件
    print("1. 初始化组件...")
    embedding_model = ZhipuEmbedding(api_key=os.getenv("ZHIPUAI_API_KEY"))
    vector_db = MilvusDB({"embedding_model": embedding_model})

    WORKING_DIR = "documents"
    loader = FolderLoader(folder_path=WORKING_DIR)
    docs = loader.get_all_chunk_content()

    vector_db.create_collection("custom_prompt_collection")
    vector_db.insert(docs=docs, collection_name="custom_prompt_collection")
    print(f"   已加载 {len(docs)} 个文档片段")

    llm_model = ZhipuLLM(api_key=os.getenv("ZHIPUAI_API_KEY"))

    # 2. 自定义提示词模板
    print("\n2. 使用自定义提示词模板...")
    custom_template = """你是一个专业的技术文档助手。请根据提供的上下文信息，准确、简洁地回答用户的问题。

问题: {question}

参考信息:
{context}

请用中文回答，如果上下文中没有相关信息，请明确说明。"""

    # 3. 构建工作流
    builder = RAGWorkflowBuilder()
    builder.add_query() \
           .add_embedding(embedding_model) \
           .add_retrieval(vector_db, top_k=3) \
           .add_prompt(template=custom_template) \
           .add_generation(llm_model)

    # 4. 执行查询
    print("\n3. 执行查询...")
    query = "BERT有哪些版本?"
    print(f"   问题: {query}")

    result = builder.execute(query)

    print("\n4. 查询结果:")
    print(f"   {result['answer']}\n")


# ============================================================================
# 示例 5: 添加重排序节点（高级功能）
# ============================================================================

def example_5_with_rerank():
    """
    添加重排序节点优化检索结果
    """
    print("\n" + "="*80)
    print("示例 5: 添加重排序节点")
    print("="*80 + "\n")

    # 1. 初始化组件
    print("1. 初始化组件...")
    embedding_model = ZhipuEmbedding(api_key=os.getenv("ZHIPUAI_API_KEY"))
    vector_db = MilvusDB({"embedding_model": embedding_model})

    WORKING_DIR = "documents"
    loader = FolderLoader(folder_path=WORKING_DIR)
    docs = loader.get_all_chunk_content()

    vector_db.create_collection("rerank_collection")
    vector_db.insert(docs=docs, collection_name="rerank_collection")
    print(f"   已加载 {len(docs)} 个文档片段")

    llm_model = ZhipuLLM(api_key=os.getenv("ZHIPUAI_API_KEY"))

    # 2. 构建带重排序的工作流
    print("\n2. 构建带重排序的工作流...")
    builder = RAGWorkflowBuilder()
    builder.add_query() \
           .add_embedding(embedding_model) \
           .add_retrieval(vector_db, top_k=5) \
           .add_rerank(top_k=3) \
           .add_prompt() \
           .add_generation(llm_model)

    # 3. 执行查询
    print("\n3. 执行查询...")
    query = "BERT的主要特点是什么?"
    print(f"   问题: {query}")

    result = builder.execute(query)

    print("\n4. 查询结果:")
    print(f"   {result['answer']}\n")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*20 + "RAG 工作流使用示例" + " "*38 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)

    # 根据需要选择运行哪个示例
    # 取消注释你想运行的示例

    example_1_builder()      # 推荐：使用 Builder 快速构建
    # example_2_manual()      # 手动创建和连接节点
    # example_3_visualize()   # 可视化工作流
    # example_4_custom_prompt()  # 自定义提示词模板
    # example_5_with_rerank()    # 添加重排序节点

    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*30 + "示例运行完成" + " "*36 + "#")
    print("#" + " "*78 + "#")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()

