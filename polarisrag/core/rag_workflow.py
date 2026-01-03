# -*- coding: utf-8 -*-
"""
RAG 工作流构建器
"""
from typing import Optional
from polarisrag.core.graph import Graph
from polarisrag.core.rag_nodes import (
    QueryNode,
    EmbeddingNode,
    RetrievalNode,
    PromptNode,
    GenerationNode,
    RerankNode
)
from polarisrag.base import BaseEmbedding, BaseVectorDB, BaseLLM


class RAGWorkflowBuilder:
    """
    RAG 工作流构建器 - 提供链式 API 快速构建 RAG 工作流

    示例:
        workflow = RAGWorkflowBuilder() \
            .add_query() \
            .add_embedding(embedding_model) \
            .add_retrieval(vector_db) \
            .add_generation(llm_model) \
            .build()

        result = workflow.execute("什么是BERT?")
    """

    def __init__(self):
        """初始化工作流构建器"""
        self.graph = Graph()
        self.last_node_name = None
        self.nodes = {}

    def add_query(self, name: str = "QueryNode") -> "RAGWorkflowBuilder":
        """
        添加查询节点

        Args:
            name: 节点名称

        Returns:
            self，支持链式调用
        """
        node = QueryNode(name=name)
        self.graph.add_node(node)
        self.nodes[name] = node
        self.last_node_name = name
        return self

    def add_embedding(
        self,
        embedding_model: BaseEmbedding,
        name: str = "EmbeddingNode"
    ) -> "RAGWorkflowBuilder":
        """
        添加嵌入节点

        Args:
            embedding_model: 嵌入模型实例
            name: 节点名称

        Returns:
            self，支持链式调用
        """
        node = EmbeddingNode(embedding_model=embedding_model, name=name)
        self.graph.add_node(node)
        self.nodes[name] = node

        # 连接到上一个节点
        if self.last_node_name:
            self.graph.add_edge(self.last_node_name, name)
        self.last_node_name = name

        return self

    def add_retrieval(
        self,
        vector_db: BaseVectorDB,
        top_k: int = 3,
        name: str = "RetrievalNode"
    ) -> "RAGWorkflowBuilder":
        """
        添加检索节点

        Args:
            vector_db: 向量数据库实例
            top_k: 检索返回的文档数量
            name: 节点名称

        Returns:
            self，支持链式调用
        """
        node = RetrievalNode(vector_db=vector_db, top_k=top_k, name=name)
        self.graph.add_node(node)
        self.nodes[name] = node

        # 连接到上一个节点
        if self.last_node_name:
            self.graph.add_edge(self.last_node_name, name)
        self.last_node_name = name

        return self

    def add_rerank(
        self,
        rerank_model=None,
        top_k: int = 3,
        name: str = "RerankNode"
    ) -> "RAGWorkflowBuilder":
        """
        添加重排序节点（可选）

        Args:
            rerank_model: 重排序模型（可选）
            top_k: 重排序后保留的文档数量
            name: 节点名称

        Returns:
            self，支持链式调用
        """
        node = RerankNode(rerank_model=rerank_model, top_k=top_k, name=name)
        self.graph.add_node(node)
        self.nodes[name] = node

        # 连接到上一个节点
        if self.last_node_name:
            self.graph.add_edge(self.last_node_name, name)
        self.last_node_name = name

        return self

    def add_prompt(
        self,
        template: Optional[str] = None,
        name: str = "PromptNode"
    ) -> "RAGWorkflowBuilder":
        """
        添加提示词节点

        Args:
            template: 自定义提示词模板（可选）
                     默认使用内置模板
                     可用变量: {question}, {context}
            name: 节点名称

        Returns:
            self，支持链式调用
        """
        node = PromptNode(template=template, name=name)
        self.graph.add_node(node)
        self.nodes[name] = node

        # 连接到上一个节点
        if self.last_node_name:
            self.graph.add_edge(self.last_node_name, name)
        self.last_node_name = name

        return self

    def add_generation(
        self,
        llm_model: BaseLLM,
        name: str = "GenerationNode"
    ) -> "RAGWorkflowBuilder":
        """
        添加生成节点

        Args:
            llm_model: LLM 模型实例
            name: 节点名称

        Returns:
            self，支持链式调用
        """
        node = GenerationNode(llm_model=llm_model, name=name)
        self.graph.add_node(node)
        self.nodes[name] = node

        # 连接到上一个节点
        if self.last_node_name:
            self.graph.add_edge(self.last_node_name, name)
        self.last_node_name = name

        return self

    def add_node(self, node) -> "RAGWorkflowBuilder":
        """
        添加自定义节点

        Args:
            node: 自定义节点实例（继承自 Node）

        Returns:
            self，支持链式调用
        """
        self.graph.add_node(node)
        self.nodes[node.name] = node

        # 连接到上一个节点
        if self.last_node_name:
            self.graph.add_edge(self.last_node_name, node.name)
        self.last_node_name = node.name

        return self

    def connect(self, from_node: str, to_node: str) -> "RAGWorkflowBuilder":
        """
        手动连接两个节点

        Args:
            from_node: 源节点名称
            to_node: 目标节点名称

        Returns:
            self，支持链式调用
        """
        if from_node not in self.nodes:
            raise ValueError(f"节点 '{from_node}' 不存在")
        if to_node not in self.nodes:
            raise ValueError(f"节点 '{to_node}' 不存在")

        self.graph.add_edge(from_node, to_node)
        return self

    def build(self) -> Graph:
        """
        构建完整的 RAG 工作流

        Returns:
            Graph 实例
        """
        if not self.nodes:
            raise ValueError("工作流中没有添加任何节点")

        # 验证图结构
        try:
            self.graph.topological_sort()
        except ValueError as e:
            raise RuntimeError(f"工作流构建失败: {e}")

        return self.graph

    def execute(self, query: str) -> dict:
        """
        快速执行 RAG 工作流

        Args:
            query: 用户查询文本

        Returns:
            工作流执行结果
        """
        # 构建工作流
        graph = self.build()

        # 准备初始输入
        # 假设第一个节点是 QueryNode
        initial_inputs = {}
        for node_name in self.nodes:
            if isinstance(self.nodes[node_name], QueryNode):
                initial_inputs[node_name] = {"text": query}
                break

        if not initial_inputs:
            raise ValueError("工作流中没有 QueryNode，请先调用 add_query()")

        # 执行工作流
        return graph.execute_workflow(initial_inputs)

    def visualize(self, save_path: str) -> None:
        """
        生成工作流可视化图

        Args:
            save_path: DOT 文件保存路径
        """
        self.graph.save_dot(save_path)
        print(f"工作流可视化已保存到: {save_path}")
        print(f"使用 Graphviz 生成图片: dot -Tpng {save_path} -o workflow.png")

    def get_graph(self) -> Graph:
        """
        获取构建的图对象

        Returns:
            Graph 实例
        """
        return self.graph

