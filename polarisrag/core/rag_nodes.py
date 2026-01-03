# -*- coding: utf-8 -*-
"""
RAG 工作流节点实现
"""
from typing import Dict, Any, List, Optional
from polarisrag.core.node import Node
from polarisrag.base import BaseEmbedding, BaseVectorDB, BaseLLM


class QueryNode(Node):
    """
    查询节点 - 处理用户查询输入
    """

    def __init__(self, name: str = "QueryNode"):
        super().__init__(name)

    def execute(self, inputs: dict) -> dict:
        """
        处理用户查询

        Args:
            inputs: 应包含 {"text": str}

        Returns:
            {"query": str}
        """
        self.logger.info(f"节点 '{self.name}' 开始处理查询")

        # 验证输入
        if "text" not in inputs:
            raise ValueError(f"节点 '{self.name}' 需要 'text' 输入")

        text = inputs["text"]
        if not isinstance(text, str):
            raise TypeError(f"节点 '{self.name}' 的 'text' 必须是字符串类型")

        if not text.strip():
            raise ValueError(f"节点 '{self.name}' 的查询文本不能为空")

        self.output = {"query": text}
        self.logger.info(f"查询文本: {text[:50]}...")

        return self.output


class EmbeddingNode(Node):
    """
    嵌入节点 - 将文本转换为向量
    """

    def __init__(self, embedding_model: BaseEmbedding, name: str = "EmbeddingNode"):
        super().__init__(name)
        self.embedding_model = embedding_model

    def execute(self, inputs: dict) -> dict:
        """
        文本向量化

        Args:
            inputs: 可以是 {"text": str} 或 {"query": str} 或 {"embedding": List[float]}

        Returns:
            {"embedding": List[float]}
        """
        self.logger.info(f"节点 '{self.name}' 开始嵌入文本")

        # 如果输入已经是向量，直接返回
        if "embedding" in inputs:
            embedding = inputs["embedding"]
            if not isinstance(embedding, list):
                raise TypeError(f"节点 '{self.name}' 的 'embedding' 必须是列表类型")
            self.output = {"embedding": embedding}
            self.logger.info(f"使用输入的向量，维度: {len(embedding)}")
            return self.output

        # 获取文本
        text = inputs.get("text") or inputs.get("query")
        if text is None:
            raise ValueError(f"节点 '{self.name}' 需要 'text' 或 'query' 或 'embedding' 输入")

        if not isinstance(text, str):
            raise TypeError(f"节点 '{self.name}' 的文本必须是字符串类型")

        # 生成嵌入
        try:
            embedding = self.embedding_model.embed_text(text)
            self.output = {"embedding": embedding}
            self.logger.info(f"嵌入生成完成，维度: {len(embedding)}")
            return self.output
        except Exception as e:
            self.logger.error(f"嵌入生成失败: {e}")
            raise RuntimeError(f"节点 '{self.name}' 嵌入生成失败: {e}")


class RetrievalNode(Node):
    """
    检索节点 - 从向量库检索相关文档
    """

    def __init__(self, vector_db: BaseVectorDB, top_k: int = 3, name: str = "RetrievalNode"):
        super().__init__(name)
        self.vector_db = vector_db
        self.top_k = top_k

    def execute(self, inputs: dict) -> dict:
        """
        向量检索

        Args:
            inputs: 可以是 {"query": str} 或 {"embedding": List[float]}

        Returns:
            {"context": str, "documents": List[str]}
        """
        self.logger.info(f"节点 '{self.name}' 开始检索")

        # 获取查询
        if "query" in inputs:
            query = inputs["query"]
            if not isinstance(query, str):
                raise TypeError(f"节点 '{self.name}' 的 'query' 必须是字符串类型")
        else:
            raise ValueError(f"节点 '{self.name}' 需要 'query' 输入")

        # 执行检索
        try:
            context = self.vector_db.query(query, limit=self.top_k)
            documents = [doc.strip() for doc in context.split("\n") if doc.strip()]

            if not documents:
                self.logger.warning(f"未检索到相关文档")
                context = "未找到相关信息"
                documents = []
            else:
                self.logger.info(f"检索到 {len(documents)} 个文档片段")

            self.output = {
                "context": context,
                "documents": documents,
                "query": query  # 保留查询供后续节点使用
            }
            return self.output

        except Exception as e:
            self.logger.error(f"检索失败: {e}")
            raise RuntimeError(f"节点 '{self.name}' 检索失败: {e}")


class PromptNode(Node):
    """
    提示词节点 - 构建提示词模板
    """

    def __init__(self, template: Optional[str] = None, name: str = "PromptNode"):
        super().__init__(name)
        self.template = template

    def execute(self, inputs: dict) -> dict:
        """
        构建提示词

        Args:
            inputs: 应包含 {"query": str, "context": str}

        Returns:
            {"prompt": str}
        """
        self.logger.info(f"节点 '{self.name}' 开始构建提示词")

        # 验证输入
        if "query" not in inputs:
            raise ValueError(f"节点 '{self.name}' 需要 'query' 输入")
        if "context" not in inputs:
            raise ValueError(f"节点 '{self.name}' 需要 'context' 输入")

        query = inputs["query"]
        context = inputs["context"]

        if not isinstance(query, str):
            raise TypeError(f"节点 '{self.name}' 的 'query' 必须是字符串类型")
        if not isinstance(context, str):
            raise TypeError(f"节点 '{self.name}' 的 'context' 必须是字符串类型")

        # 使用自定义模板或默认模板
        template = self.template
        if template is None:
            template = """使用以下上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。

问题: <question>{question}</question>

可参考的上下文：
···
<context>{context}</context>
···

如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。

有用的回答:"""

        # 构建提示词
        try:
            prompt = template.format(question=query, context=context)
            self.output = {"prompt": prompt}
            self.logger.info(f"提示词构建完成，长度: {len(prompt)} 字符")
            return self.output
        except Exception as e:
            self.logger.error(f"提示词构建失败: {e}")
            raise RuntimeError(f"节点 '{self.name}' 提示词构建失败: {e}")


class GenerationNode(Node):
    """
    生成节点 - 使用 LLM 生成答案
    """

    def __init__(self, llm_model: BaseLLM, name: str = "GenerationNode"):
        super().__init__(name)
        self.llm_model = llm_model

    def execute(self, inputs: dict) -> dict:
        """
        LLM 生成答案

        Args:
            inputs: 可以是 {"prompt": str} 或 {"query": str, "context": str}

        Returns:
            {"answer": str}
        """
        self.logger.info(f"节点 '{self.name}' 开始生成答案")

        # 获取输入
        prompt = inputs.get("prompt")
        if prompt:
            # 直接使用提供的提示词
            query_text = prompt
        elif "query" in inputs and "context" in inputs:
            # 组合查询和上下文
            query = inputs["query"]
            context = inputs["context"]
            template = """使用以下上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。

问题: <question>{question}</question>

可参考的上下文：
···
<context>{context}</context>
···

如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。

有用的回答:"""
            query_text = template.format(question=query, context=context)
        else:
            raise ValueError(f"节点 '{self.name}' 需要 'prompt' 或 ('query' + 'context') 输入")

        if not isinstance(query_text, str):
            raise TypeError(f"节点 '{self.name}' 的输入必须是字符串类型")

        # 生成答案
        try:
            answer = self.llm_model.chat(query_text)
            self.output = {"answer": answer}
            self.logger.info(f"答案生成完成，长度: {len(answer)} 字符")
            return self.output
        except Exception as e:
            self.logger.error(f"答案生成失败: {e}")
            raise RuntimeError(f"节点 '{self.name}' 答案生成失败: {e}")


class RerankNode(Node):
    """
    重排序节点 - 对检索结果进行重新排序（高级功能）
    """

    def __init__(self, rerank_model=None, top_k: int = 3, name: str = "RerankNode"):
        super().__init__(name)
        self.rerank_model = rerank_model
        self.top_k = top_k

    def execute(self, inputs: dict) -> dict:
        """
        重排序检索结果

        Args:
            inputs: 应包含 {"documents": List[str], "query": str}

        Returns:
            {"context": str, "documents": List[str]}
        """
        self.logger.info(f"节点 '{self.name}' 开始重排序")

        # 验证输入
        if "documents" not in inputs:
            raise ValueError(f"节点 '{self.name}' 需要 'documents' 输入")
        if "query" not in inputs:
            raise ValueError(f"节点 '{self.name}' 需要 'query' 输入")

        documents = inputs["documents"]
        query = inputs["query"]

        if not documents:
            self.logger.warning(f"没有文档需要重排序")
            self.output = {
                "context": inputs.get("context", ""),
                "documents": [],
                "query": query
            }
            return self.output

        # 如果没有重排序模型，直接返回
        if self.rerank_model is None:
            self.logger.info(f"未提供重排序模型，返回原结果")
            top_documents = documents[:self.top_k]
            context = "\n".join(top_documents)
            self.output = {
                "context": context,
                "documents": top_documents,
                "query": query
            }
            return self.output

        # TODO: 实现实际的重排序逻辑
        # 这里可以集成各种重排序模型，如 Cohere Rerank、BGE Reranker 等
        self.logger.info(f"重排序功能待实现")
        self.output = {
            "context": inputs.get("context", ""),
            "documents": documents[:self.top_k],
            "query": query
        }
        return self.output

