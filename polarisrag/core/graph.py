"""
图结构核心实现
"""
from collections import deque, defaultdict
from polarisrag.core.node import Node
from typing import Dict, Any

import logging
# 配置一个基础的 logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class Graph:

    def __init__(self):
        self.nodes = {}
        self.graph = defaultdict(list)
        # self.in_degree = defaultdict(int)     # 不作为核心状态，在需要时计算
        self.node_outputs = {}

    def add_node(self, node: Node):
        if node.name in self.nodes:
            raise ValueError(f"节点 '{node.name}' 已经存在")
        self.nodes[node.name] = node
        logging.info(f"节点 '{node.name}' 已添加")

    def add_edge(self, from_node_name: str, to_node_name: str):
        """
        显式地添加一条边，表示一个依赖关系
        """
        if from_node_name not in self.nodes or to_node_name not in self.nodes:
            raise ValueError("添加边时节点不存在")
        
        if to_node_name in self.graph[from_node_name]:
            return
        self.graph[from_node_name].append(to_node_name)
        # self.in_degree[to_node_name] += 1
        to_node = self.nodes[to_node_name]
        to_node.add_upstream_node_name(from_node_name)
        logging.info(f"依赖已添加: {from_node_name}->{to_node_name}")

    def _get_in_degrees(self) -> Dict[str, int]:
        """
        动态计算所有节点的入度
        """
        in_degree = {name: 0 for name in self.nodes}
        for node_name in self.graph:
            for neighbor_name in self.graph[node_name]:
                in_degree[neighbor_name] += 1
        return in_degree

    def topological_sort(self):
        """
        使用kahn算法进行拓扑排序，
        返回一个有效的节点执行顺序
        """
        in_degree = self._get_in_degrees()
        queue = deque()
        sorted_nodes = []

        # 找到所有入度为0的节点
        for name, degree in in_degree.items():
            if degree == 0:
                queue.append(name)

        while queue:
            node_name = queue.popleft()
            sorted_nodes.append(node_name)

            # 减少所有邻接节点的入度
            for neighbor_name in self.graph.get(node_name, []):
                in_degree[neighbor_name] -= 1
                if in_degree[neighbor_name] == 0:
                    queue.append(neighbor_name)

        if len(sorted_nodes) != len(self.nodes):
            # 找到循环依赖中的节点以提供更详细的错误信息
            cycle_nodes = set(self.nodes.keys()) - set(sorted_nodes)
            raise ValueError(f"图中存在循环依赖，无法执行。涉及节点: {cycle_nodes}")
        
        logging.info(f"拓扑排序完成，执行顺序: {sorted_nodes}")
        return sorted_nodes
    
    def execute_workflow(self, initial_inputs: Dict[str, Any]):
        """自动执行工作流"""
        logging.info("--- 开始执行工作流 ---")
        if initial_inputs is None:
            initial_inputs = {}

        node_outputs: Dict[str, Any] = {}   # 记录节点输出字典

        try:
            execution_order = self.topological_sort()
        except ValueError as e:
            logging.info(f"工作流执行失败: {e}")
            return None
        
        for node_name in execution_order:
            node = self.nodes[node_name]
            upstream_node_names = node.upstream_node_names
            if not upstream_node_names:
                # 起始节点，从 initial_inputs 获取输入
                inputs = initial_inputs.get(node_name, {})
            else:
                # 非起始节点
                inputs = {
                    upstream_name: node_outputs[upstream_name] 
                    for upstream_name in upstream_node_names
                }
            logging.info(f"正在执行节点 '{node_name}'...")
            node_output = node.execute(inputs)
            node_outputs[node_name] = node_output
            logging.info(f"节点 '{node_name}' 执行完成。")
        logging.info("--- 工作流执行完毕 ---")
        
        # 返回最后一个节点的输出
        if execution_order:
            last_node_name = execution_order[-1]
            return node_outputs.get(last_node_name)
        return None

    def chain(self, *nodes):
        """
        通过链式调用将多个节点连接起来
        """
        if not nodes:
            return
        
        for node in nodes:
            self.add_node(node)
        
        # 依次连接节点
        for i in range(len(nodes) - 1):
            from_node = nodes[i]
            to_node = nodes[i+1]
            self.add_edge(from_node.name, to_node.name)

        return self
    
    def to_dot(self) -> str:
        """
        生成图的DOT语言表示，用于可视化
        """
        logging.info("正在生成DOT语言描述...")
        dot_lines = [
            "digraph G {",
            "   rankdir=LR; //设置布局方向从左到右",
            "   node [shape=box, style\"rounded,filled\", fillcolor=\"#E3F2FD\", fontname=\"Arial\"];",
            "   edge [fontname=\"Arial\"];"
        ]

        # 定义所有节点
        for node_name in self.nodes:
            dot_lines.append(f'    "{node_name}";')
        
        # 定义所有边
        for from_node, to_nodes in self.graph.items():
            for to_node in to_nodes:
                dot_lines.append(f'    "{from_node}" -> "{to_node}";')
        dot_lines.append("}")
        return "\n".join(dot_lines)
    
    def save_dot(self, file_path):
        dot_text = self.to_dot()
        with open(file_path, 'w') as o:
            o.write(dot_text)