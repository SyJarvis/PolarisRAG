from abc import ABC
from collections import deque, defaultdict
import logging

class Node(ABC):
    def __init__(self, name: str):
        self.name = name
        self.output = None
        self.upstream_node_names = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_upstream_node_name(self, upstream_node_name: str):
        """记录上游节点的名称"""
        self.upstream_node_names.append(upstream_node_name)

    def execute(self, inputs: dict):
        """
        抽象方法
        """
        raise NotImplementedError("you must be use")
    
    async def aexecute(self, inputs:dict):
        """
        抽象方法，异步执行
        """
        raise NotImplementedError("you must be use")
    

class UserInputNode(Node):

    def __init__(self, name: str = "UserInputNode"):
        super().__init__(name)

    def execute(self, inputs: dict):
        # For the workflow, we'll pass through the inputs as is
        # The actual user input should be in the initial_inputs
        self.output = inputs
        return self.output