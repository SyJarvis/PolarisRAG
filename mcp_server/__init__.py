# -*- coding: utf-8 -*-
"""PolarisRAG MCP Server 包。

模块结构：
- config: 环境变量读取与启动校验
- server: MCPServer 注册 tools/resources + stdio 入口
- agent: 决策 LLM 工具循环 + 内部工具
- ingest: fast/smart 整理入库
- registry: source registry 边车
- ids: ID 生成
"""
