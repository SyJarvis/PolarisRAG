# -*- coding: utf-8 -*-
import setuptools
from setuptools import find_packages
from pathlib import Path

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setuptools.setup(
    name="PolarisRAG",
    version="0.2.0",
    description="PolarisRAG",
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email="1755115828@qq.com",
    url="https://github.com/SyJarvis/PolarisRAG",
    packages=find_packages(exclude=("tests", "tests.*", "mcp_server", "mcp_server.*")),
    include_package_data=True,
    install_requires=[
        "langchain",
        "langchain-openai",
        "langchain-text-splitters",
        "transformers>=4.44.2,<5",
        "pymilvus[milvus_lite]",
        "PyPDF2",
        "numpy",
        "PyYAML",
        "tqdm",
        "python-dotenv",
    ],
    extras_require={
        # 本地 HF Embedding（离线/私有化部署）才需要 torch
        "local-embedding": ["torch>=2.0.0", "sentence_transformers", "accelerate", "datasets"],
        # MCP 服务器（mcp_server/）
        "mcp": ["mcp==2.2.0"],
        # Web UI
        "serve": ["gradio<6"],
        # REST API（契约实现，serve/api.py）
        "api": ["fastapi>=0.110", "uvicorn>=0.29"],
    },
)
