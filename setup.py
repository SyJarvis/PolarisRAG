# -*- coding: utf-8 -*-
import setuptools
from setuptools import find_packages
from pathlib import Path

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setuptools.setup(
    name="PolarisRAG",
    version="0.0.1",
    description="PolarisRAG",
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email="1755115828@qq.com",
    url="https://github.com/SyJarvis/PolarisRAG",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=1.6.0",
        "transformers==4.44.2",
        "datasets==2.19.0",
        "accelerate>=0.20.1",
        "sentence_transformers",
        "langchain"
    ],
    extras_require={
        "finetune": ["deepspeed", "flash-attn"]
    }
)