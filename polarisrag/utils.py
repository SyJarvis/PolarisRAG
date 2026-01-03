# -*- coding: utf-8 -*-
"""
工具函数和数据加载
"""
import re
from dataclasses import dataclass, field
import numpy as np
import os
from PyPDF2 import PdfReader
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import yaml


@dataclass
class FolderLoader:
    """
    文件夹加载器

    支持加载指定文件夹中的所有文档，并进行文本切分
    """

    folder_path: str = field(
        default_factory=lambda: str
    )

    chunk_size: int = field(
        default_factory=lambda: 1000
    )

    chunk_overlap: int = field(
        default_factory=lambda: 200
    )

    file_path_list: List[str] = field(
        default_factory=list
    )

    file_ext_dict: Dict = field(
        default_factory=dict
    )

    ext_names = ["md", "txt", "pdf"]

    def set_folder_path(self, folder_path: str):
        if not os.path.exists(folder_path) and len(os.listdir(folder_path)) < 1:
            raise ValueError("Folder path does not exist or is empty")
        self.folder_path = folder_path

    def get_all_chunk_content(self, folder_path: str = None, max_len: int = 600, cover_len: int = 150):
        """
        获取所有文档的切分内容

        Args:
            folder_path: 文件夹路径
            max_len: 每个文档的最大长度
            cover_len: 切分重叠长度

        Returns:
            文档片段列表
        """
        docs = []
        if folder_path is not None:
            if isinstance(self.file_path_list, list):
                self.file_path_list.append(self.__file_list(folder_path))
            else:
                self.file_path_list = self.__file_list(folder_path)
        else:
            for ext, file_list in self.file_ext_dict.items():
                doc_list = self.read_file_content(ext, file_list)
                # 返回文件内容列表，现在要对文件内容进行切分
                for content in doc_list:
                    chunks = self.split_documents(content)
                    docs.extend(chunks)
        return docs

    def _split_text_by_length(self, text: str, length = 100):
        """
        按长度切分文本
        """
        chunks = []
        lines = text.split("\n")
        content = ''
        for line in lines:
            line = line.replace(" ", "")
            line = line.strip()
            if len(content) < length:
                content += line
            else:
                chunks.append(content)
                content = ''
        return chunks

    def split_documents(cls, text: str, chunk_size: int = 100) -> List[str]:
        """
        切分文档
        """
        chunks = cls._split_text_by_length(text, chunk_size)
        return chunks

    def read_file_content(self, ext, file_list):
        """
        读取文件内容
        """
        doc_list = []
        if ext in self.ext_func_dict.keys():
            file_func = self.ext_func_dict[ext]
            for file_path in file_list:
                content = file_func(file_path)
                doc_list.append(content)
        # 文件内容列表
        return doc_list

    def read_pdf(self, file_path):
        """
        读取 PDF 文件
        """
        reader = PdfReader(file_path)
        text_content = []
        for page in reader.pages:
            text_content.append(page.extract_text())
        return "\n".join(text_content)

    def read_txt(self, file_path):
        """
        读取文本文件
        """
        with open(file_path, "r", encoding="utf-8") as f:
            docs = f.read()
        return docs

    def read_md_file(self, file_path):
        """
        读取 Markdown 文件
        """
        docs = self.read_txt(file_path)
        return docs

    def __post_init__(self):
        if not os.path.exists(self.folder_path):
            raise Exception("folder not exist")
        self.file_path_list = self.__file_list(self.folder_path)
        self.ext_func_dict = {
            "pdf": self.read_pdf,
            "txt": self.read_txt,
            "md": self.read_md_file,

        }
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                            chunk_overlap=self.chunk_overlap)

    def __file_list(self, folder_path=None):
        """
        获取文件列表
        """
        file_list = []
        file_dict = {}
        for file_path, dir_names, file_names in os.walk(folder_path):
            for file_name in file_names:
                ext_name = file_name.split(".")[-1]
                file_abs_path = os.path.join(file_path, file_name)
                try:
                    file_dict[ext_name].append(file_abs_path)
                except KeyError:
                    file_dict[ext_name] = [file_abs_path]
                file_list.append(file_abs_path)
        self.file_lidt = file_list
        self.file_ext_dict = file_dict
        return file_list


def load_json(file_name):
    """
    加载 JSON 文件
    """
    if not os.path.exists(file_name):
        return None
    try:
        with open(file_name, "r", encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            else:
                return None
    except json.JSONDecodeError:
        return None


def write_json(json_obj, file_name):
    """
    写入 JSON 文件
    """
    with open(file_name, "w", encoding='utf-8') as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def load_yaml(file_name: str) -> dict:
    """
    加载 YAML 文件
    """
    if not os.path.exists(file_name):
        return None
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
            else:
                return None
    except yaml.YAMLError as e:
        return None
