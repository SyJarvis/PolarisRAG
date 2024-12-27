# -*- coding: utf-8 -*-
import re
from dataclasses import dataclass, field
import numpy as np
import os
from PyPDF2 import PdfReader
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json


@dataclass
class FolderLoader:

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

    def get_all_chunk_content(self, max_len:int=600, cover_len:int=150):
        docs = []
        for ext, file_list in self.file_ext_dict.items():
            doc_list = self.read_file_content(ext, file_list)
            # 返回文件内容列表，现在要对文件内容进行切分
            for content in doc_list:
                chunks = self.split_documents(content)
                docs.extend(chunks)
        return docs

    def split_documents(cls, text: str, chunk_size: int=1000) -> List[str]:
        numbered_line_pattern = re.compile(r'^\d+\.[\d, \s]*')
        chunks = []
        lines = text.split("\n")
        content = ''
        for line in lines:
            if numbered_line_pattern.match(line):
                if content == '':
                    content += line
                else:
                    chunks.append(content)
                    content = line
            else:
                content += line

        return chunks

    def read_file_content(self, ext, file_list):
        doc_list = []
        if ext in self.ext_func_dict.keys():
            file_func = self.ext_func_dict[ext]
            for file_path in file_list:
                content = file_func(file_path)
                doc_list.append(content)
        # 文件内容列表
        return doc_list

    def read_pdf(self, file_path):
        reader = PdfReader(file_path)
        text_content = []
        for page in reader.pages:
            text_content.append(page.extract_text())
        return "\n".join(text_content)

    def __post_init__(self):
        if not os.path.exists(self.folder_path):
            raise Exception("folder not exist")
        self.file_path_list = self.__file_list(self.folder_path)
        self.ext_func_dict = {
            "pdf": self.read_pdf
        }
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                            chunk_overlap=self.chunk_overlap)

    def __file_list(self, folder_path=None):
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
    if not os.path.exists(file_name):
        return None
    with open(file_name, "r", encoding='utf-8') as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)