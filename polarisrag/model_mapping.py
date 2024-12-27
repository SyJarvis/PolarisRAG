# -*- coding: utf-8 -*-
from enum import Enum
from typing import (
    Type,
    Any,
)
from dataclasses import dataclass
from collections import OrderedDict

from .embedding import (
    ZhipuEmbedding
)


@dataclass
class EmbedderConfig:
    model_class: Type[Any]
    model_type: str
    pretrained_dir: str = ""
    api_key: str = ""


@dataclass
class LLMConfig:
    model_class: Type[Any]

AUTO_LLM_MAPPING = OrderedDict([

])


AUTO_EMBEDDER_MAPPING = OrderedDict([
    (
        "zhipu-embedding",
        EmbedderConfig(ZhipuEmbedding, )
    ),
    (

    )
])

AUTO_STORAGE_MAPPING = OrderedDict([

])