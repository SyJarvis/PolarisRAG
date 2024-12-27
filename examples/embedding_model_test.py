# -*- coding: utf-8 -*-

from polarisrag.embedding import HFEmbedding


embedding_model = HFEmbedding.from_pretrained("/mnt/d/huggingface_model/m3e-base",
                                              )
vectors = embedding_model.embed_text("你好，世界", padding=True, truncation=True, return_tensors='pt')
print(len(vectors))
print(vectors)
