# -*- coding: utf-8 -*-

# from polarisrag.embedding import EmbeddingFactory
#
# embedding_model = EmbeddingFactory.create_embedding("zhipuembedding",
#                                                     api_key="123")
# print(embedding_model)

ZHIPUAI_API_KEY="537ebcdda2a19ff9ed97b57fd17d2d41.PUYzSSFzOrUHyvJH"
from polarisrag.factory import EmbeddingFactory
embedding_factory = EmbeddingFactory().run()
embedding_model = embedding_factory.create("zhipuembedding", api_key=ZHIPUAI_API_KEY)

vectors = embedding_model.embed_text("你好")
print(vectors)
print(len(vectors))