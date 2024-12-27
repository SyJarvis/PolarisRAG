# -*- coding: utf-8 -*-

# from zhipuai import ZhipuAI
# client = ZhipuAI(api_key="537ebcdda2a19ff9ed97b57fd17d2d41.PUYzSSFzOrUHyvJH") # 填写您自己的APIKey
# response = client.chat.completions.create(
#     model="glm-4-plus",  # 填写需要调用的模型编码
#     messages=[
#         {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
#         {"role": "user", "content": "农夫需要把狼、羊和白菜都带过河，但每次只能带一样物品，而且狼和羊不能单独相处，羊和白菜也不能单独相处，问农夫该如何过河。"}
#     ],
# )
# print(response.choices[0].message)


from polarisrag.llm import ZhipuLLM

model = ZhipuLLM(api_key="537ebcdda2a19ff9ed97b57fd17d2d41.PUYzSSFzOrUHyvJH",
                  model="glm-4-air",
                  is_memory=True)

model.chat("我的名字叫皮卡丘17")
print(model.chat("中国最高的山是哪一座山"))
print("++++++++++++++++++++++++++++++++++++++++++")
print(model.chat("你知道我叫什么名字吗？"))
print("++++++++++++++++++++++++++++++++++++++++++")
print(model.chat("刚才问的最后一个问题是什么？"))
print("++++++++++++++++++++++++++++++++++++++++++")
model.stream("中国的四大名山是哪几座山？")
