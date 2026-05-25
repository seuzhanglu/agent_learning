from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model="BAAI/bge-m3"  # 硅基流动上的免费embedding模型
)

# 把几句话变成向量
texts = [
    "猫是一种可爱的动物",
    "狗是人类的好朋友",
    "汽车是一种交通工具",
    "我喜欢吃苹果",
]

vectors = embeddings.embed_documents(texts)

print(f"向量数量: {len(vectors)}")
print(f"每个向量的维度: {len(vectors[0])}")
print(f"第一个向量的前5个数字: {vectors[0][:5]}")