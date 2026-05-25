from dotenv import load_dotenv
import os
import numpy as np
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings_model = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model="BAAI/bge-m3"
)

def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 实验：语义相近的句子，向量距离真的更近吗？
test_pairs = [
    ("猫是一种动物", "狗是一种动物"),      # 语义相近
    ("猫是一种动物", "今天天气很好"),       # 语义不相关
    ("机器学习", "人工智能"),              # 相关概念
    ("机器学习", "做饭食谱"),             # 无关概念
    ("我很开心", "我非常高兴"),            # 近义表达
]

print("语义相似度实验：")
print("-" * 60)
for text1, text2 in test_pairs:
    vec1 = embeddings_model.embed_query(text1)
    vec2 = embeddings_model.embed_query(text2)
    score = cosine_similarity(vec1, vec2)
    print(f"{text1!r:20} vs {text2!r:20}  相似度: {score:.4f}")