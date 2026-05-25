from dotenv import load_dotenv
import os
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage

load_dotenv()

documents = [
    "Python是一种简单易学的编程语言，广泛用于数据科学和人工智能领域。",
    "Java是一种面向对象的编程语言，以'一次编写，到处运行'著称，广泛用于企业开发。",
    "机器学习是人工智能的一个子领域，让计算机从数据中自动学习规律。",
    "深度学习使用多层神经网络处理复杂任务，在图像识别和自然语言处理上表现优异。",
    "RAG是检索增强生成技术，结合了检索系统和大语言模型，让AI能回答特定领域的问题。",
]

print(f"知识库共{len(documents)}段文本")
print("第一段：", documents[0])


# 初始化embedding模型
embeddings_model = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model="BAAI/bge-m3"
)

# 把每段文本变成向量，存在列表里
print("\n正在向量化知识库...")
doc_vectors = embeddings_model.embed_documents(documents)
print(f"向量化完成，每个向量维度: {len(doc_vectors[0])}")

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度，值越接近1表示越相似"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve(query, doc_vectors, documents, top_k=2):
    """输入问题，返回最相关的top_k段文本"""
    # 把问题也变成向量
    query_vector = embeddings_model.embed_query(query)
    
    # 计算问题和每段文本的相似度
    similarities = []
    for i, doc_vec in enumerate(doc_vectors):
        score = cosine_similarity(query_vector, doc_vec)
        similarities.append((score, i, documents[i]))
    
    # 按相似度从高到低排序
    similarities.sort(reverse=True)
    
    # 返回最相关的top_k段
    return similarities[:top_k]

# 测试检索
query = "什么是深度学习？"
results = retrieve(query, doc_vectors, documents)

print(f"\n问题：{query}")
print("检索到的相关文本：")
for score, idx, text in results:
    print(f"  相似度 {score:.4f}: {text}")



llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

def rag_answer(query):
    """完整的RAG流程：检索 + 生成"""
    # 第一步：检索相关文本
    relevant_docs = retrieve(query, doc_vectors, documents, top_k=2)
    context = "\n".join([text for _, _, text in relevant_docs])
    
    # 第二步：把检索到的文本塞进Prompt
    messages = [
        SystemMessage(content="""你是一个问答助手。
根据下面提供的参考资料回答用户问题。
如果参考资料里没有相关信息，就说"我在知识库中没有找到相关信息"。

参考资料：
""" + context),
        HumanMessage(content=query)
    ]
    
    # 第三步：LLM生成答案
    response = llm.invoke(messages)
    
    return {
        "answer": response.content,
        "sources": [text for _, _, text in relevant_docs]
    }

# 测试完整RAG
print("\n" + "="*50)
print("测试完整RAG流程")
print("="*50)

test_questions = [
    "什么是深度学习？",
    "Java和Python有什么区别？",
    "今天天气怎么样？",  # 知识库里没有的内容
]

for q in test_questions:
    print(f"\n问题：{q}")
    result = rag_answer(q)
    print(f"答案：{result['answer']}")
    print(f"参考来源：{result['sources'][0][:30]}...")